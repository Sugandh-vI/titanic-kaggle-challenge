#!/usr/bin/env python3
"""Titanic: Machine Learning from Disaster -- end-to-end ML pipeline.

What this script does
---------------------
Step 1  EDA          missing values, class balance, survival rates per feature.
Step 2  Features     Title, FamilySize/IsAlone, Deck, group-wise Age/Fare/Embarked
                     imputation, family & ticket survival-rate encoding, encodings.
Step 3  Modeling     RandomForest vs XGBoost (vs LightGBM if installed):
                     stratified 80/20 holdout + 5-fold CV, then RandomizedSearchCV
                     on the winner.
Step 4  Prediction   refit the winner on ALL of train.csv, predict test.csv and write
                     ``submission.csv`` (PassengerId, Survived; 418 rows + header).
Step 5  Reporting    feature importances + final repeated-CV accuracy.

Everything (feature engineering + imputation + encoding + model) lives inside a single
``sklearn.pipeline.Pipeline``, so all label-dependent statistics are refit on each CV
training fold only -- the reported accuracy is honest, and the final artefact is one
dumpable object (``output/titanic_pipeline.joblib``).

Usage
-----
    python titanic.py                      # full run (~4 min on 2 cores)
    python titanic.py --quick              # smaller search, fewer CV repeats
    python titanic.py --seed 7 --search-iters 80
    python titanic.py --data-dir dataset   # data/ and dataset/ are both auto-detected

Outputs
-------
    submission.csv                       <- upload this to Kaggle (repo root)
    output/eda_report.md                 <- Step 1 tables
    output/model_comparison.csv          <- Step 3 candidate scores
    output/feature_importance.csv        <- Step 5 importances
    output/submission.csv                <- copy of the submission
    output/titanic_pipeline.joblib       <- fitted pipeline (re-predict any time)

Results of the last full run (seed 42, 2 cores, ~6 min) -- see README.md for details
------------------------------------------------------------------------------------
    model comparison     RandomForest 0.8451 CV > XGBoost 0.8417 > LightGBM 0.8350
                         (holdout 80/20: 0.8212 / 0.7989 / 0.8045)
    chosen model         RandomForestClassifier(n_estimators=700, criterion="gini",
                         max_depth=6, min_samples_leaf=4, min_samples_split=3,
                         max_features=0.65, class_weight="balanced_subsample")
    local CV accuracy    0.8437 (5-fold x 4 repeats, s.e. 0.0059)
    out-of-fold accuracy 0.8384 (independent split)
    top 5 features       Title, Sex, Age_Class, FamilySurvivalRate, FarePerPerson
    submission.csv       418 rows, 169 predicted survivors (40.4%)
    baselines            "every woman survives" = 0.7868, "everyone dies" = 0.6162
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:  # XGBoost is the "stronger candidate" requested; sklearn's HGB is the fallback.
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier

    HAS_XGBOOST = False

try:  # optional third candidate
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

warnings.filterwarnings("ignore")

# ======================================================================================
# 0. Configuration -- one seed everywhere for reproducibility
# ======================================================================================
SEED = 42
REPO_ROOT = Path(__file__).resolve().parent
TARGET = "Survived"
N_FOLDS = 5
TEST_ROWS = 418

# Columns consumed by the ColumnTransformer at the end of the pipeline.
NUMERIC_FEATURES: List[str] = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "HasCabin", "TicketFreq",
    "FareLog", "FarePerPerson", "Age_Class", "IsChild", "IsMother",
    "FamilySurvivalRate", "FamilySurvivalKnown",
    "TicketSurvivalRate", "TicketSurvivalKnown",
]
CATEGORICAL_FEATURES: List[str] = ["Sex", "Embarked", "Title", "Deck", "FamilyGroup", "Sex_Pclass"]
# Raw columns that are dropped once their signal has been extracted.
DROP_RAW_COLUMNS: List[str] = ["PassengerId", "Name", "Ticket", "Cabin", "Surname", "TicketId", "FamilyKey"]

TITLE_RE = re.compile(r",\s*([^\.]+)\.")
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}   # French variants -> English ones
BASE_TITLES = ("Mr", "Mrs", "Miss", "Master")              # everything else collapses to "Rare"


def normalize_title(raw: pd.Series) -> pd.Series:
    """"Braund, Mr. Owen Harris" -> "Mr"; "Dr"/"Rev"/"the Countess"/... -> "Rare"."""
    extracted = raw.astype(str).str.extract(TITLE_RE, expand=False).str.strip()
    return extracted.map(
        lambda t: TITLE_MAP.get(t, t) if (t in BASE_TITLES or t in TITLE_MAP) else "Rare"
    )


def family_group(size: pd.Series) -> pd.Series:
    """Bucket FamilySize: solo travellers, small families, large (high-loss) families."""
    return pd.cut(size, [-1, 1, 4, 20], labels=["Alone", "Small", "Large"]).astype(str)


def _require_dataframe(X, who: str) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{who} expects a pandas DataFrame (got {type(X).__name__}) -- "
                        "the transformers work on named columns.")
    return X


# ======================================================================================
# 2. Custom transformers -- every label-dependent statistic is learned in fit() only
# ======================================================================================
class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Turn raw passenger rows into engineered columns.

    Creates
    -------
    Title       Mr / Mrs / Miss / Master / Rare, parsed from ``Name``
    Surname     family name (family group key; dropped before modelling)
    FamilySize  SibSp + Parch + 1
    IsAlone     FamilySize == 1
    FamilyGroup Alone / Small (2-4) / Large (5+)
    Deck        first letter of ``Cabin``, "U" when no cabin was recorded
    HasCabin    1 if a cabin was recorded at all
    TicketId    normalised ticket number (travelling-party key; dropped before modelling)
    TicketFreq  number of passengers in this frame sharing the ticket

    ``Name``, ``Ticket`` and ``Cabin`` are consumed and dropped here.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureExtractor":
        _require_dataframe(X, "FeatureExtractor")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = _require_dataframe(X, "FeatureExtractor").copy()

        # --- Title / family name ----------------------------------------------
        df["Title"] = normalize_title(df["Name"])
        df["Surname"] = df["Name"].astype(str).str.split(",").str[0].str.strip()

        # --- Family -------------------------------------------------------------
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FamilyGroup"] = family_group(df["FamilySize"])

        # --- Cabin -> Deck ------------------------------------------------------
        cabin = df["Cabin"].astype("string").str.strip()
        df["Deck"] = cabin.str.extract(r"([A-Za-z])", expand=False).str.upper().fillna("U")
        df["HasCabin"] = cabin.notna().astype(int)

        # --- Ticket -------------------------------------------------------------
        ticket = (
            df["Ticket"].astype(str).str.upper().str.strip()
            .str.replace(r"[./\-]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        df["TicketId"] = ticket
        df["TicketFreq"] = ticket.map(ticket.value_counts()).astype(int)

        return df.drop(columns=["Name", "Ticket", "Cabin"])


class TitanicImputer(BaseEstimator, TransformerMixin):
    """Group-aware imputation, as suggested by the EDA.

    Age       median of the (Title, Pclass) group -> Pclass median -> global median.
              A "Master" in 3rd class is a boy (~4y) and a 1st-class "Mrs" is ~40y,
              so the global median (28) would be badly wrong for both.
    Fare      median of the Pclass group. The only missing test fare belongs to a
              3rd-class passenger (median 8.05), not to the overall median (14.45).
    Embarked  most frequent port in the training data ("S").
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TitanicImputer":
        df = _require_dataframe(X, "TitanicImputer")
        self.age_by_group_: Dict[Tuple[str, int], float] = df.groupby(["Title", "Pclass"])["Age"].median().to_dict()
        self.age_by_class_: Dict[int, float] = df.groupby("Pclass")["Age"].median().to_dict()
        self.age_global_: float = float(df["Age"].median())
        self.fare_by_class_: Dict[int, float] = df.groupby("Pclass")["Fare"].median().to_dict()
        self.fare_global_: float = float(df["Fare"].median())
        self.embarked_mode_: str = str(df["Embarked"].mode(dropna=True).iloc[0])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = _require_dataframe(X, "TitanicImputer").copy()

        group_median = pd.Series(
            [self.age_by_group_.get((t, p), np.nan) for t, p in zip(df["Title"], df["Pclass"])],
            index=df.index,
        )
        df["Age"] = (
            df["Age"].fillna(group_median)
            .fillna(df["Pclass"].map(self.age_by_class_))
            .fillna(self.age_global_)
        )
        df["Fare"] = df["Fare"].fillna(df["Pclass"].map(self.fare_by_class_)).fillna(self.fare_global_)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        return df


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Interactions computed *after* imputation (they need a clean Age/Fare)."""

    def fit(self, X: pd.DataFrame, y=None) -> "DerivedFeatures":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = _require_dataframe(X, "DerivedFeatures").copy()
        df["FareLog"] = np.log1p(df["Fare"])
        df["FarePerPerson"] = df["Fare"] / df["TicketFreq"].clip(lower=1)
        df["Age_Class"] = df["Age"] * df["Pclass"]
        df["IsChild"] = (df["Age"] < 12).astype(int)
        df["IsMother"] = (
            (df["Sex"] == "female") & (df["Parch"] > 0) & (df["Age"] > 18) & (df["Title"] == "Mrs")
        ).astype(int)
        df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
        # same surname + same fare = same household travelling together
        df["FamilyKey"] = df["Surname"].astype(str) + "_" + df["Fare"].round(1).astype(str)
        return df


class GroupSurvivalEncoder(BaseEstimator, TransformerMixin):
    """Encode "did the rest of my family / ticket group survive?".

    Women and children were not rescued at random: whole households and travelling
    parties lived or died together, which is the strongest signal left after Sex and
    Pclass. Two group keys are used:

    * family = ``FamilyKey`` (Surname + rounded Fare)
    * ticket = ``TicketId`` (normalised ticket number, shared by a travelling party)

    Leakage control
    ---------------
    ``fit`` only ever sees training labels. Rows present in ``fit`` are re-encoded
    **leave-one-out** (a passenger's own label is removed from their group rate), so
    the training matrix never contains its own answer. Unseen rows (the validation
    fold, or test.csv) get their group's rate as computed from the training labels, or
    the global survival rate together with a ``...Known = 0`` flag when the group was
    never seen in training. Passengers with no group mate collapse onto that single
    global constant, so their encoded value carries no information about their own
    label -- both properties are pinned down by ``tests/test_pipeline.py``.
    """

    def __init__(self, family_col: str = "FamilyKey", ticket_col: str = "TicketId",
                 id_col: str = "PassengerId") -> None:
        self.family_col = family_col
        self.ticket_col = ticket_col
        self.id_col = id_col

    _GROUPS: Tuple[Tuple[str, str], ...] = (("Family", "family_col"), ("Ticket", "ticket_col"))

    def fit(self, X: pd.DataFrame, y=None) -> "GroupSurvivalEncoder":
        df = _require_dataframe(X, "GroupSurvivalEncoder")
        if y is None:
            raise ValueError("GroupSurvivalEncoder needs the target y in fit().")
        yv = np.asarray(y, dtype=float)
        self.global_rate_: float = float(yv.mean())
        self.seen_ids_: set = set(df[self.id_col].tolist())
        self.rate_: Dict[str, Dict[str, float]] = {}
        self.count_: Dict[str, Dict[str, int]] = {}
        self.loo_: Dict[str, Dict[object, float]] = {}

        for label, attr in self._GROUPS:
            keys = df[getattr(self, attr)].astype(str).tolist()
            stat = pd.DataFrame({"k": keys, "y": yv}).groupby("k")["y"].agg(["sum", "count"])
            sums = stat["sum"].to_dict()
            self.count_[label] = stat["count"].astype(int).to_dict()
            self.rate_[label] = (stat["sum"] / stat["count"]).to_dict()
            self.loo_[label] = {
                pid: ((sums[k] - yy) / (self.count_[label][k] - 1)
                      if self.count_[label][k] > 1
                      # Singleton groups carry no group information at all. They must all
                      # collapse onto ONE constant (the global training rate): a
                      # per-passenger value here -- even a "leave-one-out" one -- would
                      # encode the passenger's own label and let the trees memorise it.
                      else self.global_rate_)
                for pid, k, yy in zip(df[self.id_col].tolist(), keys, yv)
            }
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        src = _require_dataframe(X, "GroupSurvivalEncoder")
        df = src.drop(columns=[c for c in DROP_RAW_COLUMNS if c in src.columns])
        ids = src[self.id_col].tolist()

        for label, attr in self._GROUPS:
            keys = src[getattr(self, attr)].astype(str).tolist()
            rates: List[float] = []
            known: List[int] = []
            for pid, key in zip(ids, keys):
                if pid in self.seen_ids_:                    # fitted row -> leave-one-out rate
                    rates.append(self.loo_[label].get(pid, self.global_rate_))
                    known.append(int(self.count_[label].get(key, 1) > 1))
                else:                                        # unseen row -> fitted group rate
                    rates.append(self.rate_[label].get(key, self.global_rate_))
                    known.append(int(key in self.rate_[label]))
            df[f"{label}SurvivalRate"] = np.asarray(rates, dtype=float)
            df[f"{label}SurvivalKnown"] = np.asarray(known, dtype=int)
        return df


# ======================================================================================
# Pipeline assembly
# ======================================================================================
def build_pipeline(estimator, seed: int = SEED) -> Pipeline:
    """One reproducible object: raw csv columns in -> survival prediction out."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("extract", FeatureExtractor()),          # Title / Family / Deck / Ticket
            ("impute", TitanicImputer()),             # Age, Fare, Embarked (group-aware)
            ("derive", DerivedFeatures()),            # interactions + FamilyKey
            ("groups", GroupSurvivalEncoder()),       # family & ticket survival rates
            ("prep", preprocessor),                   # median + one-hot encoding
            ("clf", clone(estimator)),                # the model being compared
        ]
    )


def candidate_models(seed: int = SEED, n_jobs: int = 2) -> Dict[str, object]:
    """Sensible starting configs; the CV comparison below picks the winner."""
    if HAS_XGBOOST:
        booster = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.5, gamma=0.05,
            tree_method="hist", random_state=seed, n_jobs=n_jobs,
            eval_metric="logloss", verbosity=0,
        )
    else:  # pragma: no cover
        booster = XGBClassifier(
            max_iter=400, max_depth=3, learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=1.5, random_state=seed,
        )
    models: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=3, max_features=0.5,
            random_state=seed, n_jobs=n_jobs,
        ),
        "XGBoost": booster,
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=n_jobs, verbose=-1,
        )
    return models


def search_spaces() -> Dict[str, dict]:
    """Light, model-specific RandomizedSearchCV grids (keys prefixed with ``clf__``)."""
    xgb_space = (
        {
            "clf__n_estimators": randint(200, 900),
            "clf__max_depth": randint(2, 7),
            "clf__learning_rate": uniform(0.01, 0.19),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_weight": randint(1, 8),
            "clf__gamma": [0.0, 0.05, 0.1, 0.3],
            "clf__reg_lambda": uniform(0.0, 5.0),
            "clf__reg_alpha": uniform(0.0, 1.0),
        }
        if HAS_XGBOOST
        else {  # pragma: no cover
            "clf__max_iter": randint(200, 900),
            "clf__max_depth": randint(2, 7),
            "clf__learning_rate": uniform(0.01, 0.19),
            "clf__min_samples_leaf": randint(5, 60),
            "clf__max_leaf_nodes": randint(8, 64),
            "clf__l2_regularization": uniform(0.0, 5.0),
        }
    )
    spaces: Dict[str, dict] = {
        "RandomForest": {
            "clf__n_estimators": [400, 700, 1000],
            "clf__max_depth": [4, 5, 6, 7, 8, None],
            "clf__min_samples_leaf": randint(1, 6),
            "clf__min_samples_split": randint(2, 8),
            "clf__max_features": ["sqrt", 0.2, 0.3, 0.35, 0.5, 0.65, 0.8],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "XGBoost": xgb_space,
    }
    if LGBMClassifier is not None:
        spaces["LightGBM"] = {
            "clf__n_estimators": randint(200, 900),
            "clf__max_depth": randint(2, 7),
            "clf__num_leaves": randint(8, 64),
            "clf__learning_rate": uniform(0.01, 0.19),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__subsample_freq": [1],
            "clf__colsample_bytree": uniform(0.5, 0.5),
            "clf__min_child_samples": randint(5, 60),
            "clf__reg_lambda": uniform(0.0, 5.0),
        }
    return spaces


# ======================================================================================
# 1. Data loading + EDA
# ======================================================================================
def find_data_dir(explicit: str | None = None) -> Path:
    """Accept ``data/`` (this repo) or ``dataset/`` (the wording in the task brief)."""
    candidates = [Path(explicit)] if explicit else [REPO_ROOT / "data", REPO_ROOT / "dataset"]
    for cand in candidates:
        if (cand / "train.csv").exists() and (cand / "test.csv").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find train.csv/test.csv in any of: {[str(c) for c in candidates]}"
    )


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_path = data_dir / "gender_submission.csv"
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    return train, test, sample


def _survival_table(df: pd.DataFrame, by, sort: bool = True) -> pd.DataFrame:
    table = df.groupby(by, observed=True)[TARGET].agg(n="count", survival_rate="mean").round(4)
    return table.sort_values("survival_rate", ascending=False) if sort else table


def _md_table(header: Sequence[str], rows: Sequence[Sequence]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def eda_report(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Step 1: the numbers behind every feature-engineering decision in this script."""
    lines: List[str] = ["# Titanic -- exploratory data analysis", ""]
    lines += [
        f"* `train.csv`: {train.shape[0]} rows x {train.shape[1]} columns (labelled, target = `{TARGET}`)",
        f"* `test.csv`: {test.shape[0]} rows x {test.shape[1]} columns (to predict)",
        f"* `gender_submission.csv`: format reference only ({len(test)} rows) -- never used for training",
        "",
        "## Missing values",
        "",
    ]
    rows = []
    for col in train.columns:
        in_test = col in test.columns
        rows.append([
            col,
            int(train[col].isna().sum()), f"{100 * train[col].isna().mean():.2f}%",
            int(test[col].isna().sum()) if in_test else "-",
            f"{100 * test[col].isna().mean():.2f}%" if in_test else "-",
        ])
    lines += _md_table(["column", "train n", "train %", "test n", "test %"], rows)

    counts = train[TARGET].value_counts().sort_index()
    lines += [
        "## Class balance of `Survived`",
        "",
        f"* died (0): **{int(counts.get(0, 0))}** ({counts.get(0, 0) / len(train):.2%})",
        f"* survived (1): **{int(counts.get(1, 0))}** ({counts.get(1, 0) / len(train):.2%})",
        "* imbalance is mild, so plain accuracy is a fair metric; no resampling needed",
        f"* trivial baseline (always predict 0): accuracy {counts.get(0, 0) / len(train):.4f}",
        f"* `gender_submission.csv` rule (every woman survives): accuracy "
        f"{((train['Sex'] == 'female') == (train[TARGET] == 1)).mean():.4f} on train",
        "",
        "## Distributions vs survival",
        "",
    ]

    df = train.copy()
    df["Title"] = normalize_title(df["Name"])
    df["Deck"] = df["Cabin"].astype("string").str.extract(r"([A-Za-z])", expand=False).str.upper().fillna("U")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilyGroup"] = family_group(df["FamilySize"])

    for col in ["Sex", "Pclass", "Embarked", "Title", "Deck", "FamilySize", "FamilyGroup"]:
        lines += [f"### {col}", ""]
        lines += _md_table(
            [col.lower(), "n", "survival rate"],
            [[idx, int(r["n"]), f"{r['survival_rate']:.4f}"] for idx, r in _survival_table(df, col).iterrows()],
        )

    lines += ["### Age / Fare (numeric summaries)", ""]
    stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    desc = {
        "Age (train)": train["Age"].describe(), "Age (test)": test["Age"].describe(),
        "Fare (train)": train["Fare"].describe(), "Fare (test)": test["Fare"].describe(),
    }
    lines += _md_table(["stat"] + list(desc), [[s] + [f"{desc[k][s]:.2f}" for k in desc] for s in stats])

    lines += ["### Survival by age band", ""]
    bands = pd.cut(train["Age"], [0, 5, 12, 18, 30, 45, 60, 100])
    lines += _md_table(
        ["age band", "n", "survival rate"],
        [[str(idx), int(r["n"]), f"{r['survival_rate']:.4f}"]
         for idx, r in _survival_table(train, bands, sort=False).iterrows()],
    )

    lines += ["### Survival by fare quintile", ""]
    lines += _md_table(
        ["quintile", "n", "survival rate"],
        [[str(idx), int(r["n"]), f"{r['survival_rate']:.4f}"]
         for idx, r in _survival_table(train, pd.qcut(train["Fare"], 5, duplicates="drop"),
                                       sort=False).iterrows()],
    )

    lines += [
        "## Take-aways used by the pipeline",
        "",
        "* `Sex` dominates (74% of women vs 19% of men survive) -> keep it, plus the `Sex_Pclass` interaction.",
        "* `Pclass` and `Fare` are monotonic with survival -> keep both, add `FareLog` / `FarePerPerson`.",
        "* Survival peaks for toddlers (0-5: 70%) and dips for school-age kids (5-12: 36%), so",
        "  `Title` (Master/Miss) carries the age signal more reliably than raw `Age`; `IsChild`",
        "  and `IsMother` are added as weak extras.",
        "* Titles collapse neatly to Mr / Mrs / Miss / Master / Rare and separate survival well.",
        "* A recorded cabin (and its deck) flags wealthier passengers -> `Deck`, `HasCabin`.",
        "* FamilySize 2-4 does best (~55-72%), 5+ is very bad (<35%) -> `FamilyGroup`, `IsAlone`.",
        "* Age is ~20% missing -> impute by (Title, Pclass) group, not the global median.",
        "* Fare has exactly 1 missing value in test (3rd class) -> impute by Pclass median.",
        "* Embarked has 2 missing values in train -> impute with the mode (S).",
        "* Passengers travelling together share outcomes -> family / ticket survival-rate features.",
        "",
    ]
    return "\n".join(lines)


# ======================================================================================
# 3. Model comparison / tuning / reporting
# ======================================================================================
def evaluate_candidates(X: pd.DataFrame, y: np.ndarray, seed: int, n_jobs: int) -> pd.DataFrame:
    """Stratified 80/20 holdout + 5-fold CV (+ 5-fold x 2 repeats) per candidate."""
    x_tr, x_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rcv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=2, random_state=seed)

    rows = []
    for name, est in candidate_models(seed, n_jobs).items():
        pipe = build_pipeline(est, seed)
        t0 = time.time()
        pipe.fit(x_tr, y_tr)
        holdout = accuracy_score(y_va, pipe.predict(x_va))
        cv5 = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        cv5x2 = cross_val_score(pipe, X, y, cv=rcv, scoring="accuracy", n_jobs=1)
        rows.append({
            "model": name,
            "holdout_acc": round(float(holdout), 4),
            "cv5_mean": round(float(cv5.mean()), 4),
            "cv5_std": round(float(cv5.std()), 4),
            "cv5x2_mean": round(float(cv5x2.mean()), 4),
            "cv5x2_se": round(float(cv5x2.std() / np.sqrt(len(cv5x2))), 4),
            "seconds": round(time.time() - t0, 1),
        })
        r = rows[-1]
        print(f"  {name:<13s} holdout={r['holdout_acc']:.4f}   "
              f"5-fold CV={r['cv5_mean']:.4f} (+/-{r['cv5_std']:.4f})   "
              f"5x2 CV={r['cv5x2_mean']:.4f}   [{r['seconds']}s]", flush=True)
    return pd.DataFrame(rows)


def tune_model(name: str, X: pd.DataFrame, y: np.ndarray, seed: int, n_jobs: int,
               n_iter: int, top_k: int = 5) -> Tuple[Pipeline, Dict[str, object], pd.DataFrame]:
    """Two-stage hyper-parameter tuning for the winning model.

    Stage 1  ``RandomizedSearchCV`` over ``n_iter`` random configs, scored with a
             stratified 5-fold CV (fast, but noisy on only 891 rows).
    Stage 2  the ``top_k`` configs from stage 1 are re-scored with a repeated
             5-fold CV (5 x 3 = 15 folds) and the most *stable* one wins. This keeps
             the search from picking a config that merely got lucky on one split.

    Returns the best (still unfitted) pipeline, its parameters, and the stage-2 table.
    """
    base = candidate_models(seed, n_jobs)[name]
    space = search_spaces()[name]
    search = RandomizedSearchCV(
        build_pipeline(base, seed), space, n_iter=n_iter,
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
        scoring="accuracy", n_jobs=n_jobs, random_state=seed, refit=False, verbose=0,
    )
    t0 = time.time()
    search.fit(X, y)
    stage1 = time.time() - t0
    print(f"  stage 1: {n_iter} random configs x {N_FOLDS}-fold CV in {stage1:.0f}s "
          f"(best in-search accuracy {search.best_score_:.4f})")

    # --- stage 2: re-score the most promising configs with repeated CV -----------------
    rcv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=3, random_state=seed)
    candidates = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(top_k)
    rows, pipes = [], []
    for _, row in candidates.iterrows():
        params = dict(row["params"])
        pipe = build_pipeline(base, seed).set_params(**params)
        scores = cross_val_score(pipe, X, y, cv=rcv, scoring="accuracy", n_jobs=1)
        pipes.append(pipe)
        rows.append({
            "stage1_cv": round(float(row["mean_test_score"]), 4),
            "stage2_cv": round(float(scores.mean()), 4),
            "stage2_se": round(float(scores.std() / np.sqrt(len(scores))), 4),
            "params": {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v)
                       for k, v in params.items()},
        })
    order = np.argsort([-r["stage2_cv"] for r in rows])
    recheck = pd.DataFrame([rows[i] for i in order])
    best_pipeline = pipes[int(order[0])]
    for _, row in recheck.iterrows():
        print(f"  stage 2: CV={row['stage2_cv']:.4f} (+/-{row['stage2_se']:.4f})  {row['params']}", flush=True)
    info = {
        "name": name,
        "stage1_seconds": stage1,
        "stage1_best_cv": float(search.best_score_),
        "stage2_best_cv": float(recheck.iloc[0]["stage2_cv"]),
        "params": recheck.iloc[0]["params"],
    }
    print(f"  -> selected: {info['params']} (stage-2 CV {info['stage2_best_cv']:.4f})")
    return best_pipeline, info, recheck


def final_cv(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int,
             n_repeats: int) -> Tuple[float, float, float, float, np.ndarray]:
    """Stable accuracy estimate for the tuned pipeline + honest out-of-fold predictions.

    Returns ``(mean, fold_std, standard_error, oof_accuracy, oof_predictions)``.
    """
    rcv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=rcv, scoring="accuracy", n_jobs=1)
    # a different seed from the search, so this confusion matrix is not the split we tuned on
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + 1000)
    oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    return (float(scores.mean()), float(scores.std()),
            float(scores.std() / np.sqrt(len(scores))), float(accuracy_score(y, oof)), oof)


def _encoded_name_to_source(pipeline: Pipeline) -> Dict[str, str]:
    """Map every encoded column name (``cat__Title_Mr``) back to its source (``Title``)."""
    prep = pipeline.named_steps["prep"]
    source: Dict[str, str] = {f"num__{c}": c for c in NUMERIC_FEATURES}
    ohe = prep.named_transformers_["cat"]
    for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        for cat in cats:
            source[f"cat__{col}_{cat}"] = col
    # safety net for anything unexpected (e.g. a category value containing the separator)
    for name in prep.get_feature_names_out():
        if name not in source:
            matches = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
                       if name == f"num__{c}" or name.startswith(f"cat__{c}_")]
            source[name] = max(matches, key=len) if matches else name
    return source


def importance_frame(pipeline: Pipeline) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Importances per encoded feature and per source column (needs a fitted pipeline)."""
    prep = pipeline.named_steps["prep"]
    source = _encoded_name_to_source(pipeline)
    names = list(prep.get_feature_names_out())   # e.g. "num__Age", "cat__Title_Mr"
    importances = np.asarray(pipeline.named_steps["clf"].feature_importances_, dtype=float)
    frame = pd.DataFrame({
        "feature": names,
        "source_column": [source[n] for n in names],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    grouped = (
        frame.groupby("source_column", as_index=False)["importance"]
        .sum().sort_values("importance", ascending=False).reset_index(drop=True)
    )
    return frame, grouped


def write_submission(pipeline: Pipeline, test: pd.DataFrame, sample: pd.DataFrame | None,
                     path: Path) -> pd.DataFrame:
    """Predict test.csv and write the two-column submission, validating the format."""
    preds = pipeline.predict(test)
    sub = pd.DataFrame({
        "PassengerId": test["PassengerId"].to_numpy().astype(int),
        "Survived": np.asarray(preds).astype(int),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)

    print(f"\n[submission] wrote {path} -> {len(sub)} data rows + header, columns {list(sub.columns)}")
    assert list(sub.columns) == ["PassengerId", "Survived"], "columns must be PassengerId,Survived"
    assert len(sub) == len(test) == TEST_ROWS, "must hold one row per test passenger (418)"
    assert set(np.unique(sub["Survived"])) <= {0, 1}, "Survived must be binary 0/1"
    assert sub["PassengerId"].tolist() == test["PassengerId"].astype(int).tolist(), "ids/order must match test.csv"
    if sample is not None:
        assert list(sample.columns) == list(sub.columns), "columns differ from gender_submission.csv"
        assert len(sample) == len(sub), "row count differs from gender_submission.csv"
        assert sample["PassengerId"].tolist() == sub["PassengerId"].tolist(), "ids differ from gender_submission.csv"
        print(f"[submission] format matches gender_submission.csv exactly "
              f"(predicted survivors: {int(sub['Survived'].sum())} = {sub['Survived'].mean():.2%})")
    return sub


# ======================================================================================
# main
# ======================================================================================
def _make_pipeline_picklable() -> None:
    """Keep ``output/titanic_pipeline.joblib`` loadable from any other process.

    Run as a script, this module is ``__main__``, so pickle would store the custom
    transformers as ``__main__.FeatureExtractor`` -- unloadable anywhere else. Register
    the module under its real name and re-point the classes at it, so the artefact
    references ``titanic.FeatureExtractor`` and loads after a plain ``import titanic``.
    """
    if __name__ == "__main__":
        sys.modules.setdefault("titanic", sys.modules["__main__"])
    module = "titanic" if __name__ == "__main__" else __name__
    for cls in (FeatureExtractor, TitanicImputer, DerivedFeatures, GroupSurvivalEncoder):
        cls.__module__ = module


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Titanic survival prediction pipeline")
    p.add_argument("--data-dir", default=None, help="folder holding train.csv/test.csv (auto-detected)")
    p.add_argument("--out", default=str(REPO_ROOT / "submission.csv"), help="where to write submission.csv")
    p.add_argument("--seed", type=int, default=SEED, help="random seed used everywhere")
    p.add_argument("--search-iters", type=int, default=40, help="RandomizedSearchCV iterations")
    p.add_argument("--cv-repeats", type=int, default=4, help="repeats of 5-fold CV for the final score")
    p.add_argument("--n-jobs", type=int, default=2, help="parallel jobs")
    p.add_argument("--quick", action="store_true", help="smaller search + fewer CV repeats")
    p.add_argument("--no-report", action="store_true", help="skip writing the output/ artefacts")
    args = p.parse_args(argv)
    if args.quick:
        args.search_iters, args.cv_repeats = 12, 2
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    _make_pipeline_picklable()
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- Step 1: load + EDA
    data_dir = find_data_dir(args.data_dir)
    train, test, sample = load_data(data_dir)
    y = train[TARGET].to_numpy()
    X = train.drop(columns=[TARGET])

    print("=" * 90)
    print(f"TITANIC PIPELINE | data: {data_dir} | seed: {args.seed} | train: {X.shape} | test: {test.shape}")
    print(f"libraries: sklearn, pandas {pd.__version__}, numpy {np.__version__}"
          f"{', xgboost' if HAS_XGBOOST else ', HistGradientBoosting (xgboost missing)'}"
          f"{', lightgbm' if LGBMClassifier is not None else ''}")
    print("=" * 90)

    print("\n--- STEP 1: EXPLORATORY DATA ANALYSIS " + "-" * 51)
    report = eda_report(train, test)
    print(report)
    if not args.no_report:
        (out_dir / "eda_report.md").write_text(report)
        print(f"[eda] report written to {out_dir / 'eda_report.md'}")

    # --------------------------------------------------- Step 2 + 3a: compare candidates
    print("\n--- STEP 2+3a: PIPELINE & MODEL COMPARISON " + "-" * 46)
    print("  (features: Title, FamilySize/IsAlone/FamilyGroup, Deck/HasCabin, TicketFreq,")
    print("   group-imputed Age/Fare/Embarked, FareLog/FarePerPerson/Age_Class/IsChild/IsMother,")
    print("   Sex_Pclass, family & ticket survival rates; Sex/Embarked/Title/Deck one-hot encoded)")
    comparison = evaluate_candidates(X, y, args.seed, args.n_jobs)
    winner = comparison.sort_values(["cv5x2_mean", "cv5_mean"], ascending=False).iloc[0]["model"]
    print(f"  -> winner by cross-validated accuracy: {winner}")
    if not args.no_report:
        comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    # ------------------------------------------------------------- Step 3b: tune winner
    print(f"\n--- STEP 3b: HYPER-PARAMETER TUNING ({winner}) " + "-" * (48 - len(winner)))
    tuned, tune_info, recheck = tune_model(winner, X, y, args.seed, args.n_jobs, args.search_iters)
    if not args.no_report:
        recheck.to_csv(out_dir / "tuning_recheck.csv", index=False)

    # ------------------------------------------------------- Step 3c: honest final score
    print("\n--- STEP 3c: FINAL CROSS-VALIDATION " + "-" * 55)
    mean_acc, std_acc, se_acc, oof_acc, oof = final_cv(tuned, X, y, args.seed, args.cv_repeats)
    print(f"  {N_FOLDS}-fold CV x {args.cv_repeats} repeats : accuracy {mean_acc:.4f} "
          f"(fold std {std_acc:.4f}, standard error {se_acc:.4f})")
    print(f"  out-of-fold, unseen split   : accuracy {oof_acc:.4f}  <- single honest estimate")
    cm = confusion_matrix(y, oof)
    print(f"  confusion matrix            : TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")
    print(classification_report(y, oof, target_names=["died", "survived"], digits=4))

    # --------------------------- Step 4: refit on the FULL training set, then predict
    print("--- STEP 4: REFIT ON FULL train.csv AND PREDICT test.csv " + "-" * 38)
    final_pipeline = clone(tuned).fit(X, y)   # <- the model that produces submission.csv
    print(f"  refitted {winner} on all {len(X)} labelled passengers")

    frame, grouped = importance_frame(final_pipeline)
    total = float(grouped["importance"].sum()) or 1.0
    print("\n  feature importances (grouped by source column):")
    for _, row in grouped.head(10).iterrows():
        bar = "#" * int(round(40 * row["importance"] / grouped["importance"].max()))
        print(f"    {row['source_column']:<22s} {row['importance']:.4f} ({row['importance'] / total:5.1%}) {bar}")
    print("\n  top 10 individual encoded features:")
    for _, row in frame.head(10).iterrows():
        print(f"    {row['feature']:<34s} {row['importance']:.4f}")
    if not args.no_report:
        frame.to_csv(out_dir / "feature_importance.csv", index=False)

    sub_path = Path(args.out)
    sub = write_submission(final_pipeline, test, sample, sub_path)
    if not args.no_report:
        sub.to_csv(out_dir / "submission.csv", index=False)
        joblib.dump(final_pipeline, out_dir / "titanic_pipeline.joblib")
        print(f"[model] fitted pipeline saved to {out_dir / 'titanic_pipeline.joblib'}")

    # ------------------------------------------------------------------------ summary
    top5 = grouped.head(5)["source_column"].tolist()
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  chosen model           : {winner}")
    print(f"  tuned hyper-parameters : {tune_info['params']}")
    print(f"  local CV accuracy      : {mean_acc:.4f} (+/- {se_acc:.4f} s.e.) "
          f"[{N_FOLDS}-fold x {args.cv_repeats} repeats, stratified]")
    print(f"  out-of-fold accuracy   : {oof_acc:.4f} (independent split)")
    print(f"  top 5 features         : {', '.join(top5)}")
    print(f"  submission             : {sub_path} ({len(sub)} rows)")
    print(f"  leaderboard expectation: ~{mean_acc:.3f} (Kaggle public/private can differ by ~0.02 "
          f"on only {TEST_ROWS} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
