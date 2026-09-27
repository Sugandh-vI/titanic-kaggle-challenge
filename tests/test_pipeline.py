"""Regression tests for the Titanic pipeline.

Run with:  python -m pytest tests/ -q          (needs `pip install pytest`)

These tests are deliberately fast -- tiny estimators, no hyper-parameter search -- but
they pin down the three things that silently ruin a Titanic submission:

1. the model matrix is complete (no NaNs) and free of raw/id columns,
2. imputation is group-aware (Age by Title x Pclass, Fare by Pclass, Embarked by mode),
3. the family/ticket survival-rate encoder does not let a passenger's own label leak
   into their own features -- including through the singleton-group fallback, which
   once collapsed 5-fold CV from 0.845 to 0.725 when it was made per-passenger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import titanic as T  # noqa: E402

# the regularised forest the script actually starts from (sklearn's unregularised
# defaults overfit this feature set badly, which is itself worth remembering)
BASE_RF = dict(n_estimators=150, max_depth=6, min_samples_leaf=3, max_features=0.5)


@pytest.fixture(scope="module")
def data():
    train, test, sample = T.load_data(T.find_data_dir())
    return train, test, sample


@pytest.fixture(scope="module")
def prepped(data):
    """train/test frames after feature extraction + imputation + derived features."""
    train, test, _ = data
    y = train[T.TARGET].to_numpy()
    tr, te = train.drop(columns=[T.TARGET]), test
    for step in (T.FeatureExtractor(), T.TitanicImputer(), T.DerivedFeatures()):
        step.fit(tr, y)
        tr, te = step.transform(tr), step.transform(te)
    return tr, te, y


@pytest.fixture(scope="module")
def fitted_pipeline(data):
    train, _, _ = data
    X, y = train.drop(columns=[T.TARGET]), train[T.TARGET].to_numpy()
    pipe = T.build_pipeline(RandomForestClassifier(random_state=T.SEED, **BASE_RF), T.SEED)
    pipe.fit(X, y)
    return pipe, X, y


# --------------------------------------------------------------------------- Step 2
def test_title_extraction(data):
    train, _, _ = data
    feats = T.FeatureExtractor().fit(train).transform(train)
    assert set(feats["Title"]) <= {"Mr", "Mrs", "Miss", "Master", "Rare"}
    assert feats.loc[train["Name"].str.contains("Countess"), "Title"].iloc[0] == "Rare"
    assert feats.loc[train["Name"].str.contains("Mlle"), "Title"].iloc[0] == "Miss"
    assert feats.loc[train["Name"].str.contains("Dooley, Mr."), "Title"].iloc[0] == "Mr"
    assert int((feats["Title"] == "Mr").sum()) == int(train["Name"].str.contains(", Mr.", regex=False).sum())


def test_family_and_deck_features(data):
    train, _, _ = data
    feats = T.FeatureExtractor().fit(train).transform(train)
    assert (feats["FamilySize"] == train["SibSp"] + train["Parch"] + 1).all()
    assert (feats["IsAlone"] == (feats["FamilySize"] == 1).astype(int)).all()
    assert set(feats["Deck"]) <= set("ABCDEFGTU")
    assert (feats.loc[train["Cabin"].isna(), "Deck"] == "U").all()      # missing -> "Unknown"
    assert feats.loc[train["Cabin"] == "C85", "Deck"].iloc[0] == "C"    # first letter


def test_raw_columns_are_dropped(fitted_pipeline):
    pipe = fitted_pipeline[0]
    sources = set(T._encoded_name_to_source(pipe).values())
    assert sources <= set(T.NUMERIC_FEATURES) | set(T.CATEGORICAL_FEATURES)
    for forbidden in ["PassengerId", "Name", "Ticket", "Cabin", "Surname", "TicketId", "FamilyKey"]:
        assert forbidden not in sources, f"{forbidden} leaked into the model matrix"


def test_model_matrix_has_no_missing_values(data, fitted_pipeline):
    pipe, X, _ = fitted_pipeline
    _, test, _ = data
    expected_width = len(pipe.named_steps["prep"].get_feature_names_out())
    for frame in (X, test):
        matrix = np.asarray(pipe[:-1].transform(frame), dtype=float)   # all steps but the model
        assert matrix.shape == ((len(X) if frame is X else len(test)), expected_width)
        assert np.isnan(matrix).sum() == 0


def test_imputation_is_group_aware(data):
    train, test, _ = data
    extracted = T.FeatureExtractor().fit(train).transform(train)
    imputer = T.TitanicImputer().fit(extracted)
    imputed = imputer.transform(extracted)

    # a 3rd-class "Master" is a boy, a 1st-class "Mrs" an adult woman: very different medians
    boys = imputed[(imputed["Title"] == "Master") & (imputed["Pclass"] == 3) & train["Age"].isna()]
    ladies = imputed[(imputed["Title"] == "Mrs") & (imputed["Pclass"] == 1) & train["Age"].isna()]
    assert boys["Age"].median() < 10 < train["Age"].median() < ladies["Age"].median()
    assert imputed["Age"].isna().sum() == 0
    assert imputer.embarked_mode_ == "S"

    # the single missing Fare in test.csv belongs to a 3rd-class passenger
    test_imputed = imputer.transform(T.FeatureExtractor().fit(train).transform(test))
    assert test_imputed["Fare"].isna().sum() == 0 and test_imputed["Embarked"].isna().sum() == 0
    row = test_imputed.loc[test["Fare"].isna().to_numpy()]
    assert row["Pclass"].iloc[0] == 3
    assert row["Fare"].iloc[0] == pytest.approx(train.loc[train["Pclass"] == 3, "Fare"].median())
    assert row["Fare"].iloc[0] != pytest.approx(train["Fare"].median())   # not the global median


# ------------------------------------------------------------------ leakage control
def test_group_encoder_is_leave_one_out(prepped):
    """A passenger's own label must not change that passenger's own encoded rate."""
    train_features, _, y = prepped
    enc = T.GroupSurvivalEncoder().fit(train_features, y)
    out = enc.transform(train_features)

    sizes = pd.Series(y, index=train_features.index).groupby(train_features["FamilyKey"]).transform("count")
    member = train_features.index[sizes > 1][0]            # someone with a family mate
    flipped = y.copy()
    flipped[train_features.index.get_loc(member)] = 1 - flipped[train_features.index.get_loc(member)]
    out_flip = T.GroupSurvivalEncoder().fit(train_features, flipped).transform(train_features)

    assert out.loc[member, "FamilySurvivalRate"] == pytest.approx(out_flip.loc[member, "FamilySurvivalRate"])
    assert out.loc[member, "TicketSurvivalRate"] == pytest.approx(out_flip.loc[member, "TicketSurvivalRate"])
    # ... while the encoder is clearly not constant, i.e. the check above is meaningful
    assert out["FamilySurvivalRate"].nunique() > 5


def test_group_encoder_matches_manual_leave_one_out(prepped):
    train_features, _, y = prepped
    out = T.GroupSurvivalEncoder().fit(train_features, y).transform(train_features)
    yser = pd.Series(y, index=train_features.index)

    checked = 0
    keys = train_features["FamilyKey"][yser.groupby(train_features["FamilyKey"]).transform("count") > 1].unique()
    for key in keys[:5]:
        idx = train_features.index[train_features["FamilyKey"] == key]
        for i in idx:
            assert out.loc[i, "FamilySurvivalRate"] == pytest.approx(yser.loc[idx].drop(i).mean())
            assert out.loc[i, "FamilySurvivalKnown"] == 1
            checked += 1
    assert checked > 0


def test_singleton_groups_carry_no_individual_signal(prepped):
    """Every passenger without a group mate must share ONE constant encoded value.

    If this ever becomes per-passenger (e.g. a leave-one-out global mean), it encodes
    each passenger's own label and the trees memorise it: CV drops from ~0.84 to ~0.72.
    """
    train_features, _, y = prepped
    out = T.GroupSurvivalEncoder().fit(train_features, y).transform(train_features)
    sizes = pd.Series(y, index=train_features.index).groupby(train_features["FamilyKey"]).transform("count")
    singles = out.loc[train_features.index[sizes == 1]]

    assert len(singles) > 100
    assert singles["FamilySurvivalRate"].nunique() == 1
    assert singles["FamilySurvivalRate"].iloc[0] == pytest.approx(y.mean())
    assert (singles["FamilySurvivalKnown"] == 0).all()


def test_group_encoder_uses_only_train_labels_for_unseen_rows(data, prepped):
    """test.csv rows are encoded from train labels only; unseen groups -> global rate."""
    train, test, _ = data
    _, test_features, y = prepped
    train_features = prepped[0]

    enc = T.GroupSurvivalEncoder().fit(train_features, y)
    out_test = enc.transform(test_features)

    assert len(out_test) == len(test) == T.TEST_ROWS
    for col in ["FamilySurvivalRate", "TicketSurvivalRate"]:
        assert out_test[col].isna().sum() == 0
    unknown = out_test["FamilySurvivalKnown"] == 0
    assert np.allclose(out_test.loc[unknown, "FamilySurvivalRate"], y.mean())
    assert unknown.any() and (~unknown).any()          # both branches are exercised
    # a test passenger sharing a family with a train passenger gets that family's train rate
    shared = test_features.loc[~unknown.to_numpy(), "FamilyKey"].iloc[0]
    mates = train_features.index[train_features["FamilyKey"] == shared]
    assert out_test.loc[~unknown.to_numpy()].iloc[0]["FamilySurvivalRate"] == pytest.approx(
        pd.Series(y, index=train_features.index).loc[mates].mean()
    )


# --------------------------------------------------------------------------- Step 4
def test_submission_format(data, fitted_pipeline):
    pipe, _, _ = fitted_pipeline
    _, test, sample = data
    sub = pd.DataFrame({"PassengerId": test["PassengerId"].astype(int),
                        "Survived": np.asarray(pipe.predict(test)).astype(int)})

    assert list(sub.columns) == ["PassengerId", "Survived"]
    assert len(sub) == T.TEST_ROWS == 418
    assert set(sub["Survived"].unique()) <= {0, 1}
    assert sub["PassengerId"].tolist() == test["PassengerId"].astype(int).tolist()
    assert not sub.isna().any().any()
    if sample is not None:                       # gender_submission.csv is the format reference
        assert list(sample.columns) == list(sub.columns)
        assert sample["PassengerId"].tolist() == sub["PassengerId"].tolist()


def test_pipeline_beats_the_gender_baseline(data):
    """The point of the exercise: CV accuracy clearly above the 0.7868 'all women' rule."""
    train, _, _ = data
    X, y = train.drop(columns=[T.TARGET]), train[T.TARGET].to_numpy()
    pipe = T.build_pipeline(RandomForestClassifier(random_state=T.SEED, **BASE_RF), T.SEED)
    scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=T.SEED),
                             scoring="accuracy")
    baseline = ((train["Sex"] == "female") == (train[T.TARGET] == 1)).mean()
    assert scores.mean() > baseline + 0.03, (scores.mean(), baseline)


def test_saved_pipeline_can_be_reloaded(tmp_path, data):
    """output/titanic_pipeline.joblib must load in a fresh process (README workflow)."""
    train, test, _ = data
    X, y = train.drop(columns=[T.TARGET]), train[T.TARGET].to_numpy()
    pipe = T.build_pipeline(RandomForestClassifier(random_state=T.SEED, n_estimators=30), T.SEED).fit(X, y)

    T._make_pipeline_picklable()
    assert T.FeatureExtractor.__module__ != "__main__", "artefact would only load from this process"
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipe, path)
    reloaded = joblib.load(path)
    assert (reloaded.predict(test) == pipe.predict(test)).all()
