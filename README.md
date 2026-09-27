# Titanic: Machine Learning from Disaster

End-to-end solution for the Kaggle [Titanic](https://www.kaggle.com/c/titanic) competition:
EDA -> feature engineering -> model comparison -> tuning -> `submission.csv`.

Everything lives in one runnable script, **[`titanic.py`](titanic.py)**, and the whole
preprocessing + model chain is a single `sklearn.pipeline.Pipeline`, so it is refit on
every cross-validation fold (no leakage) and can be dumped/reloaded as one object.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python titanic.py          # ~6 min on 2 cores -> writes submission.csv
.venv/bin/python -m pytest tests/ -q # 12 fast regression tests
```

---

## Summary (the short version)

| | |
|---|---|
| **Chosen model** | `RandomForestClassifier` (scikit-learn), tuned with `RandomizedSearchCV` |
| **Local CV accuracy** | **0.8437** (5-fold x 4 repeats, stratified; standard error 0.0059) |
| **Independent out-of-fold accuracy** | 0.8384 (5-fold split the search never saw) |
| **Stratified 80/20 holdout** | 0.8212 |
| **Top 5 features** | `Title`, `Sex`, `Age_Class`, `FamilySurvivalRate`, `FarePerPerson` |
| **Submission** | [`submission.csv`](submission.csv) - 418 rows + header, `PassengerId,Survived` |
| **Expected on the leaderboard** | ~0.84 (public/private can differ by ~0.02 on only 418 rows) |

### Why RandomForest and not XGBoost?

All three candidates were run through the identical pipeline and scored with the same
stratified splits. Untuned, the forest was already ahead, and it stayed ahead after
tuning -- on 891 rows the bagged, heavily-regularised forest simply has less variance
than gradient boosting, which needs more data than this to pay off.

| model | holdout (80/20) | 5-fold CV | 5-fold x 2 repeats |
|---|---|---|---|
| **RandomForest** | **0.8212** | **0.8451** +/- 0.0192 | **0.8417** |
| XGBoost | 0.7989 | 0.8417 +/- 0.0170 | 0.8389 |
| LightGBM | 0.8045 | 0.8350 +/- 0.0192 | 0.8322 |

*Baseline for reference: predicting "every woman survives" (the `gender_submission.csv`
rule) scores 0.7868 on train; always predicting "died" scores 0.6162.*

Selected hyper-parameters:

```python
RandomForestClassifier(n_estimators=700, criterion="gini", max_depth=6,
                       min_samples_leaf=4, min_samples_split=3,
                       max_features=0.65, class_weight="balanced_subsample",
                       random_state=42)
```

Tuning is two-stage on purpose: `RandomizedSearchCV` (40 configs x 5-fold) proposes,
then the top 5 configs are re-scored with a repeated CV (5-fold x 3) and the most
*stable* one wins. On a dataset this small a single 5-fold score is noisy enough that
picking its argmax overfits the split (it reported 0.8563 for a config whose repeated-CV
score is 0.8437).

### Top 10 features (grouped by source column, RF impurity importance)

| # | feature | importance | share |
|---|---|---|---|
| 1 | `Title` (Mr/Mrs/Miss/Master/Rare) | 0.2867 | 28.7% |
| 2 | `Sex` | 0.1916 | 19.2% |
| 3 | `Age_Class` (Age x Pclass) | 0.0746 | 7.5% |
| 4 | `FamilySurvivalRate` | 0.0718 | 7.2% |
| 5 | `FarePerPerson` | 0.0685 | 6.8% |
| 6 | `Sex_Pclass` | 0.0357 | 3.6% |
| 7 | `TicketSurvivalRate` | 0.0344 | 3.4% |
| 8 | `FareLog` | 0.0339 | 3.4% |
| 9 | `Age` | 0.0331 | 3.3% |
| 10 | `Fare` | 0.0324 | 3.2% |

Individually, the strongest splits are `Title_Mr` (0.268), `Sex_male` (0.105) and
`Sex_female` (0.087) -- "adult man" is what decided survival, everything else is
refinement. Full list: [`output/feature_importance.csv`](output/feature_importance.csv).

---

## What the pipeline does

| stage (Pipeline step) | class | what happens |
|---|---|---|
| `extract` | `FeatureExtractor` | `Title` from `Name`, `FamilySize`/`IsAlone`/`FamilyGroup`, `Deck` + `HasCabin` from `Cabin`, `TicketId`/`TicketFreq`; drops `Name`, `Ticket`, `Cabin` |
| `impute` | `TitanicImputer` | `Age` by (Title, Pclass) median -> Pclass median -> global; `Fare` by Pclass median; `Embarked` by mode |
| `derive` | `DerivedFeatures` | `FareLog`, `FarePerPerson`, `Age_Class`, `IsChild`, `IsMother`, `Sex_Pclass`, `FamilyKey` |
| `groups` | `GroupSurvivalEncoder` | leave-one-out family / ticket survival rates + `*Known` flags; drops `PassengerId`, `Surname`, `TicketId`, `FamilyKey` |
| `prep` | `ColumnTransformer` | `SimpleImputer(median)` on the 18 numeric features, `OneHotEncoder(handle_unknown="ignore")` on `Sex`, `Embarked`, `Title`, `Deck`, `FamilyGroup`, `Sex_Pclass` -> 46 model columns |
| `clf` | the model | RandomForest / XGBoost / LightGBM |

`gender_submission.csv` is used **only** to validate the output format (columns, row
count, id order) -- never as data.

### Imputation is group-aware, and it matters

| missing value | naive fill | this pipeline |
|---|---|---|
| `Age`, 3rd-class `Master` (177 in train, 86 in test) | 28.0 (global median) | **4.0** (he is a boy) |
| `Age`, 1st-class `Mrs` | 28.0 | **40.0** |
| `Fare`, PassengerId 1044 (the one missing test fare) | 14.45 | **8.05** (3rd-class median) |
| `Embarked`, 2 train rows | drop | **"S"** (mode) |

Because the imputer is a pipeline step, its medians are learned on the training fold
only and applied to the held-out fold / `test.csv`.

### The extra feature: family & ticket survival rates

Rescue was organised by household and travelling party, so "did the rest of my group
survive?" is the strongest signal left after `Sex` and `Pclass`. Two group keys:
`Surname + rounded Fare` (household) and a normalised `Ticket` number (party). 34% of
test passengers share a household with someone in train, 36% share a ticket.

Measured with the selected forest and a 5-fold x 3 CV, simply by removing the columns
from `NUMERIC_FEATURES`:

| features | CV accuracy |
|---|---|
| full pipeline (family + ticket survival rates) | **0.8436** (s.e. 0.0063) |
| without `FamilySurvivalRate` | 0.8354 (-0.008) |
| without either group feature | 0.8223 (-0.021) |

Leakage control (this is the part that is easy to get wrong):

* group rates are computed in `fit()` from **training labels only**;
* rows seen in `fit()` are re-encoded **leave-one-out**, so no passenger's own label is
  part of their own feature;
* unseen rows (validation fold, `test.csv`) get the group's training rate, or the global
  rate plus `*Known = 0` when the group never appears in train;
* passengers with **no group mate all collapse onto one single constant**. Making that
  fallback per-passenger instead (a "leave-one-out global mean") looks more rigorous but
  quietly encodes each passenger's own label -- the trees found it instantly and 5-fold
  CV fell from 0.845 to 0.725. `tests/test_pipeline.py::test_singleton_groups_carry_no_individual_signal`
  exists to keep that from ever coming back.

---

## Repository layout

```
titanic.py                     the pipeline: EDA, features, models, tuning, submission
tests/test_pipeline.py         12 regression tests (features, imputation, leakage, format)
data/train.csv                 891 labelled passengers
data/test.csv                  418 passengers to predict
data/gender_submission.csv     format reference only
submission.csv                 <- upload this to Kaggle
requirements.txt               runtime dependencies
requirements-dev.txt           pytest, for the tests
output/eda_report.md           Step 1: missing values, class balance, survival tables
output/model_comparison.csv    Step 3: all candidates and their scores
output/tuning_recheck.csv      Step 3: stage-2 re-scoring of the top configs
output/feature_importance.csv  every encoded feature and its importance
output/submission.csv          copy of the submission
output/titanic_pipeline.joblib the fitted pipeline (git-ignored, ~4 MB)
```

## Running it

```bash
python titanic.py                          # full run: search 40 configs, CV 5x4
python titanic.py --quick                  # ~2 min: search 12 configs, CV 5x2
python titanic.py --seed 7                 # everything (splits, search, forests) re-seeded
python titanic.py --search-iters 80        # wider hyper-parameter search
python titanic.py --data-dir dataset       # if your csvs live in dataset/ instead of data/
python titanic.py --out output/submission.csv
```

The script prints the EDA tables, the model comparison, the tuning log, the final
repeated-CV accuracy, the out-of-fold confusion matrix and classification report, the
feature importances, and a closing summary. It ends with format assertions on
`submission.csv` (2 columns, 418 rows, binary `Survived`, ids in `test.csv` order,
matching `gender_submission.csv`).

Reusing the saved model without retraining:

```python
import sys; sys.path.insert(0, ".")     # repo root: makes `titanic` importable
import joblib, pandas as pd
pipe = joblib.load("output/titanic_pipeline.joblib")   # -> titanic.Pipeline
test = pd.read_csv("data/test.csv")
pd.DataFrame({"PassengerId": test.PassengerId,
              "Survived": pipe.predict(test).astype(int)}).to_csv("submission.csv", index=False)
```

## Reproducibility

One seed (`--seed`, default 42) drives the holdout split, every CV splitter, the
`RandomizedSearchCV` sampling and the forests themselves; `np.random.seed` is set in
`main()`. Re-running with the same seed on the same library versions reproduces
`submission.csv` byte for byte. Versions used for the numbers above: Python 3.11,
pandas 3.0.6, scikit-learn 1.9.1, numpy 2.4.6, xgboost 3.2.0, lightgbm 4.7.0.
XGBoost and LightGBM are optional: without XGBoost the script falls back to
`HistGradientBoostingClassifier`, without LightGBM it simply skips the third candidate.

## Ideas that did not pay off

* **Soft-voting ensemble (RandomForest + XGBoost)**: 0.8434 vs 0.8474 for the tuned
  forest alone in a development run (5-fold x 6 CV) -- the two models make similar
  errors, so averaging mostly diluted the better one.
* **Unregularised forests** (sklearn defaults, unlimited depth, `min_samples_leaf=1`):
  0.77 -- they memorise the group-survival features. Regularisation is not optional here.
* **Ticket-prefix and fare-band buckets**: no measurable gain over the raw features, so
  they were dropped to keep the matrix small.
