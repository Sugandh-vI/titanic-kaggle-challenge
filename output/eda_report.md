# Titanic -- exploratory data analysis

* `train.csv`: 891 rows x 12 columns (labelled, target = `Survived`)
* `test.csv`: 418 rows x 11 columns (to predict)
* `gender_submission.csv`: format reference only (418 rows) -- never used for training

## Missing values

| column | train n | train % | test n | test % |
|---|---|---|---|---|
| PassengerId | 0 | 0.00% | 0 | 0.00% |
| Survived | 0 | 0.00% | - | - |
| Pclass | 0 | 0.00% | 0 | 0.00% |
| Name | 0 | 0.00% | 0 | 0.00% |
| Sex | 0 | 0.00% | 0 | 0.00% |
| Age | 177 | 19.87% | 86 | 20.57% |
| SibSp | 0 | 0.00% | 0 | 0.00% |
| Parch | 0 | 0.00% | 0 | 0.00% |
| Ticket | 0 | 0.00% | 0 | 0.00% |
| Fare | 0 | 0.00% | 1 | 0.24% |
| Cabin | 687 | 77.10% | 327 | 78.23% |
| Embarked | 2 | 0.22% | 0 | 0.00% |

## Class balance of `Survived`

* died (0): **549** (61.62%)
* survived (1): **342** (38.38%)
* imbalance is mild, so plain accuracy is a fair metric; no resampling needed
* trivial baseline (always predict 0): accuracy 0.6162
* `gender_submission.csv` rule (every woman survives): accuracy 0.7868 on train

## Distributions vs survival

### Sex

| sex | n | survival rate |
|---|---|---|
| female | 314 | 0.7420 |
| male | 577 | 0.1889 |

### Pclass

| pclass | n | survival rate |
|---|---|---|
| 1 | 216 | 0.6296 |
| 2 | 184 | 0.4728 |
| 3 | 491 | 0.2424 |

### Embarked

| embarked | n | survival rate |
|---|---|---|
| C | 168 | 0.5536 |
| Q | 77 | 0.3896 |
| S | 644 | 0.3370 |

### Title

| title | n | survival rate |
|---|---|---|
| Mrs | 126 | 0.7937 |
| Miss | 185 | 0.7027 |
| Master | 40 | 0.5750 |
| Rare | 23 | 0.3478 |
| Mr | 517 | 0.1567 |

### Deck

| deck | n | survival rate |
|---|---|---|
| D | 33 | 0.7576 |
| E | 32 | 0.7500 |
| B | 47 | 0.7447 |
| F | 13 | 0.6154 |
| C | 59 | 0.5932 |
| G | 4 | 0.5000 |
| A | 15 | 0.4667 |
| U | 687 | 0.2999 |
| T | 1 | 0.0000 |

### FamilySize

| familysize | n | survival rate |
|---|---|---|
| 4 | 29 | 0.7241 |
| 3 | 102 | 0.5784 |
| 2 | 161 | 0.5528 |
| 7 | 12 | 0.3333 |
| 1 | 537 | 0.3035 |
| 5 | 15 | 0.2000 |
| 6 | 22 | 0.1364 |
| 8 | 6 | 0.0000 |
| 11 | 7 | 0.0000 |

### FamilyGroup

| familygroup | n | survival rate |
|---|---|---|
| Small | 292 | 0.5788 |
| Alone | 537 | 0.3035 |
| Large | 62 | 0.1613 |

### Age / Fare (numeric summaries)

| stat | Age (train) | Age (test) | Fare (train) | Fare (test) |
|---|---|---|---|---|
| count | 714.00 | 332.00 | 891.00 | 417.00 |
| mean | 29.70 | 30.27 | 32.20 | 35.63 |
| std | 14.53 | 14.18 | 49.69 | 55.91 |
| min | 0.42 | 0.17 | 0.00 | 0.00 |
| 25% | 20.12 | 21.00 | 7.91 | 7.90 |
| 50% | 28.00 | 27.00 | 14.45 | 14.45 |
| 75% | 38.00 | 39.00 | 31.00 | 31.50 |
| max | 80.00 | 76.00 | 512.33 | 512.33 |

### Survival by age band

| age band | n | survival rate |
|---|---|---|
| (0, 5] | 44 | 0.7045 |
| (5, 12] | 25 | 0.3600 |
| (12, 18] | 70 | 0.4286 |
| (18, 30] | 270 | 0.3556 |
| (30, 45] | 202 | 0.4257 |
| (45, 60] | 81 | 0.4074 |
| (60, 100] | 22 | 0.2273 |

### Survival by fare quintile

| quintile | n | survival rate |
|---|---|---|
| (-0.001, 7.854] | 179 | 0.2179 |
| (7.854, 10.5] | 184 | 0.2011 |
| (10.5, 21.679] | 172 | 0.4244 |
| (21.679, 39.688] | 180 | 0.4444 |
| (39.688, 512.329] | 176 | 0.6420 |

## Take-aways used by the pipeline

* `Sex` dominates (74% of women vs 19% of men survive) -> keep it, plus the `Sex_Pclass` interaction.
* `Pclass` and `Fare` are monotonic with survival -> keep both, add `FareLog` / `FarePerPerson`.
* Survival peaks for toddlers (0-5: 70%) and dips for school-age kids (5-12: 36%), so
  `Title` (Master/Miss) carries the age signal more reliably than raw `Age`; `IsChild`
  and `IsMother` are added as weak extras.
* Titles collapse neatly to Mr / Mrs / Miss / Master / Rare and separate survival well.
* A recorded cabin (and its deck) flags wealthier passengers -> `Deck`, `HasCabin`.
* FamilySize 2-4 does best (~55-72%), 5+ is very bad (<35%) -> `FamilyGroup`, `IsAlone`.
* Age is ~20% missing -> impute by (Title, Pclass) group, not the global median.
* Fare has exactly 1 missing value in test (3rd class) -> impute by Pclass median.
* Embarked has 2 missing values in train -> impute with the mode (S).
* Passengers travelling together share outcomes -> family / ticket survival-rate features.
