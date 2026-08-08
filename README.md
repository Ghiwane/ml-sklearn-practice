# ml-sklearn-practice

Self-directed practice repository covering the core scikit-learn workflow — linear regression, model evaluation metrics, feature engineering, and logistic regression for classification.

> 📌 This repo is a learning log, not a polished library — code favors clarity and comments over abstraction, since the goal was to understand every line and build stronger ML skills.

---

## Repository Structure

```
ml-sklearn-practice/
├── 01_LinearRegression_children/
│   ├── data/
│   │   └── age_vs_poids_vs_taille_vs_sexe.csv   # Children's age/height/weight/sex dataset
│   └── exercice.py
├── 02_LinearRegression_advertising/
│   ├── data/
│   │   └── advertising.csv                       # TV/radio/newspaper spend → sales
│   └── exercice.py
├── 03_LogisticRegression_classification/
│   └── exercice.py                                # Titanic survival classification
└── README.md
```

---

## What This Repo Covers

The main areas practiced here, each building on the previous one:

| # | Concept | Where it shows up |
|---|---------|--------------------|
| 1 | Simple & multiple linear regression (`LinearRegression`) | `01_LinearRegression_children`, `02_LinearRegression_advertising` |
| 2 | Model comparison via R² across feature subsets | `01_LinearRegression_children`, `02_LinearRegression_advertising` |
| 3 | `train_test_split` and evaluation on unseen data | `02_LinearRegression_advertising`, `03_LogisticRegression_classification` |
| 4 | Regression error metrics: MSE, RMSE, MAE, MAPE | `01_LinearRegression_children`, `02_LinearRegression_advertising` |
| 5 | Feature engineering (interaction terms) | `02_LinearRegression_advertising` |
| 6 | Logistic regression, classification metrics, threshold tuning, ROC/AUC | `03_LogisticRegression_classification` |

---

## Script Details

### 1. `01_LinearRegression_children/exercice.py` — Predicting Weight from Age, Sex, and Height

A first hands-on pass at linear regression: predicting a child's weight (`poids`) from progressively richer sets of predictors.

**Pipeline:**
1. Model 1: `poids ~ age` — single-feature baseline.
2. Model 2: `poids ~ sexe + age` — adds sex as a predictor.
3. Model 3: `poids ~ sexe + age + taille` — adds height.
4. R² is tracked at each step to see how much each added feature improves the fit.
5. Full model evaluated with MSE, RMSE, MAE, and MAPE (all computed on the training data here, no train/test split yet — that's introduced in the next exercise).
6. A manual prediction is made for a new, unseen child (`sexe=0, age=224, taille=175`) to sanity-check the model on a concrete example.

**What this exercise is really about:** getting comfortable with the basic `sklearn` regression API (`fit`, `predict`, `score`, `coef_`) and seeing directly, feature by feature, how R² responds as more relevant predictors are added — height turns out to matter a lot more than sex alone.

---

### 2. `02_LinearRegression_advertising/exercice.py` — Advertising Spend vs Sales

A more complete regression workflow on the classic advertising dataset (TV, radio, newspaper spend → sales), this time with proper train/test evaluation and feature engineering.

**Pipeline:**
1. Exploratory step: correlation matrix to identify which channel (TV, radio, or newspaper) correlates most strongly with sales.
2. Three models built by progressively adding channels: `TV` → `TV + radio` → `TV + radio + newspaper`, comparing R² at each step.
3. `train_test_split` (80/20) introduced here for the first time — the full model is retrained and evaluated on a held-out test set with MSE, RMSE, MAE, and MAPE, rather than scored on its own training data.
4. Feature engineering: a `tv_x_radio` interaction term is added to capture the idea that TV and radio spend might have a *combined* (synergy) effect on sales rather than a purely additive one — the same interaction term later reused in the PyTorch version of this problem.
5. The interaction model is evaluated the same way, to compare directly against the model without it.

**What this exercise is really about:** the shift from "fit on everything, report R²" to a proper train/test evaluation, and the intuition that a model's inputs aren't fixed — engineering a new feature (interaction term) can meaningfully improve fit when two variables plausibly reinforce each other.

---

### 3. `03_LogisticRegression_classification/exercice.py` — Titanic Survival Classification

First classification project: predicting passenger survival on the Titanic dataset using logistic regression, with a focus on evaluation beyond plain accuracy.

**Pipeline:**
1. Missing values in `age` and `fare` imputed with the median; `sex` encoded numerically (`male=1`, `female=0`).
2. Predictors: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare` — target: `survived`.
3. `train_test_split` (70/30), then `LogisticRegression` fit on the training set.
4. Predicted probabilities inspected for individual passengers via `predict_proba`, and visualized as a histogram across the whole test set — a reminder that the model outputs *probabilities*, not just hard 0/1 labels.
5. Threshold analysis: accuracy, precision, and recall compared across three decision thresholds (0.5, 0.3, 0.7), each with its own confusion matrix — showing how moving the threshold trades precision against recall rather than changing the model itself.
6. ROC curve plotted and ROC-AUC computed as a threshold-independent measure of the model's overall discriminative power.

**What this exercise is really about:** the point where "accuracy" alone stops being a sufficient metric. A single threshold (0.5 by default) is an arbitrary choice, not a law of nature — the threshold analysis and ROC/AUC section exist specifically to show that the same trained model can look very different (more cautious vs. more aggressive at predicting survival) depending on where that line is drawn.

---

## Requirements

See `requirements.txt`.

Install with:
```bash
pip install -r requirements.txt
```

## How to Run

```bash
# Linear regression — children's weight
python 01_LinearRegression_children/exercice.py

# Linear regression — advertising spend vs sales
python 02_LinearRegression_advertising/exercice.py

# Logistic regression — Titanic survival classification
python 03_LogisticRegression_classification/exercice.py
```

---

## Learning Notes

Code here is deliberately verbose and heavily commented — the priority was understanding *why* each step is needed (e.g., why evaluate on a held-out test set rather than training data, why a single accuracy score can hide a lot, why an interaction term can help a linear model) rather than writing the shortest possible implementation. This repo exists to build a solid, hands-on foundation in the standard scikit-learn workflow before moving on to deep learning and reinforcement learning practice.