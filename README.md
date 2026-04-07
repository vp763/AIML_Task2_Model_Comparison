# California Housing Price Prediction — Model Comparison

This project is part of Task 2 of my AI/ML internship at **Maincrafts Technology**. The focus was on building a proper regression pipeline — not just throwing models at data, but doing it in a way that actually holds up: clean preprocessing, fair evaluation, and a justified model selection.

---

## What the project covers

- Loading and exploring the California Housing dataset
- Feature scaling done correctly (fitted only on training data to avoid leakage)
- Training and comparing three regression models
- Evaluating on RMSE and R² with a proper train/test split
- Residual analysis and overfitting check
- Saving the best model using joblib

---

## Models and results

| Model | Test RMSE | Test R² | Train R² | Overfit? |
|---|---|---|---|---|
| Linear Regression | 0.7456 | 0.5758 | 0.6126 | No (gap = 0.037) |
| Ridge Regression | 0.7456 | 0.5758 | 0.6126 | No (gap = 0.037) |
| Decision Tree (max_depth=5) | 0.7050 | 0.6211 | 1.0000 | Yes (gap = 0.379) |

**Recommended model: Linear Regression**

The Decision Tree had a slightly better test R², but its training R² was a perfect 1.0 — a clear sign of memorisation. The train-test gap of 0.379 makes it unreliable for anything beyond the training set. Linear Regression generalises well, is fully interpretable, and its performance is consistent across splits.

---

## Project structure

```
├── AI_ML_Task2_Model_Comparison.ipynb    # Full pipeline notebook
├── AI_ML_Task2_Methodology_Report.pdf    # Written methodology and findings
├── best_model_linear_regression.pkl      # Saved model (joblib)
├── scaler.pkl                            # Saved StandardScaler
└── README.md
```

---

## How to run

**Clone the repo**
```bash
git clone https://github.com/your-username/california-housing-ml.git
cd california-housing-ml
```

**Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

**Open the notebook**
```bash
jupyter notebook AI_ML_Task2_Model_Comparison.ipynb
```

The dataset is loaded directly from `sklearn.datasets` so no separate download is needed. Works in Google Colab as well.

---

## Dataset

**California Housing Dataset** — 20,640 samples, 8 input features, 1 continuous target (median house value in units of $100,000).

The top predictors by feature importance (from the Decision Tree):

- `MedInc` — median income (52.8% importance)
- `AveOccup` — average household occupancy (13.1%)
- `Latitude` (9.2%)

No missing values, no categorical encoding required. Straightforward dataset for learning regression workflows end-to-end.

---

## A few things I learned along the way

Fitting the scaler before the train/test split is a common mistake I almost made here. It leaks information from the test set into training, which inflates your metrics in ways that don't show up until production. Keeping the split first fixed that.

The Decision Tree's perfect training accuracy (R² = 1.00) seemed impressive at first. But looking at the overfitting gap made it obvious that the model was just memorising the training data rather than learning anything generalizable.

Linear Regression ended up being the better choice not because it scored higher, but because it behaved predictably and its coefficients actually tell you something meaningful about the relationship between income, location, and house prices.

---

## Tech stack

Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

---

## Internship context

Maincrafts Technology AI/ML Internship — Task 2: Feature Engineering, Model Optimisation & Performance Comparison.
