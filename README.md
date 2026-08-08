# Predicting Student Test Scores

A tabular regression project that predicts an **exam score** using LightGBM and a cross-validated hyperparameter-search workflow.

## Objective

The project follows a standard supervised-regression pipeline:

```text
student features
      ↓
data preparation
      ↓
LightGBM regression
      ↓
exam-score prediction
```

The target variable is:

```text
exam_score
```

## Workflow

1. Load and explore training/test data.
2. Inspect feature types and missing values.
3. Separate predictors (`X`) and target (`y`).
4. Use LightGBM's native handling where appropriate.
5. Create train/validation data.
6. Use K-Fold cross-validation.
7. Tune `LGBMRegressor` with `GridSearchCV`.
8. Score candidates using negative RMSE.
9. Refit the best model.
10. Generate predictions for the Kaggle test set.
11. Create the submission file.

## Model

Primary model:

**LightGBM Regressor (`LGBMRegressor`)**

LightGBM is well suited to structured/tabular data because it can capture nonlinear interactions while remaining computationally efficient.

## Evaluation

The model-search procedure is based on **Root Mean Squared Error (RMSE)**.

RMSE penalizes large prediction errors more strongly than MAE:

```text
lower RMSE = better
```

The current README does not claim a final leaderboard score because the public repository does not clearly establish one in its documentation.

## Repository

The repository contains two working notebooks:

- `playground.ipynb`
- `playground-1.ipynb`

These document the exploratory and model-development process.

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- LightGBM
- scikit-learn

## Strengths

- Cross-validation
- Hyperparameter search
- Appropriate gradient-boosted tree model for tabular data
- Reproducible submission generation

## Limitations

- Final README should eventually include the exact dataset/competition citation.
- Cross-validation variance should be reported, not only the best mean score.
- Feature importance and error analysis are not yet documented here.
- Potential leakage should always be checked for educational/performance datasets.

## Future Work

- Add exact CV RMSE from the final run.
- Report fold-by-fold performance.
- Add feature importance / SHAP analysis.
- Compare LightGBM with CatBoost and XGBoost.
- Analyze largest prediction errors.
- Add explicit reproducibility seed/configuration.

## Author

**Divyadarshee Dash**
