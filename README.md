<h2>🛠️ My Approach</h2>

<ul>
  <li>Loaded and explored the training dataset to understand feature types, data distributions, and missing values.</li>

  <li>Performed a detailed missing value analysis by computing column-wise null counts to identify features requiring imputation.</li>

  <li>Handled missing values directly within the modeling pipeline, leveraging LightGBM’s native ability to deal with missing data efficiently.</li>

  <li>Separated features and target variable (<code>exam_score</code>) and split the data into training and validation sets for model evaluation.</li>

  <li>Used K-Fold cross-validation to ensure stable and reliable performance estimation.</li>

  <li>Trained a <strong>LightGBM Regressor (LGBMRegressor)</strong> with a regression objective to capture non-linear relationships in the data.</li>

  <li>Tuned hyperparameters using <code>GridSearchCV</code> with <code>neg_root_mean_squared_error</code> as the scoring metric.</li>

  <li>Selected the best-performing model based on cross-validated RMSE.</li>

  <li>Generated predictions on the test dataset using the optimized model.</li>

  <li>Created a valid Kaggle submission file containing <code>id</code> and predicted <code>exam_score</code> values.</li>
</ul>
