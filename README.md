# Safe-driver-prediction
Of course. This is an excellent, complete notebook that showcases a professional workflow. Here is a well-structured README file that clearly explains the project, your methodology, and the impressive results.

 The goal is to build a predictive model to determine whether a driver will initiate an auto insurance claim in the following year. This is a binary classification task performed on a large, anonymized, and highly imbalanced dataset.

## Project Summary

The solution follows a structured machine learning pipeline, progressing from data preparation and feature engineering to building and ensembling multiple state-of-the-art models. The final submission is a blended average of predictions from LightGBM and XGBoost, leveraging the diverse strengths of both algorithms.

### 1. Data Preparation & Cleaning

The initial phase focused on understanding and preparing the provided tabular data for modeling.

*   **Data Loading & Inspection:** Loaded `train.csv` and `test.csv` and performed initial exploratory analysis (`.head()`, `.shape`).
*   **Target and ID Separation:** The `target` variable and `id` columns were separated from the feature sets at the beginning to prevent data leakage.
*   **Consistent Preprocessing:** Training and test sets were concatenated to ensure all cleaning and feature engineering steps were applied identically. A `train` flag was added to facilitate splitting the data back after processing.
*   **Handling Missing Values:**
    *   Identified that missing values in this dataset were encoded as `-1`.
    *   Replaced all instances of `-1` with the standard `np.nan` to enable proper handling.
    *   Employed a nuanced imputation strategy:
        *   **Categorical Features (`_cat`):** Filled `NaN`s with the **mode** (most frequent value).
        *   **Numerical Features:** Analyzed the **skewness** of each column. Skewed features were imputed with the **median**, while symmetrical features were imputed with the **mean**, providing a more robust imputation than a one-size-fits-all approach.
*   **Feature Encoding:**
    *   Recognized that columns with the `_cat` suffix, despite being numerically stored, represent nominal categorical data.
    *   Applied **One-Hot Encoding** (`pd.get_dummies`) to these specific columns to correctly represent them for the models, preventing the learning of a false ordinal relationship.

### 2. Modeling & Evaluation

The project's success was measured using the competition's metric, the **Normalized Gini Coefficient**. As this is directly related to the **ROC AUC** score (`Gini = 2 * AUC - 1`), all models were trained and evaluated by optimizing for `roc_auc`.

A **5-fold Stratified Cross-Validation** strategy was used for all model training to ensure robust evaluation and to handle the significant class imbalance in the target variable.

#### a. Model 1: LightGBM (LGBM)

*   An `LGBMClassifier` was trained using the 5-fold CV strategy with `early_stopping` to find the optimal number of boosting rounds for each fold.
*   The model's out-of-fold (OOF) predictions were used to calculate a reliable validation score.

#### b. Model 2: XGBoost (XGB)

*   An `XGBClassifier` was trained following the same robust 5-fold CV and `early_stopping` methodology.
*   This allowed for a direct comparison of performance between two leading gradient boosting libraries.

### 3. Final Prediction: Blending Ensemble

Instead of relying on a single model, the final submission was created by **blending** the predictions of the two strong, diverse models.

*   The test set predictions from the LightGBM model (`sub_preds`) and the XGBoost model (`sub_preds_xgb`) were averaged with equal weight:
    `ensemble_preds = 0.5 * sub_preds + 0.5 * sub_preds_xgb`
*   This simple averaging technique leverages the unique strengths of each model, typically resulting in a more robust and accurate final prediction than either model could achieve alone.

## Results

*   **LightGBM (OOF Gini):** 0.279883
*   **XGBoost (OOF Gini):** 0.282343
*   **Final Submission:** The blended predictions were saved to `submission_ensemble.csv`. The ensemble results in a gini coefficient :
*   0.28377 
