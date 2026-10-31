# Project Report — Heart Disease Prediction Using Machine Learning

---

## 1. Title Page

**Title:** Heart Disease Prediction Using Machine Learning
**Course:** Machine Learning Mini Project
**Topic:** Binary classification on the UCI Heart Disease dataset (Cleveland subset)
**Tools:** Python, scikit-learn, Pandas, Matplotlib/Seaborn, Streamlit, Git/GitHub

---

## 2. Team Details

| Member | Responsibility |
|---|---|
| [Your Name] (solo member) | Problem definition, dataset understanding, EDA, data preprocessing, model implementation, comparison and evaluation, pipeline and hyperparameter tuning, Streamlit interface, Git/GitHub organization and deployment |

---

## 3. Abstract

This project builds an end-to-end machine-learning pipeline that predicts the presence of heart disease from 13 clinical measurements. Using the UCI Heart Disease dataset (303 patient records), we performed exploratory data analysis, preprocessing inside a reproducible scikit-learn `Pipeline`, and compared six classification algorithms with 5-fold cross-validation. The two strongest models were tuned with `GridSearchCV`. The final model — a tuned Logistic Regression (C=10) — achieved 86.9% test accuracy and 0.963 ROC-AUC on a held-out test set. The fitted pipeline was serialized with pickle and served through a Streamlit web application that accepts patient inputs and returns a risk assessment. The complete workflow is version-controlled with Git and documented for reproducibility and viva demonstration.

---

## 4. Problem Statement and Objectives

**Problem:** Predict whether a patient has heart disease (binary target) using clinical measurements available at check-up time.

**Objectives:**
1. Understand the dataset and perform meaningful EDA.
2. Build a reproducible preprocessing + model pipeline.
3. Compare at least three suitable classification algorithms.
4. Tune the best model with cross-validation and grid search.
5. Evaluate on an unseen test set with appropriate classification metrics.
6. Deploy the model in a simple, usable Streamlit application.

**Practical limitation stated up front:** the model is an academic prototype, not a substitute for professional medical, financial, or legal decisions.

---

## 5. Dataset Description and Source

- **Source:** UCI Machine Learning Repository — Heart Disease dataset (id 45), Cleveland subset.
  https://archive.ics.uci.edu/dataset/45/heart+disease
- **Records:** 303 patients | **Features:** 13 | **Target:** binary (`0` = no disease, `1` = disease present; original diagnoses 1–4 collapsed to 1)
- **Feature types:** mixed — 5 numeric (`age, trestbps, chol, thalach, oldpeak`) and 8 categorical/ordinal (`sex, cp, fbs, restecg, exang, slope, ca, thal`)
- **Missing values:** `ca` (4 rows), `thal` (2 rows) — represented as `?` in the raw file; handled by imputation inside the pipeline.
- **License:** CC BY 4.0.
- **Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

## 6. Data Understanding and EDA

EDA was performed in `notebooks/analysis.ipynb`:

- **Shape:** 303 × 14 (13 features + target); no duplicate rows.
- **Class balance:** 164 healthy vs 139 disease — reasonably balanced, so accuracy is usable, but precision/recall/F1 are also reported.
- **Numeric distributions:** histograms with target hue showed disease cases with higher `oldpeak`, higher `thalach`, and wide age spread.
- **Categorical breakdowns:** asymptomatic chest pain (`cp=4`), exercise-induced angina (`exang=1`), more colored vessels (`ca` higher), and abnormal `thal` all associate strongly with disease.
- **Correlation heatmap:** `cp`, `thalach`, `oldpeak`, `exang`, `ca`, `thal` showed the strongest relationships with the target.
- EDA directly informed the preprocessing choices (imputation strategy, scaling, one-hot encoding) — charts were not added merely for presentation.

---

## 7. Data Preprocessing

All preprocessing lives inside a `ColumnTransformer` that is part of the model `Pipeline`, so the exact same steps run during training **and** prediction (no train/prediction mismatch, no data leakage):

- **Numeric features:** `SimpleImputer(median)` → `StandardScaler` (scaling benefits distance/gradient-based models: KNN, SVM, Logistic Regression).
- **Categorical features:** `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` (so unseen categories at prediction time don't crash the app).
- **Split:** stratified 80/20 train-test split, `random_state=42` for reproducibility. The test set was **not** touched until final evaluation.

---

## 8. Algorithms Used

Six suitable classification algorithms were implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Support Vector Machine (SVM, RBF kernel)
4. Decision Tree
5. Random Forest
6. Naive Bayes (Gaussian)

---

## 9. Model Evaluation and Comparison

**Metrics (classification):** Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.

**5-fold CV accuracy on the training set:**

| Model | CV Accuracy (mean ± std) |
|---|---|
| Logistic Regression | 0.8428 ± 0.0534 |
| K-Nearest Neighbours | 0.8180 ± 0.0251 |
| Support Vector Machine | 0.8098 ± 0.0616 |
| Random Forest | 0.7974 ± 0.0405 |
| Naive Bayes | 0.7604 ± 0.0158 |
| Decision Tree | 0.7312 ± 0.0478 |

---

## 10. Cross-Validation and Hyperparameter Tuning

- **Cross-validation (cv=5):** the training data is split into 5 folds; the model trains on 4 and validates on 1, rotating 5 times. The reported score is the mean of the 5 runs — more reliable than a single split and reduces the effect of an unlucky train/test split.
- **GridSearchCV** was then run (cv=5) on the two strongest models:
  - Logistic Regression: `C ∈ {0.01, 0.1, 1, 10, 100}` → best `C=10` (CV accuracy 0.8469)
  - Random Forest: `n_estimators ∈ {50,100,200}`, `max_depth ∈ {None,5,10,20}`, `min_samples_split ∈ {2,5,10}` → best `(n_estimators=50, max_depth=None, min_samples_split=5)` (CV accuracy 0.8179)
- The overall best (Logistic Regression) was selected as the final model. The choice was not made on a single metric: Logistic Regression won on CV accuracy, is simple and interpretable (coefficients), and delivered strong recall and ROC-AUC — important for a medical screening problem.

---

## 11. Final Pipeline / Model

- **Final model:** Pipeline = `ColumnTransformer (impute + scale / impute + one-hot)` + `LogisticRegression(C=10)`
- **Held-out test set evaluation (61 records, evaluated only after model selection):**

| Metric | Score |
|---|---|
| Accuracy | 0.8689 |
| Precision | 0.8333 |
| Recall | 0.8929 |
| F1-score | 0.8621 |
| ROC-AUC | 0.9632 |

- **Confusion matrix:** `[[28, 5], [3, 25]]` — 53/61 correct; only 3 sick patients missed (high recall 0.89).
- The complete fitted pipeline (preprocessing + model) was saved to **`model.pkl`** with `pickle`. The app loads this file — training is never repeated at prediction time.

---

## 12. Streamlit Application

`app.py` provides a simple user-facing interface:

- Title, description, and an input control for every feature (sliders for numeric values, labeled select boxes for categories — e.g. chest pain types shown by name, not raw codes).
- **Predict** button; inputs are converted back to the training codes, assembled into a one-row DataFrame, and passed through the saved pipeline.
- Output: a clear risk statement ("High risk" / "Low risk") with the estimated disease probability and a short explanation — not an unexplained class code.
- "How was this calculated?" expander explains preprocessing → model → threshold.
- **Disclaimer** stating it is an educational prototype, not a medical device (the domain is health — a consequential domain).

---

## 13. Git/GitHub and Deployment

- The project is maintained with Git; each logical stage is a separate commit with a meaningful message (dataset added → EDA notebook → training script → Streamlit app → documentation).
- Pushed to a GitHub repository showing development history (not a single final upload).
- **Deployment:** Streamlit Community Cloud — connect the GitHub repo, set the main file to `app.py`; the platform installs `requirements.txt` and serves the app including `model.pkl`. The deployed link is tested in a separate browser session and added to the README.
- **Deployed app link:** _\<paste after deployment\>_

---

## 14. Results and Screenshots

*(Add after running the app: screenshots of the input form, a "High risk" prediction result, and a "Low risk" prediction result.)*

- Final model: Logistic Regression (C=10) — test accuracy **86.9%**, ROC-AUC **0.963**
- App runs locally via `streamlit run app.py` and deploys without retraining.

---

## 15. Limitations

- Small dataset (303 records collected in the 1980s) — limited generalizability to modern, diverse populations.
- The model is an **educational prototype**, not a diagnostic tool; real diagnosis requires clinicians and additional tests.
- Only 13 features are used; patient history, medications, lifestyle, and genetics are absent.
- Predictions depend on the correctness of user-entered values.
- Class imbalance is mild but present; for clinical use, threshold tuning and recall-focused optimization would be needed.

---

## 16. Conclusion and Future Scope

**Conclusion:** A complete ML workflow — from problem definition through EDA, pipeline-based preprocessing, six-model comparison, grid-search tuning, honest test-set evaluation, pickle serialization, and a working Streamlit deployment — was successfully implemented, with the tuned Logistic Regression model providing strong discrimination (ROC-AUC 0.963) on unseen data.

**Future scope:**
- Try ensemble stacking and calibrated probabilities.
- Add SHAP values for per-patient feature explanations.
- Collect modern, larger, more diverse data and retrain.
- Add a threshold slider in the app to trade precision vs recall.
- Secure the deployed app and add input sanity checks against clinical ranges.

---

## 17. References

1. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
2. UCI Heart Disease dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
4. Streamlit documentation: https://docs.streamlit.io

---

## 18. Individual Contribution Statement

This is a solo project. I personally completed every stage of the project: problem framing, dataset sourcing and citation, data loading and cleaning, exploratory data analysis, preprocessing design, implementation and comparison of six algorithms, cross-validation and hyperparameter tuning, final pipeline training and evaluation, pickle serialization, the Streamlit application, Git/GitHub organization, deployment, and all documentation. I understand every part of the submitted code and can explain each decision during the viva.
