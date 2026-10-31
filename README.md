# Heart Disease Prediction — ML Mini Project

A complete machine-learning workflow that predicts whether a patient has heart disease from 13 clinical measurements. Built with Python, scikit-learn, and Streamlit.

> **Educational prototype only.** This app is not a medical device and must not be used for real clinical decisions.

## Problem Statement

**Predict** the presence of heart disease (binary: disease / no disease) using **input features** such as age, chest pain type, blood pressure, cholesterol, maximum heart rate, and exercise results. The prediction can help flag high-risk patients for further clinical review. It is a **classification** task. Limitation: the model is an academic prototype trained on a small historical dataset — not a substitute for professional medical diagnosis.

## Dataset

| Item | Value |
|---|---|
| Source | [UCI Machine Learning Repository — Heart Disease (id=45)](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Subset used | Cleveland database (`processed.cleveland.data`) |
| Records | 303 |
| Features | 13 (plus binary target) |
| Target | `0` = no disease, `1` = disease present (original diagnoses 1–4 collapsed to 1) |
| Missing values | `ca` (4), `thal` (2) — handled by imputation inside the model pipeline |
| License | CC BY 4.0 |

**Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

### Feature meanings

| Column | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `restecg` | Resting ECG result: 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | 3 = normal, 6 = fixed defect, 7 = reversible defect |

## Repository Structure

```
ML_Mini_Project/
├── data/
│   └── dataset.csv          # UCI Cleveland data (303 rows)
├── notebooks/
│   └── analysis.ipynb       # EDA, model comparison, tuning, evaluation
├── train_model.py           # Reproducible training script → model.pkl
├── app.py                   # Streamlit application
├── model.pkl                # Saved fitted Pipeline (preprocessing + model)
├── requirements.txt
├── README.md
├── report.md                # Project report
├── VIVA_NOTES.md            # Viva preparation Q&A
└── .gitignore
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Re)train the model — creates/updates model.pkl
python train_model.py

# 3. Launch the web app
streamlit run app.py
```

The EDA notebook can be opened with Jupyter or VS Code:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Approach Summary

1. **Preprocessing** (inside a `Pipeline`): median imputation + scaling for numeric features; most-frequent imputation + one-hot encoding for categorical features.
2. **Split:** stratified 80/20 train-test split (`random_state=42`).
3. **Comparison:** 6 algorithms scored by 5-fold cross-validation on the training set — Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Naive Bayes.
4. **Tuning:** `GridSearchCV` (cv=5) on the two strongest models (Logistic Regression, Random Forest).
5. **Final evaluation** on the untouched test set; best pipeline saved with `pickle`.

### Results (final model: Logistic Regression, C=10)

| Metric | Score |
|---|---|
| CV Accuracy (train, 5-fold) | 0.8469 |
| Test Accuracy | 0.8689 |
| Test Precision | 0.8333 |
| Test Recall | 0.8929 |
| Test F1-score | 0.8621 |
| Test ROC-AUC | 0.9632 |

Confusion matrix (test set, 61 rows):

```
[[28  5]   ← correctly healthy / healthy predicted sick
 [ 3 25]]  ← sick predicted healthy / correctly sick
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (your own account).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Deploy — Streamlit installs `requirements.txt` automatically and serves `model.pkl` from the repo.
5. Copy the deployed URL and add it here: **Deployed app link:** _\<paste after deployment\>_
6. Test the deployed link in a fresh browser/incognito session with a few valid inputs.

## Limitations

- Small dataset (303 records, from 1988) — results may not generalize to today's populations.
- The app is an **educational prototype**, not a diagnostic tool; real diagnosis requires clinicians and further tests.
- No patient history, medications, or lifestyle features are included.
- Predictions depend on user-entered values; wrong inputs give meaningless outputs.

## Team & Contribution

Solo project — all work (problem framing, data preparation, EDA, modelling, evaluation, pipeline/tuning, Streamlit app, Git organization, documentation, deployment) done by the single team member.
