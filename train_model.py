"""
Heart Disease Prediction - Model Training Script
=================================================
This script:
  1. Loads the UCI Heart Disease dataset (Cleveland subset)
  2. Splits it into train (80%) and test (20%) sets
  3. Builds preprocessing + model Pipelines
  4. Compares 6 classification algorithms using 5-fold cross-validation
  5. Tunes the best model with GridSearchCV
  6. Evaluates the tuned model on the held-out test set
  7. Saves the best fitted Pipeline to model.pkl

Run:  python train_model.py
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
DATA_PATH = "data/dataset.csv"

# Feature groups (decided during EDA in notebooks/analysis.ipynb)
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

print("=" * 60)
print("HEART DISEASE PREDICTION - MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset loaded: {df.shape[0]} records, {df.shape[1] - 1} features")

X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------------------------------------------
# 2. Train-test split (stratify keeps class ratio same in both sets)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train set: {X_train.shape[0]} records | Test set: {X_test.shape[0]} records")

# ------------------------------------------------------------------
# 3. Preprocessing inside a ColumnTransformer
#    - Numeric:     impute missing values with median, then scale
#    - Categorical: impute missing values with most frequent, then one-hot encode
#    This same preprocessing is part of the Pipeline, so it also runs at prediction time.
# ------------------------------------------------------------------
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC_FEATURES),
    ("cat", categorical_pipe, CATEGORICAL_FEATURES),
])

# ------------------------------------------------------------------
# 4. Candidate models (at least 3 required; we compare 6)
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
}

print("\n--- 5-Fold Cross-Validation Comparison (on training set) ---")
print(f"{'Model':<25} {'CV Accuracy (mean ± std)':<30}")
print("-" * 55)

cv_results = {}
for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name:<25} {scores.mean():.4f} ± {scores.std():.4f}")

# ------------------------------------------------------------------
# 5. Hyperparameter tuning with GridSearchCV on the two strongest models
# ------------------------------------------------------------------
print("\n--- Hyperparameter Tuning (GridSearchCV, cv=5) ---")

param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
    },
    "Random Forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
    },
}

best_estimators = {}
for name, grid in param_grids.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", models[name]),
    ])
    grid_search = GridSearchCV(
        pipe, grid, cv=5, scoring="accuracy", n_jobs=-1, refit=True
    )
    grid_search.fit(X_train, y_train)
    best_estimators[name] = grid_search
    print(f"\n{name}:")
    print(f"  Best CV accuracy : {grid_search.best_score_:.4f}")
    print(f"  Best parameters  : {grid_search.best_params_}")

# Pick the single best tuned model overall
best_name = max(best_estimators, key=lambda k: best_estimators[k].best_score_)
best_pipeline = best_estimators[best_name].best_estimator_
print(f"\n>> Selected final model: {best_name}")

# ------------------------------------------------------------------
# 6. Final evaluation on the held-out test set
#    (test set is touched ONLY now, after all model selection is done)
# ------------------------------------------------------------------
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Final Test Set Evaluation ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ------------------------------------------------------------------
# 7. Save the complete fitted Pipeline (preprocessing + model) with pickle
# ------------------------------------------------------------------
MODEL_PATH = "model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f)
print(f"\nSaved fitted pipeline -> {MODEL_PATH}")
print("The Streamlit app (app.py) loads this file for predictions.")
