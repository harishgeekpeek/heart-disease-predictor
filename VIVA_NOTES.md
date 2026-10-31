# Viva Preparation — Heart Disease Prediction

Everything you need to explain this project confidently. Read this 2–3 times before the viva.
**Rule:** if you don't know an answer, say what you *do* know and reason out loud — examiners like reasoning more than memorized answers.

---

## Part 1 — Your 2-Minute Opening Statement (memorize this)

> "This is my machine-learning mini project: **heart disease prediction**. I used the **UCI Heart Disease dataset** — 303 patient records with 13 clinical features like age, blood pressure, cholesterol, and chest pain type — and the goal is **binary classification**: predict whether a patient has heart disease or not.
>
> I started with **EDA** in a Jupyter notebook to understand distributions, class balance, and which features relate to the target. Then I built a **scikit-learn Pipeline** that does preprocessing — median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for categorical features — all inside the pipeline so the same steps run at prediction time.
>
> I **compared six algorithms** using 5-fold cross-validation: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, and Naive Bayes. Then I ran **GridSearchCV** on the two best models. **Logistic Regression with C=10** won with 84.7% CV accuracy.
>
> On the **held-out test set**, the final model got **86.9% accuracy, 0.89 recall, and 0.963 ROC-AUC**. The whole fitted pipeline was saved as **`model.pkl`** using pickle, and I built a **Streamlit app** that loads this file and predicts from user inputs — training is never repeated in the app.
>
> Everything is in a **Git repository** with meaningful commits, and the app is deployed on **Streamlit Community Cloud**."

*(Then walk them through the live demo — see Part 8.)*

---

## Part 2 — The Big Picture: How I Built It (step by step)

| Step | What I did | Tool/file |
|---|---|---|
| 1. Problem | Defined a binary classification problem + stated limitations | README, report |
| 2. Data | Downloaded UCI Cleveland data, parsed it, added column names, converted `?` to missing values, collapsed target 1–4 → 1 | `data/dataset.csv` |
| 3. EDA | Distributions, class balance, categorical vs target, correlation heatmap, insights | `notebooks/analysis.ipynb` |
| 4. Preprocessing | Imputer+scaler (numeric), imputer+one-hot (categorical) inside a ColumnTransformer inside a Pipeline | notebook + `train_model.py` |
| 5. Split | Stratified 80/20, random_state=42 | notebook + `train_model.py` |
| 6. Compare | 6 models, 5-fold CV accuracy | notebook + `train_model.py` |
| 7. Tune | GridSearchCV (cv=5) on Logistic Regression + Random Forest | notebook + `train_model.py` |
| 8. Evaluate | Final metrics + confusion matrix + ROC curve on untouched test set | notebook + `train_model.py` |
| 9. Save | `pickle.dump(best_pipeline, "model.pkl")` | `train_model.py` |
| 10. App | Streamlit UI → converts inputs to a DataFrame → pipeline predicts → readable output | `app.py` |
| 11. Version control | Git repo, logical commits, pushed to GitHub | `.git` |
| 12. Deploy | Streamlit Community Cloud serves `app.py` + `model.pkl` from GitHub | deployed link |

**What tools I used and why:**
- **Python** — standard language for ML.
- **Pandas/NumPy** — load, clean, and manipulate the data.
- **Matplotlib/Seaborn** — EDA charts.
- **scikit-learn** — preprocessing, models, cross-validation, GridSearchCV, metrics.
- **Pickle** — serialize the fitted pipeline to a file so the app can reuse it.
- **Streamlit** — quick way to turn a Python model into a web app without writing HTML/JS.
- **Jupyter notebook** — for EDA and experimentation with visible outputs.
- **Git/GitHub** — history, collaboration-ready, and Streamlit Cloud deploys directly from GitHub.

---

## Part 3 — The Questions from the Manual (exact list)

### Q: What problem does the model solve?
It predicts the **presence of heart disease** in a patient (yes/no) from 13 routine clinical measurements. It's a **binary classification** task. Intended use: an educational screening aid that could flag high-risk patients for a doctor's review — not a diagnosis.

### Q: What does each important input feature represent?
- **age** — patient's age in years.
- **sex** — 1 male, 0 female.
- **cp (chest pain type)** — 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic. Counter-intuitively, *asymptomatic* patients had the highest disease rate in this dataset.
- **trestbps** — resting blood pressure (mm Hg) on admission.
- **chol** — serum cholesterol (mg/dl).
- **fbs** — fasting blood sugar > 120 mg/dl (1/0).
- **restecg** — resting ECG result: 0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy.
- **thalach** — maximum heart rate achieved during exercise.
- **exang** — exercise-induced angina (1/0).
- **oldpeak** — ST depression induced by exercise relative to rest (float).
- **slope** — slope of the peak-exercise ST segment: 1 upsloping, 2 flat, 3 downsloping.
- **ca** — number of major vessels (0–3) colored by fluoroscopy.
- **thal** — 3 normal, 6 fixed defect, 7 reversible defect.

### Q: Why was the selected preprocessing required?
- **Missing values:** `ca` had 4 missing and `thal` had 2 (marked `?` in the raw file). Models can't handle NaN, so I **impute** — median for numeric (robust to outliers), most-frequent for categorical (the safe "typical value" choice).
- **Scaling:** Logistic Regression, KNN, and SVM are distance/gradient-based. Without scaling, `chol` (~200–400) would dominate `oldpeak` (~0–6). `StandardScaler` puts everything on a common scale.
- **One-hot encoding:** Categorical codes like cp=1..4 or thal={3,6,7} are **not ordered numbers** — treating them as numbers would invent fake distances (e.g. thal 6 "twice" thal 3). OneHotEncoder creates binary indicator columns so the model treats categories correctly. `handle_unknown="ignore"` prevents crashes if an unseen category appears at prediction time.
- **Why inside the pipeline?** So preprocessing is fitted only on training data (no leakage) and is **re-applied identically** at prediction time — the app doesn't need separate preprocessing code.

### Q: Why were the chosen algorithms compared?
The manual requires comparing **at least three** suitable algorithms. I compared **six** spanning different families — linear (Logistic Regression), instance-based (KNN), margin-based (SVM), tree-based (Decision Tree, Random Forest), and probabilistic (Naive Bayes) — because no single algorithm is guaranteed best for a small mixed-feature dataset. Comparing with the **same CV protocol** makes the comparison fair. Decision Trees alone tend to overfit; ensembles like Random Forest usually generalize better — worth showing.

### Q: How are the evaluation metrics interpreted?
- **Accuracy** — fraction of all predictions that are correct. OK here because classes are fairly balanced (164 vs 139), but misleading if imbalanced.
- **Precision** (for class "disease") — of patients the model *flagged* sick, how many actually were sick. High precision = few false alarms.
- **Recall (sensitivity)** — of patients who *are* sick, how many the model caught. In medical screening, **missing a sick patient is worse than a false alarm**, so recall matters a lot. My model: 0.89 (caught 25 of 28).
- **F1-score** — harmonic mean of precision and recall; balances both (0.86 for me).
- **ROC-AUC** — probability that the model ranks a random sick patient above a random healthy one. 0.5 = random guessing, 1.0 = perfect. Mine: **0.963** — excellent ranking ability.
- **Confusion matrix** — `[[28, 5], [3, 25]]`: 28 correct healthy, 5 false alarms, 3 missed sick patients, 25 correct sick.

### Q: What does cv=5 mean?
5-fold cross-validation. The 242-row training set is divided into 5 equal folds. The model trains on 4 folds and validates on the remaining 1, repeating 5 times so **every row is used once for validation**. The reported score is the **mean ± standard deviation** of the 5 runs. It's more reliable than one train/test split because the score doesn't depend on a single lucky/unlucky split. Important: CV is done **only on the training set** — the test set is never part of CV.

### Q: What did GridSearchCV do?
It is an **automated hyperparameter tuner**. I gave it a grid of values to try — for Logistic Regression: `C ∈ {0.01, 0.1, 1, 10, 100}`; for Random Forest: `n_estimators ∈ {50,100,200}` × `max_depth ∈ {None,5,10,20}` × `min_samples_split ∈ {2,5,10}`. It trained every combination with cv=5, scored each, and **refit the best combination on the whole training set**. Best scores: Logistic Regression 0.8469 (C=10), Random Forest 0.8179 (n_estimators=50, max_depth=None, min_samples_split=5).

### Q: What were the best hyperparameters?
- **Final model — Logistic Regression: `C = 10`.** `C` is the inverse regularization strength: small C = stronger regularization (simpler model, avoids overfitting); large C = weaker regularization (fits training data more closely). 10 gave the best CV accuracy.
- Runner-up Random Forest: `n_estimators=50`, `max_depth=None`, `min_samples_split=5`.

### Q: Why was a Pipeline used?
1. **Reproducibility** — the exact preprocessing travels with the model; nobody can forget a step.
2. **No data leakage** — scalers/imputers are fitted only on the training fold inside CV, not on the whole dataset.
3. **One object to save** — `model.pkl` contains preprocessing + model, so the app just calls `predict()` on raw inputs.
4. **CV-safe** — GridSearchCV can cross-validate the entire preprocess+train process as a single estimator.

### Q: What was saved in the Pickle file?
The **complete fitted pipeline object**: the `ColumnTransformer` (imputers, scaler, one-hot encoder — already fitted) **plus** the tuned `LogisticRegression` model (already trained with C=10 on the training data). Pickle is Python's standard serialization format. The file is ~small (a few hundred KB) and is loaded by `app.py` with `pickle.load()`.

### Q: How does Streamlit send user input to the model?
1. The sidebar/main page has sliders and select boxes — one control per feature.
2. When you click **Predict**, the chosen values are converted back to the training codes (e.g. "Asymptomatic" → cp=4, "Male" → sex=1) and put into a **one-row pandas DataFrame** with the same column names as training.
3. `pipeline.predict_proba(input_df)` runs the saved pipeline on that row.
4. The result is shown as "High risk / Low risk" **with the probability** and a short explanation — not just a class code.

### Q: Why is training/tuning NOT repeated on every Predict click?
Training and GridSearchCV take minutes; prediction takes milliseconds. The model is trained **once** offline (`train_model.py` / notebook) and the fitted object is **saved to `model.pkl`**. The app only **loads** that file (cached with `@st.cache_resource` so even the load happens once) and calls `predict`. Retraining on every click would be slow, wasteful, and would give different results per user.

### Q: How was Git used during development?
The project was built in logical stages and **committed after each stage** with meaningful messages (e.g. "Add UCI dataset", "Add EDA notebook", "Add training script", "Add Streamlit app"). This creates a visible development history (required by the manual — not one final dump), lets me roll back mistakes, and the GitHub repo is what Streamlit Cloud deploys from.

### Q: How does the deployed app differ from the local app?
**Functionally identical** — same code, same `model.pkl`, same predictions. Differences:
- Local: runs on my machine (`streamlit run app.py`), localhost URL, I control the environment.
- Deployed: Streamlit Community Cloud clones my GitHub repo, installs `requirements.txt`, and serves it on a public HTTPS URL that anyone can open. The model file is committed to the repo so the cloud app doesn't need to retrain.

### Q: What are the limitations of the final model?
1. **Small, old dataset** — 303 records from 1988; may not generalize to today's diverse populations.
2. **Educational prototype, not a diagnostic tool** — the app shows a clear disclaimer; real diagnosis needs clinicians and further tests.
3. **Limited features** — no family history, medications, diet, or exercise habits.
4. **Input-dependent** — wrong user inputs give meaningless predictions.
5. **Mild class imbalance**; for clinical use you'd tune the decision threshold toward recall.
6. **Logistic Regression assumes linear decision boundaries** — complex non-linear relationships may need tree/ensemble models (though ROC-AUC 0.963 suggests it works well here).

---

## Part 4 — Likely Follow-Up Questions (be ready!)

**Why did you binarize the target?**
The original `num` is 0–4 (disease severity from angiography). All published experiments distinguish absence (0) from presence (1–4). Binarizing matches the literature and makes a clean classification task.

**Why not use all 303 rows for training?**
The held-out test set gives an **unbiased** estimate of performance on unseen data. If I trained and evaluated on everything, the metrics would be optimistically biased.

**Why stratify the split?**
`stratify=y` keeps the healthy/sick ratio the same in train and test (≈54%/46%), so the test set fairly represents both classes.

**What is `random_state=42`?**
A seed that makes the split (and stochastic models) reproducible — anyone re-running my code gets the same split and same numbers. 42 is just a popular convention.

**Why one-hot and not label encoding?**
Label encoding assigns arbitrary integers (cp: 1<2<3<4), inventing a fake order/distance. One-hot creates independent binary columns — correct for unordered categories.

**Could you have used SMOTE/class weighting for imbalance?**
The imbalance is mild (164/139) and recall is already 0.89, so it wasn't needed. For a more skewed dataset I would consider class weights or resampling — inside the CV loop, never before the split, to avoid leakage.

**What is data leakage? Give an example.**
Fitting a scaler on the full dataset *before* splitting — then test information leaks into training and inflates scores. My pipeline avoids this by fitting preprocessing inside each CV fold / on the training set only.

**Why Logistic Regression over Random Forest if both are good?**
Higher CV accuracy (0.847 vs 0.818), simpler and interpretable (each feature has a coefficient showing direction of influence), faster, and on small tabular data linear models often generalize as well as or better than complex ones. Not chosen *only* because of one metric.

**What does the ROC curve show?**
It plots True Positive Rate vs False Positive Rate across all probability thresholds. The AUC summarizes ranking quality. A model at (0,1) is perfect; the diagonal is random. Mine hugged the top-left → AUC 0.963.

**What would you do with more data/time?**
Try XGBoost/stacking, SHAP explanations per patient, probability calibration, threshold tuning for high recall, and a larger modern dataset.

**Is the app safe for real patients?**
No — it's an academic prototype; the disclaimer states this. Presenting it as authoritative medical advice would be unethical (the manual explicitly restricts that).

**How do you know the app uses the same preprocessing as training?**
Because preprocessing is *inside* the pickle file — the app calls the pipeline directly; there is no separate app-side preprocessing that could drift.

**What does `handle_unknown="ignore"` do?**
If a category appears at prediction time that wasn't seen during training, one-hot encoding would normally crash; this option encodes it as all-zeros instead.

**Why is the Decision Tree the worst performer?**
A single tree easily overfits (high variance) — its CV accuracy had notable spread. Random Forest (many trees, averaging) reduces that variance, and the smooth probabilistic boundary of Logistic Regression fits this small dataset well.

---

## Part 5 — Live Demo Flow (practice this)

1. Open the app (local `streamlit run app.py` or the deployed link).
2. Show the inputs and explain: "one control per feature, labeled for humans — the codes are converted back before prediction."
3. Enter a **high-risk** example: age 65, Male, Asymptomatic, BP 145, Chol 260, FBS Yes, RestECG Normal, Max HR 110, Angina Yes, Oldpeak 2.5, Flat slope, ca=2, Reversible defect → Predict → show "High risk" + probability.
4. Enter a **low-risk** example: age 35, Female, Non-anginal pain, BP 115, Chol 190, FBS No, Normal ECG, Max HR 175, No angina, Oldpeak 0.0, Upsloping, ca=0, Normal thal → Predict → show "Low risk".
5. Open the "How was this calculated?" expander → explain preprocessing → model → threshold.
6. Point to the disclaimer → "educational prototype, not a medical device."
7. Optionally open `train_model.py` and show: same feature lists, same pipeline structure — "the app and the script share one source of truth: model.pkl."

---

## Part 6 — Quick-Fire Numbers to Remember

| Thing | Value |
|---|---|
| Dataset | UCI Heart Disease, Cleveland, 303 rows, 13 features |
| Target split | 164 healthy / 139 disease (after binarizing) |
| Missing values | ca: 4, thal: 2 |
| Train/test | 242 / 61 (80/20, stratified, seed 42) |
| Models compared | 6 |
| CV | 5-fold |
| Best CV accuracy | 0.8469 (LogReg, C=10) |
| Test accuracy | 0.8689 |
| Test recall | 0.8929 |
| Test ROC-AUC | 0.9632 |
| Confusion matrix | [[28, 5], [3, 25]] |
| Final model file | model.pkl (Pipeline: preprocessing + LogisticRegression) |

---

## Part 7 — Files Cheat Sheet (if asked "show me the code")

- **`data/dataset.csv`** — the 303-row dataset.
- **`notebooks/analysis.ipynb`** — EDA + full modelling story with charts (open it and scroll through before the viva).
- **`train_model.py`** — the reproducible training script; section comments (numbered 1–7) match the workflow steps.
- **`app.py`** — Streamlit app; `load_model()` at the top, inputs in the middle, `build_input_frame()` converts UI → codes, predict block at the bottom, disclaimer last.
- **`model.pkl`** — the serialized fitted pipeline.
- **`report.md`** — full project report.
- **`README.md`** — overview, how to run, results, deployment steps.

**If asked to explain a specific line:** say what it does, and *why it's there* — e.g. `@st.cache_resource` → "loads model.pkl once and keeps it in memory, instead of reloading on every rerun."
