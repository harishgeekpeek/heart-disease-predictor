"""
Heart Disease Prediction - Streamlit App
========================================
Loads the saved Pipeline (model.pkl) and lets a user enter patient
measurements to get a heart-disease risk prediction.

Run:  streamlit run app.py
"""

import pickle

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Load the saved model pipeline (done once and cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# ------------------------------------------------------------------
# Page settings
# ------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Risk Predictor")
st.markdown(
    "Enter the patient's clinical measurements below and click "
    "**Predict** to estimate the likelihood of heart disease."
)
st.divider()

# ------------------------------------------------------------------
# Input controls (one control per feature)
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 29, 77, 54)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic",
        ],
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    restecg = st.selectbox(
        "Resting ECG Result",
        options=["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
    )

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
    exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=["Upsloping", "Flat", "Downsloping"],
    )
    ca = st.selectbox("Major Vessels Colored (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia Result",
        options=["Normal", "Fixed defect", "Reversible defect"],
    )

# ------------------------------------------------------------------
# Convert the UI choices back to the numeric codes used in training
# ------------------------------------------------------------------
def build_input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp) + 1,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": ["Upsloping", "Flat", "Downsloping"].index(slope) + 1,
            "ca": ca,
            "thal": {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal],
        }]
    )

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
if st.button("Predict", type="primary", use_container_width=True):
    input_df = build_input_frame()
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"**High risk of heart disease** (estimated probability: {probability:.0%})")
        st.markdown(
            "The model finds patterns in the patient's measurements that are "
            "similar to patients diagnosed with heart disease in the dataset. "
            "A cardiologist's review and further tests are recommended."
        )
    else:
        st.success(f"**Low risk of heart disease** (estimated probability of disease: {probability:.0%})")
        st.markdown(
            "The model finds patterns in the patient's measurements that are "
            "similar to healthy patients in the dataset. This does not rule out "
            "future risk — regular check-ups are still advised."
        )

    with st.expander("How was this calculated?"):
        st.markdown(
            "- The inputs go through the **same preprocessing** (imputation, scaling, "
            "one-hot encoding) that was fitted during training.\n"
            "- The tuned **Logistic Regression** model then outputs a probability.\n"
            "- The probability is compared to the 0.5 threshold to decide the class."
        )

# ------------------------------------------------------------------
# Limitations disclaimer
# ------------------------------------------------------------------
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This is an educational machine-learning prototype trained on the "
    "UCI Heart Disease dataset (303 records). It is NOT a medical device and must not be "
    "used for real clinical, financial, or legal decisions. Always consult a qualified "
    "healthcare professional."
)
