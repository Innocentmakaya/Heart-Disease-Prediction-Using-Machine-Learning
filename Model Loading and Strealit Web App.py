import streamlit as st
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="❤️ Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    with open("model_columns.json", "r") as f:
        columns = json.load(f)
    return model, columns

rf_model, model_columns = load_model()

# ============ SIDEBAR ============
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🧍 Predict", "ℹ️ About"])

# ============ HOME PAGE ============
if page == "🏠 Home":
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("""
    ### Welcome!
    This web app predicts the likelihood of heart disease using a trained Random Forest machine learning model.

    🔍 **How it works:**
    - Enter the patient's health data in the *Predict* section.  
    - The app will analyze the data and provide a risk prediction.  
    - You can view model accuracy and insights in the *Model Insights* tab.

    """)
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=250)

# ============ PREDICTION PAGE ============
elif page == "🧍 Predict":
    st.title("🧍 Heart Disease Risk Prediction")
    st.markdown("Fill in the patient details below:")

    # Splitting the input form into columns for better layout
    col1, col2 = st.columns(2)
    user_input = []

    # Alternating features in columns
    for i, feature in enumerate(model_columns):
        if i % 2 == 0:
            val = col1.number_input(f"{feature}", step=0.01)
        else:
            val = col2.number_input(f"{feature}", step=0.01)
        user_input.append(val)

    # Predict button
    if st.button("🔍 Predict Risk"):
        input_data = np.array(user_input).reshape(1, -1)
        prediction = rf_model.predict(input_data)[0]
        proba = rf_model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("📈 Prediction Result:")

        if prediction == 1:
            st.error(
                f"⚠️ **High Risk of Heart Disease**\n\nProbability: **{proba*100:.2f}%**",
                icon="🚨"
            )
        else:
            st.success(
                f"✅ **Low Risk of Heart Disease**\n\nProbability: **{proba*100:.2f}%**",
                icon="💚"
            )

        st.markdown("---")
        st.caption("Model: Random Forest Classifier | Developed by Innocent Makaya")

# ============ ABOUT PAGE ============
elif page == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.markdown("""
    **Developer:** Innocent Makaya  
    **Model:** Random Forest Classifier  
    **Purpose:** Predict the likelihood of heart disease based on clinical data.  
    **Framework:** Streamlit + scikit-learn  

    """)
