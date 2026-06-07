# ======================================================================
# STREAMLIT APP – Telco Customer Churn Prediction System
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# -------------------------------------------------------------
# LOAD MODEL + METADATA
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    best_model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return best_model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# -------------------------------------------------------------
# UI CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align:center;color:#2c3e50;'>📉 Telco Customer Churn Prediction App</h1>
    <p style='text-align:center;font-size:17px;color:#34495e;'>
    Predict churn probability and risk level using the trained ML model.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# USER INPUT FORM (SIDEBAR)
# -------------------------------------------------------------
st.sidebar.header("Customer Profile Input")
st.sidebar.markdown("Provide customer details to get churn probability.")

# Basic Fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen?", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

# Service Fields
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
stream_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
stream_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Contract / Billing
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# Numeric Fields
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 1.0, 150.0, 65.0)
total_charges = tenure * monthly_charges

# -------------------------------------------------------------
# BUILD INPUT ROW → MATCH TRAINING FEATURES
# -------------------------------------------------------------
input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "TechSupport": tech_support,
    "StreamingTV": stream_tv,
    "StreamingMovies": stream_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Apply same preprocessing
# One-hot encode
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Scale numerical fields
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

# -------------------------------------------------------------
# PREDICT
# -------------------------------------------------------------
if st.sidebar.button("Predict Churn"):
    probability = model.predict_proba(input_encoded)[0][1]
    probability_pct = probability * 100

    # Risk category
    if probability < 0.30:
        risk = "LOW RISK"
        color = "#2ecc71"
    elif probability < 0.60:
        risk = "MEDIUM RISK"
        color = "#f1c40f"
    else:
        risk = "HIGH RISK"
        color = "#e74c3c"

    # ---------------------------------------------------------
    # DISPLAY OUTPUT (CENTER)
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <h2 style='text-align:center;color:#2c3e50;'>Prediction Results</h2>
        <h3 style='text-align:center;color:{color};'>
            Churn Probability: {probability_pct:.2f}%
        </h3>
        <h3 style='text-align:center;color:{color};'>
            {risk}
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # GAUGE METER
    # ---------------------------------------------------------
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            title={"text": "Churn Likelihood (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 60], "color": "#f1c40f"},
                    {"range": [60, 100], "color": "#e74c3c"},
                ],
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # BUSINESS INTERPRETATION SECTION
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <hr>
        <h2 style='color:#2c3e50;'>📊 Business Interpretation</h2>
        <p style='font-size:16px;color:#2c3e50;'>
        The predicted churn probability for this customer is <b>{probability_pct:.2f}%</b>.
        Based on this value, the customer falls under the <b style='color:{color};'>{risk}</b> segment.
        </p>

        <h3 style='color:#2c3e50;'>Recommended Actions:</h3>
        <ul style='font-size:16px;color:#2c3e50;'>
            <li>Enhance customer support outreach.</li>
            <li>Offer personalized discounts or service upgrades.</li>
            <li>Promote autopay and long-term contract options.</li>
            <li>Monitor service usage and satisfaction levels.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<p style='text-align:center;color:#7f8c8d;'>Fill the details and click Predict to see results.</p>",
        unsafe_allow_html=True,
    )
