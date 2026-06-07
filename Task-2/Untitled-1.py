# ======================================================================
# STREAMLIT APP – Telco Customer Churn Prediction System (FINAL VERSION)
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------------------------
# LOAD ARTIFACTS (MODEL + SCALER + FEATURE COLUMNS)
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# -------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn System",
    page_icon="📉",
    layout="wide",
)

st.markdown("""
    <h1 style='text-align:center;color:#2c3e50;'>📉 Telco Customer Churn Prediction System</h1>
    <p style='text-align:center;font-size:17px;color:#34495e;'>
    ML-Powered Real-Time Customer Retention Intelligence
    </p>
    <hr>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE NAVIGATION
# -------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Churn Prediction", "Analytics Dashboard"])



# =====================================================================
# PAGE 1 — CHURN PREDICTION
# =====================================================================
if page == "Churn Prediction":

    st.sidebar.header("Customer Input Form")

    # ------------------------- BASIC INPUTS -------------------------
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen?", ["Yes", "No"])
    partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

    # ------------------------- SERVICE INPUTS -------------------------
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # ------------------------- CONTRACT & BILLING -------------------------
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

    # ------------------------- NUMERIC INPUTS -------------------------
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 1.0, 150.0, 65.0)
    total_charges = tenure * monthly_charges

    # ------------------------- ASSEMBLE RAW INPUT -------------------------
    input_df = pd.DataFrame([{
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
    }])


    # =====================================================================
    # FEATURE ENGINEERING (MUST MATCH TRAINING EXACTLY)
    # =====================================================================
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    for col in service_cols:
        if col not in input_df.columns:
            input_df[col] = "No"

    input_df["TotalServices"] = (input_df[service_cols] == "Yes").sum(axis=1)
    input_df["AvgSpendPerService"] = input_df["MonthlyCharges"] / (input_df["TotalServices"] + 1)

    contract_map = {"Month-to-month": 1, "One year": 2, "Two year": 3}
    input_df["ContractScore"] = input_df["Contract"].map(contract_map)

    input_df["HasAutoPayment"] = input_df["PaymentMethod"].apply(
        lambda x: "Yes" if "automatic" in x.lower() else "No"
    )


    # =====================================================================
    # ONE-HOT ENCODING + COLUMN ALIGNMENT
    # =====================================================================
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)


    # =====================================================================
    # SCALING NUMERIC FIELDS (MUST MATCH TRAINING)
    # =====================================================================
    num_cols = [
        "tenure", 
        "MonthlyCharges", 
        "TotalCharges",
        "TotalServices", 
        "AvgSpendPerService", 
        "ContractScore"
    ]
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])


    # =====================================================================
    # PREDICTION
    # =====================================================================
    if st.sidebar.button("Predict Churn"):

        prob = model.predict_proba(input_encoded)[0][1]
        prob_pct = prob * 100

        if prob < 0.30:
            risk = "LOW RISK"
            color = "#2ecc71"
        elif prob < 0.60:
            risk = "MEDIUM RISK"
            color = "#f1c40f"
        else:
            risk = "HIGH RISK"
            color = "#e74c3c"

        st.markdown(
            f"""
            <h2 style='text-align:center;color:{color};'>{risk}</h2>
            <h3 style='text-align:center;color:{color};'>Churn Probability: {prob_pct:.2f}%</h3>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------- GAUGE ----------------------
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#2ecc71"},
                        {"range": [30, 60], "color": "#f1c40f"},
                        {"range": [60, 100], "color": "#e74c3c"},
                    ],
                },
                title={"text": "Churn Probability (%)"},
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------- INTERPRETATION ----------------------
        st.markdown(
            f"""
            <h3 style='color:#2c3e50;'>Business Interpretation</h3>
            <p style='font-size:16px;'>
            This customer is categorized as <b style='color:{color};'>{risk}</b>.
            The predicted churn probability is <b>{prob_pct:.2f}%</b>.
            </p>
            """,
            unsafe_allow_html=True,
        )



# =====================================================================
# PAGE 2 — ANALYTICS DASHBOARD
# =====================================================================
elif page == "Analytics Dashboard":

    st.markdown("<h2>📊 Model Insights & Analytics</h2><hr>", unsafe_allow_html=True)

    try:
        imp = pd.read_csv("feature_importance.csv")

        fig = px.bar(
            imp.sort_values("importance", ascending=False).head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances",
            color="importance",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.warning("feature_importance.csv missing. Export it from the notebook.")
