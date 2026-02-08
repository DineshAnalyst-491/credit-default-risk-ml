import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Credit Risk Assessment", layout="wide")

# -------------------------------
# Load model & scaler
# -------------------------------
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("💳 Credit Card Default Risk Assessment System")
st.markdown("Predict customer default risk using Machine Learning")

st.sidebar.header("Customer Information")

# -------------------------------
# Mappings for categorical fields
# -------------------------------
sex_map = {"Male": 1, "Female": 2}

education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}

marriage_map = {
    "Married": 1,
    "Single": 2,
    "Divorced": 3
}

# -------------------------------
# User Inputs
# -------------------------------

inputs = []

# LIMIT_BAL
inputs.append(st.sidebar.number_input("Credit Limit (LIMIT_BAL)", value=50000.0))

# SEX
sex_ui = st.sidebar.selectbox("Sex", list(sex_map.keys()))
inputs.append(sex_map[sex_ui])

# EDUCATION
education_ui = st.sidebar.selectbox("Education", list(education_map.keys()))
inputs.append(education_map[education_ui])

# MARRIAGE
marriage_ui = st.sidebar.selectbox("Marriage Status", list(marriage_map.keys()))
inputs.append(marriage_map[marriage_ui])

# AGE
inputs.append(st.sidebar.number_input("Age", value=30))

# Repayment status
st.sidebar.subheader("Repayment Status (Months Delay)")
inputs.append(st.sidebar.number_input("PAY_1", value=0))
inputs.append(st.sidebar.number_input("PAY_2", value=0))
inputs.append(st.sidebar.number_input("PAY_3", value=0))
inputs.append(st.sidebar.number_input("PAY_4", value=0))
inputs.append(st.sidebar.number_input("PAY_5", value=0))
inputs.append(st.sidebar.number_input("PAY_6", value=0))

# Bill amounts
st.sidebar.subheader("Bill Amounts")
for i in range(1, 7):
    inputs.append(st.sidebar.number_input(f"BILL_AMT{i}", value=0.0))

# Payment amounts
st.sidebar.subheader("Payment Amounts")
for i in range(1, 7):
    inputs.append(st.sidebar.number_input(f"PAY_AMT{i}", value=0.0))

# -------------------------------
# Prediction
# -------------------------------
if st.button("Assess Risk"):

    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("## Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk Customer\n\nProbability of Default: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk Customer\n\nProbability of Default: {probability:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("### Developed by: Dinesh S")
