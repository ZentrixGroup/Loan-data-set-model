import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page header
st.set_page_config(page_title="Loan Prediction App")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter client details to check loan eligibility.")

# Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    applicant_income = st.number_input("Applicant Income ($)", value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", value=0)
    loan_amount = st.number_input("Loan Amount", value=150)
    loan_term = st.number_input("Loan Amount Term", value=360)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Pre-processing inputs to match model's training data
if st.button("Predict Loan Status"):
    # Convert categorical inputs to numeric (Matching your factorize order)
    d_gender = 0 if gender == "Male" else 1
    d_married = 0 if married == "No" else 1
    d_dep = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    d_edu = 0 if education == "Graduate" else 1
    d_self = 0 if self_employed == "No" else 1
    d_area = {"Urban": 0, "Rural": 1, "Semiurban": 2}[property_area]
    
    # Prepare the feature array
    features = np.array([[d_gender, d_married, d_dep, d_edu, d_self, 
                          applicant_income, coapplicant_income, loan_amount, 
                          loan_term, credit_history, d_area]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display result
    if prediction[0] == 0:
        st.success("✅ Loan Status: APPROVED")
    else:
        st.error("❌ Loan Status: REJECTED")