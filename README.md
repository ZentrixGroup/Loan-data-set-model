# Loan-data-set-model
# 🏦 Loan Approval Prediction System

This repository contains a Machine Learning web application that predicts whether a loan application will be **Approved** or **Rejected** based on applicant data. The model is built using a **Random Forest Classifier** and deployed via **Streamlit**.

## 📊 Project Overview
The goal of this project is to automate the loan eligibility process based on customer details provided while filling out online application forms. These details include Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others.

- **Model Accuracy:** ~78%
- **Algorithm:** Random Forest Classifier
- **Frontend:** Streamlit

## 📁 File Structure
- `loan_data_set.csv`: The raw dataset used for training.
- `train.py`: Python script for data cleaning, feature encoding, and model training.
- `app.py`: Streamlit web application script.
- `loan_model.pkl`: The saved (serialized) model file.
- `requirements.txt`: List of required libraries.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ZentrixGroup/loan-prediction-app.git](https://github.com/your-username/loan-prediction-app.git)
   cd loan-prediction-app
