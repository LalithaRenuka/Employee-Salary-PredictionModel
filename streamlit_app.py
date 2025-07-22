import streamlit as st
import pandas as pd
import joblib

# Load model correctly with joblib
model = joblib.load("salary_model.joblib")

st.set_page_config(page_title="Indian Salary Predictor", layout="centered")

st.title("💼 Indian Salary Prediction App")
st.markdown("Enter your job details to predict the expected salary.")

# Input fields
company = st.selectbox("Company Name", ["TCS", "Infosys", "Wipro", "Accenture", "Capgemini", "Other"])
job_title = st.text_input("Job Title", value="Software Engineer")
location = st.selectbox("Location", ["Bangalore", "Hyderabad", "Pune", "Chennai", "Mumbai", "Other"])
Experience = st.slider("Experience", 1, 22, value=10)

# Predict button
if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        'Company Name': [company],
        'Job Title': [job_title],
        'Experience': [Experience],
        'Location': [location]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: ₹{int(prediction):,}/yr")
