import streamlit as st
import joblib
import numpy as np
import os

# Auto-train model if not already trained
if not os.path.exists("salary_model.pkl") or not os.path.exists("label_encoders.pkl"):
    from model import train_model
    train_model()

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼")
st.title("💼 Employee Salary Prediction")

# Input fields
exp = st.slider("Years of Experience", 0, 40, 2)
edu = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
job_title = st.selectbox("Job Title", ["Data Analyst", "Software Engineer", "ML Engineer", "Web Developer", "Manager"])
loc = st.selectbox("Location", ["New York", "San Francisco", "Austin", "Seattle", "Remote"])
company = st.selectbox("Company Size", ["Small", "Medium", "Large"])
skill_score = st.slider("Skill Score (1-10)", 1, 10, 5)
cert = st.selectbox("Certifications", ["Yes", "No"])

# Prediction only on button click
if st.button("Predict Salary 💰"):
    try:
        input_data = np.array([
            exp,
            encoders['Education_Level'].transform([edu])[0],
            encoders['Job_Title'].transform([job_title])[0],
            encoders['Location'].transform([loc])[0],
            encoders['Company_Size'].transform([company])[0],
            skill_score,
            encoders['Certifications'].transform([cert])[0]
        ]).reshape(1, -1)

        prediction = model.predict(input_data)
        st.success(f"Estimated Salary: ${int(prediction[0]):,}")

    except ValueError as e:
        st.error(f"❌ Input Error: {e}")
