import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

# App Title
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write("Enter details below to predict the **Final Exam Score**")

# Input fields
attendance = st.slider("📌 Attendance (%)", 50, 100, 75)
study_hours = st.slider("📘 Study Hours per Week", 1, 15, 7)
past_score = st.slider("📊 Past Score (%)", 40, 100, 70)

# Predict button
if st.button("🔮 Predict Final Score"):
    # Create input DataFrame
    input_data = pd.DataFrame([[attendance, study_hours, past_score]],
                              columns=["Attendance", "StudyHours", "PastScore"])
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"🎯 Predicted Final Exam Score: **{prediction:.2f}%**")

    # Advice based on prediction
    if prediction < 50:
        st.warning("⚠️ At risk! Needs serious improvement in study habits and attendance.")
    elif prediction < 70:
        st.info("📘 Doing okay, but increasing study hours could help improve performance.")
    else:
        st.success("✅ Excellent performance! Keep up the good work.")
