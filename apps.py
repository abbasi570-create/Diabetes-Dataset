import streamlit as st
import numpy as np
import joblib

# Load saved objects
model = joblib.load("diabetes_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Predict whether a patient is Diabetic or Non-Diabetic using SVM model (Accuracy: 74.03%)")

# User Inputs
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prediction
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)
    
    if prediction[0] == 1:
        st.error("🟥 Patient is Diabetic")
    else:
        st.success("🟩 Patient is Non-Diabetic")
