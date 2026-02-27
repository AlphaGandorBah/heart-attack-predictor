import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Heart Attack Risk Prediction System")

st.write("Enter patient details below:")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.number_input("Chest Pain Type", 0, 3)
trtbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol Level")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0,1])
restecg = st.number_input("Rest ECG", 0, 2)
thalachh = st.number_input("Max Heart Rate")
exng = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
oldpeak = st.number_input("Oldpeak")
slp = st.number_input("Slope", 0, 2)
caa = st.number_input("Number of Major Vessels", 0, 3)
thall = st.number_input("Thalassemia", 0, 3)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs,
                            restecg, thalachh, exng, oldpeak,
                            slp, caa, thall]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
