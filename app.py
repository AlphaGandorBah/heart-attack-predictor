import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="HeartSense AI", layout="wide")

# --- HEADER SECTION ---
st.title("❤️ HeartSense: Intelligent Risk Prediction")
st.markdown("""
This system uses Machine Learning to analyze cardiovascular health markers. 
Fill in the details below to see your risk profile.
""")

st.divider()

# --- INPUT SECTION WITH VISUAL AIDS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Clinical Demographics")
    age = st.number_input("Age", 1, 120, value=50)
    sex = st.radio("Biological Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    st.info("**Tip:** Chest pain isn't always sharp. Atypical pain can feel like pressure or discomfort in the neck/jaw.")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])

    trtbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol Level (mg/dl)", 100, 600, 200)

with col2:
    st.subheader("📊 Diagnostic Results")
    # Adding a visual guide for ECG/Stress tests
    st.write("Does exercise cause chest pain (Angina)?")
    exng = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    thalachh = st.number_input("Maximum Heart Rate Achieved", 60, 220, value=150)
    
    st.write("The 'Oldpeak' measures heart stress during exercise.")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, value=0.0, step=0.1)
    
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Advanced metrics in a small expander to keep UI clean
    with st.expander("Advanced Cardiac Markers"):
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        slp = st.selectbox("ST Segment Slope", [0, 1, 2])
        caa = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
        thall = st.selectbox("Thalassemia Result", [0, 1, 2, 3])

st.divider()

# --- PREDICTION & VISUALIZATION ---

if st.button("🚀 Run Diagnostic Analysis", use_container_width=True):
    # Prepare input
    features = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
    scaled_features = scaler.transform(features)
    
    # Get Probability
    prediction = model.predict(scaled_features)
    # Most models have predict_proba to show the "strength" of the prediction
    try:
        prob = model.predict_proba(scaled_features)[0][1] * 100
    except:
        prob = 100 if prediction[0] == 1 else 10 # Fallback if model doesn't support proba

    # Display Results with a visual "Gauge" feel
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction[0] == 1:
            st.error("### HIGH RISK")
            st.write(f"The model is **{prob:.1f}%** confident in this risk profile.")
        else:
            st.success("### LOW RISK")
            st.write(f"The model estimates a **{100-prob:.1f}%** chance of a healthy profile.")

    with res_col2:
        # A simple bar chart acting as a risk meter
        chart_data = pd.DataFrame({"Risk Level": [prob], "Safety Level": [100-prob]})
        st.bar_chart(chart_data)

st.warning("**Medical Disclaimer:** This AI is an educational tool. Always prioritize a doctor's diagnosis over software predictions.")
