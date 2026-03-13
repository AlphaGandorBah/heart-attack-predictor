import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="HeartSense AI",
    page_icon="❤️",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

body{
background-color:#0e1117;
}

.main-title{
font-size:42px;
font-weight:700;
text-align:center;
color:#ff4b4b;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
}

.card{
padding:25px;
border-radius:12px;
background:#1f2933;
box-shadow:0px 6px 18px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOGIN SYSTEM
# ------------------------------------------------

def login():

    st.title("🔐 HeartSense Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "doctor" and password == "1234":
            st.session_state["login"] = True
        else:
            st.error("Invalid Credentials")


if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():

    model = pickle.load(open("heart_model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))

    return model, scaler

model, scaler = load_model()

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown('<p class="main-title">❤️ HeartSense AI</p>', unsafe_allow_html=True)

st.markdown(
"""
<p class="subtitle">
AI-Powered Cardiovascular Risk Prediction System
</p>
""",
unsafe_allow_html=True
)

st.divider()

# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("👤 Patient Information")

    age = st.number_input("Age",1,120,50)

    sex = st.selectbox("Sex",["Male","Female"])
    sex = 1 if sex=="Male" else 0

    cp = st.selectbox("Chest Pain Type",[
    "Typical Angina",
    "Atypical Angina",
    "Non-Anginal Pain",
    "Asymptomatic"
    ])

    cp_map = {
    "Typical Angina":0,
    "Atypical Angina":1,
    "Non-Anginal Pain":2,
    "Asymptomatic":3
    }

    cp = cp_map[cp]

    trtbps = st.slider("Blood Pressure",80,200,120)

    chol = st.slider("Cholesterol",100,600,200)

with col2:

    st.subheader("🧪 Medical Indicators")

    fbs = st.selectbox("Fasting Blood Sugar >120",["No","Yes"])
    fbs = 1 if fbs=="Yes" else 0

    thalachh = st.number_input("Max Heart Rate",60,220,150)

    exng = st.selectbox("Exercise Angina",["No","Yes"])
    exng = 1 if exng=="Yes" else 0

    oldpeak = st.number_input("ST Depression",0.0,10.0,0.0)

    with st.expander("Advanced Markers"):

        restecg = st.selectbox("Rest ECG",[0,1,2])
        slp = st.selectbox("Slope",[0,1,2])
        caa = st.selectbox("Major Vessels",[0,1,2,3])
        thall = st.selectbox("Thalassemia",[0,1,2,3])

st.divider()

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.button("🧠 Run AI Diagnosis", use_container_width=True):

    features = np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    try:
        prob = model.predict_proba(scaled)[0][1]*100
    except:
        prob = 100 if prediction[0]==1 else 10

    colA, colB = st.columns([1,2])

# ------------------------------------------------
# RESULT
# ------------------------------------------------

    with colA:

        if prediction[0]==1:

            st.error("⚠️ HIGH RISK")
            st.write(f"Risk Probability: **{prob:.2f}%**")

        else:

            st.success("✅ LOW RISK")
            st.write(f"Healthy Probability: **{100-prob:.2f}%**")

# ------------------------------------------------
# GAUGE CHART
# ------------------------------------------------

    with colB:

        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Heart Disease Risk"},
        gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "red"},
        'steps':[
        {'range':[0,30],'color':'green'},
        {'range':[30,60],'color':'yellow'},
        {'range':[60,100],'color':'red'}
        ]
        }
        ))

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# AI HEALTH RECOMMENDATIONS
# ------------------------------------------------

    st.subheader("💡 AI Health Recommendations")

    if prob > 60:

        st.write("""
        • Consult a cardiologist immediately  
        • Reduce cholesterol intake  
        • Exercise regularly  
        • Avoid smoking  
        • Monitor blood pressure
        """)

    elif prob > 30:

        st.write("""
        • Improve diet  
        • Exercise regularly  
        • Monitor blood pressure  
        """)

    else:

        st.write("""
        • Maintain healthy lifestyle  
        • Continue regular health checkups
        """)

# ------------------------------------------------
# SAVE HISTORY
# ------------------------------------------------

    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.session_state["history"].append({
    "date": datetime.now(),
    "risk": prob
    })

# ------------------------------------------------
# HISTORY DASHBOARD
# ------------------------------------------------

st.divider()

st.subheader("📊 Patient Risk History")

if "history" in st.session_state:

    df = pd.DataFrame(st.session_state["history"])

    st.line_chart(df.set_index("date"))

# ------------------------------------------------
# PDF REPORT
# ------------------------------------------------

def generate_pdf(prob):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial",size=16)
    pdf.cell(200,10,"HeartSense AI Report",ln=True)

    pdf.set_font("Arial",size=12)

    pdf.cell(200,10,f"Risk Score: {prob:.2f}%",ln=True)
    pdf.cell(200,10,f"Generated: {datetime.now()}",ln=True)

    file = "report.pdf"
    pdf.output(file)

    return file

if st.button("📄 Download Medical Report"):

    file = generate_pdf(prob)

    with open(file,"rb") as f:

        st.download_button(
        "Download PDF",
        f,
        file_name="heart_report.pdf"
        )

# ------------------------------------------------
# DISCLAIMER
# ------------------------------------------------

st.warning("""
⚠️ Medical Disclaimer

This system is an AI educational tool.
It does NOT replace professional medical diagnosis.
""")
