
# streamlit_app.py
import streamlit as st
import requests
import json
import pandas as pd

# Load dropdown options
with open("dropdown_values.json") as f:
    dropdown_options = json.load(f)

st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f8f9fa; padding: 30px; border-radius: 15px;}
    .title {font-size: 2.5em; color: #343a40; text-align: center; margin-bottom: 30px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🎯 Employee Promotion Predictor</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    st.subheader("🔍 Fill in the employee information:")

    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox("Department", dropdown_options["department"])
        region = st.selectbox("Region", dropdown_options["region"])
        education = st.selectbox("Education", dropdown_options["education"])
        gender = st.radio("Gender", dropdown_options["gender"])
        recruitment_channel = st.radio("Recruitment Channel", dropdown_options["recruitment_channel"])

    with col2:
        no_of_trainings = st.slider("Number of Trainings", 1, 10, 1)
        age = st.slider("Age", 18, 60, 30)
        previous_year_rating = st.slider("Previous Year Rating", 1.0, 5.0, 3.0, step=0.1)
        length_of_service = st.slider("Length of Service (years)", 1, 40, 5)
        awards_won = st.selectbox("Awards Won?", [0, 1])
        avg_training_score = st.slider("Average Training Score", 40, 100, 60)

    submit = st.form_submit_button("🔮 Predict Promotion")

    if submit:
        payload = {
            "department": department,
            "region": region,
            "education": education,
            "gender": gender,
            "recruitment_channel": recruitment_channel,
            "no_of_trainings": no_of_trainings,
            "age": age,
            "previous_year_rating": previous_year_rating,
            "length_of_service": length_of_service,
            "awards_won": awards_won,
            "avg_training_score": avg_training_score
        }

        try:
            response = requests.post("http://localhost:7860/predict", json=payload)
            result = response.json()

            if result["status"] == "ok":
                prediction = "Yes ✅" if result["prediction"] == 1 else "No ❌"
                st.success(f"**Promotion Prediction: {prediction}**")
            else:
                st.error(f"Error: {result['message']}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
