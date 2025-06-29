import streamlit as st
import requests
import json

# Load dropdown choices
with open("dropdown_values.json") as f:
    options = json.load(f)

st.title("🚗 Car Price Prediction")

with st.form("predict_form"):
    brand = st.selectbox("Brand", options['brand'])
    model = st.selectbox("Model", options['model'])
    title_status = st.selectbox("Title Status", options['title_status'])
    color = st.selectbox("Color", options['color'])
    state = st.selectbox("State", options['state'])
    condition = st.selectbox("Condition", options['condition'])
    
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
    mileage = st.number_input("Mileage", min_value=0, value=30000)
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = {
        "brand": brand,
        "model": model,
        "title_status": title_status,
        "color": color,
        "state": state,
        "condition": condition,
        "year": year,
        "mileage": mileage
    }

    # GANTI URL BERIKUT JIKA API DIPISAHKAN
    url = "http://localhost:5000/predict"
    
    try:
        res = requests.post(url, json=input_data)
        result = res.json()
        if result["status"] == "ok":
            st.success(f"💰 Predicted Price: ${result['prediction']:,.2f}")
        else:
            st.error(f"Prediction failed: {result['message']}")
    except Exception as e:
        st.error(f"Error contacting API: {e}")
