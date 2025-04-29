# app/streamlit_app.py

import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# Resolve paths relative to repo root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'model' / 'mlp_cnc_model.pkl'

# Load model and scaler
if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}")
else:
    with open(MODEL_PATH, 'rb') as f:
        model, scaler = pickle.load(f)

# Streamlit UI
st.title('🛠️ CNC Machine Maintenance Prediction')
st.markdown('Predict if a CNC machine needs maintenance based on operating conditions.')

# Input fields
operating_hours = st.number_input('Operating Hours (hours)', min_value=0.0)
temperature = st.number_input('Internal Temperature (°C)', min_value=0.0)
vibration_level = st.number_input('Vibration Level (mm/s)', min_value=0.0)
coolant_flow = st.number_input('Coolant Flow Rate (liters/min)', min_value=0.0)
load_percentage = st.number_input('Load Percentage (%)', min_value=0.0)
acoustic_emission = st.number_input('Acoustic Emission (dB)', min_value=0.0)
previous_failures = st.number_input('Previous Failures (count)', min_value=0)
ambient_humidity = st.number_input('Ambient Humidity (%)', min_value=0.0)
ambient_temperature = st.number_input('Ambient Temperature (°C)', min_value=0.0)

# Predict button
def predict_condition(data):
    data_arr = np.array([data])
    data_scaled = scaler.transform(data_arr)
    return model.predict(data_scaled)[0]

if st.button('Predict Maintenance Condition'):
    sample = [
        operating_hours, temperature, vibration_level, coolant_flow,
        load_percentage, acoustic_emission, previous_failures,
        ambient_humidity, ambient_temperature
    ]
    try:
        pred = predict_condition(sample)
        if pred == 0:
            st.success('✅ Machine Status: Normal')
        elif pred == 1:
            st.warning('⚠️ Machine Status: Maintenance Required Soon')
        elif pred == 2:
            st.error('🚨 Machine Status: Immediate Maintenance Required')
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.markdown('---')
st.caption('Developed as part of CNC Machine Predictive Maintenance MLOps Project')


