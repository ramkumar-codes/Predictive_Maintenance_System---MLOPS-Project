# app/streamlit_app.py

import streamlit as st
import numpy as np
import pickle

# Load the saved MLP model and scaler
with open('mlp_cnc_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

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
if st.button('Predict Maintenance Condition'):
    input_data = np.array([[operating_hours, temperature, vibration_level, coolant_flow,
                            load_percentage, acoustic_emission, previous_failures,
                            ambient_humidity, ambient_temperature]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 0:
        st.success('✅ Machine Status: Normal')
    elif prediction[0] == 1:
        st.warning('⚠️ Machine Status: Maintenance Required Soon')
    elif prediction[0] == 2:
        st.error('🚨 Machine Status: Immediate Maintenance Required')

# Add footer
st.markdown('---')
st.caption('Developed as part of CNC Machine Predictive Maintenance MLOps Project')
