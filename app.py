import streamlit as st
import numpy as np
import joblib

st.title("Chronic Liver Disease Detection System")

st.write("Enter patient details:")

# Example inputs (adjust based on your dataset)
age = st.number_input("Age")
bilirubin = st.number_input("Total Bilirubin")
alk_phosphate = st.number_input("Alkaline Phosphotase")

if st.button("Predict"):
    try:
        model = joblib.load("model.pkl")  # make sure this file exists
        data = np.array([[age, bilirubin, alk_phosphate]])
        prediction = model.predict(data)
        st.success(f"Prediction: {prediction[0]}")
    except:
        st.error("Model file not found or input mismatch")
