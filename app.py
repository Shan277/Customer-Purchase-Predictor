import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("🛒 Customer Purchase Predictor")

st.write("Predict whether a customer is likely to purchase.")

age = st.number_input("Age", min_value=18, max_value=100)
salary = st.number_input("Estimated Salary", min_value=0)
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "EstimatedSalary": [salary],
        "Gender": [gender]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Customer is likely to purchase!")
    else:
        st.error("Customer is unlikely to purchase.")