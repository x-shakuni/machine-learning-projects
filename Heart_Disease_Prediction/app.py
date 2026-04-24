import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')


st.title('Heart Stroke Prediction by Ayush')
st.markdown("Provide the following details to predict the likelihood of a heart stroke:")
age = st.slider('Age', 18, 100, 40)
sex = st.selectbox("SEX", ['M', 'F'])
pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.number_input('Cholesterol (mg/dl)', 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
rest_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.number_input('ST Depression', 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])


if st.button('Predict'):
    raw_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex'+ sex: 1,
        'ChestPainType' + pain: 1,
        'RestECG'+ rest_ecg: 1,
        'ExerciseAngina'+ exercise_angina: 1,
        'ST_Slope'+ st_slope: 1

    }
    input_df = pd.DataFrame([raw_data])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("High likelihood of heart stroke. Please consult a doctor.")
    else:
        st.success("Low likelihood of heart stroke. Keep up the healthy lifestyle!")