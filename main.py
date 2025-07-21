import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the new dataset
df = pd.read_csv("student_multi.csv")

# Split features and target
X = df[['Hours', 'Attendance(%)', 'SleepHours', 'PreviousScore']]
y = df['FinalScore']

# Train the model using scikit-learn
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🎯 Student Score Predictor (Multiple Features)")
st.write("Enter student details to predict final exam score:")

# User Inputs
hours = st.slider("📘 Hours Studied", 0.0, 12.0, 6.0)
attendance = st.slider("🎓 Attendance (%)", 50, 100, 85)
sleep = st.slider("🛏️ Sleep Hours", 3.0, 10.0, 6.5)
prev_score = st.slider("📄 Previous Exam Score", 0, 100, 65)

# Make prediction
input_data = np.array([[hours, attendance, sleep, prev_score]])
prediction = model.predict(input_data)[0]
st.success(f"📊 Predicted Final Exam Score: {prediction:.2f}")

# Optional: show full dataset and model coefficients
with st.expander("See model details and dataset"):
    st.dataframe(df)
    st.write("Model Coefficients:")
    for col, coef in zip(X.columns, model.coef_):
        st.write(f"- {col}: {coef:.2f}")
