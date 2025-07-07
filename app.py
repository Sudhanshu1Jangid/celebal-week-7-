import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = joblib.load('model.pkl')
feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                 'petal length (cm)', 'petal width (cm)']
target_names = ['Setosa', 'Versicolor', 'Virginica']

st.title("🌸 Iris Flower Prediction App")
st.write("Enter the flower measurements to predict the species.")

# Sidebar inputs
sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                          columns=feature_names)

# Show input
st.subheader("🔍 Input Data")
st.write(input_data)

# Predict
prediction = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]

st.subheader("🎯 Prediction")
st.success(f"Predicted Species: **{target_names[prediction]}**")

# Show probability
st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame([probs], columns=target_names)
st.bar_chart(prob_df.T)

# Show feature importance
st.subheader("🧠 Feature Importance")
importances = model.feature_importances_
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
imp_df = imp_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
st.pyplot(fig)
