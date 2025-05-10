
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and preprocessing tools
model = tf.keras.models.load_model('D:\ANN-project\models\model.h5')

with open('D:\ANN-project\models/Oht_Geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('D:\ANN-project\models/Label_Encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('D:\ANN-project\models\sclaer.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Customer Churn Prediction App 📊</h1>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📝 Customer Details")
geography = st.sidebar.selectbox('🌎 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('🧑 Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92)
credit_score = st.sidebar.number_input('💳 Credit Score')
balance = st.sidebar.number_input('💰 Balance')
estimated_salary = st.sidebar.number_input('💼 Estimated Salary')
tenure = st.sidebar.slider('⏳ Tenure', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('🟢 Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale and predict
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Result display
st.subheader("📈 Prediction Result")

if prediction_proba > 0.5:
    st.markdown(f"""
        <div style='padding: 20px; background-color: #ffcccc; border-radius: 10px; text-align: center;'>
            <h2 style='color: red;'>⚠️ High Risk of Churn</h2>
            <p style = 'color: red' >Probability: <strong>{prediction_proba:.2f}</strong></p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div style='padding: 20px; background-color: #ccffcc; border-radius: 10px; text-align: center;'>
            <h2 style='color: green;'>✅ Low Risk of Churn</h2>
            <p style = 'color: green' >Probability: <strong>{prediction_proba:.2f}</strong></p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr><center>Made with ❤️ by Achraf</center>", unsafe_allow_html=True)
