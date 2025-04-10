import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model('sms_fraud_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # Must match training

# Streamlit UI
st.set_page_config(page_title="SMS Fraud Detector", layout="centered")
st.title("\U0001F4E9 SMS Fraud Detection")
st.write("Enter your SMS or OTP text below:")

user_input = st.text_area("Message Text", height=150)

if st.button("🔍 Detect"):
    if user_input:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
        prediction = model.predict(padded)[0][0]
        label = "⚠️ Fraud Detected" if prediction > 0.5 else "✅ Legit Message"
        st.subheader("Prediction:")
        st.markdown(f"### {label}")
    else:
        st.warning("Please enter a message first.")
