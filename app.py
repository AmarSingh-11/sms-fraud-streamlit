from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('sms_fraud_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # same as used during training

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pred = model.predict(padded)[0][0]
        prediction = "⚠️ Fraud Detected" if pred > 0.5 else "✅ Legit Message"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

