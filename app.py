import xgboost
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Define dummy class to handle pickle loading
class XGBoostLabelEncoder:
    pass

# Patch the missing class into xgboost
xgboost.compat.XGBoostLabelEncoder = XGBoostLabelEncoder

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Example prediction logic (adjust based on your form inputs)
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Loan Default Prediction: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
