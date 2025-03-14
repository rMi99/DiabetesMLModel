from flask import Flask, request, jsonify
import joblib
import numpy as np
from data_preprocessing import preprocess_input

app = Flask(__name__)

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('expected_columns.pkl')

def interpret_prediction(prediction):
    if prediction == 0:
        return "No diabetes"
    elif prediction == 1:
        return "Low diabetic"
    elif prediction == 2:
        return "High diabetic"
    else:
        return "Unknown"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = {
            'gender': data['gender'],
            'age': data['age'],
            'hypertension': data['hypertension'],
            'heart_disease': data['heart_disease'],
            'smoking_history': data['smoking_history'],
            'bmi': data['bmi'],
            'HbA1c_level': data['HbA1c_level'],
            'blood_glucose_level': data['blood_glucose_level']
        }
        features = preprocess_input(input_data, scaler, expected_columns)
        prediction = model.predict(features)
        prediction_label = interpret_prediction(int(prediction[0]))
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)