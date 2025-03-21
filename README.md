# Diabetic Prediction ML with API

This project aims to create a Machine Learning model to predict the likelihood of diabetes and expose the model via an API.

## Steps

1. **Data Collection**: Gather relevant data for diabetic prediction.
2. **Data Preprocessing**: Clean and preprocess the data.
3. **Model Training**: Build and train the ML model.
4. **Model Evaluation**: Evaluate the model's performance.
5. **API Development**: Develop an API to expose the trained model.
6. **Deployment**: Deploy the API to a server or cloud service.

## Requirements

- Python 3.x
- Flask
- Scikit-learn
- Pandas
- Numpy

## Setup

1. Clone the repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   python app.py
   ```

## API Endpoints

- **POST /predict**: Predict the likelihood of diabetes based on input data.

## Example Request

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "gender": "Female",
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 28.5,
  "HbA1c_level": 5.5,
  "blood_glucose_level": 130
}' http://localhost:5000/predict
```