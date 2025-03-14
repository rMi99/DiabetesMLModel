import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Handle categorical data
    data = pd.get_dummies(data, columns=['gender', 'smoking_history'], drop_first=True)
    
    # Preprocess the data
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.to_list()

def preprocess_input(input_data, scaler, expected_columns):
    df = pd.DataFrame([input_data])
    
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    df_scaled = scaler.transform(df)
    
    return df_scaled

def interpret_prediction(prediction):
    if prediction == 0:
        return "No diabetes"
    elif prediction == 1:
        return "Low diabetic"
    elif prediction == 2:
        return "High diabetic"
    else:
        return "Unknown"