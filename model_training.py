from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from data_preprocessing import load_and_preprocess_data

def train_model(data_filepath):
    print("Training model...")
    X_train, X_test, y_train, y_test, scaler, expected_columns = load_and_preprocess_data(data_filepath)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(expected_columns, 'expected_columns.pkl')

if __name__ == "__main__":
    train_model('diabetes_prediction_dataset.csv')