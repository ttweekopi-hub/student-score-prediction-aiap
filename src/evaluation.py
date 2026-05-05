import pandas as pd # Data manipulation
import joblib # To load the saved model
from sklearn.metrics import mean_squared_error, r2_score # Metrics
import numpy as np # For square root calculation

def evaluate_model():
    # 1. Load the cleaned data
    # We use the same data we preprocessed in Task 2
    try:
        df = pd.read_csv('data/cleaned_score.csv')
    except FileNotFoundError:
        print("Error: data/cleaned_score.csv not found. Run preprocessing first.")
        return

    # 2. Split data into Features and Target
    X = df.drop(columns=['final_test'])
    y = df['final_test']

    # 3. Load the trained model
    try:
        model = joblib.load('models/student_model.pkl')
    except FileNotFoundError:
        print("Error: models/student_model.pkl not found. Run training first.")
        return

    # 4. Make Predictions
    # The pipeline handles the OneHotEncoding automatically!
    predictions = model.predict(X)

    # 5. Calculate Metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    # 6. Print the results
    print("--- Model Evaluation Results ---")
    print(f"RMSE: {rmse:.2f} marks")
    print(f"R2 Score: {r2:.4f}")
    print("--------------------------------")

if __name__ == "__main__":
    evaluate_model()