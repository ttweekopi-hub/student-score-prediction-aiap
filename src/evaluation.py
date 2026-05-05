import pandas as pd # Data manipulation
import joblib # To load the saved model
from sklearn.metrics import mean_squared_error, r2_score # Metrics
import numpy as np # For square root calculation
import json # Config parser
import argparse # For parsing command line arguments

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def evaluate_model():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Evaluate Student Score Prediction Pipeline")
    parser.add_argument(
        '--model', 
        type=str, 
        help="Override the model to evaluate (e.g., RandomForestRegressor, LinearRegression)"
    )
    args = parser.parse_args()

    # 2. Load the base JSON config
    config = load_config()

    # 3. Determine the model path to load
    if args.model:
        model_path = f"models/{args.model.lower()}_model.pkl"
        print(f"Checking custom command-line model: {model_path}")
    else:
        model_path = config["model"]["save_path"]
        print(f"Checking default model: {model_path}")

    # 4. Load the cleaned evaluation data
    try:
        df = pd.read_csv(config["data"]["cleaned_csv_path"])
    except FileNotFoundError:
        print(f"Error: {config['data']['cleaned_csv_path']} not found. Run preprocessing first.")
        return

    # 5. Split data into Features and Target
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    # 6. Load the trained model pipeline
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find model at '{model_path}'. Did you train it first?")
        return

    # 7. Make Predictions
    predictions = model.predict(X)

    # 8. Calculate Metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    # 9. Print the results
    print("\n" + "="*40)
    print(f"   Evaluation: {model_path.split('/')[-1]}")
    print("="*40)
    print(f" RMSE:     {rmse:.4f} marks")
    print(f" R2 Score: {r2:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()