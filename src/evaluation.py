import pandas as pd # Data manipulation
import joblib # To load the saved model
from sklearn.metrics import mean_squared_error, r2_score # Metrics
from sklearn.model_selection import train_test_split # NEW: To split out the test set
import numpy as np # For square root calculation
import json # Config parser
import argparse # For parsing command line arguments
from sklearn.metrics import mean_absolute_error

def load_config(config_path="config.json"):
    """Loads the pipeline configuration parameters from a JSON file.

    Args:
        config_path (str, optional): Relative path to the configuration JSON file. 
            Defaults to "config.json".

    Returns:
        dict: A nested dictionary containing data ingestion, preprocessing, 
            and modeling configurations.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        json.JSONDecodeError: If the configuration file is not valid JSON.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def evaluate_model():
    """Evaluates a trained model pipeline on the unseen test set partition.

    This function executes the following steps:
    1. Parses CLI model overrides, if any.
    2. Loads the configuration parameters.
    3. Loads the preprocessed data and splits out the 20% validation partition.
    4. Loads the requested serialized pipeline (preprocessor + estimator).
    5. Generates predictions on the unseen features.
    6. Computes and displays evaluation metrics: RMSE, MAE, and R2 Score.

    Returns:
        None: Outputs results directly to the standard output console.

    Raises:
        FileNotFoundError: If the cleaned CSV dataset or serialized model `.pkl` 
            file cannot be found.
    """
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Evaluate Student Score Prediction Pipeline")
    parser.add_argument(
        '--model', 
        type=str, \
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

    # 4. Load the cleaned data
    try:
        df = pd.read_csv(config["data"]["cleaned_csv_path"])
    except FileNotFoundError:
        print(f"Error: {config['data']['cleaned_csv_path']} not found. Run preprocessing first.")
        return

    # 5. Split data into Features and Target
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    # 6. Split into Train and Test sets (using the exact same parameters as train.py)
    # This isolates the test set to ensure we only evaluate on unseen data!
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Load the trained model pipeline
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find model at '{model_path}'. Did you train it first?")
        return

    # 8. Make Predictions on UNSEEN test data only
    predictions = model.predict(X_test)

    # 9. Calculate Metrics using the actual test targets
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # 10. Print the results
    print("\n" + "="*40)
    print("        MODEL EVALUATION METRICS")
    print("="*40)
    print(f" RMSE:     {rmse:.4f} marks")
    print(f" MAE:      {mae:.4f} marks")
    print(f" R2 Score: {r2:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()