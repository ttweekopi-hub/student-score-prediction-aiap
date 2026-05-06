import pandas as pd # Data manipulation
import joblib # To load the saved model
from sklearn.metrics import mean_squared_error, r2_score # Metrics
from sklearn.model_selection import train_test_split # To split out the test set
import numpy as np # For square root calculation
import json # Config parser
import argparse # For parsing command line arguments
from sklearn.metrics import mean_absolute_error
from src.utils import setup_logger  # <--- IMPORT THE LOGGER HELPER

# Initialize the logger for the evaluation module
logger = setup_logger("evaluation")

def load_config(config_path="config.json"):
    """Loads the pipeline configuration parameters from a JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found at {config_path}: {e}")
        raise

def evaluate_model():
    """Evaluates a trained model pipeline on the unseen test set partition."""
    logger.info("=========================================")
    logger.info("        Evaluating Model Pipeline        ")
    logger.info("=========================================")

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
        logger.info(f"Targeting custom CLI model: {model_path}")
    else:
        model_path = config["model"]["save_path"]
        logger.info(f"Targeting default model path from config: {model_path}")

    # 4. Load the cleaned data
    try:
        df = pd.read_csv(config["data"]["cleaned_csv_path"])
    except FileNotFoundError as e:
        logger.error(f"Cleaned dataset not found at '{config['data']['cleaned_csv_path']}'. Run preprocessing first. Error: {e}")
        return

    # 5. Split data into Features and Target
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    # 6. Split into Train and Test sets (using the exact same parameters as train.py)
    # This isolates the test set to ensure we only evaluate on unseen data!
    logger.info("Isolating unseen 20% test partition for evaluation.")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Load the trained model pipeline
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully deserialized model from {model_path}")
    except FileNotFoundError as e:
        logger.error(f"Could not find model file at '{model_path}'. Did you run training first? Error: {e}")
        return

    # 8. Make Predictions on UNSEEN test data only
    logger.info("Generating predictions on validation features...")
    predictions = model.predict(X_test)

    # 9. Calculate Metrics using the actual test targets
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # 10. Display the results using logger.info to save to pipeline.log as well!
    logger.info("Evaluation complete. Logging final performance metrics:")
    logger.info(f"   >>> RMSE:     {rmse:.4f} marks")
    logger.info(f"   >>> MAE:      {mae:.4f} marks")
    logger.info(f"   >>> R2 Score: {r2:.4f}")

if __name__ == "__main__":
    evaluate_model()