import pandas as pd  # Data manipulation
import joblib  # To load the saved model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Metrics
from sklearn.model_selection import train_test_split  # To split out the test set
import numpy as np  # For square root calculation
import json  # Config parser
import argparse  # For parsing command line arguments
from src.utils import setup_logger  # Import the timezone-locked logger

# Initialize the logger for the evaluation module
logger = setup_logger("evaluation")

# Define the user-friendly command line nicknames mapped to full scikit-learn classes
MODEL_ALIASES = {
    "lr": "LinearRegression",
    "rf": "RandomForestRegressor",
    "gbm": "GradientBoostingRegressor",
    "gbr": "GradientBoostingRegressor",
    "svr": "SVR"
}

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
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found at {config_path}: {e}")
        raise

def evaluate_model():
    """Evaluates a trained model pipeline on both train and test partitions.

    This function isolates the validation split (the same 20% validation split 
    defined during training), restores the serialized preprocessor/model pipeline,
    generates predictions on both training and validation splits to explicitly 
    diagnose overfitting, calculates comprehensive regression metrics (RMSE, MAE, R2), 
    and outputs a sorted Feature Importance leaderboard to provide stakeholder 
    interpretability.

    Returns:
        None: Logs output metrics directly to the console and appends to the log file.

    Raises:
        FileNotFoundError: If the cleaned CSV dataset or the saved model binary 
            cannot be resolved at their defined destinations.
    """
    logger.info("=========================================")
    logger.info("        Evaluating Model Pipeline        ")
    logger.info("=========================================")

    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Evaluate Student Score Prediction Pipeline")
    parser.add_argument(
        '--model', 
        type=str, 
        help="Override the model to evaluate using its name or alias (e.g., lr, rf, gbm, svr)"
    )
    args = parser.parse_args()

    # 2. Load the base JSON config
    config = load_config()

    # 3. Determine the model path to load
    if args.model:
        # Resolve command line alias to find correct .pkl filename
        user_input = args.model.lower()
        if user_input in MODEL_ALIASES:
            algorithm = MODEL_ALIASES[user_input]
            logger.info(f"Resolved evaluation alias '{args.model}' to '{algorithm}'.")
        else:
            algorithm = args.model  # Fallback to literal entry if user typed the full class name
            
        model_path = f"models/{algorithm.lower()}_model.pkl"
        logger.info(f"Targeting custom CLI model: {model_path}")
    else:
        model_path = config["model"]["save_path"]
        logger.info(f"Targeting default model path from config: {model_path}")

    # 4. Load the cleaned data
    try:
        df = pd.read_csv(config["data"]["cleaned_csv_path"], keep_default_na=False)
    except FileNotFoundError as e:
        logger.error(f"Cleaned dataset not found at '{config['data']['cleaned_csv_path']}'. Run preprocessing first. Error: {e}")
        return

    # 5. Split data into Features and Target
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    # 6. Split into Train and Test sets (using the exact same parameters as train.py)
    # This isolates the test set to ensure we only evaluate on unseen data!
    logger.info("Isolating train (80%) and test (20%) partitions for diagnostic evaluation.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Load the trained model pipeline
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully deserialized model from {model_path}")
    except FileNotFoundError as e:
        logger.error(f"Could not find model file at '{model_path}'. Did you run training first? Error: {e}")
        return

    # 8. Generate Predictions on BOTH partitions to check for overfitting
    logger.info("Generating predictions for comparative metrics...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # 9. Calculate Train Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_mae = mean_absolute_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    # 10. Calculate Test Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    # 11. Log and Contrast Performance (The Overfitting Check)
    logger.info("-----------------------------------------")
    logger.info("        Training vs. Testing Metrics     ")
    logger.info("-----------------------------------------")
    logger.info(f"   [TRAIN] RMSE: {train_rmse:.4f} marks  |  [TEST] RMSE: {test_rmse:.4f} marks")
    logger.info(f"   [TRAIN] MAE:  {train_mae:.4f} marks  |  [TEST] MAE:  {test_mae:.4f} marks")
    logger.info(f"   [TRAIN] R2:   {train_r2:.4f}        |  [TEST] R2:   {test_r2:.4f}")
    logger.info("-----------------------------------------")

    # Diagnose performance health
    if train_r2 - test_r2 > 0.15:
        logger.warning("🚨 WARNING: High variance detected! The model is likely OVERFITTING the training data.")
    elif test_r2 < 0.40:
        logger.warning("🚨 WARNING: Low predictive power detected! The model may be UNDERFITTING.")
    else:
        logger.info("✅ Model generalization appears acceptable with no immediate evidence of severe overfitting (low variance between Train and Test performance).")

    # 12. Residual Bias Diagnostics
    residuals = y_test - test_preds
    mean_residual = np.mean(residuals)
    logger.info(f"   >>> Mean Prediction Error (Residual Mean): {mean_residual:.4f} marks")
    
    if abs(mean_residual) > 0.5:
        logger.warning("⚠️ BIAS DETECTED: The model's predictions are systematically biased (consistently over/under-estimating).")
    else:
        logger.info("✅ Unbiased predictions: The error distribution is centered tightly around zero.")

    # 13. Extract and Print Feature Importance for Interpretability
    logger.info("=========================================")
    logger.info("       Feature Importance Analysis       ")
    logger.info("=========================================")
    
    try:
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']
        
        # Verify if estimator supports feature importance attributes
        if hasattr(regressor, 'feature_importances_'):
            feature_names = []
            
            # Map structural transformer columns back to their post-transformed names
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'cat':
                    cat_features = transformer.get_feature_names_out(columns)
                    feature_names.extend(cat_features)
                elif name == 'remainder':
                    feature_names.extend(columns)
            
            # Fallback in case columns map mismatch
            if not feature_names or len(feature_names) != len(regressor.feature_importances_):
                feature_names = [f"Feature {i}" for i in range(len(regressor.feature_importances_))]

            # Pack, sort, and log structural importance
            importances = regressor.feature_importances_
            feature_importance_list = sorted(
                zip(feature_names, importances), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            logger.info("Relative influence of factors on student final score:")
            for rank, (feature, importance) in enumerate(feature_importance_list, 1):
                logger.info(f"   {rank}. {feature:<25} : {importance * 100:>6.2f}%")
        else:
            logger.warning(f"The model '{type(regressor).__name__}' does not natively support feature importances.")
            
    except Exception as e:
        logger.error(f"Could not extract feature importances: {e}")

if __name__ == "__main__":
    evaluate_model()