import pandas as pd  # Data manipulation
import json  # Config parser
import os  # Folder generation
import joblib  # Model serialization
import argparse  # For parsing command line arguments
from sklearn.model_selection import train_test_split  # Splitter
from sklearn.preprocessing import OneHotEncoder  # Encoder
from sklearn.compose import ColumnTransformer  # Feature pipelines
from sklearn.pipeline import Pipeline  # Pipeline builder
import importlib  # Used to load any sklearn class dynamically
from src.utils import setup_logger  # Import the timezone-locked logger

# Initialize the logger for the training module
logger = setup_logger("train")

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

def get_regressor(algorithm_name, params):
    """Dynamically imports and instantiates an scikit-learn regressor class.

    This dynamically routes the algorithm request to its correct submodule
    within scikit-learn, preventing hardcoded package-level imports.

    Args:
        algorithm_name (str): The exact class name of the scikit-learn regressor 
            (e.g., "RandomForestRegressor").
        params (dict): A dictionary of hyperparameters to instantiate the model with.

    Returns:
        RegressorMixin: An initialized scikit-learn regressor instance.

    Raises:
        ModuleNotFoundError: If the module associated with the algorithm name cannot be found.
        AttributeError: If the class name does not exist within the imported module.
    """
    logger.info(f"Dynamically instantiating {algorithm_name} with parameters: {params}")
    modules = {
        "RandomForestRegressor": "sklearn.ensemble",
        "GradientBoostingRegressor": "sklearn.ensemble",
        "LinearRegression": "sklearn.linear_model",
        "SVR": "sklearn.svm"
    }
    module_name = modules.get(algorithm_name, "sklearn.linear_model")
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, algorithm_name)
        return model_class(**params)
    except (ModuleNotFoundError, AttributeError) as e:
        logger.error(f"Failed to dynamically load algorithm class '{algorithm_name}': {e}")
        raise

if __name__ == "__main__":
    logger.info("=========================================")
    logger.info("         Starting Model Training         ")
    logger.info("=========================================")

    # 1. Set up Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Train Student Score Prediction Pipeline")
    parser.add_argument(
        '--model', 
        type=str, 
        help="Override the default model algorithm using its name or a short alias (e.g., lr, rf, gbm, svr)"
    )
    args = parser.parse_args()

    # 2. Load configurations
    config = load_config()

    # Determine algorithm, parameters, and save paths
    if args.model:
        # Resolve command line alias to full scikit-learn class name
        user_input = args.model.lower()
        if user_input in MODEL_ALIASES:
            algorithm = MODEL_ALIASES[user_input]
            logger.info(f"Resolved command-line alias '{args.model}' to '{algorithm}'.")
        else:
            algorithm = args.model  # Fallback to literal entry if user typed the full class name
            
        params = {"random_state": 42} if algorithm != "LinearRegression" else {}
        logger.warning(f"CLI Override detected. Training specified model: {algorithm}")
        save_path = f"models/{algorithm.lower()}_model.pkl"
    else:
        algorithm = config["model"]["algorithm"]
        params = config["model"]["parameters"]
        logger.info(f"No CLI overrides. Using config.json default: {algorithm}")
        save_path = config["model"]["save_path"]

    # 4. Load cleaned data
    logger.info(f"Loading cleaned dataset from: {config['data']['cleaned_csv_path']}")
    try:
        df = pd.read_csv(config["data"]["cleaned_csv_path"], keep_default_na=False)
    except FileNotFoundError as e:
        logger.error(f"Cleaned dataset not found. Please run preprocessing first. Error: {e}")
        raise SystemExit(1)
    
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    
    # Identify categorical columns dynamically
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Dynamic categorical features detected: {cat_cols}")
    
    # Preprocessing pipeline step (one-hot encode the categorical factors)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ], remainder='passthrough')
    
    # Dynamic algorithm fetch
    regressor = get_regressor(algorithm, params)
    
    # Chain preprocessor and model into a unified Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Split train and validation sets (using fixed seed 42)
    logger.info("Splitting dataset into 80% train and 20% test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the dynamic pipeline
    logger.info(f"Fitting {algorithm} model pipeline on training features...")
    model_pipeline.fit(X_train, y_train)
    logger.info("Model pipeline fit completed successfully.")
    
    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the pipeline object
    joblib.dump(model_pipeline, save_path)
    logger.info(f"Trained model serialized and saved to: {save_path}")