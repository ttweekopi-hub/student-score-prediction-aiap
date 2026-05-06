import pandas as pd # Data manipulation
import json # Config parser
import os # Folder generation
import joblib # Model serialization
import argparse # NEW: For parsing command line arguments
from sklearn.model_selection import train_test_split # Splitter
from sklearn.preprocessing import OneHotEncoder # Encoder
from sklearn.compose import ColumnTransformer # Feature pipelines
from sklearn.pipeline import Pipeline # Pipeline builder
import importlib # Used to load any sklearn class dynamically

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

def get_regressor(algorithm_name, params):
    """Dynamically imports and instantiates an scikit-learn regressor class.

    This enables the pipeline to support multiple algorithms (e.g., Random Forest,
    Gradient Boosting, Linear Regression) without hardcoding imports.

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
    modules = {
        "RandomForestRegressor": "sklearn.ensemble",
        "GradientBoostingRegressor": "sklearn.ensemble",
        "LinearRegression": "sklearn.linear_model",
        "SVR": "sklearn.svm"
    }
    module_name = modules.get(algorithm_name, "sklearn.linear_model")
    module = importlib.import_module(module_name) 
    model_class = getattr(module, algorithm_name) 
    return model_class(**params) 

if __name__ == "__main__":
    # 1. Set up Command Line Argument Parsing
    parser = argparse.ArgumentParser(description="Train Student Score Prediction Pipeline")
    parser.add_argument(
        '--model', 
        type=str, 
        help="Override the default model algorithm (e.g., LinearRegression, GradientBoostingRegressor)"
    )
    args = parser.parse_args()

    # 2. Load basic configurations
    config = load_config()

    # Determine algorithm and parameters
    if args.model:
        # CLI Override logic
        algorithm = args.model
        params = {"random_state": 42} if algorithm != "LinearRegression" else {}
        print(f"CLI Override detected! Training model: {algorithm}")
        # Create a unique path so we don't overwrite other model files
        save_path = f"models/{algorithm.lower()}_model.pkl"
    else:
        # Default Configuration logic
        algorithm = config["model"]["algorithm"]
        params = config["model"]["parameters"]
        print(f"Loading default config. Training model: {algorithm}")
        save_path = config["model"]["save_path"]

    # 4. Load cleaned data
    df = pd.read_csv(config["data"]["cleaned_csv_path"])
    
    # Separate Features and Target
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    
    # Identify categorical columns dynamically
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Set up preprocessing pipeline step
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ], remainder='passthrough')
    
    # Dynamically fetch the algorithm model
    regressor = get_regressor(algorithm, params)
    
    # Chain preprocessor and model together
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the dynamic pipeline
    model_pipeline.fit(X_train, y_train)
    
    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save pipeline object (which contains the preprocessor AND the model)
    joblib.dump(model_pipeline, save_path)
    print(f"Model trained and saved successfully to: {save_path}")