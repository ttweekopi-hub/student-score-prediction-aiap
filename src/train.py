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
    with open(config_path, "r") as f:
        return json.load(f)

def get_regressor(algorithm_name, params):
    """Dynamically imports and instantiates an sklearn model class."""
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
        help="Override the model to use (e.g., RandomForestRegressor, LinearRegression, GradientBoostingRegressor)"
    )
    args = parser.parse_args()

    # 2. Load the base JSON config
    config = load_config() 
    
    # 3. Handle Command Line Overrides
    algorithm = config["model"]["algorithm"]
    params = config["model"]["parameters"]
    save_path = config["model"]["save_path"]

    if args.model:
        print(f"⚠️ Command Line Override Active: Switching from {algorithm} to {args.model}")
        algorithm = args.model
        # Reset parameters if we swap models, as parameters for RF won't work on Linear Regression
        params = {} 
        # Update save path so we don't overwrite other model files
        save_path = f"models/{algorithm.lower()}_model.pkl"

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
        
    # Serialize pipeline
    joblib.dump(model_pipeline, save_path)
    print(f"Success! Model Training Complete ({algorithm}): Saved to {save_path}")