"""
FastAPI Server for Student Score Prediction.
Integrates with existing preprocessing and trained models.
"""

import joblib
import pandas as pd
import json
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from src.utils import setup_logger
from src.preprocessing import process_data
import traceback
import sys

# Initialize logger using your existing utility
logger = setup_logger("api_serve")

app = FastAPI(
    title="Student Exam Score Predictor",
    description="This API uses Random Forest/Linear Regression/Gradient Boosting models to predict final test scores based on student stats.",
    version="1.0.0"
)

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL ERROR DETECTED BY MIDDLEWARE:")
        traceback.print_exc(file=sys.stderr)
        print("="*50 + "\n")
        raise e

# Container for models
models = {}

class StudentInput(BaseModel):
    """
    Data schema for student features. 
    Field names match the raw CSV columns exactly.
    """
    number_of_siblings: int = Field(..., example=1, ge=0, le=10)
    direct_admission: str = Field(..., example="No")
    CCA: str = Field(..., example="Sports")
    learning_style: str = Field(..., example="Visual")
    tuition: str = Field(..., example="Yes")
    n_male: int = Field(..., example=10, ge=0, le=50)
    student_age: int = Field(..., example=16, ge=13, le=18)
    hours_per_week: float = Field(..., example=15.0, ge=0, le=80)
    attendance_rate: float = Field(..., example=95.0, ge=0, le=100)
    sleep_duration: int = Field(..., example=8, ge=0, le=15)
    
@app.on_event("startup")
def load_models():
    """
    Loads trained scikit-learn pipelines into memory on startup.
    Ensures that the models/ directory contains the .pkl files.
    """
    model_paths = {
        "rf": "models/student_model.pkl",
        "lr": "models/linearregression_model.pkl",
        "gbr": "models/gradientboostingregressor_model.pkl"
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            logger.info(f"Successfully loaded model: {name}")
        else:
            logger.warning(f"Model file not found: {path}. {name} will be unavailable.")

@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: StudentInput):
    """
    Prediction endpoint that applies preprocessing and model inference.
    
    Args:
        model_name (str): The identifier ('rf', 'lr', or 'gbr').
        input_data (StudentInput): The JSON feature set.
    
    Returns:
        dict: The prediction result and model metadata.
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not available.")

    try:
        # Convert input to DataFrame (using aliases to catch 'CCA')
        raw_df = pd.DataFrame([input_data.dict(by_alias=True)])

        # Load config to use your existing preprocessing logic
        with open("config.json", "r") as f:
            config = json.load(f)

        # RENAME COLUMNS TO MATCH TRAINING DATA
        # Adjust these mappings based on what your model expects!
        rename_map = {
            "student_age": "age",
            "n_male": "gender",  # Assuming n_male was used to determine gender
            "hours_studied": "hours_per_week" # Just an example, check your CSV!
        }
        raw_df = raw_df.rename(columns=rename_map)

        # If the model expects an 'index' column, we might need to add a dummy one
        if 'index' not in raw_df.columns:
            raw_df['index'] = 0

        # Apply your project's existing feature engineering (sleep duration, etc.)
        # process_data returns the DF ready for the pipeline's ColumnTransformer
        processed_df = process_data(raw_df, config)

        # Add missing columns back as the CORRECT type
        if 'index' not in processed_df.columns:
            processed_df['index'] = 0
            
        if 'n_male' not in processed_df.columns:
            # Ensure this matches the type in your training CSV (likely int or float)
            processed_df['n_male'] = float(input_data.n_male)

        # Get the expected feature names from the model
        expected_columns = models[model_name].feature_names_in_ 

        # Reorder and filter columns
        # Use .copy() to avoid SettingWithCopy warnings later
        processed_df = processed_df[expected_columns].copy()

        # Convert the entire DataFrame to object type 
        # This is a 'brute force' way to ensure NumPy doesn't try to 
        # coerce strings and numbers into a single numeric type
        processed_df = processed_df.astype(object)

        # FORCE CATEGORICAL COLUMNS TO OBJECT TYPE
        # We use 'object' because Scikit-Learn pipelines prefer it for categorical logic
        categorical_cols = ['direct_admission', 'CCA', 'learning_style', 'tuition']
        for col in categorical_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(object)

        # Handle NaNs
        # It's safer to use a dictionary or fill specific types to avoid 
        # accidentally putting a '0' in a string column
        processed_df = processed_df.fillna(0)

        # Check for any 'None' or 'NaN' hidden in the data
        if processed_df.isna().any().any():
            print("\n!!! FOUND HIDDEN NaNs !!!")
            print(processed_df.isna().sum())
            # Fill them immediately
            processed_df = processed_df.fillna(0)

        # Check if any categorical column has the word "string" or is empty
        for col in ['direct_admission', 'CCA', 'learning_style', 'tuition']:
            if (processed_df[col] == "string").any() or (processed_df[col] == "").any():
                print(f"\n!!! WARNING: Column {col} contains invalid placeholder values !!!")

        try:
            # Assuming your pipeline has a step named 'preprocessor' and a transformer named 'cat'
            cat_categories = models[model_name].named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()
        except:
            print("\nCould not extract categories, but the 200 OK means the model is guessing anyway!")

        # Convert your processed data to a DataFrame if it isn't one
        if not isinstance(processed_df, pd.DataFrame):
            # Use the expected columns from the model
            expected_columns = models[model_name].feature_names_in_
            processed_df = pd.DataFrame(processed_df, columns=expected_columns)

        # Explicitly cast to object to avoid the 'isnan' string error we saw earlier
        processed_df = processed_df.astype(object)

        # Inference
        try:
            prediction = models[model_name].predict(processed_df)
        except TypeError:
            # If it still complains about isnan, try the raw values
            prediction = models[model_name].predict(processed_df.values)
        
        return {
            "model": model_name,
            "prediction": round(float(prediction[0]), 2),
            "status": "success"
        }

    except Exception as e:
        import traceback
        import sys
        # This prints to the standard error stream, which WSL always shows
        traceback.print_exc(file=sys.stderr) 
        
        # This will send the full error string to your browser UI
        raise HTTPException(status_code=500, detail=f"TRACEBACK: {str(e)}")

@app.get("/health")
def health():
    return {"status": "online", "active_models": list(models.keys())}