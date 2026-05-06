import pandas as pd # Data manipulation
import sqlite3 # Database connection
from datetime import datetime # Date/Time operations
import json # To read the configuration file
from src.utils import setup_logger  # <--- IMPORT THE LOGGER HELPER
import os

# Initialize the logger for the preprocessing step
logger = setup_logger("preprocessing")

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

def load_data(db_path):
    """Establishes a connection to the SQLite database and loads raw student data.

    Args:
        db_path (str): Relative path to the SQLite database file containing the raw data.

    Returns:
        pd.DataFrame: A pandas DataFrame holding the raw records from the 'score' table.

    Raises:
        sqlite3.Error: If a database connection error or query execution failure occurs.
    """
    try:
        logger.info(f"Connecting to raw database at: {db_path}")
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM score"
        df = pd.read_sql_query(query, conn)
        conn.close()
        logger.info(f"Successfully loaded {len(df)} raw records from database.")
        return df
    except sqlite3.Error as e:
        logger.error(f"Database connection or query failed: {e}")
        raise

def process_data(df, config):
    """Cleans, transforms, and prepares raw data for training and evaluation."""
    logger.info("Starting data preprocessing...")
    
    # 1. Drop rows where target variable is missing
    target = config["data"]["target_column"]
    initial_len = len(df)
    df = df.dropna(subset=[target])
    dropped_target = initial_len - len(df)
    if dropped_target > 0:
        logger.warning(f"Dropped {dropped_target} rows missing the target column '{target}'.")
    
    # 2. Drop columns defined in the config
    cols_to_drop = config["data"]["drop_columns"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    logger.info(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # 3. Standardize Categorical Redundancies
    if 'tuition' in df.columns:
        logger.info("Standardizing categorical representations for 'tuition'.")
        df['tuition'] = df['tuition'].replace({'Y': 'Yes', 'N': 'No'})
        
    if 'CCA' in df.columns:
        logger.info("Standardizing categorical representations for 'CCA'.")
        df['CCA'] = df['CCA'].replace({'NONE': 'None'})
    
    # 4. Feature Engineering: Sleep Duration
    if 'sleep_time' in df.columns and 'wake_time' in df.columns:
        logger.info("Engineering feature: 'sleep_duration' from raw sleep/wake timestamps.")
        def calc_sleep(row):
            fmt = '%H:%M'
            start = datetime.strptime(row['sleep_time'], fmt)
            end = datetime.strptime(row['wake_time'], fmt)
            diff = (end - start).total_seconds() / 3600
            return diff + 24 if diff < 0 else diff
        
        df['sleep_duration'] = df.apply(calc_sleep, axis=1)
        df = df.drop(columns=['sleep_time', 'wake_time'])
    
    # 5. Dynamic Imputation Strategy
    strategy = config["data"]["imputation_strategy"]
    logger.info(f"Applying missing numerical value imputation strategy: '{strategy}'")
    if strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        
    logger.info("Data preprocessing finished successfully.")
    return df

if __name__ == "__main__":
    logger.info("=========================================")
    logger.info("   Starting Data Preprocessing Pipeline  ")
    logger.info("=========================================")
    
    config = load_config()
    raw_df = load_data(config["data"]["db_path"])
    cleaned_df = process_data(raw_df, config)
    
    # NEW: Extract the directory path (e.g., 'data') and create it if missing
    output_path = config["data"]["cleaned_csv_path"]
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Directory '{output_dir}' not found. Creating it dynamically.")
        os.makedirs(output_dir)
    
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to: {output_path}")