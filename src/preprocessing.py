import pandas as pd # Data manipulation
import sqlite3 # Database connection
from datetime import datetime # Date/Time operations
import json # To read the configuration file
from src.utils import setup_logger  # <--- IMPORT THE LOGGER HELPER
import os
import urllib.request  # Native python library to download files

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

    # Start a 'try' block to handle potential errors that might happen while reading the file
    try:
        # Open the file at the specified path in 'read' mode and ensure it closes automatically
        with open(config_path, "r") as f:
            # Parse the file's content as JSON data and return it as a Python object
            return json.load(f)
    # Catch the specific error that occurs if the file doesn't exist at the given path
    except FileNotFoundError as e:
        # Record a detailed error message in the logs including the path and the system error
        logger.error(f"Configuration file not found at {config_path}: {e}")
        # Re-raise the error so the rest of the program knows something went wrong and stops
        raise

def load_data(db_path):
    """Establishes a connection to the SQLite database and loads raw student data.

    If the database file is not found at the specified path, this function 
    attempts to dynamically download the database from a remote hosting URL 
    to ensure automated pipeline execution does not fail.

    Args:
        db_path (str): Relative path to the SQLite database file containing the raw data.

    Returns:
        pd.DataFrame: A pandas DataFrame holding the raw records from the 'score' table.

    Raises:
        FileNotFoundError: If the local database is missing and the remote download fails.
        sqlite3.Error: If a database connection error or query execution failure occurs.
    """
    # Check if the database file actually exists at the provided path
    if not os.path.exists(db_path):
        # Log a warning message if the file is missing
        logger.warning(f"Database not found at '{db_path}'.")
        
        # Define your public raw database download link here (if you have one)
        download_url = "https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db" 
        
        try:
            logger.info(f"Attempting to download raw database from {download_url}...")
            # Create the necessary folders (like 'data/') if they don't exist yet
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            # Download the database from the URL and save it directly to the database path
            urllib.request.urlretrieve(download_url, db_path)
            logger.info("Download complete!")
        except Exception as e:
            logger.error(f"Failed to auto-download database: {e}")
            logger.error("Please manually copy 'score.db' into the 'data/' folder.")
            raise FileNotFoundError(f"Database missing at {db_path} and download failed.")

    # Proceed with loading database
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM score"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        logger.error(f"Failed to query the database table: {e}")
        raise

def process_data(df, config):
    """Cleans, transforms, and prepares raw data for training and evaluation."""
    logger.info("Starting data preprocessing...")
    
    # Clean and standardize all Column Names (e.g. "CCA " -> "CCA")
    df.columns = df.columns.str.strip()
    
    # 1. Drop rows where target variable is missing
    target = config["data"]["target_column"]
    initial_len = len(df)
     # Only drop NaNs from the target if we are in "Training Mode" (target exists)
    if target in df.columns:
        df = df.dropna(subset=[target])
    else:
        # If target is missing, we are likely in "Inference/API Mode"
        # Just skip this step!
        pass
    # df = df.dropna(subset=[target])
    dropped_target = initial_len - len(df)
    if dropped_target > 0:
        logger.warning(f"Dropped {dropped_target} rows missing the target column '{target}'.")
    
    # 2. Drop columns defined in the config
    cols_to_drop = config["data"]["drop_columns"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    logger.info(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # NEW: Specific Imputation for attendance_rate as per README documentation
    if 'attendance_rate' in df.columns:
        null_count = df['attendance_rate'].isnull().sum()
        if null_count > 0:
            attendance_median = df['attendance_rate'].median()
            df['attendance_rate'] = df['attendance_rate'].fillna(attendance_median)
            logger.info(f"Imputed {null_count} missing values in 'attendance_rate' using median: {attendance_median}")
    
    # 3. Standardize Categorical Redundancies
    if 'tuition' in df.columns:
        logger.info("Standardizing categorical representations for 'tuition'.")
        df['tuition'] = df['tuition'].replace({'Y': 'Yes', 'N': 'No'})
        
    if 'CCA' in df.columns:
        logger.info("Standardizing categorical representations for 'CCA'...")
        
        # 1. Convert to string, make Title Case, and strip hidden spaces (e.g. " Clubs " -> "Clubs")
        df['CCA'] = df['CCA'].astype(str).str.strip().str.title()
        
        # 2. Standardize all variations of "None" and string "Nan" representations to a unified "None"
        logger.info("Normalizing null and missing string variations to 'None'.")
        df['CCA'] = df['CCA'].replace({
            'None': 'None',
            'Nan': 'None',
            '<Null>': 'None',  # In case the database has literal null text
            '': 'None'         # In case there are empty strings
        })
        
        # 3. Safety Net: Fill any actual pandas NaN values that somehow survived string conversion
        df['CCA'] = df['CCA'].fillna('None')

    # 4. Feature Engineering: Sleep Duration
    if 'sleep_time' in df.columns and 'wake_time' in df.columns:
        logger.info("Engineering feature: 'sleep_duration' from raw sleep/wake timestamps.")
        def calc_sleep(row):
            """
            Calculates the duration of sleep in hours from sleep and wake timestamps.

            Args:
                row (pd.Series): A row of the DataFrame containing 'sleep_time' 
                                and 'wake_time' strings in 'HH:MM' format.

            Returns:
                float: Total sleep duration in hours. Handles overnight sleep 
                    by adding 24 hours if the end time is before the start time.
            """

            # Define the format: %H is 24-hour hour, %M is minutes (e.g., '23:30')
            fmt = '%H:%M'
            # Convert the 'sleep_time' string from the data row into a Python datetime object
            start = datetime.strptime(row['sleep_time'], fmt)
            # Convert the 'wake_time' string into a Python datetime object
            end = datetime.strptime(row['wake_time'], fmt)
            # Subtract start from end to get a 'timedelta', get total seconds,
            diff = (end - start).total_seconds() / 3600
            # Handle the "Midnight Problem": If the result is negative (meaning you woke up 
            # the next day), add 24 hours to get the correct duration. Otherwise, keep it as is.
            return diff + 24 if diff < 0 else diff
        
        # Vectorize the calculation across all rows in the DataFrame
        df['sleep_duration'] = df.apply(calc_sleep, axis=1)
        # Remove original timestamp columns to keep the dataset clean and focused on features
        df = df.drop(columns=['sleep_time', 'wake_time'])
    
    # 5. Dynamic Imputation Strategy
    # Look up which method to use for filling gaps in the data from the configuration file
    strategy = config["data"]["imputation_strategy"]
    # Record a log message showing which specific strategy is being applied
    logger.info(f"Applying missing numerical value imputation strategy: '{strategy}'")
    # Check if the chosen strategy is to use the middle value (median)
    if strategy == "median":
        # Fill all empty spots in numeric columns with the median value of those columns
        df = df.fillna(df.median(numeric_only=True))
    # Check if the chosen strategy is to use the average value (mean)
    elif strategy == "mean":
        # Fill all empty spots in numeric columns with the average value of those columns
        df = df.fillna(df.mean(numeric_only=True))

    # Record that the data cleaning process is done    
    logger.info("Data preprocessing finished successfully.")
    return df

if __name__ == "__main__":
    logger.info("=========================================")
    logger.info("   Starting Data Preprocessing Pipeline  ")
    logger.info("=========================================")
    
    # Call the function that reads and loads the configuration settings into a dictionary
    config = load_config()
    # Use the database path stored in the config to load the raw data into a DataFrame
    raw_df = load_data(config["data"]["db_path"])
    # Pass the raw data and settings into the processing function to clean and prepare the data
    cleaned_df = process_data(raw_df, config)
    
    # Get the file path where the cleaned data should be saved from the configuration
    output_path = config["data"]["cleaned_csv_path"]
    # Extract just the folder part of that path (e.g., "data/processed" from "data/processed/clean.csv")
    output_dir = os.path.dirname(output_path)
    # Check if the folder path exists and if that folder is actually missing from the system
    if output_dir and not os.path.exists(output_dir):
        # Log a message explaining that the missing folder is being created
        logger.info(f"Directory '{output_dir}' not found. Creating it dynamically.")
        # Create the folder (and any parent folders needed) so the save operation won't crash
        os.makedirs(output_dir)
    
    # Save the cleaned data to the file path we defined earlier
    # The index=False part ensures we don't add an extra column for row numbers in our CSV file
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved successfully to: {output_path}")