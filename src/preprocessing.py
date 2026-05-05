import pandas as pd # Data manipulation
import sqlite3 # Database connection
from datetime import datetime # Date/Time operations
import json # To read the configuration file

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f) # Parse and return JSON as a dictionary

def load_data(db_path):
    conn = sqlite3.connect(db_path) # Establish DB connection
    query = "SELECT * FROM score" # Query statement
    df = pd.read_sql_query(query, conn) # Pull data to DataFrame
    conn.close() # Close connection
    return df

def process_data(df, config):
    # 1. Drop rows where target variable is missing
    target = config["data"]["target_column"]
    df = df.dropna(subset=[target])
    
    # 2. Drop columns defined in the config
    cols_to_drop = config["data"]["drop_columns"]
    # Only drop columns that actually exist in the dataframe to prevent errors
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # 3. Standardize Categorical Redundancies
    # Standardize 'tuition' formatting
    if 'tuition' in df.columns:
        df['tuition'] = df['tuition'].replace({'Y': 'Yes', 'N': 'No'})
        
    # Standardize 'CCA' formatting (Replace 'NONE' with 'None')
    if 'CCA' in df.columns:
        df['CCA'] = df['CCA'].replace({'NONE': 'None'})
    
    # 4. Feature Engineering: Sleep Duration
    if 'sleep_time' in df.columns and 'wake_time' in df.columns:
        def calc_sleep(row):
            fmt = '%H:%M'
            start = datetime.strptime(row['sleep_time'], fmt)
            end = datetime.strptime(row['wake_time'], fmt)
            diff = (end - start).total_seconds() / 3600
            return diff + 24 if diff < 0 else diff
        
        df['sleep_duration'] = df.apply(calc_sleep, axis=1)
        df = df.drop(columns=['sleep_time', 'wake_time'])
    
    # 5. Dynamic Imputation Strategy based on config
    strategy = config["data"]["imputation_strategy"]
    if strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        
    return df

if __name__ == "__main__":
    config = load_config() # Load experimental parameters
    raw_df = load_data(config["data"]["db_path"]) # Ingest raw data
    clean_df = process_data(raw_df, config) # Process data
    clean_df.to_csv(config["data"]["cleaned_csv_path"], index=False) # Save output
    print(f"Preprocessing Complete: Saved to {config['data']['cleaned_csv_path']}")