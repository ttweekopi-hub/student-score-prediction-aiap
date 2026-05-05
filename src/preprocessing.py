import pandas as pd  # Import pandas for data manipulation
import sqlite3  # Import sqlite3 to connect to the database
from datetime import datetime  # Import datetime for time calculations


def load_data(db_path):
    conn = sqlite3.connect(db_path)  # Establish connection to the database
    query = "SELECT * FROM score"  # Define SQL query to fetch all data
    df = pd.read_sql_query(query, conn)  # Load data into a DataFrame
    conn.close()  # Close the database connection
    return df  # Return the loaded dataframe


def process_data(df):
    # 0. Drop rows where the target (final_test) is missing
    # We cannot train a model on data that has no answer key
    df = df.dropna(subset=['final_test'])  # NEW LINE HERE

    # 1. Drop Red Herrings and Multicollinear features
    cols_to_drop = ['student_id', 'bag_color', 'mode_of_transport', 'n_female']
    df = df.drop(columns=cols_to_drop)

    # 2. Standardize Categorical Redundancies
    df['tuition'] = df['tuition'].replace({'Y': 'Yes', 'N': 'No'})

    # 3. Feature Engineering: Sleep Duration
    def calc_sleep(row):
        fmt = '%H:%M'
        start = datetime.strptime(row['sleep_time'], fmt)
        end = datetime.strptime(row['wake_time'], fmt)
        diff = (end - start).total_seconds() / 3600
        return diff + 24 if diff < 0 else diff

    df['sleep_duration'] = df.apply(calc_sleep, axis=1)
    df = df.drop(columns=['sleep_time', 'wake_time'])

    # 4. Final Cleanup: Handle missing values in FEATURES
    # If study hours or attendance are missing, we fill them with the median
    df = df.fillna(df.median(numeric_only=True))

    return df

import os

if __name__ == "__main__":
    # Get the directory where preprocessing.py actually lives
    script_dir = os.path.dirname(__file__)

    # Build the path to the database (go up to src, then into data)
    db_path = os.path.join(script_dir, '..', 'data', 'score.db')
    save_path = os.path.join(script_dir, '..', 'data', 'cleaned_score.csv')

    raw_df = load_data(db_path)
    clean_df = process_data(raw_df)
    clean_df.to_csv(save_path, index=False)
    print(f"Preprocessing Complete: Saved to {save_path}")