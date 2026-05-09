import sqlite3
import os


def generate_mock():
    os.makedirs('data', exist_ok=True)
    db_path = 'data/score.db'

    # SAFETY CHECK:
    # If the file exists and we are NOT in a GitHub Action, do not overwrite.
    # GitHub Actions sets an environment variable called 'GITHUB_ACTIONS' to 'true'.
    is_github = os.getenv('GITHUB_ACTIONS') == 'true'

    if os.path.exists(db_path) and not is_github:
        print(f"⚠️  Safety Triggered: '{db_path}' already exists.")
        print("I will not overwrite your real data locally.")
        print("If you really want to generate mock data, delete the file manually first.")
        return  # Exit the function safely

    # If we are in GitHub, or the file doesn't exist, proceed:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create the table with the core columns your code expects
    cursor.execute('''
        CREATE TABLE score (
            student_id TEXT,
            gender TEXT,
            student_age INTEGER,
            number_of_siblings INTEGER,
            direct_admission TEXT,
            CCA TEXT,
            learning_style TEXT,
            tuition TEXT,
            hours_per_week REAL,
            attendance_rate REAL,
            sleep_time TEXT,
            wake_time TEXT,
            n_male INTEGER,
            final_test REAL
        )
    ''')

    # 2. Define data for those specific 14 columns
    # (Note: I removed 'index_col' to keep it simple,
    # adjust if your code strictly requires an index column)
    sample_data = [
        ('STU001', 'Female', 15, 2, 'Yes', 'Sports', 'Visual', 'Y', 20.5, 95.0, '23:00', '07:00', 10, 85.0),
        ('STU002', 'Male', 16, 0, 'No', 'None', 'Auditory', 'N', 10.0, 80.0, '00:00', '06:00', 15, 65.0)
    ]

    # 3. Explicitly name the columns in the INSERT statement
    # This prevents the "18 vs 15" error
    query = '''
        INSERT INTO score (
            student_id, gender, student_age, number_of_siblings, 
            direct_admission, CCA, learning_style, tuition, 
            hours_per_week, attendance_rate, sleep_time, wake_time, 
            n_male, final_test
        ) 
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    '''

    cursor.executemany(query, sample_data)
    conn.commit()
    conn.close()
    print("✅ Mock score.db created successfully with explicit columns.")


if __name__ == "__main__":
    generate_mock()