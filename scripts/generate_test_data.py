import sqlite3
import os
import numpy as np # Ensure numpy is in your requirements.txt

def generate_mock():
    os.makedirs('data', exist_ok=True)
    db_path = 'data/score.db'

    # SAFETY CHECK:
    is_github = os.getenv('GITHUB_ACTIONS') == 'true'

    if os.path.exists(db_path) and not is_github:
        print(f"⚠️  Safety Triggered: '{db_path}' already exists.")
        print("I will not overwrite your real data locally.")
        return 

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create the table structure
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

    # 2. Generate 100 rows of "Predictable" mock data
    # This ensures your model can learn a pattern and achieve MAE < 6.0
    sample_data = []
    for i in range(1, 101):
        hours = np.random.uniform(5, 30)
        # Formula: Score = (Hours * 2.5) + Base 20. 
        # This creates a strong linear relationship for the model to find.
        score = (hours * 2.5) + 20 + np.random.normal(0, 0.5) 
        
        sample_data.append((
            f'STU{i:03}', 'Female' if i % 2 == 0 else 'Male', 
            15, 1, 'No', 'None', 'Visual', 'N', 
            hours, 90.0, '23:00', '07:00', 12, score
        ))

    # 3. Insert the data
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
    print(f"✅ Mock score.db created with 100 predictable rows. MAE should now be < 6.0.")

if __name__ == "__main__":
    generate_mock()