# Student Score Prediction Pipeline

## 1. Overview and Folder Structure

This case study was taken from the AIAP Technical Assessment Past Year Series repository:
https://github.com/AISG-AIAP/AIAP-Technical-Assessment-Past-Years-Series

I have selected to use the "Student Score Prediction" case study from the Past Year Series to practise building an end-to-end ML pipeline covering EDA/ML Pipeline as requirements to pass the AIAP Technical Assessment.

The pipeline is designed to predict O-level mathematics scores for U.A. Secondary School. The goal is to identify students who may require additional academic support before the examination.

To download the database used in this case study is score.db which can be downloaded from here:
https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db

# Project Structure

```text
student_score_project/
├── data/               # Contains cleaned_score.csv (score.db must be placed here)
├── models/             # Stores the trained student_model.pkl
├── notebooks/          # Stores the Jupyter notebook file
│   └── eda.ipynb       # Exploratory Data Analysis and visualizations
├── src/                # Python modules for the pipeline
│   ├── preprocessing.py # Data ingestion, cleaning, and feature engineering
│   ├── train.py         # Model training and pipeline serialization
│   └── evaluation.py    # Performance metrics calculation
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation and usage instructions
```

## 2. Instructions for execution

To run the pipeline, ensure you have the score.db file placed in the data/ folder.

1. Install Dependencies:
pip install -r requirements.txt

2. Run Preprocessing:
python src/preprocessing.py
This fetches data via SQLite, cleans features, and performs feature engineering.

3. Run Training:
python src/train.py
This trains the model and saves the .pkl file in the models/ directory.

4. Run Evaluation:
python src/evaluation.py
This outputs the final RMSE and R² scores.

## 3. Pipeline flow and Logical Steps

The pipeline follows a sequential flow from raw data to evaluation:

1. Data Ingestion: Fetches raw data from score.db using SQLite relative paths.  

2. Data Cleaning: Handles missing values, drops "Red Herring" features (e.g., bag_color), and standardizes categorical inconsistencies.  

3. Feature Engineering: Converts sleep_time and wake_time into a continuous sleep_duration feature to capture threshold effects on performance.  

4. Encoding & Modeling: Uses a Scikit-Learn Pipeline to handle OneHotEncoding for categorical features and feeds them into a RandomForestRegressor.

5. Serialization: Saves the entire pipeline (preprocessor + model) as a single object for reusability.

## 4. Key EDA findings & Feature Processing

**Findings Summary**
- Attendance & Study Hours: Strongest positive correlation with test scores.- Tuition: Significant performance boost for students with external tuition.- Sleep Threshold: EDA revealed that students with $<7$ hours of sleep rarely score above 50.

| Feature | Type | Processing Action | Reason |
| :--- | :--- | :--- | :--- |
| `student_id` | ID | Dropped | No predictive signal. |
| `tuition` | Categorical | Replace 'Y'/'N' with 'Yes'/'No' | Standardize labels for encoding. |
| `n_female` | Numerical | Dropped | High multicollinearity with `n_male`. |
| `sleep_time` | String | Combined into `sleep_duration` | Convert raw time to usable duration. |
| `attendance_rate` | Numerical | Imputed with Median | Handle missing values without dropping rows. |

## 5. Choice of Model and Evaluation

**Model Choice: Random Forest Regressor**
While the assessment requires exploring multiple models, Random Forest was selected as the final model due to:  

- Non-linearity: Its ability to capture "threshold" effects (like the 7-hour sleep rule) that Linear Regression misses.

- Robustness: It handles various data types and multi-modal distributions effectively without extensive scaling.

**Evaluation Metrics**
- RMSE (Root Mean Squared Error): Chosen to quantify the average deviation in test marks. Our result of 4.49 indicates high precision for school interventions.

- R² (R-Squared): Chosen to measure the proportion of variance explained. Our result of 0.8968 proves the features selected have strong explanatory power.

