# Student Score Prediction Pipeline

## 1. Overview and Folder Structure

This case study was taken from the AIAP Technical Assessment Past Year Series repository:
https://github.com/AISG-AIAP/AIAP-Technical-Assessment-Past-Years-Series

I have selected to use the "Student Score Prediction" case study from the Past Year Series collection to practise building an end-to-end ML pipeline covering EDA/ML Pipeline as requirements to pass the AIAP Technical Assessment.

The pipeline I built is designed to predict O-level mathematics scores for U.A. Secondary School. The goal is to identify students who may require additional academic support before the examination.

The sqlite db (score.db) used in this ML pipeline can be downloaded from here:
https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db

### Project Structure

```text
student-score-prediction-aiap/
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
└── config.json         # central control file to change model settings
```

## 2. Instructions for Execution & Parameter Modification

The pipeline is highly configurable and supports execution with different machine learning algorithms and parameters using a combination of a configuration file (`config.json`) and command-line interface (CLI) parameter overrides.

To run the pipeline, ensure you have the *score.db* file downloaded and placed in the data/ folder.<br>

## 2.1. Install Dependencies:<br>
    pip install -r requirements.txt

## 2.2 Run Preprocessing:<br>
This fetches data via SQLite, cleans features, and performs feature engineering.<br>


    python src/preprocessing.py

## 2.3 Model Training (With Model Swapping)<br>
You can train models using the default settings in config.json or swap algorithms directly using command-line arguments.<br>
<br>
**Option A: Train the Default Model (Random Forest)**<br>
This runs using the default hyperparameters and configuration specified inside config.json.<br>
    
    python src/train.py

Output is saved to: models/student_model.pkl (as specified in config.json)<br>
<br>
**Option B: Train a Linear Regression Baseline**<br>
Overrides the default model from the terminal. This automatically resets the hyperparameters to suit a linear baseline.<br>

    python src/train.py --model LinearRegression

Output is saved to: models/LinearRegression_model.pkl<br>
<br>
**Option C: Train a Gradient Boosting Regressor**<br>

    python src/train.py --model GradientBoostingRegressor

Output is saved to: models/gradientboostingregressor_model.pkl<br>

## 2.4 Model Evaluation:<br>
Evaluate your trained models dynamically by passing matching command-line arguments.<br>
<br>
**Evaluate Default Model (Random Forest):**<br>

    python src/evaluation.py

**Evaluate Linear Regression:**

    python src/evaluation.py --model LinearRegression

**Evaluate Gradient Boosting:**<br>

    python src/evaluation.py --model GradientBoostingRegressor

All evaluation will output the final RMSE and R² scores.

## 3. Pipeline flow and Logical Steps

The pipeline follows a sequential flow from raw data to evaluation:

1. **Data Ingestion**: Fetches raw data from score.db using SQLite relative paths.  

2. **Data Cleaning**: Handles missing values, drops "Red Herring" features (e.g., bag_color), and standardizes categorical inconsistencies.  

3. **Feature Engineering**: Converts sleep_time and wake_time into a continuous sleep_duration feature to capture threshold effects on performance.  

4. **Encoding & Modeling**: Uses a Scikit-Learn Pipeline to handle OneHotEncoding for categorical features and feeds them into a RandomForestRegressor.

5. **Serialization**: Saves the entire pipeline (preprocessor + model) as a single object for reusability.

## 4. Key EDA findings & Feature Processing

### Findings Summary
- **Attendance & Study Hours**: Strongest positive correlation with test scores.<br>
- **Tuition**: Significant performance boost for students with external tuition.<br>
- **Sleep Threshold**: EDA revealed that students with <7 hours of sleep rarely score above 50.

| Feature | Type | Processing Action | Reason |
| :--- | :--- | :--- | :--- |
| `student_id` | ID | Dropped | No predictive signal. |
| `tuition` | Categorical | Replace 'Y'/'N' with 'Yes'/'No' | Standardize labels for encoding. |
| `n_female` | Numerical | Dropped | High multicollinearity with `n_male`. |
| `sleep_time` | String | Combined into `sleep_duration` | Convert raw time to usable duration. |
| `attendance_rate` | Numerical | Imputed with Median | Handle missing values without dropping rows. |

## 5. Choice of Model and Evaluation

To promote easy experimentation, all model training configurations are centralized. The workflow can be fine-tuned using the following parameters:<br>
<br>
**The Configuration Command Center (config.json)**<br>
The *config.json* file controls directories, database targets, column modifications, and default model hyperparameters:<br>
<br>
```json
{
  "data": {
    "db_path": "data/score.db",
    "cleaned_csv_path": "data/cleaned_score.csv",
    "target_column": "final_test",
    "drop_columns": ["student_id", "bag_color", "mode_of_transport", "n_female"],
    "imputation_strategy": "median"
  },
  "model": {
    "algorithm": "RandomForestRegressor",
    "parameters": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    },
    "save_path": "models/student_model.pkl"
  }
}
```

### Choice of Evaluated Algorithms:<br>

**Random Forest Regressor (Default Ensemble Model)**<br>
*Why Chosen*: Excellent at capturing complex interaction thresholds in student habits (such as the non-linear relationship where score performance drops sharply below 7 hours of sleep).
<br>
*Strengths*: Extremely robust against multi-modal distributions and requires no scaling.<br>
<br>
**Gradient Boosting Regressor (Sequential Ensemble Model)**<br>
*Why Chosen*: Builds trees sequentially to minimize residual errors, often achieving tighter fitting and lower error margins than Random Forest.<br>
<br>
**Linear Regression (Baseline Parametric Model)**<br>
*Why Chosen*: Serves as a vital baseline to quantify the performance boost gained by transitioning to non-linear tree-based models.

## 6. Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Chosen to quantify the average deviation in test marks. The result of 4.49 indicates high precision for school interventions.

- **R² (R-Squared)**: Chosen to measure the proportion of variance explained. An R² score of 0.8968 confirms that roughly 90% of a student's score variance is explained by their academic habits, attendance, and sleep duration.

