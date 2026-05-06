# Student Score Prediction Pipeline

## 1. Overview and Folder Structure

This case study was taken from the AIAP Technical Assessment Past Year Series repository:
https://github.com/AISG-AIAP/AIAP-Technical-Assessment-Past-Years-Series

I have selected to use the "Student Score Prediction" case study from the Past Year Series collection to practise building an end-to-end ML pipeline covering EDA/ML Pipeline as requirements to pass the AIAP Technical Assessment.

The pipeline is designed to predict final mathematics scores using student demographic and academic indicators to proactively identify at-risk individuals. The system optimizes for a Mean Absolute Error (MAE) < 6.0 marks while prioritizing actionable interpretability to enable timely, targeted academic interventions.

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
│   └── utils.py         # Logging feature
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation and usage instructions
└── config.json         # central control file to change model settings
└── run.sh              # run script to auto-execute pipeline
└── pipeline.log        # Generated automatically on script execution
```

## 2. Logging & Pipeline Monitoring (Production Features)

To align with robust software engineering practices, this pipeline completely avoids unformatted `print()` statements. Instead, it utilizes Python's native `logging` module to output standardized, traceable logs to both the console and a persistent log file.

### Key Logging Configurations
* **Timezone-Aware Timestamps:** All log entries are locked to **Singapore Standard Time (SGT, UTC+8)** using a custom timezone formatter. This ensures consistency regardless of whether the pipeline is executed locally, in a cloud VM, or on a grader's machine.
* **Date & Time Format:** Timestamps follow the Singapore-localized date format: `DD-MM-YYYY HH:MM:SS SGT`.
* **Multi-Destination Logging:** Logs are simultaneously streamed to the active console window (standard output) and appended to a centralized file.

### File Structure & Output Destination
Logs are stored in a centralized file located at the project root:
```text
my_project_root/
└── pipeline.log             # Generated automatically on script execution
```


## 3. Instructions for Execution

This pipeline supports two execution methods: a **one-click automated runner** (recommended for quick grading) and **manual step-by-step commands** (designed for custom experiments and model-swapping).

### Option A: Quickstart (One-Click Automated Run)

The repository includes a cross-platform pipeline runner script (`run.sh`). It automatically checks for the database, creates and isolates a virtual environment (`venv`), installs dependencies, runs preprocessing, trains the optimized Random Forest model, and runs the evaluation.

#### For macOS / Linux / Git Bash (Windows):
```bash
# Grant execution permissions (Mac/Linux only)
chmod +x run.sh

# Execute the pipeline
./run.sh
```
### Option B: Step-by-Step Execution (For Custom Experimentation)

This is the step-by-step execution option for running the pipeline.  It is highly configurable (unlike the one-click automated option which runs just our best recommended Random Forest model) and supports execution with different machine learning algorithms and parameters using a combination of a configuration file (`config.json`) and command-line interface (CLI) parameter overrides.

**Ensure you have the *score.db* file downloaded and placed in the data/ folder before starting.**<br>

### 3.1. Install Dependencies:<br>
    pip install -r requirements.txt

### 3.2 Run Preprocessing:<br>
This fetches data via SQLite, cleans features, and performs feature engineering.<br>


    python src/preprocessing.py

### 3.3 Model Training (With Model Swapping)<br>
You can train models using the default settings in config.json or swap algorithms directly using command-line arguments.<br>
<br>
**<u>Option A: Train the Default Model (Random Forest)</u>**<br>
This runs using the default hyperparameters and configuration specified inside config.json.<br>
Output is saved to: models/student_model.pkl (as specified in config.json)<br>
    
    python src/train.py

**<u>Option B: Train a Linear Regression Baseline</u>**<br>
Overrides the default model from the terminal. This automatically resets the hyperparameters to suit a linear baseline.<br>
Output is saved to: models/LinearRegression_model.pkl<br>

    python src/train.py --model LinearRegression


**<u>Option C: Train a Gradient Boosting Regressor</u>**<br>
Output is saved to: models/gradientboostingregressor_model.pkl<br>

    python src/train.py --model GradientBoostingRegressor



### 3.4 Model Evaluation:<br>
Evaluate your trained models dynamically by passing matching command-line arguments.<br>
<br>
**<u>Evaluate Default Model (Random Forest):</u>**<br>

    python src/evaluation.py

**<u>Evaluate Linear Regression:</u>**

    python src/evaluation.py --model LinearRegression

**<u>Evaluate Gradient Boosting:</u>**<br>

    python src/evaluation.py --model GradientBoostingRegressor

All evaluation will output the final RMSE and R² scores.

## 4. Pipeline flow and Logical Steps

The pipeline follows a sequential flow from raw data to evaluation:

1. **Data Ingestion**: Fetches raw data from score.db using SQLite relative paths.  

2. **Data Cleaning**: Handles missing values, drops "Red Herring" features (e.g., bag_color), and standardizes categorical inconsistencies.  

3. **Feature Engineering**: Converts sleep_time and wake_time into a continuous sleep_duration feature to capture threshold effects on performance.  

4. **Encoding & Modeling**: Uses a Scikit-Learn Pipeline to handle OneHotEncoding for categorical features and feeds them into a RandomForestRegressor.

5. **Serialization**: Saves the entire pipeline (preprocessor + model) as a single object for reusability.

## 5. Key EDA findings & Feature Processing

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

## 6. Choice of Model and Evaluation

To promote easy experimentation, all model training configurations are centralized. The workflow can be fine-tuned using the following parameters:<br>
<br>
**<u>The Configuration Command Center (config.json)</u>**<br>
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

## 7. Choice of Evaluated Algorithms: <br>

**<u>Random Forest Regressor (Default Ensemble Model)</u>**<br>
*Why Chosen*: Excellent at capturing complex interaction thresholds in student habits (such as the non-linear relationship where score performance drops sharply below 7 hours of sleep).
<br>
*Strengths*: Extremely robust against multi-modal distributions and requires no scaling.<br>
<br>
**<u>Gradient Boosting Regressor (Sequential Ensemble Model)</u>**<br>
*Why Chosen*: Builds trees sequentially to minimize residual errors, often achieving tighter fitting and lower error margins than Random Forest.<br>
<br>
**<u>Linear Regression (Baseline Parametric Model)</u>**<br>
*Why Chosen*: Serves as a vital baseline to quantify the performance boost gained by transitioning to non-linear tree-based models.

## 8. Model Evaluation Results (Test Set Evaluation)

To ensure strict validation and prevent data leakage, all models are evaluated **solely on an unseen test set partition (20% split)**. The project success criterion is established at a **Mean Absolute Error (MAE) < 6.0 marks** to ensure predictions are tight enough to provide reliable, actionable early interventions for school educators.

### Experimental Performance Metrics Comparison

| Model | Evaluation Command Override | RMSE | MAE | R² Score | Target (MAE < 6.0) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | *Default (no flag)* | **7.9057** | **5.8297** | **0.6780** | **Met** | **Selected Model** |
| **Gradient Boosting** | `--model GradientBoostingRegressor` | 8.2876 | 6.3934 | 0.6462 | Failed | Rejected |
| **Linear Regression** | `--model LinearRegression` | 10.0892 | 8.0855 | 0.4756 | Failed | Rejected |

### Technical Analysis & Insights

1. **Why the Random Forest Won (MAE: 5.83 marks, R²: 0.6780):**
   The Random Forest Regressor successfully met our strict acceptance criteria. It explains **67.8% of the variance** on unseen data, keeping predictions within an average margin of **$\pm$ 5.8 marks**. This high precision allows teachers to confidently identify at-risk students who need remedial help without suffering from excessive false alarms.

2. **Why the Linear Model Struggled (MAE: 8.09 marks, R²: 0.4756):**
   Linear Regression left more than 52% of the variance unexplained. Because it is forced to fit relationships to a straight line, it cannot model non-linear boundaries. For example, losing an hour of sleep does not degrade performance linearly; instead, performance drops sharply past the 7-hour threshold. Linear Regression averages these trends out, resulting in a much wider margin of error ($\pm$ 8.1 marks) which is too loose for practical school intervention.

3. **Why Gradient Boosting Fell Behind Random Forest (MAE: 6.39 marks, R²: 0.6462):**
   While Gradient Boosting is highly capable, its sequential optimization was slightly more sensitive to noise in the categorical indicators (such as tuition and CCA types) compared to the parallel, variance-reducing nature of Random Forest's bootstrap aggregation (bagging).

### Explanation of Evaluation Metrics
* **MAE (Mean Absolute Error):** Chosen as our primary business metric because it represents the average prediction error in physical units (marks). An MAE of 5.83 means our predictions are, on average, within 5.8 marks of the student's true O-level score.
* **RMSE (Root Mean Squared Error):** Chosen to penalize larger prediction errors more heavily. Our winning model's RMSE of 7.91 indicates that we have managed to keep large, outlier prediction errors to a minimum.
* **R² (Coefficient of Determination):** Measures the proportion of variance in O-level math scores that our features can predict.