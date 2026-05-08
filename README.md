# Student Score Prediction Pipeline

## 1. Overview and Folder Structure

This case study was taken from the AIAP Technical Assessment Past Year Series repository:
https://github.com/AISG-AIAP/AIAP-Technical-Assessment-Past-Years-Series

I have selected to use the "Student Score Prediction" case study from the Past Year Series collection to practise building an end-to-end ML pipeline covering EDA/ML Pipeline as requirements to pass the AIAP Technical Assessment.

The pipeline is designed to predict final mathematics scores using student demographic and academic indicators to proactively identify at-risk individuals. The system targets an MAE below 6.0 marks (≤6% error on a 100-point scale), balancing predictive accuracy with interpretability for practical educational intervention. This threshold was selected because errors within ±6 marks preserve meaningful distinction between performance bands while remaining actionable for identifying at-risk students.

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
└── .gitignore          # files/folders to be ignored by Git
└── Dockerfile          # Docker config file
└── Makefile            # Orchestration script to standardize Docker and local commands
└── README.md           # Project documentation and usage instructions
└── config.json         # central control file to change model settings
└── requirements.txt    # Python dependencies
└── run.sh              # run script to auto-execute pipeline
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

**The pipeline also supports containerized execution via Docker — see Option A in Section 3.**

## 3. Instructions for Execution

This pipeline supports three execution methods: Containerized (via Docker) for guaranteed reproducibility, a **one-click automated runner** (recommended for quick grading) and **manual step-by-step commands** (designed for custom experiments and model-swapping).

### Option A (Docker)
Before running via Docker, you must build the environment. This ensures all dependencies and SGT-localized logging utilities are baked in.

```bash
make build
```

The following commands work identically across Windows (WSL), Mac, and Linux. They utilize a "self-healing" Makefile that automatically creates local log files and mounts data/model volumes.

| Alias | Machine Learning Algorithm |
| :--- | :--- |
| `lr` | Linear Regression |
| `rf` | Random Forest Regressor (Default) |
| `gbm` / `gbr` | Gradient Boosting Regressor |
| `svr` | Support Vector Regressor |

| Action | Docker Command | Local Python Command |
| :--- | :--- | :--- |
| **Preprocess Data** | `make preprocess` | `python -m src.preprocessing` |
| **Train (Default RF)** | `make train` | `python -m src.train` |
| **Evaluate Model (Default RF)** | `make evaluate` | `python -m src.evaluation` |
| **Train (Linear Reg)** | `make train model=lr` | `python -m src.train --model lr` |
| **Evaluate Model (Linear Reg)** | `make evaluate model=lr` | `python -m src.evaluation --model lr` |
 **Train (Gradient Boost Regressor)** | `make train model=gbr` | `python -m src.train --model gbr` |
| **Evaluate Model (Gradien Boost Regressor)** | `make evaluate model=gbr` | `python -m src.evaluation --model gbr` |

### Option B: Quickstart (One-Click Automated Run)

The repository includes a cross-platform pipeline runner script (`run.sh`). It automatically checks for the database, creates and isolates a virtual environment (`venv`), installs dependencies, runs preprocessing, trains the optimized Random Forest model, and runs the evaluation.

#### For macOS / Linux / Git Bash (Windows):
```bash
# Grant execution permissions (Mac/Linux only)
chmod +x run.sh

# Execute the pipeline
./run.sh
```
### Option C: Step-by-Step Execution (For Custom Experimentation)

This is the alternative execution option for running the pipeline.  It is highly configurable (unlike the one-click automated option which runs just our best recommended Random Forest model) and supports execution with different machine learning algorithms and parameters using a combination of a configuration file (`config.json`) and command-line interface (CLI) parameter overrides.<br>
**Run all terminal commands from the root folder of the cloned repository (/student-score-prediction-aiap).**

### 3.1. Install Dependencies:<br>
    pip install -r requirements.txt

### 3.2 Run Preprocessing:<br>
This fetches data via SQLite, cleans features, and performs feature engineering.<br>

    python -m src.preprocessing

### 3.3 Model Training (With Model Swapping)<br>
You can train models using the default settings in config.json or swap algorithms directly using command-line arguments.<br>
<br>
**<u>Option A: Train the Default Model (Random Forest)</u>**<br>
This runs using the default hyperparameters and configuration specified inside config.json.<br>
Output is saved to: models/student_model.pkl (as specified in config.json)<br>
    
    python -m src.train

**<u>Option B: Train a Linear Regression Baseline</u>**<br>
Overrides the default model from the terminal. This automatically resets the hyperparameters to suit a linear baseline.<br>
Output is saved to: models/LinearRegression_model.pkl<br>

    python -m src.train --model lr


**<u>Option C: Train a Gradient Boosting Regressor</u>**<br>
Output is saved to: models/gradientboostingregressor_model.pkl<br>

    python -m src.train --model gbr



### 3.4 Model Evaluation:<br>
Evaluate your trained models dynamically by passing matching command-line arguments.<br>
<br>
**<u>Evaluate Default Model (Random Forest):</u>**<br>

    python -m src.evaluation

**<u>Evaluate Linear Regression:</u>**

    python -m src.evaluation --model lr

**<u>Evaluate Gradient Boosting:</u>**<br>

    python -m src.evaluation --model gbr

All evaluation will output the final RMSE and R² scores.

## 4. Pipeline flow and Logical Steps

The pipeline follows a sequential flow from raw data to evaluation:

1. **Data Ingestion**: Fetches raw data from score.db using SQLite relative paths.  

2. **Data Cleaning**: Handles missing values, drops "Red Herring" features (e.g., bag_color), and standardizes categorical inconsistencies.  

3. **Feature Engineering**: Converts sleep_time and wake_time into a continuous sleep_duration feature to capture threshold effects on performance.  

4. **Encoding & Modeling**: Uses a Scikit-Learn Pipeline to handle OneHotEncoding for categorical features and feeds them into a RandomForestRegressor.

5. **Serialization**: Saves the entire pipeline (preprocessor + model) as a single object for reusability.

## 5. Feature Processing

| Feature | Type | Processing Action | Reason |
| :--- | :--- | :--- | :--- |
| `student_id` | ID | Dropped | No predictive signal. |
| `index` | ID | Dropped | Duplicate row index artifact from the database export; no predictive signal. |
| `tuition` | Categorical | Replace 'Y'/'N' with 'Yes'/'No' | Standardize labels for encoding. |
| `n_female` | Numerical | Dropped | High multicollinearity with `n_male`. |
| `sleep_time` | String | Combined into `sleep_duration` | Convert raw time to usable duration. |
| `attendance_rate` | Numerical | Imputed with Median | Handle missing values without dropping rows. |
| `mode_of_transport` | Categorical | Dropped | No correlation with academic performance observed in EDA; logistical detail with no predictive signal. |
| `CCA` | Categorical | Normalized into consistent categories: 'Clubs', 'Sports', 'None' | To ensure the OneHotEncoder correctly interpreted "None" as a valid behavioral state rather than a missing value. |
| `bag_color` | Categorical | Dropped | Unlikely owning a red bag will make a student better at math. |

## 📊 Data Quality & Cleaning Findings<br>
Target Variable Integrity: During the initial data audit, 495 rows were identified as missing the target variable (final_test).

Action Taken: These rows were deliberately dropped from the pipeline rather than using imputation.

Justification: In predictive modeling, imputing a target variable introduces "synthetic noise" and artificial bias that can lead to misleading accuracy metrics. Dropping these records ensures the model is trained only on ground-truth performance data, maintaining the scientific integrity of the results.

Multicollinearity (Gender Distribution): Analysis of classroom demographics revealed that n_male and n_female were perfectly negatively correlated. To prevent the Dummy Variable Trap and ensure model stability, n_female was dropped, and n_male was retained as the representative feature for classroom gender composition.

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
    "drop_columns": ["student_id", "bag_color", "mode_of_transport", "n_female", "index"],
    "imputation_strategy": "median"
  },
  "model": {
    "algorithm": "RandomForestRegressor",
        "parameters": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "random_state": 42
  },
    "save_path": "models/student_model.pkl"
  }
}
```

## 7. Choice of Evaluated Algorithms: <br>

**<u>Random Forest Regressor (Default Ensemble Model)</u>**<br>
*Why Chosen*: Excellent at capturing complex interaction thresholds in student habits (such as the non-linear relationship of the high importance of a lack of CCA vs. the low importance of specific CCAs).
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
| **Random Forest** | *Default (no flag)* | **7.9083** | **5.8300** | **0.6778** | **Met** | **Selected Model** |
| **Gradient Boosting** | `--model gbr` | 8.2750 | 6.3895 | 0.6472 | Failed | Rejected |
| **Linear Regression** | `--model lr` | 10.0943 | 8.0861 | 0.4751 | Failed | Rejected |

### Technical Analysis & Insights

1. **Why the Random Forest Won (MAE: 5.83 marks, R²: 0.6778):**
   The Random Forest Regressor successfully met the acceptance criteria of < 6 MAE score. It explains **67.7% of the variance** on unseen data, keeping predictions within an average margin of **$\pm$ 5.8 marks**. This high precision allows teachers to confidently identify at-risk students who need remedial help without suffering from excessive false alarms.

2. **Why Gradient Boosting Fell Behind Random Forest (MAE: 6.38 marks, R²: 0.6472):**
   While Gradient Boosting is highly capable, its sequential optimization was slightly more sensitive to noise in the categorical indicators (such as tuition and CCA types) compared to the parallel, variance-reducing nature of Random Forest's bootstrap aggregation (bagging).

3. **Why the Linear Model Struggled (MAE: 8.08 marks, R²: 0.4751):**
   Linear Regression left more than 52% of the variance unexplained. Because it is forced to fit relationships to a straight line, it cannot model non-linear boundaries. For example, losing an hour of sleep does not degrade performance linearly; instead, performance drops sharply past the 7-hour threshold. Linear Regression averages these trends out, resulting in a much wider margin of error ($\pm$ 8.0 marks) which is too loose for practical school intervention.

### Feature Importance Analysis (Random Forest)

| Rank | Feature / Predictive Factor | Relative Influence (%) |
| :---: | :--- | :---: |
| 1 | Number of Siblings | 32.67% |
| 2 | Hours Studied per Week | 19.21% |
| 3 | Attendance Rate | 15.20% |
| 4 | Direct Admission (Yes) | 11.34% |
| 5 | Co-Curricular Activity: None | 6.45% |
| 6 | Learning Style: Visual | 5.07% |
| 7 | Extra Tuition (Yes) | 4.95% |
| 8 | n_male | 3.86% |
| 9 | gender_Male | 0.38% |
| 10 | Student Age | 0.36% |
| 11 | Co-Curricular Activity: Clubs | 0.26% |
| 12 | Co-Curricular Activity: Sports | 0.18% |
| 13 | Sleep Duration | 0.08% |

### Feature Importance Analysis (Gradient Boosting Regressor)

| Rank | Feature / Predictive Factor | Relative Influence (%) |
| :---: | :--- | :---: |
| 1 | Number of Siblings | 31.09% |
| 2 | Hours Studied per Week | 19.43% |
| 3 | Attendance Rate | 13.80% |
| 4 | Co-Curricular Activity: None | 11.21% |
| 5 | Direct Admission (Yes) | 9.67% |
| 6 | Extra Tuition (Yes) | 6.93% |
| 7 | Learning Style: Visual | 5.33% |
| 8 | Class Gender Ratio (Number of Males) | 2.28% |
| 9 | Sleep Duration | 0.10% |
| 10 | Gender: Male | 0.08% |
| 11 | Student Age | 0.04% |
| 12 | Co-Curricular Activity: Clubs | 0.02% |
| 13 | Co-Curricular Activity: Sports | 0.01% |

Linear Regression model does not natively support feature importances.

## 📊 Key Findings: What Drives Final Math Scores?

Using a **Random Forest Regressor**, I analyzed which factors have the greatest predictive power on a student's final math score. 

### 1. The "Big Three" Predictors (Over 67% of Total Influence)
Just three features account for **67.08%** of the model's decision-making power when predicting math scores:
* **Number of Siblings (32.67%):** Unexpectedly, the size of a student's family is the single strongest predictor in this model. This may be linked to other background factors in a student's home environment, but the model cannot prove that having more siblings directly affects math scores.
* **Hours Studied per Week (19.21%):** Unsurprisingly, active effort and study time are critical drivers of academic success.
* **Attendance Rate (15.20%):** Showing up matters. Consistent school attendance is the third most vital pillar of performance.

### 2. Moderate Influencers (The School & Social Environment)
* **Direct Admission & Class Demographics:** Whether a student was admitted directly (11.34%) hold moderate weight, suggesting that admission pathways may shape outcomes.
* **Lack of Activities (6.45%):** Having *no* Co-Curricular Activities (CCAs) is more influential than participating in specific ones (like sports or clubs), indicating that being completely disengaged from school activities may impact academic focus.

### 3. Low-Impact Factors (Surprisingly Weak Predictors)
Several factors that intuitively seem important actually have **almost zero predictive power** in this model:
* **Sleep & Age:** Student age (0.36%) and sleep duration (0.08%) barely register as important to the final math score.
* **Specific CCAs (<1%):** While having *no* activity had some impact, whether a student chose Sports (0.18%) vs. Clubs (0.26%) does not meaningfully alter the prediction.<br>

While sleep_duration shows low overall predictive importance (0.08%), EDA reveals it acts as a critical threshold requirement. Data indicates that 7+ hours of sleep is a necessary condition for achieving scores above 50, even though it does not guarantee a high result on its own.


## Explanation of Evaluation Metrics
* **MAE (Mean Absolute Error):** Chosen as our primary business metric because it represents the average prediction error in physical units (marks). An MAE of 5.83 means our predictions are, on average, within 5.8 marks of the student's true O-level score.
* **RMSE (Root Mean Squared Error):** Chosen to penalize larger prediction errors more heavily. Our winning model's RMSE of 7.91 indicates that we have managed to keep large, outlier prediction errors to a minimum.
* **R² (Coefficient of Determination):** Measures the proportion of variance in O-level math scores that our features can predict.
