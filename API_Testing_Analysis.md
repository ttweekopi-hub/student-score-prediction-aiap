# Model Prediction & Deep-Dive Analysis

This page documents the behavioral validation of the Student Exam Score Prediction API. Three models (**Random Forest**, **Gradient Boosting**, and **Linear Regression**) were tested across various scenarios to ensure logical consistency and evaluate performance nuances.

---

## 1. Baseline Scenario (Standard Student)
**Description:** A student with 95% attendance, 15 hours of weekly study, and active Tuition support.

**Full JSON Payload:**
```json
{
  "number_of_siblings": 2,
  "direct_admission": "No",
  "CCA": "Sports",
  "learning_style": "Visual",
  "tuition": "Yes",
  "n_male": 10,
  "student_age": 16,
  "hours_per_week": 15,
  "attendance_rate": 95,
  "sleep_duration": 8
}
```

| Model | Predicted Score | Interpretation |
| :--- | :--- | :--- |
| **Random Forest (RF)** | 65.59 | The most optimistic; likely rewards the combination of high attendance and tuition. |
| **Gradient Boosting (GBR)** | 63.30 | Conservative; focuses on the most frequent patterns in the data. |
| **Linear Regression (LR)** | 62.94 | The mathematical "average" expectation for these inputs. |

🔍 ***Test Findings: Model Consensus***<br>
In the baseline test, all three models produced results within a narrow 2.65-point range — well within the model's MAE of 5.83. This high level of consensus confirms that the dataset has a strong signal. The slight lead of the Random Forest suggests it is capturing non-linear "bonuses" for students who have both high attendance and tuition.

## 2. Stress Test: Attendance Sensitivity
**Description:** Extreme scenario where attendance is dropped from 95% → 10% to test model logic.

**Full JSON Payload:**
```json
{
  "number_of_siblings": 2,
  "direct_admission": "No",
  "CCA": "Sports",
  "learning_style": "Visual",
  "tuition": "Yes",
  "n_male": 10,
  "student_age": 16,
  "hours_per_week": 15,
  "attendance_rate": 10,
  "sleep_duration": 8
}
```
| Model | Predicted Score | Point Drop |
| :--- | :--- | :--- |
| **Random Forest (RF)** | 43.58 | -22.01 |
| **Gradient Boosting (GBR)** | 46.13 | -17.17 |
| **Linear Regression (LR)** | 34.58 | -28.36 |

🔍 ***Test Findings: The Impact of Presence***<br>
This confirms attendance as a significant driver, consistent with its 3rd-place feature importance ranking.

Linear Regression shows the most aggressive penalty (-28.36). Since LR is a linear function, it assumes that every 1% of missing attendance results in a fixed point deduction.

GBR is the most "forgiving" at 46.13. It likely assumes that the 15 hours of study still provides a "failing but non-zero" floor for the student.

RF sits in the middle, identifying a clear fail state while acknowledging other input factors.

## 3. Support Analysis: The "Tuition Premium"
**Description:** Comparing outcomes when external tuition is toggled from Yes to No.

**Full JSON Payload (Tuition: No):**
```json
{
  "number_of_siblings": 2,
  "direct_admission": "No",
  "CCA": "Sports",
  "learning_style": "Visual",
  "tuition": "No",
  "n_male": 10,
  "student_age": 16,
  "hours_per_week": 15,
  "attendance_rate": 95,
  "sleep_duration": 8
}
```
| Model | Score (Tuition: Yes) | Score (Tuition: No) | Net Impact |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | 65.59 | 58.83 | +6.76 |
| **Linear Regression (LR)** | 62.94 | 57.08 | +5.86 |
| **Gradient Boosting (GBR)** | 63.30 | 60.96 | +2.34 |

***🔍 Test Findings: Feature Weighting***<br>
Random Forest places the highest value on tuition (+6.76). This suggests that in the training data, the presence of tuition was a meaningful categorical indicator of passing. GBR is significantly less sensitive to tuition, suggesting it relies more heavily on numerical features (hours/attendance) rather than categorical labels.

## 4. Engagement Analysis: CCA Variability
**Description:** Testing the impact of different Co-Curricular Activities (Sports, Clubs, or None).

**Full JSON Payload (Example - CCA: None):**
```json
{
  "number_of_siblings": 2,
  "direct_admission": "No",
  "CCA": "None",
  "learning_style": "Visual",
  "tuition": "Yes",
  "n_male": 10,
  "student_age": 16,
  "hours_per_week": 15,
  "attendance_rate": 95,
  "sleep_duration": 8
}
```
| Model | Clubs | Sports | None |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | 65.55 | 65.59 | 66.01 |
| **Gradient Boosting (GBR)** | 63.30 | 63.30 | 63.30 |
| **Linear Regression (LR)** | 62.66 | 62.94 | 62.49 |

***🔍 Test Findings: Complex Category Handling***<br>
This section reveals a major technical distinction between the models:

Gradient Boosting (GBR) Invariance: The GBR score remains exactly 63.30 across all CCAs. This indicates that the GBR model likely pruned these features during training, considering them "noise". This is a notable limitation: the GBR model cannot differentiate students based on CCA engagement, reducing its utility for holistic student profiling.

Random Forest Logic: RF ranks "None" as the highest (66.01). This suggests the model found that students without CCAs might have more uninterrupted focus for exams. This is consistent with the feature importance table, where CCA: None ranks 5th at 6.45% — far above CCA: Clubs (0.26%) and CCA: Sports (0.18%)

Linear Regression Logic: LR actually ranks "Sports" (62.94) higher than "None" (62.49). This implies a positive linear correlation between physical activity and academic performance within the LR framework.

## Conclusion & Recommendation
Based on the deep-dive analysis, the Random Forest (RF) model is recommended as the primary production model for the following reasons:

**Superior Feature Sensitivity**: Unlike GBR, which ignored CCA variability, Random Forest successfully captured the subtle impacts of every feature. This ensures the model utilizes the full context of a student's profile rather than relying solely on numerical averages.

**Balanced Robustness**: While Linear Regression reacted too aggressively to low attendance (dropping to 34.58), Random Forest provided a more realistic failing "floor" (43.58). This indicates the model correctly balances primary drivers with secondary support features.

**Holistic Prediction**: The RF model's responsiveness to categorical changes (like Tuition and CCA) results in a more sophisticated and "intelligent" prediction, making it the most reliable tool for identifying both high-performing and at-risk students.