# EMPLOYEE_EXIT_PREDICTION_AND_ANALYSIS_SYSTEM

This project provides an end-to-end solution for analyzing reasons behind premature employee exits in a multinational company. The goal is to support HR teams with actionable insights by combining data exploration, pattern analysis, and predictive modeling using Machine Learning.

---

## Project Overview

This project addresses the following key HR requirements:

- Explore the dataset and check if it is suitable for analysis.
- Determine the relationship between satisfaction level and working hours of employees who have left the organization.
- Understand the effect of satisfaction level, department, recent promotions, and salary on employee exits.
- Build a machine learning model to predict which employees are likely to leave.
- Visualize key results and generate comprehensive reports for HR decision-making.

The dataset consists of ~15,000 employee records with features such as satisfaction, evaluation scores, project counts, average monthly hours, length of service, accident history, promotion status, department, and salary.

---

## Project Structure

EMPLOYEE_EXIT_PREDICTION_AND_ANALYSIS_SYSTEM/
├── dataset/
│ └── HR_comma_sep.csv # HR employee dataset
├── output/
│ ├── plots/ # Visualizations (PNG files)
│ │ ├── satisfaction_vs_hours_left.png
│ │ ├── satisfaction_level_distribution.png
│ │ ├── department_vs_exit.png
│ │ ├── promotion_vs_exit.png
│ │ ├── salary_vs_exit.png
│ │ ├── confusion_matrix.png
│ │ └── feature_importance.png
│ ├── reports/ # Reports, metrics, models, CSVs
│ │ ├── dataset_usability_summary.txt
│ │ ├── missing_values_report.csv
│ │ ├── effect_analysis_summary.txt
│ │ ├── effect_analysis_combined.csv
│ │ ├── model_metrics.txt
│ │ ├── feature_importance.csv
│ │ ├── employee_exit_model.pkl
│ │ └── final_report.md
├── scripts/ # Python analysis scripts
│ ├── explore_dataset_check_usability.py
│ ├── satisfaction_vs_hours.py
│ ├── employee_exit_effect_analysis.py
│ └── employee_exit_prediction.py
text

---

## Key Functionalities

- **explore_dataset_check_usability.py:**  
  Loads and checks data, verifies usability, and saves missing values report.

- **satisfaction_vs_hours.py:**  
  Generates scatter plot showing the relationship between satisfaction level and monthly hours for former employees.

- **employee_exit_effect_analysis.py:**  
  Performs exploratory analysis on exit reasons, including department, salary, and promotions. Produces visualizations and summary statistics.

- **employee_exit_prediction.py:**  
  Trains and evaluates machine learning models (Logistic Regression and Random Forest) to predict employee departures. Outputs model metrics and feature importance.

---

## Getting Started

1. **Install dependencies:**
pip install -r requirements.txt

text

2. **Run the scripts in order:**
python scripts/explore_dataset_check_usability.py
python scripts/satisfaction_vs_hours
py python scripts/employee_exit_effect_analy
text

All outputs (plots, reports, models) will be located within the relevant `output/plots/` and `output/reports/` folders.

---

## Dataset Features

- satisfaction_level
- last_evaluation
- number_project
- average_monthly_hours
- time_spend_company
- Work_accident
- left
- promotion_last_5years
- Department
- salary

---

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, openpyxl

---

## Model Performance and Key Features

This project evaluated two machine learning models to predict employee exits:

1. Logistic Regression

  - Accuracy: 75.77%

  - Precision: 46.92%

  - Recall: 22.66%

  - F1 Score: 30.56%

2. Random Forest

  - Accuracy: 98.83%

  - Precision: 98.69%

  - Recall: 96.32%

  - F1 Score: 97.49%

The Random Forest model demonstrated superior prediction performance and was selected for further use.

---

## Top Features Influencing Employee Exit

The most important features identified by the Random Forest model were:

- satisfaction_level (importance: 0.360)

- time_spend_company (importance: 0.184)

- number_project (importance: 0.165)

- average_monthly_hours (importance: 0.149)

- last_evaluation (importance: 0.116)

These attributes significantly contribute to predicting whether an employee is likely to leave the organization, providing actionable insights for HR management.

---

## Future Work

- Interactive dashboards (Streamlit/Dash)
- More advanced ML models (XGBoost, LightGBM)
- Automated hyperparameter tuning
- REST API deployment for predictions

---

## License

This project is for educational use and free to modify.

---

## Author

Gollapalli Abhiram

_Last updated: November 2025_

