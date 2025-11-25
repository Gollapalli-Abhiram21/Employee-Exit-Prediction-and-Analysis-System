# EMPLOYEE_EXIT_PREDICTION_AND_ANALYSIS_SYSTEM

An end‑to‑end solution for analyzing reasons behind premature employee exits in a multinational company. Designed to support HR teams with actionable insights through data exploration, pattern analysis, and predictive modeling.

---
## 🚀 Project Overview
- Validates HR dataset (~15,000 records) for usability
- Explores satisfaction, workload, tenure, promotions, and salary impact on exits
- Builds predictive models to identify employees likely to leave
- Provides visualizations and reports for HR decision‑making

---
## 📂 Project Structure
```
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

```
---

## ⚙️ Installation & Usage

1. Clone the repository
2. Create and activate a virtual environment
```
python -m venv .myenv
source .myenv/bin/activate   # macOS/Linux
.myenv\Scripts\activate      # Windows
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run scripts in order:
```
python scripts/explore_dataset_check_usability.py
python scripts/satisfaction_vs_hours.py
python scripts/employee_exit_effect_analysis.py
python scripts/employee_exit_prediction.py
```
5. Outputs (plots, reports, models) are saved in `output/plots/` and `output/reports/`.
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

## Key Functionalities

**explore_dataset_check_usability.py**: Loads and checks data, verifies usability, and saves missing values report.

**satisfaction_vs_hours.py**: Generates scatter plot showing the relationship between satisfaction level and monthly hours for former employees.

**employee_exit_effect_analysis.py**: Performs exploratory analysis on exit reasons, including department, salary, and promotions. Produces visualizations and summary statistics.

**employee_exit_prediction.py**: Trains and evaluates machine learning models (Logistic Regression and Random Forest) to predict employee departures. Outputs model metrics and feature importance.

---

## 🔎 Workflow
- **Data Validation** – check completeness & missing values
- **Exploratory Analysis** – visualize satisfaction, salary, promotions, departments
- **Modeling** – train Logistic Regression & Random Forest classifiers
- **Evaluation & Reporting** – accuracy, precision, recall, F1, feature importance
---

## 🛠️ Technologies
- Python 3.8+
- pandas, numpy – data processing
- matplotlib, seaborn – visualization
- scikit‑learn – modeling & metrics
- joblib – model persistence
---

## 📊 Results

This project evaluated two machine learning models to predict employee exits:
1. Logistic Regression
  - Accuracy: 75.77%
  - Precision: 46.92%
  - Recall: 22.66%
  - F1 Score: 30.56%
2. Random Forest (selected model):
  - Accuracy: 98.8%
  - Precision: 98.7%
  - Recall: 96.3%
  - F1 Score: 97.5%

3. Top Features Influencing Exit:
  - Satisfaction level
  - Time spent in company
  - Number of projects
  - Average monthly hours
  - Last evaluation
---

## 🔮 Future Work

- Interactive dashboards (Streamlit/Dash)
- Advanced ML models (XGBoost, LightGBM)
- Automated hyperparameter tuning
- REST API deployment for predictions
---
## 📜 License

Educational use, free to modify.

---
## 👤 Author

**Gollapalli Abhiram**

_Last updated: November 2025_

