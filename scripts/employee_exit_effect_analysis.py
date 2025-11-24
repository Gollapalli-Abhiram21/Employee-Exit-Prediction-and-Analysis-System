import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
csv_path = "dataset/HR_comma_sep.csv"
output_dir = "output"
plots_dir = os.path.join(output_dir, "plots")
reports_dir = os.path.join(output_dir, "reports")

# Ensure directories exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)

# Filter employees who have left
left_df = df[df['left'] == 1]

# -------------------------------
# 1. Satisfaction Level Analysis
# -------------------------------
plt.figure(figsize=(8, 6))
sns.histplot(left_df['satisfaction_level'], bins=20, kde=True, color='blue')
plt.title('Distribution of Satisfaction Level (Employees Who Left)')
plt.xlabel('Satisfaction Level')
plt.ylabel('Count')
plt.savefig(os.path.join(plots_dir, "satisfaction_level_distribution.png"))
plt.close()

# -------------------------------
# 2. Department vs Exit
# -------------------------------
plt.figure(figsize=(12, 8))
sns.countplot(x='Department', data=left_df,
              order=left_df['Department'].value_counts().index)
plt.title('Employees Who Left by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig(os.path.join(plots_dir, "department_vs_exit.png"))
plt.close()

# -------------------------------
# 3. Promotion in Last 5 Years
# -------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(x='promotion_last_5years', data=left_df)
plt.title('Promotion in Last 5 Years (Employees Who Left)')
plt.xlabel('Promotion in Last 5 Years')
plt.ylabel('Count')
plt.savefig(os.path.join(plots_dir, "promotion_vs_exit.png"))
plt.close()

# -------------------------------
# 4. Salary Level vs Exit
# -------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(x='salary', data=left_df, order=['low', 'medium', 'high'])
plt.title('Salary Level of Employees Who Left')
plt.xlabel('Salary Level')
plt.ylabel('Count')
plt.savefig(os.path.join(plots_dir, "salary_vs_exit.png"))
plt.close()

# -------------------------------
# Summary Insights
# -------------------------------
summary = []
summary.append("Effect Analysis for Employees Who Left:\n")
summary.append(
    f"Average Satisfaction Level: {left_df['satisfaction_level'].mean():.2f}\n")
summary.append("\nDepartment Distribution:\n")
summary.append(str(left_df['Department'].value_counts()))
summary.append("\nPromotion in Last 5 Years:\n")
summary.append(str(left_df['promotion_last_5years'].value_counts()))
summary.append("\nSalary Distribution:\n")
summary.append(str(left_df['salary'].value_counts()))

# Save summary in reports folder
summary_file = os.path.join(reports_dir, "effect_analysis_summary.txt")
with open(summary_file, "w") as f:
    f.write("\n".join(summary))

# -------------------------------
# Combined CSV Report
# -------------------------------
report_data = {
    "Metric": ["Average Satisfaction Level"],
    "Value": [round(left_df['satisfaction_level'].mean(), 2)]
}

# Convert to DataFrame
report_df = pd.DataFrame(report_data)

# Add Department counts
dept_counts = left_df['Department'].value_counts().reset_index()
dept_counts.columns = ['Department', 'Count']

# Add Salary counts
salary_counts = left_df['salary'].value_counts().reset_index()
salary_counts.columns = ['Salary', 'Count']

# Add Promotion counts
promo_counts = left_df['promotion_last_5years'].value_counts().reset_index()
promo_counts.columns = ['Promotion_Last_5_Years', 'Count']

# Save all as separate sheets in one Excel file OR combine into one CSV
combined_csv_path = os.path.join(reports_dir, "effect_analysis_combined.csv")

with open(combined_csv_path, "w") as f:
    f.write("Summary Metrics\n")
report_df.to_csv(combined_csv_path, mode='a', index=False)
f = open(combined_csv_path, "a")
f.write("\nDepartment Distribution\n")
dept_counts.to_csv(f, index=False)
f.write("\nSalary Distribution\n")
salary_counts.to_csv(f, index=False)
f.write("\nPromotion Distribution\n")
promo_counts.to_csv(f, index=False)
f.close()

print(
    f"Analysis completed. Visualizations saved in '{plots_dir}', summary in '{summary_file}', and combined CSV in '{combined_csv_path}'.")


