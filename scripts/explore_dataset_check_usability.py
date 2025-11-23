# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "dataset/HR_comma_sep.csv"
output_dir = "output"
plots_dir = os.path.join(output_dir, "plots")
reports_dir = os.path.join(output_dir, "reports")

# Ensure directories exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
print(f"Directories ensured: {plots_dir}, {reports_dir}")

# Load data
df = pd.read_csv(csv_path)
print(f"\n Columns are: {list(df.columns)}")
print(f"\n first five rows are:\n{df.head()}")
print("Successfully imported dataset.")

# Check for missing values
'''print(df.isnull())    # returns false if missing values doesnt exists '''
missing_values = df.isnull().sum(
)      # returns the columns with count of the missing values

print("Missing values per column:\n", missing_values)

# Decision logic
if missing_values.sum() == 0:
    print("\nNo missing values found. Data can be used as it is.")
else:
    print("Missing values detected. Handling missing values...")

    # Example handling: fill numeric columns with mean
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    print("Missing values handled successfully.")


# Save missing values report
missing_values.to_csv(os.path.join(reports_dir, "missing_values_report.csv"))

