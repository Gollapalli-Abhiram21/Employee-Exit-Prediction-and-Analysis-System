import pandas as pd
import matplotlib.pyplot as plt
import os

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
left_employees = df[df['left'] == 1]

# Generate scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(left_employees['satisfaction_level'],
            left_employees['average_montly_hours'],
            alpha=0.5, color='blue')
plt.title(
    'Relationship: Satisfaction Level vs Average Monthly Hours (Employees Left)')
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Monthly Hours')
plt.grid(True)

# Save plot in plots folder
plot_file = os.path.join(plots_dir, 'satisfaction_vs_hours_left.png')
plt.savefig(plot_file)
plt.close()

print(f"Scatter plot saved at: {plot_file}")

