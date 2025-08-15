import os
import pandas as pd
import matplotlib.pyplot as plt

# --- List CSV files ---
csv_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the folder.")

print("Available stress CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("Enter the number of the CSV file to use: ")) - 1
if choice < 0 or choice >= len(csv_files):
    raise ValueError("Invalid file selection.")

selected_csv = os.path.join(csv_folder, csv_files[choice])
print(f"\nLoading data from: {selected_csv}")

# --- Load CSV ---
df = pd.read_csv(selected_csv)

# Parse Time column - only time, no date
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

# To group by minute, convert Time to datetime with dummy date
df['DateTime'] = pd.to_datetime(df['Time'].astype(str))

# Floor to minute bins
df['TimeBin'] = df['DateTime'].dt.floor('T')

# Aggregate counts of each stress level per minute
stress_counts = df.groupby(['TimeBin', 'Emotion Detection']).size().unstack(fill_value=0)

# Calculate proportions
stress_props = stress_counts.divide(stress_counts.sum(axis=1), axis=0)

# Plot stacked area chart
plt.figure(figsize=(12, 5))
stress_props.plot.area(colormap='Set2', alpha=0.8)
plt.title('Stress Level Proportions Over Time (1-minute bins)')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend(title='Stress Level', loc='upper right')
plt.tight_layout()
plt.show()

# Pie chart for overall stress distribution
stress_totals = df['Emotion Detection'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    stress_totals,
    labels=stress_totals.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Set2.colors
)
plt.title('Overall Stress Level Distribution')
plt.tight_layout()
plt.show()
