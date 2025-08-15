import os
import pandas as pd
import matplotlib.pyplot as plt

# --- List CSV files ---
csv_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the folder.")

print("Available emotion CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("Enter the number of the CSV file to use: ")) - 1
if choice < 0 or choice >= len(csv_files):
    raise ValueError("Invalid file selection.")

selected_csv = os.path.join(csv_folder, csv_files[choice])
print(f"\nLoading data from: {selected_csv}")

# --- Load CSV ---
df = pd.read_csv(selected_csv)
df['Time'] = pd.to_datetime(df['Time'])

# Aggregate into 1-minute bins
df['TimeBin'] = df['Time'].dt.floor('T')  # 'T' = minute

# Calculate emotion counts per time bin
emotion_counts = df.groupby(['TimeBin', 'Emotion Detection']).size().unstack(fill_value=0)

# Convert counts to proportions (for stacked area plot)
emotion_props = emotion_counts.divide(emotion_counts.sum(axis=1), axis=0)

# Plot stacked area chart
plt.figure(figsize=(12, 5))
emotion_props.plot.area(colormap='Pastel1', alpha=0.8)
plt.title('Emotion Proportions Over Time (1-minute bins)')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend(title='Emotion', loc='upper right')
plt.tight_layout()
plt.show()

# Pie chart (overall emotion distribution)
emotion_totals = df['Emotion Detection'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    emotion_totals,
    labels=emotion_totals.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Pastel1.colors
)
plt.title('Overall Emotion Distribution')
plt.tight_layout()
plt.show()
