import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Step 1: Select a CSV from the posture directory ---
csv_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the 'posture' folder.")

print("Available posture CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("Enter the number of the CSV file to use: ")) - 1

if choice < 0 or choice >= len(csv_files):
    raise ValueError("Invalid file selection.")

selected_csv = os.path.join(csv_folder, csv_files[choice])
print(f"\nLoading data from: {selected_csv}")

# --- Step 2: Load and process the selected posture CSV ---
df = pd.read_csv(selected_csv)

# Standardize posture labels
df['posture'] = df['posture'].replace({
    'Bad - Slouching': 'Slouching',
    'Bad - Forward Head': 'Forward Head'
})

# Count posture occurrences
posture_counts = df['posture'].value_counts()
posture_counts_scaled = posture_counts // 100  # for bar chart

# --- Step 3: Plot bar chart ---
plt.figure(figsize=(8, 5))
posture_counts_scaled.plot(kind='bar', color='lightgreen')
plt.title('Posture Frequency (Bar Chart)')
plt.xlabel('Posture')
plt.ylabel('Count (x100)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Step 4: Plot pie chart ---
plt.figure(figsize=(6, 6))
plt.pie(
    posture_counts,
    labels=posture_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Pastel1.colors
)
plt.title('Posture Frequency Distribution (Pie Chart)')
plt.tight_layout()
plt.show()
