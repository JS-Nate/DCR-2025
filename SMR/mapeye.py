import os
import pandas as pd
import matplotlib.pyplot as plt

# --- List CSV files ---
csv_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the folder.")

print("Available eye tracking CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("Enter the number of the CSV file to use: ")) - 1
if choice < 0 or choice >= len(csv_files):
    raise ValueError("Invalid file selection.")

selected_csv = os.path.join(csv_folder, csv_files[choice])
print(f"\nLoading data from: {selected_csv}")

# --- Load CSV ---
df = pd.read_csv(selected_csv)

# Parse Time column to datetime (dummy date to enable grouping)
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

# Aggregate into 1-minute bins (change 'T' to '5T' for 5-minute bins)
df['TimeBin'] = df['Time'].dt.floor('T')

# Function to interpret states
def interpret_state(eye):
    if eye == 'CENTER':
        return 'Focus'
    elif eye == 'Blinking':
        return 'Distraction'
    else:
        return 'Stress'

df['Interpreted State'] = df['Eye Detection'].apply(interpret_state)

# --- Raw Eye Detection proportions over time ---
eye_counts = df.groupby(['TimeBin', 'Eye Detection']).size().unstack(fill_value=0)
eye_props = eye_counts.divide(eye_counts.sum(axis=1), axis=0)

plt.figure(figsize=(12, 5))
eye_props.plot.area(colormap='Pastel1', alpha=0.8, ax=plt.gca())
plt.title('Eye Detection State Proportions Over Time (1-minute bins)')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend(title='Eye Detection')
plt.tight_layout()
plt.show()

# --- Interpreted State proportions over time ---
state_counts = df.groupby(['TimeBin', 'Interpreted State']).size().unstack(fill_value=0)
state_props = state_counts.divide(state_counts.sum(axis=1), axis=0)

plt.figure(figsize=(12, 5))
state_props.plot.area(colormap='Set2', alpha=0.8, ax=plt.gca())
plt.title('Interpreted State Proportions Over Time (1-minute bins)')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend(title='State')
plt.tight_layout()
plt.show()

# --- Pie and Bar charts for totals (unchanged) ---

# Raw Eye Detection totals
total_eye_counts = df['Eye Detection'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    total_eye_counts,
    labels=total_eye_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Pastel1.colors
)
plt.title('Overall Eye Detection Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
total_eye_counts.plot(kind='bar', color='skyblue')
plt.title('Total Eye Detection Counts')
plt.xlabel('Eye Detection State')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Interpreted State totals
total_state_counts = df['Interpreted State'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    total_state_counts,
    labels=total_state_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Set2.colors
)
plt.title('Overall Interpreted State Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
total_state_counts.plot(kind='bar', color='lightcoral')
plt.title('Total Interpreted State Counts')
plt.xlabel('State')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
