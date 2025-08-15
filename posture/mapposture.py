import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# --- Load CSV file ---
csv_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV files found in the folder.")

print("Available posture CSV files:")
for i, file in enumerate(csv_files):
    print(f"{i + 1}. {file}")

choice = int(input("Enter the number of the CSV file to use: ")) - 1
if choice < 0 or choice >= len(csv_files):
    raise ValueError("Invalid file selection.")

selected_csv = os.path.join(csv_folder, csv_files[choice])
print(f"\nLoading data from: {selected_csv}")

df = pd.read_csv(selected_csv)

# --- Standardize posture labels ---
df['posture'] = df['posture'].replace({
    'Bad - Slouching': 'Slouching',
    'Bad - Forward Head': 'Forward Head'
})

# --- Posture plots (existing) ---
posture_counts = df['posture'].value_counts()
posture_counts_scaled = posture_counts // 100

plt.figure(figsize=(8, 5))
posture_counts_scaled.plot(kind='bar', color='lightgreen')
plt.title('Posture Frequency (Bar Chart)')
plt.xlabel('Posture')
plt.ylabel('Count (x100)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

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

# --- Calculate displacements for shoulders ---
for side in ['left_shoulder', 'right_shoulder']:
    df[f'dx_{side}'] = df[f'{side}_x'].diff()
    df[f'dy_{side}'] = df[f'{side}_y'].diff()
    df[f'displacement_{side}'] = np.sqrt(df[f'dx_{side}']**2 + df[f'dy_{side}']**2)

# 1. Histogram of movement magnitudes
plt.figure(figsize=(8,5))
plt.hist(df['displacement_left_shoulder'].dropna(), bins=30, alpha=0.6, label='Left Shoulder')
plt.hist(df['displacement_right_shoulder'].dropna(), bins=30, alpha=0.6, label='Right Shoulder')
plt.title('Distribution of Movement Magnitudes')
plt.xlabel('Displacement')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# KDE plot with seaborn
plt.figure(figsize=(8,5))
sns.kdeplot(df['displacement_left_shoulder'].dropna(), shade=True, label='Left Shoulder')
sns.kdeplot(df['displacement_right_shoulder'].dropna(), shade=True, label='Right Shoulder')
plt.title('Movement Magnitude Density')
plt.xlabel('Displacement')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Boxplots of displacement
movement_data = pd.DataFrame({
    'Left Shoulder': df['displacement_left_shoulder'],
    'Right Shoulder': df['displacement_right_shoulder']
})

plt.figure(figsize=(6,5))
sns.boxplot(data=movement_data)
plt.title('Movement Magnitude Summary')
plt.ylabel('Displacement')
plt.tight_layout()
plt.show()

# 3. Heatmap of left shoulder movement frequency
plt.figure(figsize=(6,6))
plt.hist2d(df['left_shoulder_x'], df['left_shoulder_y'], bins=50, cmap='viridis')
plt.colorbar(label='Frequency')
plt.title('Left Shoulder Movement Heatmap')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.tight_layout()
plt.show()

# 4. Cumulative movement over time (left shoulder as example)
df['cumulative_disp_left'] = df['displacement_left_shoulder'].cumsum()

plt.figure(figsize=(10,4))
plt.plot(df['frame'], df['cumulative_disp_left'], color='purple')
plt.title('Cumulative Left Shoulder Movement Over Time')
plt.xlabel('Frame')
plt.ylabel('Cumulative Displacement')
plt.tight_layout()
plt.show()

# 5. Movement status pie chart
movement_counts = df['movement_status'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(movement_counts, labels=movement_counts.index, autopct='%1.1f%%', colors=plt.cm.Pastel2.colors)
plt.title('Movement Status Distribution')
plt.tight_layout()
plt.show()
