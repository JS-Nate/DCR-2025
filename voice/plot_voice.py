import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Prompt for CSV file in script directory ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]

if not csv_files:
    print("No CSV files found in script directory.")
    exit()

print("Select a CSV file:")
for idx, file in enumerate(csv_files):
    print(f"{idx + 1}: {file}")

try:
    choice = int(input("Enter the number of the file to use: ")) - 1
    if choice < 0 or choice >= len(csv_files):
        raise ValueError
except ValueError:
    print("Invalid selection.")
    exit()

selected_file = os.path.join(script_dir, csv_files[choice])

# --- Load CSV, skipping comment lines (e.g., lines starting with /) ---
df = pd.read_csv(selected_file, comment='/')

# --- Option 2: WPM Histogram ---
plt.figure(figsize=(8,4))
df['WPM'].dropna().plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('WPM (Words Per Minute) Distribution')
plt.xlabel('WPM')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# --- Option 5: Clarity Feedback Counts ---
plt.figure(figsize=(10,4))
df['Clarity'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Clarity Feedback Counts')
plt.xlabel('Clarity Feedback')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Option 6: Pitch (Hz) Over Time ---
plt.figure(figsize=(12,4))
df_sorted = df.sort_values('Timestamp')
plt.plot(pd.to_datetime(df_sorted['Timestamp']), df_sorted['Pitch (Hz)'], marker='o', linestyle='-', color='green')
plt.title('Pitch (Hz) Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Pitch (Hz)')
plt.tight_layout()
plt.show()

# --- Option 9: Voice State Distribution (Pie Chart) ---
plt.figure(figsize=(6,6))

# Hide percentages below 1%
def autopct_func(pct):
    return f'{pct:.1f}%' if pct >= 1 else ''

df['Voice State'].value_counts().plot(
    kind='pie',
    autopct=autopct_func,
    startangle=90,
    colors=['#66b3ff','#99ff99','#ffcc99'],
    textprops={'fontsize': 10}
)

plt.title('Voice State Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# --- Display detected spoken text and their timestamps (only when text is present) ---
spoken = df[['Text', 'Timestamp']].dropna(subset=['Text'])
spoken = spoken[spoken['Text'].astype(str).str.strip() != '']

if not spoken.empty:
    print("\nDetected Spoken Text Segments:")
    for _, row in spoken.iterrows():
        print(f"[{row['Timestamp']}] {row['Text']}")
