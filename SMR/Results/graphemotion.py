import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# List CSV files in that directory
csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]

if not csv_files:
    print("No CSV files found in the script directory.")
    exit()

# Show CSVs to user
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

# Load the selected CSV
selected_file = os.path.join(script_dir, csv_files[choice])
df = pd.read_csv(selected_file)

# Ensure the column exists
if 'Emotion Detection' not in df.columns:
    print("Column 'Emotion Detection' not found in the selected file.")
    exit()

# Bar Chart of Emotion Frequencies
emotion_counts = df['Emotion Detection'].value_counts()
plt.figure(figsize=(10, 5))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title('Emotion Frequency')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.show()







# Pie Chart of Emotion Distribution (Hide <1%)
plt.figure(figsize=(8, 8))

# Define a custom autopct function to hide small slices (<1%)
def autopct_func(pct):
    return f'{pct:.1f}%' if pct >= 1 else ''

# Plot pie chart without inline labels, legend on the side
wedges, texts, autotexts = plt.pie(
    emotion_counts,
    labels=None,  # Emotion labels shown in legend instead
    autopct=autopct_func,
    startangle=90,
    colors=plt.cm.Pastel1.colors,
    textprops={'fontsize': 10}
)

# Add a side legend for emotion names
plt.legend(wedges, emotion_counts.index, title="Emotions", loc="center left", bbox_to_anchor=(1, 0.5))

plt.title('Emotion Distribution')
plt.tight_layout()
plt.show()








import matplotlib.ticker as ticker

# Line Graph: Emotion Progression Over Time (Final Clean Version)
plt.figure(figsize=(14, 6))

# Convert emotion labels to category codes
emotions = df['Emotion Detection'].astype('category')
codes = emotions.cat.codes
labels = emotions.cat.categories

# Downsample if too many points (e.g., show only every 10th emotion)
max_points = 300  # Adjust for clarity
if len(codes) > max_points:
    step = len(codes) // max_points
    codes = codes[::step]
    index_range = range(0, len(df), step)
else:
    index_range = range(len(df))

# Plot with steps to highlight transitions
plt.step(index_range, codes, where='post', color='slateblue')

# Map emotion codes to labels
plt.yticks(ticks=range(len(labels)), labels=labels)

# Reduce number of x-axis ticks
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

plt.title('Progression of Emotions Over Time')
plt.xlabel('Time Index')
plt.ylabel('Emotion')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
