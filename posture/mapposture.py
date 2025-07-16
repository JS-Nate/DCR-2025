import pandas as pd
import matplotlib.pyplot as plt

col_names = ['Index', 'Val1', 'Val2', 'Val3', 'Val4', 'Val5', 'Val6', 'Val7', 'Val8', 'Posture']
df = pd.read_csv('C:/Users/natt4/Documents/GitHub/Summer2025/posture/posture_readings.csv', names=col_names, header=None)

# Remove any accidental header rows (case insensitive)
df = df[~df['Posture'].str.lower().eq('posture')]

# Rename "Bad - Slouching" to "Slouching"
df['Posture'] = df['Posture'].replace({'Bad - Slouching': 'Slouching'})

# Get counts and scale down (e.g., divide by 100)
posture_counts = df['Posture'].value_counts() // 100

ax = posture_counts.plot(kind='bar', color='lightgreen')
plt.title('Posture Frequency')
plt.xlabel('Posture')
plt.ylabel('Count (x100)')
plt.xticks(rotation=0)  # Make x-axis labels horizontal

plt.show()