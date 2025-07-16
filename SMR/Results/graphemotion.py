import pandas as pd
import matplotlib.pyplot as plt

# Use the correct relative path from your script location
df = pd.read_csv('c:/Users/natt4/Documents/GitHub/Summer2025/SMR/Results/emotion_detection-finalll-2025-07-15_13-25-45.csv')

# Count occurrences of each emotion
emotion_counts = df['Emotion Detection'].value_counts()

# Plot bar chart
emotion_counts.plot(kind='bar', color='skyblue')
plt.title('Emotion Frequency')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.show()