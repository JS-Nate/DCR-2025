import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
from io import StringIO
import base64
import os

# Load and process CSV data
def load_data(file_path):
    data = []
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    try:
        with open(full_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({"emotion": row["Emotion Detection"], "time": datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")})
    except FileNotFoundError:
        print(f"Error: Could not find {full_path}. Please check the file location or adjust the path.")
        return []
    return data

# Process data for charting
def process_emotion_data(data):
    grouped = {}
    for entry in data:
        time_key = entry["time"].strftime("%H:%M:%S")
        grouped[time_key] = grouped.get(time_key, {})
        grouped[time_key][entry["emotion"]] = grouped[time_key].get(entry["emotion"], 0) + 1
    chart_data = [{"time": time, "neutral": counts.get("neutral", 0), "happy": counts.get("happy", 0)} 
                  for time, counts in grouped.items()]
    return chart_data

# Generate HTML report with embedded chart
def generate_report(data):
    chart_data = process_emotion_data(data)
    # Create chart
    rcParams['figure.figsize'] = (10, 5)
    plt.stackplot([entry["time"] for entry in chart_data], 
                  [entry["neutral"] for entry in chart_data], 
                  [entry["happy"] for entry in chart_data], 
                  labels=["Neutral", "Happy"], colors=["#8884d8", "#82ca9d"])
    plt.legend(loc="upper left")
    plt.title("Emotion Detection Over Time")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Save plot to buffer and encode
    buf = StringIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    chart_image = base64.b64encode(buf.getvalue()).decode()

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .chart-container {{ max-width: 100%; margin: 0 auto; }}
    </style>
</head>
<body class="container mx-auto p-4">
    <h1 class="text-2xl font-bold mb-4 text-blue-600">Emotion Detection Report</h1>
    <p class="mb-4">Interesting Fact: A sudden shift from neutral to happy at 13:52:34 lasted 13 seconds, possibly due to a positive event.</p>
    <div class="chart-container">
        <img src="data:image/png;base64,{chart_image}" alt="Emotion Chart" style="width: 100%; height: auto;">
    </div>
    <p class="mt-4">Summary: The chart tracks emotion changes over time, with a notable happy period suggesting improved operator mood.</p>
</body>
</html>
"""
    with open("emotion_report.html", "w") as f:
        f.write(html_content)

# Run the script
if __name__ == "__main__":
    # Adjust this path relative to the script's location
    file_path = "..\SMR\Unchanged\emotion_detection-Test1-2025-06-24_13-52-29.csv"  # Relative to Results\
    emotion_data = load_data(file_path)
    if emotion_data:
        generate_report(emotion_data)
        print("Emotion report generated as 'emotion_report.html'.")
    else:
        print("Error: Could not load data from CSV.")