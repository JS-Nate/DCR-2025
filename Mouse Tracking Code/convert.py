import os
import csv
from datetime import datetime

# List all CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if not csv_files:
    print("No CSV files found in the current directory.")
    exit()

print("CSV files found:")
for idx, filename in enumerate(csv_files, 1):
    print(f"{idx}: {filename}")

# Ask user to pick a file
while True:
    choice = input(f"Enter the number of the CSV file to process (1-{len(csv_files)}): ")
    if choice.isdigit():
        choice_idx = int(choice)
        if 1 <= choice_idx <= len(csv_files):
            input_file = csv_files[choice_idx - 1]
            break
    print("Invalid input, please try again.")

output_file = f"converted_{input_file}"

with open(input_file, newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
    reader = csv.DictReader(csvfile_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in reader:
        ts_float = float(row['Timestamp'])
        dt = datetime.utcfromtimestamp(ts_float)
        row['Timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # milliseconds precision
        writer.writerow(row)

print(f"Converted timestamps saved to {output_file}")
