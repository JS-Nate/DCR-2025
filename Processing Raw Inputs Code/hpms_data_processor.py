import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class HPMSDataProcessor:
    def __init__(self):
        self.posture_mapping = {
            'Bad - Slouching': 'slouching',
            'Good': 'upright',
            'Forward Head': 'leaning_forward',
            'Leaning': 'leaning_forward'
        }
        
        self.eye_mapping = {
            'CENTER': 'focused',
            'LEFT': 'distracted',
            'RIGHT': 'distracted', 
            'UP': 'distracted',
            'DOWN': 'distracted',
            'BLINKING': 'blinking_frequently'
        }
        
        self.voice_mapping = {
            'Normal': 'normal',
            'Raised': 'raised',
            'Strained': 'strained',
            'Whisper': 'whisper'
        }
        
        self.emotion_mapping = {
            'neutral': 'neutral',
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'surprised': 'surprised',
            'disgust': 'disgust'
        }
        
        self.stress_mapping = {
            'non stress': 'low',
            'low stress': 'low', 
            'medium stress': 'medium',
            'high stress': 'high'
        }
        
        self.task_mapping = {
            'Began Step 1': 'monitoring',
            'Step 1': 'monitoring',
            'Step 2': 'adjusting_controls',
            'Step 3': 'reporting',
            'Emergency': 'emergency_shutdown',
            'Alert': 'emergency_shutdown'
        }

    def clean_column_names(self, df):
        """Clean and standardize column names"""
        # Create a mapping for known problematic columns
        column_mapping = {
            'Heart Rate (bpm)': 'heart_rate',
            'temperature_C': 'room_temp',
            'skin_temperature_C': 'skin_temp',
            'humidity_%': 'humidity',
            'pressure_hPa': 'pressure',
            'light_lux': 'light_intensity',
            'CCT_K': 'cct_temp',
            'Eye Detection': 'eye_tracking',
            'Emotion Detection': 'face_emotion',
            'movement_status': 'face_stress_raw',
            'Voice State': 'voice',
            'Task Timestamp': 'task_raw'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        return df

    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime object"""
        try:
            return pd.to_datetime(timestamp_str)
        except:
            return pd.NaT

    def interpret_eye_tracking(self, eye_data):
        """Interpret eye tracking data over a 5-second window"""
        if eye_data.empty:
            return 'unknown'
        
        # Count occurrences of each eye state
        eye_counts = eye_data.value_counts()
        
        # If blinking is frequent (more than 2 times in 5 seconds)
        if 'BLINKING' in eye_counts and eye_counts['BLINKING'] > 2:
            return 'blinking_frequently'
        
        # If mostly centered, considered focused
        if 'CENTER' in eye_counts and eye_counts['CENTER'] / len(eye_data) > 0.6:
            return 'focused'
        
        # Otherwise, considered distracted
        return 'distracted'

    def interpret_posture(self, posture_data):
        """Interpret posture data over a 5-second window"""
        if posture_data.empty:
            return 'unknown'
        
        # Get the most common posture
        most_common = posture_data.mode().iloc[0] if not posture_data.mode().empty else 'unknown'
        return self.posture_mapping.get(most_common, 'upright')

    def interpret_voice(self, voice_data, clarity_data=None):
        """Interpret voice characteristics"""
        if voice_data.empty:
            return 'normal'
        
        # Check for voice state indicators
        voice_states = voice_data.dropna().unique()
        
        if any('strained' in str(v).lower() for v in voice_states):
            return 'strained'
        elif any('raised' in str(v).lower() for v in voice_states):
            return 'raised'
        elif clarity_data is not None and not clarity_data.empty:
            # Check clarity indicators
            clarity_issues = clarity_data.str.contains('hesitation|fatigue', case=False, na=False).sum()
            if clarity_issues > len(clarity_data) / 2:
                return 'strained'
        
        return 'normal'

    def interpret_emotion(self, emotion_data):
        """Interpret facial emotion over a 5-second window"""
        if emotion_data.empty:
            return 'neutral'
        
        # Get the most frequent emotion
        emotion_counts = emotion_data.value_counts()
        most_common = emotion_counts.index[0] if not emotion_counts.empty else 'neutral'
        return self.emotion_mapping.get(most_common, 'neutral')

    def interpret_stress(self, stress_data, heart_rate=None):
        """Interpret stress level"""
        if not stress_data.empty:
            # Use the most common stress level from movement_status
            stress_counts = stress_data.value_counts()
            most_common = stress_counts.index[0] if not stress_counts.empty else 'low stress'
            return self.stress_mapping.get(most_common, 'low')
        
        # Fallback to heart rate if available
        if heart_rate is not None and not pd.isna(heart_rate):
            if heart_rate > 110:
                return 'high'
            elif heart_rate > 90:
                return 'medium'
            else:
                return 'low'
        
        return 'low'

    def interpret_task(self, task_data):
        """Interpret current task from task timestamp data"""
        if task_data.empty:
            return 'monitoring', 5
        
        # Look for task indicators in the data
        task_str = ' '.join(task_data.dropna().astype(str))
        
        if 'Step 1' in task_str or 'Began Step 1' in task_str:
            return 'monitoring', 5
        elif 'Step 2' in task_str:
            return 'adjusting_controls', 3
        elif 'Step 3' in task_str:
            return 'reporting', 7
        elif 'Emergency' in task_str or 'Alert' in task_str:
            return 'emergency_shutdown', 2
        else:
            return 'monitoring', 5

    def process_raw_data(self, df):
        """Process raw data into 5-second intervals - optimized for large datasets"""
        print(f"Processing {len(df)} rows of raw data...")
        
        # Clean column names
        df = self.clean_column_names(df)
        
        # Parse timestamps more efficiently
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"After cleaning: {len(df)} valid rows")
        
        # Create 5-second grouping intervals (fixed deprecated 'S' to 's')
        df['group_time'] = df['timestamp'].dt.floor('5s')
        
        # Pre-check which columns exist to avoid repeated checks
        col_exists = {
            'heart_rate': 'heart_rate' in df.columns,
            'skin_temp': 'skin_temp' in df.columns,
            'room_temp': 'room_temp' in df.columns,
            'humidity': 'humidity' in df.columns,
            'pressure': 'pressure' in df.columns,
            'light_intensity': 'light_intensity' in df.columns,
            'cct_temp': 'cct_temp' in df.columns,
            'posture': 'posture' in df.columns,
            'eye_tracking': 'eye_tracking' in df.columns,
            'voice': 'voice' in df.columns,
            'face_emotion': 'face_emotion' in df.columns,
            'face_stress_raw': 'face_stress_raw' in df.columns,
            'task_raw': 'task_raw' in df.columns,
            'clarity': 'Clarity' in df.columns
        }
        
        # Use vectorized operations where possible - fixed aggregation
        numeric_cols = []
        if col_exists['heart_rate']:
            numeric_cols.append('heart_rate')
        if col_exists['skin_temp']:
            numeric_cols.append('skin_temp')
        if col_exists['room_temp']:
            numeric_cols.append('room_temp')
        if col_exists['humidity']:
            numeric_cols.append('humidity')
        if col_exists['pressure']:
            numeric_cols.append('pressure')
        if col_exists['light_intensity']:
            numeric_cols.append('light_intensity')
        if col_exists['cct_temp']:
            numeric_cols.append('cct_temp')
        
        # Aggregate numeric columns efficiently
        if numeric_cols:
            numeric_results = df.groupby('group_time')[numeric_cols].mean().round(2)
        else:
            numeric_results = pd.DataFrame(index=df['group_time'].unique())
        
        processed_data = []
        group_count = 0
        total_groups = df['group_time'].nunique()
        
        for group_time, group in df.groupby('group_time'):
            group_count += 1
            if group_count % 100 == 0:
                print(f"Processing group {group_count}/{total_groups}")
            
            # Get pre-calculated numeric values
            if group_time in numeric_results.index:
                numeric_row = numeric_results.loc[group_time]
                avg_heart_rate = numeric_row.get('heart_rate', 97)
                avg_skin_temp = numeric_row.get('skin_temp', 34.0)
                avg_room_temp = numeric_row.get('room_temp', 24.0)
                avg_humidity = numeric_row.get('humidity', 45)
                avg_pressure = numeric_row.get('pressure', 995)
                avg_light = numeric_row.get('light_intensity', 500)
                avg_cct = numeric_row.get('cct_temp', 5000)
            else:
                avg_heart_rate, avg_skin_temp, avg_room_temp = 97, 34.0, 24.0
                avg_humidity, avg_pressure, avg_light, avg_cct = 45, 995, 500, 5000
            
            # Interpret behavioral data
            posture = self.interpret_posture(group['posture'] if col_exists['posture'] else pd.Series())
            eye_tracking = self.interpret_eye_tracking(group['eye_tracking'] if col_exists['eye_tracking'] else pd.Series())
            voice = self.interpret_voice(
                group['voice'] if col_exists['voice'] else pd.Series(),
                group['Clarity'] if col_exists['clarity'] else None
            )
            face_emotion = self.interpret_emotion(group['face_emotion'] if col_exists['face_emotion'] else pd.Series())
            face_stress = self.interpret_stress(
                group['face_stress_raw'] if col_exists['face_stress_raw'] else pd.Series(),
                avg_heart_rate
            )
            
            # Interpret task
            task, duration = self.interpret_task(group['task_raw'] if col_exists['task_raw'] else pd.Series())
            
            processed_row = {
                'timestamp': group_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'heart_rate': int(round(avg_heart_rate)) if not pd.isna(avg_heart_rate) else 85,
                'skin_temp': round(avg_skin_temp, 1) if not pd.isna(avg_skin_temp) else 34.0,
                'posture': posture,
                'eye_tracking': eye_tracking,
                'voice': voice,
                'face_emotion': face_emotion,
                'face_stress': face_stress,
                'task': task,
                'task_duration': duration,
                'room_temp': round(avg_room_temp, 1) if not pd.isna(avg_room_temp) else 24.0,
                'cct_temp': int(round(avg_cct)) if not pd.isna(avg_cct) else 5000,
                'light_intensity': int(round(avg_light)) if not pd.isna(avg_light) else 500,
                'humidity': int(round(avg_humidity)) if not pd.isna(avg_humidity) else 45,
                'pressure': round(avg_pressure, 2) if not pd.isna(avg_pressure) else 995.0
            }
            
            processed_data.append(processed_row)
        
        print(f"Processing complete! Generated {len(processed_data)} 5-second intervals")
        return pd.DataFrame(processed_data)

    def save_processed_data(self, processed_df, output_filename):
        """Save processed data to CSV"""
        processed_df.to_csv(output_filename, index=False)
        print(f"Processed data saved to {output_filename}")

# Usage example
def main():
    # Initialize processor
    processor = HPMSDataProcessor()
    
    # Load raw data with optimized reading
    print("Loading raw data...")
    try:
        # Read in chunks for very large files if needed
        raw_df = pd.read_csv('raw.csv', low_memory=False)
    except MemoryError:
        print("File too large for memory, reading in chunks...")
        chunk_list = []
        for chunk in pd.read_csv('raw.csv', chunksize=5000):
            chunk_list.append(chunk)
        raw_df = pd.concat(chunk_list, ignore_index=True)
        del chunk_list  # Free memory
    
    print(f"Raw data shape: {raw_df.shape}")
    print(f"Raw data date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
    
    # Estimate memory usage
    memory_usage = raw_df.memory_usage(deep=True).sum() / 1024**2
    print(f"Raw data memory usage: {memory_usage:.2f} MB")
    
    # Process the data
    print("Processing data into 5-second intervals...")
    processed_df = processor.process_raw_data(raw_df)
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Compression ratio: {len(raw_df)/len(processed_df):.1f}:1")
    print("\nFirst few rows of processed data:")
    print(processed_df.head())
    
    # Save processed data
    processor.save_processed_data(processed_df, 'hpms_processed_output.csv')
    
    return processed_df

if __name__ == "__main__":
    processed_data = main()