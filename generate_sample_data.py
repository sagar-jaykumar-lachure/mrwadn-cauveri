"""
Generate sample flood data for testing and demonstration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps
start_date = datetime(2020, 1, 1)
num_samples = 5000
timestamps = [start_date + timedelta(hours=i) for i in range(num_samples)]

# Generate features
data = {
    'timestamp': timestamps,
    
    # Weather features
    'rainfall': np.random.exponential(5, num_samples),  # mm/hour
    'temperature': 25 + 10 * np.sin(np.arange(num_samples) * 2 * np.pi / 8760) + np.random.normal(0, 2, num_samples),
    'humidity': np.clip(60 + np.random.normal(0, 10, num_samples), 0, 100),
    'wind_speed': np.abs(np.random.normal(10, 5, num_samples)),
    'pressure': 1013 + np.random.normal(0, 5, num_samples),
    
    # River features
    'water_level': np.zeros(num_samples),
    'flow_rate': np.zeros(num_samples),
    'upstream_level': np.zeros(num_samples),
}

# Generate water level based on rainfall with some lag
for i in range(num_samples):
    if i < 10:
        rainfall_sum = np.sum(data['rainfall'][:i+1])
    else:
        rainfall_sum = np.sum(data['rainfall'][i-10:i+1])
    
    # Base water level
    base_level = 2.0
    
    # Add rainfall contribution
    rainfall_contribution = rainfall_sum * 0.05
    
    # Add seasonal component
    seasonal = 0.5 * np.sin(i * 2 * np.pi / 8760)
    
    # Add noise
    noise = np.random.normal(0, 0.1)
    
    data['water_level'][i] = base_level + rainfall_contribution + seasonal + noise
    
    # Flow rate correlates with water level
    data['flow_rate'][i] = data['water_level'][i] * 50 + np.random.normal(0, 10)
    
    # Upstream level is slightly higher
    data['upstream_level'][i] = data['water_level'][i] + np.random.uniform(0.1, 0.5)

# Generate flood events based on conditions
flood_threshold = 4.0
data['flood_event'] = (data['water_level'] > flood_threshold).astype(int)

# Add some realistic complexity: floods also depend on rate of change
for i in range(10, num_samples):
    water_level_change = data['water_level'][i] - data['water_level'][i-5]
    if water_level_change > 1.0 and data['water_level'][i] > 3.5:
        data['flood_event'][i] = 1

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values (realistic scenario)
missing_indices = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
for idx in missing_indices:
    col = np.random.choice(['rainfall', 'temperature', 'humidity', 'wind_speed'])
    df.loc[idx, col] = np.nan

# Save the data
df.to_csv('data/sample_data.csv', index=False)

print(f"Sample data generated: {len(df)} samples")
print(f"Flood events: {df['flood_event'].sum()} ({df['flood_event'].sum()/len(df)*100:.2f}%)")
print(f"No flood events: {(1-df['flood_event']).sum()} ({(1-df['flood_event']).sum()/len(df)*100:.2f}%)")
print("\nData saved to: data/sample_data.csv")
print("\nSample data:")
print(df.head(10))
print("\nData statistics:")
print(df.describe())
