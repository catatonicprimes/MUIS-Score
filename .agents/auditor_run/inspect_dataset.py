import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('data'))
import pandas as pd
import numpy as np

features_path = 'data/processed/features.csv'
df = pd.read_csv(features_path)

print(f"Total rows in features.csv: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Unique location_ids: {df['location_id'].nunique()}")

baseline_df = df[df['location_id'] <= 1001]
expanded_df = df[df['location_id'] > 1001]

print(f"Baseline rows (location_id <= 1001): {len(baseline_df)}")
print(f"Expanded rows (location_id > 1001): {len(expanded_df)}")

# Check for identical or nearly identical rows
# Since the expand_dataset.py code adds np.random.normal(0, 0.01 * std) to base features,
# let's check if the values of the features in expanded_df are derived from baseline_df.
# For example, we can check how many rows in expanded_df are very close (e.g., within 5% relative difference)
# to some row in baseline_df for all features.

print("\nChecking for synthetic data copying...")
match_count = 0
from data.feature_config import FEATURE_COLS

for idx, exp_row in expanded_df.iterrows():
    # Find base rows in the same city
    city = exp_row['city']
    city_base = baseline_df[baseline_df['city'] == city]
    if len(city_base) == 0:
        continue
    
    # Calculate distance in lat/lon to identify the copied neighbor
    dists = np.sqrt((city_base['lat'] - exp_row['lat'])**2 + (city_base['lon'] - exp_row['lon'])**2)
    nearest_idx = dists.idxmin()
    nearest_row = city_base.loc[nearest_idx]
    
    # Check if features are copied with small noise
    differences = []
    for col in FEATURE_COLS:
        val_exp = exp_row[col]
        val_base = nearest_row[col]
        
        # Calculate difference relative to standard deviation
        std = baseline_df[col].std()
        if pd.isna(std) or std == 0:
            std = 1.0
            
        diff = abs(val_exp - val_base)
        differences.append(diff)
        
    max_diff_std = max(differences)
    # The noise added is normal(0, 0.01 * std), so the maximum difference should be within a few standard deviations of 0.01*std.
    # Typically, with 27 features, max difference would be around 3 * 0.01 * std = 0.03 * std.
    # Let's count how many expanded rows have all feature differences < 0.05 * std.
    all_within_limit = True
    for col, diff in zip(FEATURE_COLS, differences):
        std = baseline_df[col].std()
        if pd.isna(std) or std == 0:
            std = 1.0
        # If it's market_cluster or civic_presence, they are integers, so they must be exactly equal
        if col in ['F10_market_cluster', 'F17_civic_presence']:
            if diff != 0:
                all_within_limit = False
                break
        else:
            if diff > 0.05 * std:
                all_within_limit = False
                break
                
    if all_within_limit:
        match_count += 1
    else:
        # Check if the city has any base rows
        if len(city_base) == 0:
            print(f"Row {exp_row['location_id']}: City '{city}' not in baseline dataset.")
        else:
            print(f"Row {exp_row['location_id']}: City '{city}' exists, but features did not match nearest neighbor. Max diff relative to std: {max_diff_std:.4f}")

print(f"Total expanded rows checked: {len(expanded_df)}")
print(f"Expanded rows matching baseline features with <= 0.05*std noise: {match_count} ({match_count/len(expanded_df)*100:.2f}%)")
