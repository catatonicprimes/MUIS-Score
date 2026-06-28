import os
import pandas as pd
import numpy as np

features_path = 'data/processed/features.csv'
if not os.path.exists(features_path):
    print(f"Error: {features_path} does not exist!")
    sys.exit(1)

df = pd.read_csv(features_path)
print(f"Total rows in features.csv: {len(df)}")

baseline_df = df[df['location_id'] <= 1001]
expanded_df = df[df['location_id'] > 1001]

print(f"Baseline rows (location_id <= 1001): {len(baseline_df)}")
print(f"Expanded rows (location_id > 1001): {len(expanded_df)}")

# Import FEATURE_COLS
from data.feature_config import FEATURE_COLS

match_count = 0
for idx, exp_row in expanded_df.iterrows():
    city = exp_row['city']
    city_base = baseline_df[baseline_df['city'] == city]
    if len(city_base) == 0:
        city_base = baseline_df
        
    dists = np.sqrt((city_base['lat'] - exp_row['lat'])**2 + (city_base['lon'] - exp_row['lon'])**2)
    nearest_idx = dists.idxmin()
    nearest_row = city_base.loc[nearest_idx]
    
    differences = []
    for col in FEATURE_COLS:
        val_exp = exp_row[col]
        val_base = nearest_row[col]
        
        std = baseline_df[col].std()
        if pd.isna(std) or std == 0:
            std = 1.0
            
        diff = abs(val_exp - val_base)
        differences.append(diff)
        
    all_within_limit = True
    for col, diff in zip(FEATURE_COLS, differences):
        std = baseline_df[col].std()
        if pd.isna(std) or std == 0:
            std = 1.0
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

print(f"Total expanded rows checked: {len(expanded_df)}")
print(f"Expanded rows matching baseline features with <= 0.05*std noise: {match_count} ({match_count/max(len(expanded_df), 1)*100:.2f}%)")

# Let's also check X_train and X_test shape in model/
x_train_path = 'model/X_train.npy'
x_test_path = 'model/X_test.npy'
if os.path.exists(x_train_path) and os.path.exists(x_test_path):
    x_train = np.load(x_train_path)
    x_test = np.load(x_test_path)
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"X_train.shape[0] + X_test.shape[0] = {x_train.shape[0] + x_test.shape[0]}")
else:
    print("Model numpy files not found!")
