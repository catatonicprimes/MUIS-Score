import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/features.csv')
baseline_df = df[df['location_id'] <= 1001]
row_1002 = df[df['location_id'] == 1002].iloc[0]

ny_base = baseline_df[baseline_df['city'] == 'New York']
print(f"Number of baseline rows for New York: {len(ny_base)}")
print(f"Row 1002: {row_1002['neighbourhood']}, {row_1002['city']}, osm_element_count={row_1002['osm_element_count']}")
print("Row 1002 features:")
for col in ['F01_poi_density', 'F02_land_use_entropy', 'F03_retail_gf_ratio', 'F12_gamma_index']:
    print(f"  {col}: {row_1002[col]}")

if len(ny_base) > 0:
    dists = np.sqrt((ny_base['lat'] - row_1002['lat'])**2 + (ny_base['lon'] - row_1002['lon'])**2)
    nearest_idx = dists.idxmin()
    nearest_row = ny_base.loc[nearest_idx]
    print(f"Nearest baseline row in NY: {nearest_row['neighbourhood']}, dist={dists.min()}")
    print("Nearest row features:")
    for col in ['F01_poi_density', 'F02_land_use_entropy', 'F03_retail_gf_ratio', 'F12_gamma_index']:
        print(f"  {col}: {nearest_row[col]}")
else:
    print("No baseline rows for New York.")
