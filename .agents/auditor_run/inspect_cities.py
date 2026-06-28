import pandas as pd

df = pd.read_csv('data/raw/training_locations.csv')
new_locs = df[df['location_id'] > 1001]
print("New locations cities and counts:")
print(new_locs['city'].value_counts())
