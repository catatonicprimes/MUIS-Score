import os
import sys
import time
import pandas as pd
import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Ignore deprecation warnings
warnings.filterwarnings('ignore')

# Single source of truth paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_data import fetch_osm_features, fetch_google_places, fetch_network_metrics, engineer_features
from feature_config import FEATURES_CSV, TRAINING_LOCATIONS_CSV

# Lock for appending results and writing to files
write_lock = threading.Lock()



def process_single_location(row):
    loc_id = int(row['location_id'])
    city = row['city']
    neighbourhood = row['neighbourhood']
    country = row['country']
    expected_class = row['expected_class']
    lat = float(row['lat'])
    lon = float(row['lon'])
    
    # Offline cache disabled to ensure genuine crawl
    
    # Pause between processing locations to prevent rate limiting
    time.sleep(0.5)
    
    try:
        t_start = time.time()
        
        features = None
        osm_len = 0
        google_len = 0
        
        try:
            # Step 1: Fetch OSM features
            osm_elements = fetch_osm_features(lat, lon)
            
            # Step 2: Fetch Google Places
            google_places = fetch_google_places(lat, lon)
            
            # Step 3: Fetch network metrics
            network_metrics = fetch_network_metrics(lat, lon)
            
            # Step 4: Engineer features
            features = engineer_features(osm_elements, google_places, network_metrics, lat, lon)
            osm_len = len(osm_elements)
            google_len = len(google_places)
            
        except Exception as api_err:
            print(f"[{loc_id}] WARNING: Genuine API fetch failed ({api_err}).", flush=True)
            raise api_err
        
        # Build record
        record = {
            'location_id': loc_id,
            'city': city,
            'neighbourhood': neighbourhood,
            'country': country,
            'lat': lat,
            'lon': lon,
            'expected_class': expected_class,
            'osm_element_count': osm_len,
            **features,
        }
        
        elapsed = time.time() - t_start
        print(f"[{loc_id}] Done: {neighbourhood}, {city} in {elapsed:.3f}s | OSM: {osm_len} | Google: {google_len}", flush=True)
        return record
        
    except Exception as e:
        print(f"[{loc_id}] ERROR processing: {e}", flush=True)
        return None

def main():
    if not os.path.exists(TRAINING_LOCATIONS_CSV):
        print(f"Error: {TRAINING_LOCATIONS_CSV} not found.")
        return
        
    locs_df = pd.read_csv(TRAINING_LOCATIONS_CSV)
    print(f"Loaded {len(locs_df)} total locations from training_locations.csv")
    
    # Identify which locations are new (location_id > 1001)
    new_locs = locs_df[locs_df['location_id'] > 1001]
    print(f"New locations to process: {len(new_locs)}")
    
    # Check what is already done in features.csv
    already_done = set()
    if os.path.exists(FEATURES_CSV):
        try:
            feat_df = pd.read_csv(FEATURES_CSV)
            already_done = set(feat_df['location_id'].values)
            print(f"Found {len(already_done)} locations already in {FEATURES_CSV}")
        except Exception as e:
            print(f"Warning reading features.csv: {e}")
            
    # Filter out locations that are already processed
    to_process = new_locs[~new_locs['location_id'].isin(already_done)]
    
    # Limit to the exact number of new locations needed to reach 2010 rows in features.csv
    target_total = 2010
    needed = target_total - len(already_done)
    if needed <= 0:
        print(f"Already have {len(already_done)} rows in features.csv, which is >= target of {target_total}.")
        return
        
    print(f"Locations left to crawl overall: {len(to_process)}")
    to_process = to_process.head(needed)
    print(f"Limiting this run to fetch {len(to_process)} locations to reach total of {target_total} rows.")
    
    # Run a ProcessPoolExecutor
    max_workers = 8
    print(f"Starting parallel crawler with {max_workers} worker processes...")
    
    records = []
    checkpoint_interval = 10
    
    # Start process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_location, row): row for _, row in to_process.iterrows()}
        
        for future in as_completed(futures):
            record = future.result()
            if record is not None:
                with write_lock:
                    records.append(record)
                    
                    # Save checkpoint every checkpoint_interval
                    if len(records) % checkpoint_interval == 0:
                        temp_df = pd.DataFrame(records)
                        if os.path.exists(FEATURES_CSV):
                            try:
                                existing_df = pd.read_csv(FEATURES_CSV)
                                combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset='location_id', keep='last')
                                combined_df.to_csv(FEATURES_CSV, index=False)
                                # Clear buffer records so we don't duplicate on next checkpoint
                                records.clear()
                                print(f"  [Checkpoint] Checkpoint saved: total {len(combined_df)} locations in {FEATURES_CSV}", flush=True)
                            except Exception as e:
                                print(f"  Warning: Checkpoint save failed: {e}", flush=True)
                                
    # Write any remaining records
    if records:
        with write_lock:
            temp_df = pd.DataFrame(records)
            if os.path.exists(FEATURES_CSV):
                try:
                    existing_df = pd.read_csv(FEATURES_CSV)
                    combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset='location_id', keep='last')
                    combined_df.to_csv(FEATURES_CSV, index=False)
                    print(f"  [Save] Final results saved: total {len(combined_df)} locations in {FEATURES_CSV}", flush=True)
                except Exception as e:
                    print(f"  Warning: Final save failed: {e}", flush=True)
            else:
                temp_df.to_csv(FEATURES_CSV, index=False)
                print(f"  [Save] Final results saved to new features.csv: {len(temp_df)} locations", flush=True)
                
    print("Crawler run complete.")

if __name__ == '__main__':
    main()
