# Mock torch before any backend/model imports to bypass Windows DLL initialization failures (OSError: [WinError 1114])
import sys
from unittest.mock import MagicMock

class DummyTensor:
    pass

class DummyFloatTensor(DummyTensor):
    def __init__(self, *args, **kwargs):
        pass
    def item(self):
        return 5.5

class MockModule:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return DummyFloatTensor()
    def load_state_dict(self, *args, **kwargs):
        pass
    def eval(self, *args, **kwargs):
        pass

class MockNN:
    Module = MockModule
    Linear = lambda *a, **kw: MagicMock()
    ReLU = lambda *a, **kw: MagicMock()
    LeakyReLU = lambda *a, **kw: MagicMock()
    ELU = lambda *a, **kw: MagicMock()
    Dropout = lambda *a, **kw: MagicMock()
    Sequential = lambda *a, **kw: MagicMock()

# Create mock torch
mock_torch = MagicMock()
mock_torch.Tensor = DummyTensor
mock_torch.FloatTensor = DummyFloatTensor
mock_torch.nn = MockNN
mock_torch.no_grad = MagicMock()

# no_grad must act as a context manager
class MockNoGrad:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

mock_torch.no_grad.return_value = MockNoGrad()

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MockNN


# Now proceed with other imports
import os
import math
from collections import Counter
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure project directories are in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
data_dir = os.path.join(PROJECT_ROOT, 'data')
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)
backend_dir = os.path.join(PROJECT_ROOT, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from data.feature_config import FEATURE_COLS, NORMALIZATION_STRATEGY, LANDUSE_CATEGORIES
from data.collect_data import engineer_features, fetch_google_places
from data.prepare_training_data import fit_normalise_features, transform_features
from backend.app import app


# Area of analysis circle in km2
AREA_KM2 = math.pi * (800 / 1000) ** 2
SMALL_AREA_KM2 = math.pi * (200 / 1000) ** 2

# Helper to compute expected features to validate tests
def calculate_expected_feature(feature_name, osm_elements, google_places, network_metrics, lat=0.0, lon=0.0):
    amenity_counts = {}
    shop_count = 0
    office_count = 0
    craft_count = 0
    building_types = []
    landuse_types = []
    footway_count = 0
    transit_stops = 0

    for element in osm_elements:
        tags = element.get('tags', {}) if isinstance(element, dict) else {}
        if not tags:
            continue
        if 'amenity' in tags:
            amenity = tags['amenity']
            if amenity:
                amenity_counts[amenity] = amenity_counts.get(amenity, 0) + 1
        if 'shop' in tags:
            shop_count += 1
        if 'office' in tags:
            office_count += 1
        if 'craft' in tags:
            craft_count += 1
        if tags.get('building') and tags['building'] != 'nan':
            building_types.append(tags['building'])
        if tags.get('landuse') and tags['landuse'] != 'nan':
            landuse_types.append(tags['landuse'])
        if tags.get('highway') in ('footway', 'pedestrian'):
            footway_count += 1
        if (tags.get('highway') == 'bus_stop' or tags.get('railway') in ('station', 'subway_entrance')):
            transit_stops += 1

    google_restaurant_count = sum(1 for p in google_places if isinstance(p, dict) and 'types' in p and p['types'] and 'restaurant' in p['types'])
    google_cafe_count = sum(1 for p in google_places if isinstance(p, dict) and 'types' in p and p['types'] and 'cafe' in p['types'])
    google_store_count = sum(1 for p in google_places if isinstance(p, dict) and 'types' in p and p['types'] and 'store' in p['types'])
    google_health_count = sum(1 for p in google_places if isinstance(p, dict) and 'types' in p and p['types'] and ('hospital' in p['types'] or 'pharmacy' in p['types']))
    google_school_count = sum(1 for p in google_places if isinstance(p, dict) and 'types' in p and p['types'] and 'school' in p['types'])
    nightlife_count = sum(1 for p in google_places if isinstance(p, dict) and 'types' in p and p['types'] and ('night_club' in p['types'] or 'bar' in p['types']))

    total_google = max(len(google_places), 1)
    total_buildings = max(len(building_types), 1)

    residential_set = {'residential', 'house', 'apartments', 'terrace', 'dormitory', 'detached', 'semidetached_house'}
    residential_buildings = sum(1 for bt in building_types if bt in residential_set)
    total_poi = len(osm_elements) + len(google_places)

    if feature_name == 'F01_poi_density':
        return round(total_poi / AREA_KM2, 2)
    elif feature_name == 'F02_land_use_entropy':
        if landuse_types:
            lu_counts = Counter(landuse_types)
            lu_total = sum(lu_counts.values())
            lu_probs = [c / lu_total for c in lu_counts.values()]
            entropy = -sum(p * math.log2(p) for p in lu_probs if p > 0)
            max_entropy = math.log2(15)
            return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0
        return 0.0
    elif feature_name == 'F03_retail_gf_ratio':
        osm_restaurant_count = amenity_counts.get('restaurant', 0)
        osm_cafe_count = amenity_counts.get('cafe', 0)
        ground_floor_uses = shop_count + office_count + osm_restaurant_count + osm_cafe_count
        return round(min(ground_floor_uses / total_buildings, 1.0), 4)
    elif feature_name == 'F04_work_residential_ratio':
        work_uses = office_count + craft_count + shop_count
        _total = work_uses + residential_buildings + 1
        return round(1.0 - abs(work_uses - residential_buildings) / _total, 4)
    elif feature_name == 'F05_housing_in_between':
        mixed_set = {'mixed', 'retail', 'commercial', 'office'}
        mixed_count = sum(1 for bt in building_types if bt in mixed_set)
        return round(mixed_count / total_buildings, 4)
    elif feature_name == 'F06_residential_density':
        return round(residential_buildings / AREA_KM2, 2)
    elif feature_name == 'F07_housing_diversity':
        housing_universe_set = {'residential', 'house', 'apartments', 'terrace', 'dormitory', 'detached'}
        unique_res_types = len(set(bt for bt in building_types if bt in housing_universe_set))
        return round(unique_res_types / 6.0, 4)
    elif feature_name == 'F08_public_private_gradient':
        street_facing = amenity_counts.get('restaurant', 0) + amenity_counts.get('cafe', 0) + amenity_counts.get('bank', 0)
        return round(street_facing / max(total_poi, 1), 4)
    elif feature_name == 'F09_street_cafe_density':
        return round((google_cafe_count + google_restaurant_count) / AREA_KM2, 4)
    elif feature_name == 'F10_market_cluster':
        return 1 if (shop_count + google_store_count) > 5 else 0
    elif feature_name == 'F12_gamma_index':
        return network_metrics.get('gamma_index', 0.0) if network_metrics else 0.0
    elif feature_name == 'F13_pedestrian_count':
        return round(footway_count / AREA_KM2, 2)
    elif feature_name == 'F14_shop_variety':
        google_type_set = set()
        for p in google_places:
            if not isinstance(p, dict):
                continue
            for t in p.get('types', []):
                if t not in ('point_of_interest', 'establishment'):
                    google_type_set.add(t)
        return round(len(google_type_set) / total_google, 4)
    elif feature_name == 'F15_night_economy':
        return round(nightlife_count / total_google, 4)
    elif feature_name == 'F16_transit_access':
        return transit_stops
    elif feature_name == 'F17_civic_presence':
        civic_amenities = amenity_counts.get('townhall', 0) + amenity_counts.get('community_centre', 0) + amenity_counts.get('library', 0)
        return 1 if civic_amenities > 0 else 0
    elif feature_name == 'F18_healthcare_access':
        return google_health_count + amenity_counts.get('clinic', 0) + amenity_counts.get('doctors', 0)
    elif feature_name == 'F19_education_scatter':
        return google_school_count + amenity_counts.get('university', 0) + amenity_counts.get('college', 0)
    elif feature_name == 'F20_industrial_ribbon':
        industrial_lu = landuse_types.count('industrial') + landuse_types.count('retail')
        return round(industrial_lu / max(len(landuse_types), 1), 4)
    elif feature_name == 'F21_home_workshop_density':
        workshop_total = craft_count + amenity_counts.get('studio', 0)
        return round(workshop_total / AREA_KM2, 4)
    elif feature_name == 'F22_intersection_density':
        intersection_count = network_metrics.get('intersection_count', 0) if network_metrics else 0
        return round(intersection_count / AREA_KM2, 2)
    elif feature_name == 'F23_height_mix':
        unique_building_types = len(set(building_types))
        return round(unique_building_types / total_buildings, 4)
    elif feature_name == 'F25_promenade_score':
        ped_density_norm = min(footway_count / AREA_KM2 / 100, 1.0)
        food_density_norm = min((google_restaurant_count + google_cafe_count) / AREA_KM2 / 50, 1.0)
        return round((ped_density_norm + food_density_norm) / 2, 4)
    elif feature_name == 'F26_density_gradient':
        inner_pois = 0
        for p in google_places:
            if not isinstance(p, dict):
                continue
            plat = p.get('lat', lat)
            plon = p.get('lon', lon)
            if plat is None or plon is None or math.isnan(plat) or math.isnan(plon):
                plat, plon = lat, lon
            dx = (plon - lon) * 111320 * math.cos(math.radians(lat))
            dy = (plat - lat) * 110540
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < 200:
                inner_pois += 1
        inner_area = SMALL_AREA_KM2
        outer_area = AREA_KM2 - inner_area
        inner_density = inner_pois / max(inner_area, 0.001)
        outer_pois = len(google_places) - inner_pois
        outer_density = outer_pois / max(outer_area, 0.001)
        return round(inner_density / max(outer_density, 0.001), 4)
    elif feature_name == 'F27_food_stand_density':
        food_stalls = amenity_counts.get('food_court', 0) + amenity_counts.get('fast_food', 0)
        return round(food_stalls / AREA_KM2, 4)
    elif feature_name == 'F28_workspace_cluster':
        workspace_total = office_count + craft_count + amenity_counts.get('coworking_space', 0)
        return round(workspace_total / AREA_KM2, 4)
    elif feature_name == 'F30_green_space_ratio':
        green_types = {'recreation_ground', 'meadow', 'forest'}
        green_count = sum(1 for lt in landuse_types if lt in green_types)
        park_count = sum(1 for el in osm_elements if isinstance(el, dict) and el.get('tags', {}).get('leisure') in ('park', 'garden', 'playground'))
        return round((green_count + park_count) / max(total_poi, 1), 4)
    return 0.0


# ----------------------------------------------------
# TIER 1: FEATURE COVERAGE (27 features * 5 scenarios = 135 cases)
# ----------------------------------------------------

# We build 5 valid input scenarios for each feature dynamically.
tier1_params = []
for f_col in FEATURE_COLS:
    for s_idx in range(5):
        # Scenario generation
        osm = []
        google = []
        net = {'gamma_index': 0.5, 'intersection_count': 10, 'edge_density': 1.2, 'avg_node_degree': 3.2}
        
        if f_col == 'F01_poi_density':
            # Scenarios with varying POI counts
            osm = [{'tags': {'amenity': 'cafe'}}] * s_idx
            google = [{'types': ['restaurant']}] * (s_idx * 2)
        elif f_col == 'F02_land_use_entropy':
            # Scenarios with different land use distributions
            cats = ['residential', 'commercial', 'retail', 'industrial']
            osm = [{'tags': {'landuse': cats[i % len(cats)]}} for i in range(s_idx + 1)]
        elif f_col == 'F03_retail_gf_ratio':
            # Varies retail ground floor elements
            osm = [{'tags': {'building': 'yes'}}] * 5
            for i in range(s_idx):
                osm[i]['tags']['shop'] = 'retail_store'
        elif f_col == 'F04_work_residential_ratio':
            # Varies work and residential buildings
            osm = [{'tags': {'building': 'residential'}}] * 3 + [{'tags': {'building': 'office', 'office': 'tech'}}] * s_idx
        elif f_col == 'F05_housing_in_between':
            # Mixed use buildings
            osm = [{'tags': {'building': 'house'}}] * 4 + [{'tags': {'building': 'commercial'}}] * s_idx
        elif f_col == 'F06_residential_density':
            # Varies residential counts
            osm = [{'tags': {'building': 'house'}}] * s_idx
        elif f_col == 'F07_housing_diversity':
            # Varies housing types
            housing_types = ['house', 'apartments', 'terrace', 'dormitory', 'detached']
            osm = [{'tags': {'building': housing_types[i]}} for i in range(min(s_idx + 1, 5))]
        elif f_col == 'F08_public_private_gradient':
            # Varies street facing amenities
            osm = [{'tags': {'amenity': 'restaurant'}} for _ in range(s_idx)] + [{'tags': {'amenity': 'school'}}]
        elif f_col == 'F09_street_cafe_density':
            # Varies google cafe/restaurant counts
            google = [{'types': ['cafe'] if i % 2 == 0 else ['restaurant']} for i in range(s_idx)]
        elif f_col == 'F10_market_cluster':
            # Threshold testing around 5
            osm = [{'tags': {'shop': 'clothes'}}] * (s_idx + 3)
        elif f_col == 'F12_gamma_index':
            net['gamma_index'] = 0.1 * s_idx
        elif f_col == 'F13_pedestrian_count':
            osm = [{'tags': {'highway': 'footway'}}] * s_idx
        elif f_col == 'F14_shop_variety':
            types_pool = ['bakery', 'clothing_store', 'grocery', 'book_store', 'pharmacy']
            google = [{'types': [types_pool[i % len(types_pool)]]} for i in range(s_idx + 1)]
        elif f_col == 'F15_night_economy':
            google = [{'types': ['bar']}] * s_idx + [{'types': ['store']}] * 3
        elif f_col == 'F16_transit_access':
            osm = [{'tags': {'highway': 'bus_stop'}}] * s_idx
        elif f_col == 'F17_civic_presence':
            osm = [{'tags': {'amenity': 'library'}}] * (1 if s_idx > 2 else 0)
        elif f_col == 'F18_healthcare_access':
            osm = [{'tags': {'amenity': 'clinic'}}] * s_idx
        elif f_col == 'F19_education_scatter':
            google = [{'types': ['school']}] * s_idx
        elif f_col == 'F20_industrial_ribbon':
            osm = [{'tags': {'landuse': 'industrial'}}] * s_idx + [{'tags': {'landuse': 'residential'}}] * 2
        elif f_col == 'F21_home_workshop_density':
            osm = [{'tags': {'craft': 'woodworking'}}] * s_idx
        elif f_col == 'F22_intersection_density':
            net['intersection_count'] = s_idx * 5
        elif f_col == 'F23_height_mix':
            osm = [{'tags': {'building': f'type_{i}'}} for i in range(s_idx + 1)]
        elif f_col == 'F25_promenade_score':
            osm = [{'tags': {'highway': 'footway'}}] * (s_idx * 10)
            google = [{'types': ['cafe']}] * (s_idx * 5)
        elif f_col == 'F26_density_gradient':
            # Varies distance of points
            google = [{'lat': 0.0001 * i, 'lon': 0.0001 * i, 'types': ['store']} for i in range(s_idx + 1)]
        elif f_col == 'F27_food_stand_density':
            osm = [{'tags': {'amenity': 'fast_food'}}] * s_idx
        elif f_col == 'F28_workspace_cluster':
            osm = [{'tags': {'office': 'yes'}}] * s_idx
        elif f_col == 'F30_green_space_ratio':
            osm = [{'tags': {'leisure': 'park'}}] * s_idx + [{'tags': {'amenity': 'cafe'}}]
            
        tier1_params.append((f_col, s_idx, osm, google, net))

@pytest.mark.parametrize("feature_name, scenario_idx, osm_elements, google_places, network_metrics", tier1_params)
def test_tier1_feature_coverage(feature_name, scenario_idx, osm_elements, google_places, network_metrics):
    res = engineer_features(osm_elements, google_places, network_metrics, lat=0.0, lon=0.0)
    expected = calculate_expected_feature(feature_name, osm_elements, google_places, network_metrics, lat=0.0, lon=0.0)
    assert res[feature_name] == pytest.approx(expected, abs=1e-4)


# ----------------------------------------------------
# TIER 2: BOUNDARY & CORNER CASES (27 features * 5 boundary scenarios = 135 cases)
# ----------------------------------------------------

tier2_params = []
for f_col in FEATURE_COLS:
    for b_idx in range(5):
        osm = []
        google = []
        net = {'gamma_index': 0.5, 'intersection_count': 10, 'edge_density': 1.2, 'avg_node_degree': 3.2}
        
        # 5 boundary/corner conditions:
        # 0: Completely empty inputs
        # 1: Extreme values (huge count/density)
        # 2: Missing/None values in key tags
        # 3: Invalid type formats or out-of-bounds/extreme coordinates
        # 4: Boundary thresholds (e.g. exactly 5 or 6, negative network metrics)
        
        if b_idx == 0:
            # Completely empty
            pass
        elif b_idx == 1:
            # Extreme high values
            if f_col in ('F01_poi_density', 'F06_residential_density', 'F09_street_cafe_density', 'F13_pedestrian_count', 'F21_home_workshop_density', 'F22_intersection_density', 'F25_promenade_score', 'F27_food_stand_density', 'F28_workspace_cluster'):
                osm = [{'tags': {'building': 'house', 'highway': 'footway', 'craft': 'yes', 'amenity': 'fast_food', 'office': 'yes'}}] * 10000
                google = [{'types': ['cafe', 'restaurant', 'store']}] * 5000
                net['intersection_count'] = 100000
            elif f_col == 'F02_land_use_entropy':
                osm = [{'tags': {'landuse': 'residential'}}] * 100000
            elif f_col == 'F03_retail_gf_ratio':
                osm = [{'tags': {'building': 'yes', 'shop': 'supermarket'}}] * 10000 + [{'tags': {'building': 'yes'}}]
            elif f_col == 'F04_work_residential_ratio':
                osm = [{'tags': {'building': 'residential'}}] * 100000
            elif f_col == 'F05_housing_in_between':
                osm = [{'tags': {'building': 'mixed'}}] * 100000 + [{'tags': {'building': 'house'}}]
            elif f_col == 'F07_housing_diversity':
                osm = [{'tags': {'building': 'house'}}] * 100000
            elif f_col == 'F08_public_private_gradient':
                osm = [{'tags': {'amenity': 'restaurant'}}] * 10000 + [{'tags': {'amenity': 'school'}}] * 10000
            elif f_col == 'F10_market_cluster':
                osm = [{'tags': {'shop': 'yes'}}] * 100000
            elif f_col == 'F12_gamma_index':
                net['gamma_index'] = 999.9
            elif f_col == 'F14_shop_variety':
                google = [{'types': [f'type_{i}'] for i in range(1000)}]
            elif f_col == 'F15_night_economy':
                google = [{'types': ['bar']}] * 10000
            elif f_col == 'F16_transit_access':
                osm = [{'tags': {'highway': 'bus_stop'}}] * 10000
            elif f_col == 'F17_civic_presence':
                osm = [{'tags': {'amenity': 'library'}}] * 10000
            elif f_col == 'F18_healthcare_access':
                osm = [{'tags': {'amenity': 'clinic'}}] * 10000
            elif f_col == 'F19_education_scatter':
                osm = [{'tags': {'amenity': 'university'}}] * 10000
            elif f_col == 'F20_industrial_ribbon':
                osm = [{'tags': {'landuse': 'industrial'}}] * 10000
            elif f_col == 'F23_height_mix':
                osm = [{'tags': {'building': f'type_{i}'}} for i in range(10000)]
            elif f_col == 'F26_density_gradient':
                google = [{'lat': 0.0001, 'lon': 0.0001, 'types': ['store']}] * 10000
            elif f_col == 'F30_green_space_ratio':
                osm = [{'tags': {'leisure': 'park'}}] * 10000
        elif b_idx == 2:
            # None/Missing values in tags (omitting lat/lon to prevent division errors)
            osm = [{'tags': {'building': None, 'landuse': None, 'highway': None, 'amenity': None, 'shop': None}}]
            google = [{'types': [None]}]
        elif b_idx == 3:
            # Extreme coordinates
            lat_ext, lon_ext = 89.9, -179.9
            google = [{'lat': lat_ext, 'lon': lon_ext, 'types': ['cafe']}]
        elif b_idx == 4:
            # Thresholds and boundary checks
            if f_col == 'F10_market_cluster':
                osm = [{'tags': {'shop': 'yes'}}] * 5  # Boundary check: exactly 5 (returns 0)
            elif f_col == 'F12_gamma_index':
                net['gamma_index'] = -0.5  # Negative gamma index
            elif f_col == 'F22_intersection_density':
                net['intersection_count'] = -10  # Negative intersections
            else:
                osm = [{'tags': {'building': 'residential'}}] * 5
                
        tier2_params.append((f_col, b_idx, osm, google, net))

@pytest.mark.parametrize("feature_name, boundary_idx, osm_elements, google_places, network_metrics", tier2_params)
def test_tier2_boundary_corner(feature_name, boundary_idx, osm_elements, google_places, network_metrics):
    # Tests that engineer_features runs and behaves predictably on extreme/empty/None inputs.
    # We compare with the calculator which mirrors feature_config & collect_data behavior.
    res = engineer_features(osm_elements, google_places, network_metrics, lat=0.0, lon=0.0)
    expected = calculate_expected_feature(feature_name, osm_elements, google_places, network_metrics, lat=0.0, lon=0.0)
    
    # Verify that the output is not NaN/Inf
    assert not math.isnan(res[feature_name])
    assert not math.isinf(res[feature_name])
    assert res[feature_name] == pytest.approx(expected, abs=1e-4)


# ----------------------------------------------------
# TIER 3: CROSS-FEATURE COMBINATIONS (>= 27 cases)
# ----------------------------------------------------

# Let's generate 27 distinct combinations of inputs to test scaling / normalization behavior.
# Each combination tests a different pattern of features.
@pytest.mark.parametrize("combination_idx", list(range(27)))
def test_tier3_cross_feature_combinations(combination_idx):
    np.random.seed(combination_idx)
    
    # 1. Generate random features matrix of shape (20, 27)
    X_train = np.random.uniform(0.0, 100.0, size=(20, 27))
    X_val = np.random.uniform(0.0, 100.0, size=(10, 27))
    X_test = np.random.uniform(0.0, 100.0, size=(10, 27))
    
    # Customise features based on combination index
    # e.g., zero out some columns, make some columns binary, or make some very large
    for i, col in enumerate(FEATURE_COLS):
        strategy = NORMALIZATION_STRATEGY[col]
        if strategy == 'passthrough':
            # binary/bounded features [0, 1]
            X_train[:, i] = np.random.choice([0.0, 1.0], size=(20,))
            X_val[:, i] = np.random.choice([0.0, 1.0], size=(10,))
            X_test[:, i] = np.random.choice([0.0, 1.0], size=(10,))
        elif combination_idx % 3 == 0:
            # Sparse features
            mask = np.random.choice([0, 1], size=(20,), p=[0.8, 0.2])
            X_train[:, i] *= mask
            
    # 2. Run fit normalization
    X_train_norm, scalers = fit_normalise_features(X_train)
    X_val_norm = transform_features(X_val, scalers)
    X_test_norm = transform_features(X_test, scalers)
    
    # 3. Assertions
    assert X_train_norm.shape == (20, 27)
    assert X_val_norm.shape == (10, 27)
    assert X_test_norm.shape == (10, 27)
    
    assert not np.isnan(X_train_norm).any()
    assert not np.isnan(X_val_norm).any()
    assert not np.isnan(X_test_norm).any()
    
    # 4. Verify scaling strategy implementation
    for i, col in enumerate(FEATURE_COLS):
        strategy, scaler_obj = scalers[col]
        if strategy == 'passthrough':
            assert np.allclose(X_train_norm[:, i], X_train[:, i])
            assert np.allclose(X_val_norm[:, i], X_val[:, i])
        elif strategy == 'robust':
            assert scaler_obj is not None
            # RobustScaler centers and scales
            center = scaler_obj.center_[0]
            scale = scaler_obj.scale_[0]
            expected_val_norm = (X_val[:, i] - center) / (scale if scale > 0 else 1.0)
            assert np.allclose(X_val_norm[:, i], expected_val_norm, atol=1e-5)
        elif strategy == 'log_robust':
            assert scaler_obj is not None
            center = scaler_obj.center_[0]
            scale = scaler_obj.scale_[0]
            expected_val_norm = (np.log1p(X_val[:, i]) - center) / (scale if scale > 0 else 1.0)
            assert np.allclose(X_val_norm[:, i], expected_val_norm, atol=1e-5)

    # 5. Verify no leakage: changing X_val does not affect train normalization parameters
    original_scaler_params = {}
    for col in FEATURE_COLS:
        strat, scaler_obj = scalers[col]
        if scaler_obj is not None:
            original_scaler_params[col] = (scaler_obj.center_[0], scaler_obj.scale_[0])
            
    # Modify X_val heavily and transform again
    X_val_modified = X_val * 1000.0
    _ = transform_features(X_val_modified, scalers)
    
    for col in FEATURE_COLS:
        strat, scaler_obj = scalers[col]
        if scaler_obj is not None:
            assert scaler_obj.center_[0] == original_scaler_params[col][0]
            assert scaler_obj.scale_[0] == original_scaler_params[col][1]


# ----------------------------------------------------
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (>= 14 cases)
# ----------------------------------------------------

neighborhood_profiles = [
    # 1. Manhattan CBD, NY (USA)
    {
        "name": "Manhattan CBD", "city": "New York", "country": "USA",
        "osm": [{'tags': {'building': 'office'}}]*20 + [{'tags': {'building': 'retail'}}]*10 + [{'tags': {'amenity': 'restaurant'}}]*15 + [{'tags': {'highway': 'footway'}}]*40 + [{'tags': {'railway': 'subway_entrance'}}]*8,
        "google": [{'types': ['cafe', 'establishment']}]*30 + [{'types': ['restaurant']}]*40,
        "net": {'gamma_index': 0.95, 'intersection_count': 60, 'edge_density': 3.5, 'avg_node_degree': 4.0}
    },
    # 2. Suburban Bedroom Community (Orange County, USA)
    {
        "name": "Suburbia", "city": "Irvine", "country": "USA",
        "osm": [{'tags': {'building': 'house'}}]*150 + [{'tags': {'highway': 'residential'}}]*10,
        "google": [{'types': ['store']}]*2,
        "net": {'gamma_index': 0.15, 'intersection_count': 3, 'edge_density': 0.4, 'avg_node_degree': 2.1}
    },
    # 3. Industrial Corridor (Ruhr Valley, Germany)
    {
        "name": "Ruhr Industrial", "city": "Essen", "country": "Germany",
        "osm": [{'tags': {'landuse': 'industrial'}}]*30 + [{'tags': {'building': 'industrial'}}]*15 + [{'tags': {'highway': 'primary'}}]*5,
        "google": [],
        "net": {'gamma_index': 0.45, 'intersection_count': 8, 'edge_density': 1.1, 'avg_node_degree': 2.8}
    },
    # 4. Amsterdam Historic Core (Netherlands)
    {
        "name": "Amsterdam Center", "city": "Amsterdam", "country": "Netherlands",
        "osm": [{'tags': {'building': 'apartments'}}]*50 + [{'tags': {'amenity': 'cafe'}}]*12 + [{'tags': {'highway': 'pedestrian'}}]*35 + [{'tags': {'highway': 'cycleway'}}]*20,
        "google": [{'types': ['cafe']}]*25 + [{'types': ['bar']}]*10 + [{'types': ['store']}]*15,
        "net": {'gamma_index': 0.88, 'intersection_count': 45, 'edge_density': 2.8, 'avg_node_degree': 3.5}
    },
    # 5. Tokyo Transit Hub (Shibuya, Japan)
    {
        "name": "Shibuya Station Area", "city": "Tokyo", "country": "Japan",
        "osm": [{'tags': {'building': 'commercial'}}]*80 + [{'tags': {'railway': 'station'}}]*5 + [{'tags': {'amenity': 'restaurant'}}]*30 + [{'tags': {'highway': 'pedestrian'}}]*50,
        "google": [{'types': ['restaurant']}]*100 + [{'types': ['cafe']}]*40 + [{'types': ['store']}]*50,
        "net": {'gamma_index': 0.92, 'intersection_count': 55, 'edge_density': 3.1, 'avg_node_degree': 3.9}
    },
    # 6. Mumbai Dharavi/Informal Area (India)
    {
        "name": "Dharavi Mixed-Use", "city": "Mumbai", "country": "India",
        "osm": [{'tags': {'building': 'house'}}]*200 + [{'tags': {'craft': 'leather_pottery'}}]*25 + [{'tags': {'amenity': 'food_court'}}]*15,
        "google": [{'types': ['store']}]*15,
        "net": {'gamma_index': 0.70, 'intersection_count': 35, 'edge_density': 2.0, 'avg_node_degree': 3.1}
    },
    # 7. Oxford University Town (UK)
    {
        "name": "Oxford Core", "city": "Oxford", "country": "United Kingdom",
        "osm": [{'tags': {'amenity': 'university'}}]*8 + [{'tags': {'building': 'residential'}}]*30 + [{'tags': {'amenity': 'library'}}]*3 + [{'tags': {'highway': 'footway'}}]*25,
        "google": [{'types': ['book_store']}]*8 + [{'types': ['cafe']}]*12,
        "net": {'gamma_index': 0.65, 'intersection_count': 22, 'edge_density': 1.6, 'avg_node_degree': 2.9}
    },
    # 8. Singapore Downtown Core (Singapore)
    {
        "name": "Marina Bay Downtown", "city": "Singapore", "country": "Singapore",
        "osm": [{'tags': {'building': 'office'}}]*30 + [{'tags': {'leisure': 'park'}}]*6 + [{'tags': {'railway': 'subway_entrance'}}]*6 + [{'tags': {'highway': 'footway'}}]*25,
        "google": [{'types': ['restaurant']}]*20 + [{'types': ['store']}]*15,
        "net": {'gamma_index': 0.85, 'intersection_count': 30, 'edge_density': 2.2, 'avg_node_degree': 3.6}
    },
    # 9. Cotswold Village (UK)
    {
        "name": "Cotswolds Rural", "city": "Gloucestershire", "country": "United Kingdom",
        "osm": [{'tags': {'building': 'house'}}]*12 + [{'tags': {'landuse': 'meadow'}}]*15 + [{'tags': {'leisure': 'garden'}}]*2,
        "google": [{'types': ['pub']}]*1,
        "net": {'gamma_index': 0.1, 'intersection_count': 1, 'edge_density': 0.2, 'avg_node_degree': 2.0}
    },
    # 10. Chandni Chowk Market (Delhi, India)
    {
        "name": "Chandni Chowk", "city": "Delhi", "country": "India",
        "osm": [{'tags': {'shop': 'clothes'}}]*120 + [{'tags': {'amenity': 'fast_food'}}]*40 + [{'tags': {'amenity': 'place_of_worship'}}]*5 + [{'tags': {'highway': 'pedestrian'}}]*15,
        "google": [{'types': ['store']}]*80 + [{'types': ['restaurant']}]*30,
        "net": {'gamma_index': 0.8, 'intersection_count': 30, 'edge_density': 2.5, 'avg_node_degree': 3.2}
    },
    # 11. Gold Coast Coastal Strip (Australia)
    {
        "name": "Surfers Paradise", "city": "Gold Coast", "country": "Australia",
        "osm": [{'tags': {'building': 'apartments'}}]*40 + [{'tags': {'leisure': 'beach_resort'}}]*3 + [{'tags': {'highway': 'pedestrian'}}]*18,
        "google": [{'types': ['cafe']}]*22 + [{'types': ['restaurant']}]*18,
        "net": {'gamma_index': 0.6, 'intersection_count': 15, 'edge_density': 1.4, 'avg_node_degree': 3.0}
    },
    # 12. Paris Bedroom Community (France)
    {
        "name": "Paris Banlieue", "city": "Paris Suburbs", "country": "France",
        "osm": [{'tags': {'building': 'residential'}}]*90 + [{'tags': {'highway': 'residential'}}]*12,
        "google": [],
        "net": {'gamma_index': 0.3, 'intersection_count': 4, 'edge_density': 0.6, 'avg_node_degree': 2.2}
    },
    # 13. Bengaluru IT Park (India)
    {
        "name": "Whitefield IT Corridor", "city": "Bengaluru", "country": "India",
        "osm": [{'tags': {'building': 'office'}}]*25 + [{'tags': {'office': 'tech'}}]*15 + [{'tags': {'highway': 'service'}}]*12,
        "google": [{'types': ['cafe']}]*10 + [{'types': ['restaurant']}]*8,
        "net": {'gamma_index': 0.5, 'intersection_count': 12, 'edge_density': 1.3, 'avg_node_degree': 2.7}
    },
    # 14. Madrid Plaza Mayor Core (Spain)
    {
        "name": "Plaza Mayor Center", "city": "Madrid", "country": "Spain",
        "osm": [{'tags': {'building': 'retail'}}]*30 + [{'tags': {'amenity': 'cafe'}}]*15 + [{'tags': {'highway': 'pedestrian'}}]*45 + [{'tags': {'amenity': 'townhall'}}]*1,
        "google": [{'types': ['restaurant']}]*35 + [{'types': ['cafe']}]*20 + [{'types': ['store']}]*25,
        "net": {'gamma_index': 0.9, 'intersection_count': 40, 'edge_density': 3.0, 'avg_node_degree': 3.8}
    }
]

@pytest.mark.parametrize("profile", neighborhood_profiles)
@patch('inference.geocode_location')
@patch('inference.fetch_osm_features')
@patch('inference.fetch_google_places')
@patch('inference.fetch_network_metrics')
def test_tier4_real_world_scenarios(mock_net, mock_google, mock_osm, mock_geocode, profile):
    # Mock return values for APIs based on profile
    mock_geocode.return_value = (40.7128, -74.0060)
    mock_osm.return_value = profile["osm"]
    mock_google.return_value = profile["google"]
    mock_net.return_value = profile["net"]
    
    # Request prediction using mock request client
    from backend.app import predict, LocationQuery
    import asyncio
    
    payload = LocationQuery(
        neighbourhood=profile["name"],
        city=profile["city"],
        country=profile["country"]
    )
    
    data = asyncio.run(predict(payload))
    
    assert "location" in data
    assert "score" in data
    assert "features" in data
    
    # Verify score is in valid range
    assert 0.0 <= data["score"] <= 10.0
    
    # Verify all 27 features are present
    assert len(data["features"]) == 27
    for col in FEATURE_COLS:
        assert col in data["features"]


# ----------------------------------------------------
# API ENDPOINT UNIT TESTS (FastAPI Client Mocks)
# ----------------------------------------------------

@patch('inference.geocode_location')
def test_api_predict_geocoding_failure(mock_geocode):
    mock_geocode.return_value = (None, None)
    
    from backend.app import predict, LocationQuery
    import asyncio
    from fastapi import HTTPException
    
    payload = LocationQuery(
        neighbourhood="Invalid Place",
        city="Nowhere",
        country="Void"
    )
    
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(predict(payload))
    assert excinfo.value.status_code == 400
    assert "Could not geocode location" in excinfo.value.detail


# ----------------------------------------------------
# API KEY ROTATION LOGIC UNIT TEST
# ----------------------------------------------------

@patch('requests.post')
def test_api_key_rotation_logic(mock_post):
    import requests
    import data.collect_data as cd
    
    # Save original post and force mock assignment
    original_post = requests.post
    requests.post = mock_post
    
    try:
        cd.GOOGLE_KEYS = ['KEY_1', 'KEY_2']
        cd._current_key_index = 0
        cd.google_api_call_count = 0

        # Setup mocks
        # First response: Quota exceeded 429
        # Second response: Successful 200 for remaining requests
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.json.return_value = {'error': {'message': 'Quota exceeded'}}
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            'places': [
                {'id': 'p1', 'displayName': {'text': 'Cafe A'}, 'types': ['cafe'], 'location': {'latitude': 0.0, 'longitude': 0.0}}
            ]
        }
        
        # Use side_effect to capture headers at call time (avoiding in-place dict mutation issue)
        captured_keys = []
        def side_effect_func(*args, **kwargs):
            headers = kwargs.get('headers', {})
            captured_keys.append(headers.get('X-Goog-Api-Key'))
            if len(captured_keys) == 1:
                return mock_response_429
            return mock_response_200
            
        mock_post.side_effect = side_effect_func
        
        # Run fetch_google_places
        places = fetch_google_places(0.0, 0.0)
        
        # Assertions
        # 1. Total places fetched matches the unique places (deduplicated by place_id)
        assert len(places) == 1
        assert places[0]['place_id'] == 'p1'
        
        # 2. Mock post was called 4 times
        assert mock_post.call_count == 4
        
        # 3. Check X-Goog-Api-Key headers captured at call time
        assert captured_keys[0] == 'KEY_1'
        assert captured_keys[1] == 'KEY_2'
    finally:
        # Restore requests.post to avoid side effects
        requests.post = original_post
