"""
MUIS Project — Central Configuration
=====================================
Every path, constant, feature definition, and expert weight lives here.
All other scripts import from this module — single source of truth.

Author: Swaastak
"""

import os

# ============================================================
# PATHS — derived from this file's location, works on any machine
# ============================================================
_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(_THIS_DIR)

DATA_DIR       = _THIS_DIR
RAW_DIR        = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR  = os.path.join(DATA_DIR, 'processed')
MODEL_DIR      = os.path.join(PROJECT_ROOT, 'model')
CACHE_DIR      = os.path.join(PROJECT_ROOT, 'cache')

# Auto-create directories
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

TRAINING_LOCATIONS_CSV = os.path.join(RAW_DIR, 'training_locations.csv')
FEATURES_CSV           = os.path.join(PROCESSED_DIR, 'features.csv')
FEATURES_LABELLED_CSV  = os.path.join(PROCESSED_DIR, 'features_labelled.csv')
FEATURES_TEMP_CSV      = os.path.join(PROCESSED_DIR, 'features_temp.csv')
SCALER_PKL             = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
ENV_FILE               = os.path.join(PROJECT_ROOT, '.env')

# ============================================================
# SPATIAL PARAMETERS
# ============================================================
RADIUS_M       = 800    # Primary analysis radius (metres) — 10-minute walk
SMALL_RADIUS_M = 200    # Inner core for density gradient measurement

# ============================================================
# API CONFIGURATION
# ============================================================
GOOGLE_PLACES_DAILY_LIMIT = 20000    # Set to a high value to allow key rotation and scale up to 2000+ locations
NOMINATIM_USER_AGENT = 'MUIS-Paper-Research/1.0 (academic@spa.edu.in)'

# Google Places (New) — 3 semantic batches instead of 12 individual types
# Each batch = 1 API call, total = 3 calls per location
# 1,001 locations × 3 = 3,003 calls/day — fits free tier comfortably
GOOGLE_TYPE_GROUPS = [
    ['restaurant', 'cafe', 'bar', 'night_club'],         # Food & nightlife
    ['supermarket', 'store', 'bank', 'pharmacy'],         # Retail & services
    ['hospital', 'school', 'gym', 'shopping_mall'],       # Community & health
]

# ============================================================
# MASTER LAND-USE TYPOLOGY
# ============================================================
# WHY: Shannon entropy must be normalized by the TOTAL number of
# possible categories, not just the observed ones. If we normalize
# by observed count, a place with 2 equally-split types gets entropy
# = 1.0, same as a place with 15 equally-split types. That's wrong.
#
# This list defines ALL land-use types we recognise from OSM.
# log2(15) = 3.91 is our maximum possible entropy.
LANDUSE_CATEGORIES = [
    'residential', 'commercial', 'retail', 'industrial',
    'institutional', 'education', 'religious', 'recreation_ground',
    'meadow', 'forest', 'farmland', 'cemetery',
    'military', 'construction', 'brownfield',
]
N_LANDUSE_CATEGORIES = len(LANDUSE_CATEGORIES)  # 15

# Building types that count as "residential"
RESIDENTIAL_BUILDING_TYPES = [
    'residential', 'house', 'apartments', 'terrace',
    'dormitory', 'detached', 'semidetached_house',
]

# Building types that indicate mixed-use / commercial
MIXED_BUILDING_TYPES = [
    'mixed', 'retail', 'commercial', 'office',
]

# All possible housing types for diversity calculation
HOUSING_TYPE_UNIVERSE = [
    'residential', 'house', 'apartments', 'terrace',
    'dormitory', 'detached',
]

# ============================================================
# FEATURE DEFINITIONS (27 features — 2 removed, 1 added vs original)
# ============================================================
# Removed: F11 (= 1 - F03, zero new information)
#          F24 (= F22, identical computation)
# Added:   Gini-Simpson diversity index (complements Shannon entropy)
#          Green space ratio (Alexander Pattern 60)

FEATURE_COLS = [
    'F01_poi_density',              # POIs per km² — Activity intensity
    'F02_land_use_entropy',         # Shannon H' (normalised) — Use diversity
    'F03_retail_gf_ratio',          # Ground-floor retail as fraction of buildings
    'F04_work_residential_ratio',   # Jobs-housing balance
    'F05_housing_in_between',       # Mixed-type buildings as % of total
    'F06_residential_density',      # Residential buildings per km²
    'F07_housing_diversity',        # Housing type variety (0–1)
    'F08_public_private_gradient',  # Street-facing amenities / total POI
    'F09_street_cafe_density',      # Cafes + restaurants per km²
    'F10_market_cluster',           # Binary: is there a market cluster?
    'F12_gamma_index',              # Street network connectivity
    'F13_pedestrian_count',         # Footway/pedestrian element count per km²
    'F14_shop_variety',             # Unique shop types / total shops
    'F15_night_economy',            # Nightlife POIs / total Google POIs
    'F16_transit_access',           # Transit stops count
    'F17_civic_presence',           # Binary: any civic amenity present?
    'F18_healthcare_access',        # Healthcare POIs count
    'F19_education_scatter',        # Education POIs count
    'F20_industrial_ribbon',        # Industrial land use fraction
    'F21_home_workshop_density',    # Craft/workshop POIs per km²
    'F22_intersection_density',     # Intersections per km² (block granularity)
    'F23_height_mix',               # Building type variety / total buildings
    'F25_promenade_score',          # Pedestrian infra + food density (additive)
    'F26_density_gradient',         # Inner/outer POI density ratio
    'F27_food_stand_density',       # Fast food / food courts per km²
    'F28_workspace_cluster',        # Offices + coworking per km²
    'F30_green_space_ratio',        # Parks/gardens as fraction of total land use
]

N_FEATURES = len(FEATURE_COLS)  # 27

# ============================================================
# EXPERT MUIS WEIGHTS — Urban Planning Rationale
# ============================================================
#
# These weights determine how the 28 features combine into a single
# MUIS score (0–10). They are derived from foundational urban planning
# literature and calibrated for mixed-use interaction measurement.
#
# THEORETICAL FRAMEWORK:
# ─────────────────────
# Jane Jacobs (1961) argued that DIVERSITY OF PRIMARY USES is the
# single most critical factor for vibrant urban life. A neighbourhood
# needs at least 2-3 primary functions (residential + commercial +
# institutional) to generate foot traffic at different times of day.
#
# Christopher Alexander (1977) encoded this into 253 patterns. For
# mixed-use, the most critical are:
#   Pattern 9  (Scattered Work)   -> work-residential balance
#   Pattern 14 (Identifiable Neighbourhood) -> block variation
#   Pattern 30 (Activity Nodes)   -> POI concentration
#   Pattern 31 (Promenade)        -> pedestrian infrastructure
#   Pattern 36 (Degrees of Publicness) -> public-private gradient
#   Pattern 100 (Pedestrian Street) -> street connectivity
#
# Jan Gehl (1971) showed that pedestrian-scale design and street
# activation (ground-floor retail, cafes) drive social interaction.
#
# WEIGHT TIERS:
# ─────────────
# Tier 1 (Critical — 55%): Land use diversity, activity density,
#         work-residential balance, street connectivity, retail activation
# Tier 2 (Important — 30%): Pedestrian infra, transit, shop variety,
#         housing diversity, night economy, density gradient
# Tier 3 (Supporting — 15%): Civic, healthcare, education, food,
#         industrial, workshops, green space
#
MUIS_WEIGHTS = {
    # ── Tier 1: Core Mixed-Use Indicators (55%) ──────────────────
    'F02_land_use_entropy':         0.18,   # THE fundamental measure of mixing
    'F01_poi_density':              0.11,   # Activity intensity — more POIs = more interaction potential
    'F04_work_residential_ratio':   0.09,   # Jobs-housing balance — the most basic form of mixed use
    'F12_gamma_index':              0.08,   # Street connectivity — connected grids enable mixed use
    'F03_retail_gf_ratio':          0.06,   # Active ground floors — makes streets alive (Gehl)
    'F22_intersection_density':     0.04,   # Block granularity — smaller blocks = more walkable mixing

    # ── Tier 2: Supporting Mixed-Use Indicators (30%) ────────────
    'F13_pedestrian_count':         0.05,   # Walkability infrastructure
    'F16_transit_access':           0.05,   # Transit enables diverse users
    'F14_shop_variety':             0.04,   # Depth of commercial diversity
    'F07_housing_diversity':        0.04,   # Demographic mixing through housing types
    'F15_night_economy':            0.04,   # Temporal diversity — place lives day AND night
    'F26_density_gradient':         0.03,   # Proper centre-to-edge gradation (Pattern 29)
    'F09_street_cafe_density':      0.03,   # Street-level social spaces (Pattern 88)
    'F08_public_private_gradient':  0.02,   # Transition spaces (Pattern 36)

    # ── Tier 3: Supplementary Indicators (15%) ───────────────────
    'F05_housing_in_between':       0.02,   # Vertical mixing indicator
    'F06_residential_density':      0.01,   # Population base
    'F10_market_cluster':           0.02,   # Cluster economy presence
    'F17_civic_presence':           0.01,   # Institutional completeness
    'F18_healthcare_access':        0.01,   # Essential services
    'F19_education_scatter':        0.01,   # Institutional diversity
    'F20_industrial_ribbon':       -0.02,   # NEGATIVE: mono-industrial = anti-mixed-use
    'F21_home_workshop_density':    0.01,   # Artisan/maker economy
    'F23_height_mix':               0.01,   # Visual variety
    'F25_promenade_score':          0.02,   # Pedestrian amenity
    'F27_food_stand_density':       0.01,   # Street food culture
    'F28_workspace_cluster':        0.02,   # Work community presence
    'F30_green_space_ratio':        0.01,   # Green amenity (too much = mono-use park)
}

# Verify weights sum to ~1.0
_weight_sum = sum(MUIS_WEIGHTS.values())
assert abs(_weight_sum - 1.0) < 0.001, f"Weights must sum to 1.0, got {_weight_sum}"

# ============================================================
# NORMALIZATION STRATEGY PER FEATURE TYPE
# ============================================================
# WHY: StandardScaler assumes Gaussian distribution. But our features
# include binary (0/1), bounded [0,1], and heavy-tailed counts.
# Applying z-score to a binary feature is mathematically inappropriate.
#
# Strategy:
#   'log_robust'  — Apply log1p() then RobustScaler (for skewed counts)
#   'robust'      — RobustScaler only (for continuous, roughly symmetric)
#   'passthrough' — Leave as-is (for binary 0/1 or already bounded [0,1])

NORMALIZATION_STRATEGY = {
    'F01_poi_density':          'log_robust',   # Heavy right skew (count-based)
    'F02_land_use_entropy':     'passthrough',  # Already bounded [0, 1]
    'F03_retail_gf_ratio':      'passthrough',  # Already bounded [0, 1]
    'F04_work_residential_ratio':'passthrough',  # Unbounded ratio, right skew
    'F05_housing_in_between':   'passthrough',  # Already bounded [0, 1]
    'F06_residential_density':  'log_robust',   # Count per area, right skew
    'F07_housing_diversity':    'passthrough',  # Already bounded [0, 1]
    'F08_public_private_gradient':'passthrough', # Already bounded [0, 1]
    'F09_street_cafe_density':  'log_robust',   # Count per area
    'F10_market_cluster':       'passthrough',  # Binary 0/1
    'F12_gamma_index':          'robust',       # Continuous, roughly normal
    'F13_pedestrian_count':     'log_robust',   # Count per area
    'F14_shop_variety':         'passthrough',  # Bounded ratio [0, ~2]
    'F15_night_economy':        'passthrough',  # Bounded ratio [0, 1]
    'F16_transit_access':       'log_robust',   # Count, right skew
    'F17_civic_presence':       'passthrough',  # Binary 0/1
    'F18_healthcare_access':    'log_robust',   # Count
    'F19_education_scatter':    'log_robust',   # Count
    'F20_industrial_ribbon':    'passthrough',  # Bounded [0, 1]
    'F21_home_workshop_density':'log_robust',   # Count per area
    'F22_intersection_density': 'log_robust',   # Count per area
    'F23_height_mix':           'passthrough',  # Bounded [0, 1]
    'F25_promenade_score':      'log_robust',   # Composite, right skew
    'F26_density_gradient':     'robust',       # Ratio, can be > 1
    'F27_food_stand_density':   'log_robust',   # Count per area
    'F28_workspace_cluster':    'log_robust',   # Count per area
    'F30_green_space_ratio':    'passthrough',  # Bounded [0, 1]
}

# ============================================================
# MUIS SCORE CLASS BOUNDARIES
# ============================================================
# For calibrating the weighted MUIS score against expected_class labels.
# These define the "soft target" ranges — used to verify the formula
# produces scores in the right ballpark, not to force labels.
MUIS_CLASS_RANGES = {
    'High':   (4.0, 10.0),    # Top 33%
    'Medium': (2.75, 4.0),    # Middle 33%
    'Low':    (0.0, 2.75),    # Bottom 33%
}
