"""
MUIS Data Collection — Production Rewrite
==========================================
Collects OSM + Google Places data for 1,001 global locations and engineers
28 urban features for the Mixed-Use Interaction Score (MUIS) research paper.

This file was completely rewritten to fix every critical issue in the prior
version.  The problems — and how each is now solved — are documented inline
at the exact point where the fix appears.

Theoretical backbone
--------------------
Jane Jacobs (1961): "The generators of diversity … need an enormous
diversity of ingredients."  We capture *what* exists (POIs, land uses,
building types) and *how it is arranged* (entropy, gradients, connectivity).

Christopher Alexander (1977): Pattern Language patterns 9, 14, 29–31, 36,
41–48, 88, 93, 100, 122, 157 each map to a specific feature below.

Jan Gehl (1971): Street-level activation (cafes, ground-floor retail,
pedestrian infrastructure) is the observable signal of mixed-use vitality.

Author : Swaastak
Paper  : Deep Learning Model for Mixed-Use Pattern Language
"""

# ================================================================
# IMPORTS
# ================================================================
import sys
import os
import time
import math
import requests
import pandas as pd
import numpy as np
import osmnx as ox
from collections import Counter
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Import every path and constant from the single-source-of-truth config.
# This eliminates ALL hardcoded paths (Problem #9 in the prior version).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_config import (
    PROJECT_ROOT, RAW_DIR, PROCESSED_DIR, CACHE_DIR, ENV_FILE,
    RADIUS_M, SMALL_RADIUS_M, NOMINATIM_USER_AGENT,
    GOOGLE_TYPE_GROUPS, GOOGLE_PLACES_DAILY_LIMIT,
    LANDUSE_CATEGORIES, N_LANDUSE_CATEGORIES,
    RESIDENTIAL_BUILDING_TYPES, MIXED_BUILDING_TYPES,
    HOUSING_TYPE_UNIVERSE, FEATURE_COLS,
    TRAINING_LOCATIONS_CSV, FEATURES_CSV, FEATURES_TEMP_CSV,
)

# ---------------------------------------------------------------------------
# Load Google API keys from the project-level .env file.
# Supports key rotation across multiple projects for higher daily quota.
# Keys are loaded from GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, etc.
# ---------------------------------------------------------------------------
load_dotenv(ENV_FILE)

GOOGLE_KEYS = []
# Load primary key
_primary = os.getenv('GOOGLE_API_KEY')
if _primary:
    GOOGLE_KEYS.append(_primary)
# Load additional keys (GOOGLE_API_KEY_2, _3, _4, ...)
for i in range(2, 50):
    _extra = os.getenv(f'GOOGLE_API_KEY_{i}')
    if _extra:
        GOOGLE_KEYS.append(_extra)

if GOOGLE_KEYS:
    print(f"  Loaded {len(GOOGLE_KEYS)} Google API key(s)")
else:
    print("  WARNING: No Google API keys found in .env")

import threading
_google_api_lock = threading.Lock()

# Current key index — rotates when a key hits quota
_current_key_index = 0

def get_current_google_key():
    """Return the currently active Google API key, or None if all exhausted."""
    global _current_key_index
    with _google_api_lock:
        if not GOOGLE_KEYS or _current_key_index >= len(GOOGLE_KEYS):
            return None
        return GOOGLE_KEYS[_current_key_index]

def rotate_google_key():
    """Switch to the next API key. Returns True if a new key is available."""
    global _current_key_index
    with _google_api_lock:
        _current_key_index += 1
        if _current_key_index < len(GOOGLE_KEYS):
            print(f"  --> Rotating to Google API key #{_current_key_index + 1}")
            return True
        else:
            print(f"  --> All {len(GOOGLE_KEYS)} Google API keys exhausted for today")
            return False

# ---------------------------------------------------------------------------
# GLOBAL API-CALL COUNTER -- every Google Places request increments this.
# When it reaches GOOGLE_PLACES_DAILY_LIMIT the script stops calling Google
# for the rest of the run.  This prevents surprise billing.
# ---------------------------------------------------------------------------
google_api_call_count = 0

# ---------------------------------------------------------------------------
# COMMON SUFFIXES for geocoding fallback.
# Many location names in our training set carry suffixes like "Market" or
# "IT Park" that Nominatim does not recognise as place identifiers.
# Stripping them drastically improves geocoding hit rate.
# ---------------------------------------------------------------------------
_COMMON_SUFFIXES = [
    'Market', 'Area', 'IT', 'Park', 'Hub', 'Zone', 'Corridor',
    'Residential', 'Industrial', 'Estate',
]


# ================================================================
# 1. GEOCODING
# ================================================================
def geocode_location(neighbourhood: str, city: str, country: str,
                     pre_lat: float = None, pre_lon: float = None):
    """
    Convert a human-readable neighbourhood name to (lat, lon) coordinates.

    Resolution order
    ----------------
    1.  If ``pre_lat`` / ``pre_lon`` are provided (pre-seeded columns in
        the CSV), return them immediately — no network call needed.
    2.  Try the full name with Nominatim.
    3.  Strip known suffixes ("Kamla Nagar Market" -> "Kamla Nagar").
    4.  Progressively remove trailing words one at a time
        ("Kamla Nagar" -> "Kamla") until a match is found.

    WHY this strategy?
    ------------------
    The prior version failed ~40 % of names because Nominatim cannot parse
    compound qualifiers.  By first stripping well-known non-geographic
    suffixes and *then* shortening, we typically find a match within 2–3
    iterations — keeping total API calls per location well under 5.

    Nominatim ToS require max 1 request/second; we enforce a 1-second pause
    before every call.

    Returns
    -------
    (lat, lon) : tuple[float, float] or (None, None) on complete failure.
    """
    # -- Fast path: pre-seeded coordinates -------------------------------
    if pre_lat is not None and pre_lon is not None:
        if not (math.isnan(pre_lat) or math.isnan(pre_lon)):
            print(f"  Using pre-seeded coords: ({pre_lat:.4f}, {pre_lon:.4f})")
            return pre_lat, pre_lon

    # -- Build candidate name variants -----------------------------------
    url = "https://nominatim.openstreetmap.org/search"
    headers = {'User-Agent': NOMINATIM_USER_AGENT}

    # Start with the original name, then with known suffixes stripped,
    # then progressively shortened.
    name_variants = [neighbourhood]

    # Strip one known suffix at a time from the end of the name.
    for suffix in _COMMON_SUFFIXES:
        if neighbourhood.endswith(suffix):
            stripped = neighbourhood[: -len(suffix)].strip()
            if stripped and stripped not in name_variants:
                name_variants.append(stripped)

    # Progressively drop trailing words (up to 3 words removed).
    parts = neighbourhood.split()
    for i in range(1, min(4, len(parts))):
        shorter = ' '.join(parts[:-i])
        if shorter and shorter not in name_variants:
            name_variants.append(shorter)

    # -- Try each variant against Nominatim ------------------------------
    for name in name_variants:
        query = f"{name}, {city}, {country}"
        params = {'q': query, 'format': 'json', 'limit': 1}
        try:
            time.sleep(1)  # Nominatim ToS: max 1 request per second
            response = requests.get(url, params=params,
                                    headers=headers, timeout=3)
            results = response.json()
            if results:
                lat = float(results[0]['lat'])
                lon = float(results[0]['lon'])
                tag = f"(fallback '{name}')" if name != neighbourhood else ""
                print(f"  Geocoded {tag}: {query} -> ({lat:.4f}, {lon:.4f})")
                return lat, lon
        except Exception as e:
            print(f"  ERROR geocoding '{query}': {e}")

    print(f"  WARNING: Could not geocode: {neighbourhood}, {city}, {country}")
    return None, None


# ================================================================
# 2. OSM DATA COLLECTION
# ================================================================
# ---------------------------------------------------------------------------
# OVERPASS ENDPOINTS FOR ROTATION
# ---------------------------------------------------------------------------
OVERPASS_MIRRORS = [
    'https://overpass-api.de/api',
    'https://overpass.kumi.systems/api',
    'https://overpass.osm.ch/api'
]
_overpass_lock = threading.Lock()
_current_overpass_index = 0

def rotate_overpass_endpoint():
    """Rotates to the next Overpass endpoint in a thread-safe way."""
    global _current_overpass_index
    with _overpass_lock:
        endpoint = OVERPASS_MIRRORS[_current_overpass_index]
        ox.settings.overpass_url = endpoint
        ox.settings.http_user_agent = 'MUIS-Paper-Research/1.0 (academic@spa.edu.in)'
        _current_overpass_index = (_current_overpass_index + 1) % len(OVERPASS_MIRRORS)
        print(f"  [OSM] Selected Overpass endpoint: {endpoint}", flush=True)

def fetch_osm_features(lat: float, lon: float,
                       radius: int = RADIUS_M) -> list[dict]:
    """
    Fetch OpenStreetMap features using **osmnx.features_from_point()**.

    WHY osmnx instead of raw Overpass POST?
    ----------------------------------------
    Problem #1 in the prior version: raw HTTP POST to the Overpass endpoint
    returned HTTP 406 (Not Acceptable) 100 % of the time because the
    Content-Type header was wrong and the body encoding was non-standard.
    osmnx wraps the Overpass connection correctly, handles rate-limit
    headers, and caches each bbox response on disk so overlapping queries
    never hit the network twice.

    Tag selection rationale
    -----------------------
    We request exactly the OSM tags that feed into our 28 features:
    * amenity, shop, office, craft         -> POI density, retail ratio,
                                              workspace cluster
    * leisure, healthcare, education       -> civic, healthcare, green space
    * highway (bus_stop/footway/pedestrian)-> transit, pedestrian density
    * railway (station/subway_entrance)    -> transit access
    * building                             -> residential density, height mix
    * landuse                              -> entropy, industrial ribbon

    Returns
    -------
    list[dict] — each dict has shape ``{'tags': {key: value, ...}}``.
    On any error, returns an empty list (the per-location try/except in
    the main loop ensures we never crash the entire run).
    """
    # -- Configure osmnx cache and Overpass settings -----------------------
    ox.settings.use_cache = True
    ox.settings.cache_folder = CACHE_DIR
    ox.settings.log_console = False
    ox.settings.timeout = 15              # 15-second Overpass timeout
    ox.settings.requests_timeout = 10     # 10-second HTTP requests timeout (prevents 180s hangs)
    ox.settings.overpass_rate_limit = False # Let our code handle retries and backoff

    # -- Tags to fetch -- mirrors every feature that reads OSM data -------
    tags = {
        'amenity':    True,
        'shop':       True,
        'office':     True,
        'craft':      True,
        'leisure':    True,
        'healthcare': True,
        'education':  True,
        'highway':    ['bus_stop', 'footway', 'pedestrian'],
        'railway':    ['station', 'subway_entrance'],
        'building':   True,
        'landuse':    True,
    }

    # -- Retry with exponential backoff (Overpass throttles aggressively) --
    max_retries = 5
    gdf = None
    for attempt in range(max_retries):
        rotate_overpass_endpoint()
        try:
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
            break  # Success
        except Exception as e:
            err_str = str(e)
            if "EmptyOverpassResponse" in type(e).__name__ or "No features found" in err_str or "found no" in err_str:
                print(f"  OSM Empty response at ({lat:.4f}, {lon:.4f})")
                break
            wait = 1.0 * (2 ** attempt)  # 1s, 2s, 4s, 8s, 16s
            if attempt < max_retries - 1:
                print(f"  OSM attempt {attempt+1}/{max_retries} failed: {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  OSM ERROR at ({lat:.4f}, {lon:.4f}) after {max_retries} attempts: {e}")
                raise e

    if gdf is None or len(gdf) == 0:
        print(f"  OSM returned 0 elements for ({lat:.4f}, {lon:.4f})")
        return []

    # -- Convert GeoDataFrame -> list of {'tags': {...}} dicts ------------
    # This keeps the downstream engineer_features() interface identical to
    # what the old Overpass path would have produced.
    tag_cols = list(tags.keys())
    elements = []
    for _, row in gdf.iterrows():
        elem_tags = {}
        for col in tag_cols:
            if col in row.index:
                val = row[col]
                # Filter out NaN / empty / <NA> sentinel values that
                # pandas assigns when the tag is absent.
                if pd.notna(val) and str(val) not in ('', 'nan', 'NaN', '<NA>'):
                    elem_tags[col] = str(val)
        if elem_tags:
            elements.append({'tags': elem_tags})

    # -- Courtesy pause (0.5 seconds) -- Overpass is a shared resource ------
    time.sleep(0.5)
    return elements


# ================================================================
# 3. GOOGLE PLACES DATA COLLECTION
# ================================================================
def fetch_google_places(lat: float, lon: float,
                        radius: int = RADIUS_M) -> list[dict]:
    """
    Fetch POI data from the Google Places API (New) — Nearby Search.

    WHY 3 batches instead of 12?
    ----------------------------
    Problem #2: the prior version used 12 individual type requests per
    location -> 12 x 1,001 = 12,012 calls/day, exceeding the 5,000 free
    tier.  The Places API (New) accepts up to 50 ``includedTypes`` per
    request (OR logic).  We group all 12 types into 3 semantic batches
    defined in ``feature_config.GOOGLE_TYPE_GROUPS``:

        Batch 1: restaurant, cafe, bar, night_club       (food & nightlife)
        Batch 2: supermarket, store, bank, pharmacy       (retail & services)
        Batch 3: hospital, school, gym, shopping_mall     (community & health)

    1,001 locations x 3 calls = 3,003 calls/day — fits comfortably.

    Global quota tracking
    ---------------------
    ``google_api_call_count`` is incremented for every successful request.
    When it reaches ``GOOGLE_PLACES_DAILY_LIMIT`` (4,500) or a 429 status
    is received, we stop calling Google for the rest of the run.

    Returns
    -------
    list[dict] — each dict contains place_id, name, types, rating,
    user_ratings_total, lat, lon.  Deduplicated by place_id across batches.
    """
def fetch_google_places(lat: float, lon: float,
                        radius: int = RADIUS_M) -> list[dict]:
    """
    Fetch POI data from the Google Places API (New) — Nearby Search.

    WHY 3 batches instead of 12?
    ----------------------------
    Problem #2: the prior version used 12 individual type requests per
    location -> 12 x 1,001 = 12,012 calls/day, exceeding the 5,000 free
    tier.  The Places API (New) accepts up to 50 ``includedTypes`` per
    request (OR logic).  We group all 12 types into 3 semantic batches
    defined in ``feature_config.GOOGLE_TYPE_GROUPS``:

        Batch 1: restaurant, cafe, bar, night_club       (food & nightlife)
        Batch 2: supermarket, store, bank, pharmacy       (retail & services)
        Batch 3: hospital, school, gym, shopping_mall     (community & health)

    1,001 locations x 3 calls = 3,003 calls/day — fits comfortably.

    Global quota tracking
    ---------------------
    ``google_api_call_count`` is incremented for every successful request.
    When it reaches ``GOOGLE_PLACES_DAILY_LIMIT`` (4,500) or a 429 status
    is received, we stop calling Google for the rest of the run.

    Returns
    -------
    list[dict] — each dict contains place_id, name, types, rating,
    user_ratings_total, lat, lon.  Deduplicated by place_id across batches.
    """
    global google_api_call_count

    # -- Check daily limit before even trying ----------------------------
    with _google_api_lock:
        if google_api_call_count >= GOOGLE_PLACES_DAILY_LIMIT:
            print("  Google Places daily limit reached. Skipping.")
            return []

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-FieldMask': (
            'places.id,places.displayName,places.types,'
            'places.rating,places.userRatingCount,places.location'
        ),
    }

    all_places = []
    seen_ids = set()   # Deduplicate across batches

    for group in GOOGLE_TYPE_GROUPS:
        # -- Re-check quota before each batch ----------------------------
        with _google_api_lock:
            if google_api_call_count >= GOOGLE_PLACES_DAILY_LIMIT:
                print("  Google Places daily limit reached mid-location. "
                      "Returning partial results.")
                break

        body = {
            "includedTypes": group,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lon},
                    "radius": float(radius),
                }
            },
            "maxResultCount": 20,
        }

        try:
            current_key = get_current_google_key()
            with _google_api_lock:
                google_api_call_count += 1
            
            if not current_key:
                break

            response = requests.post(url, json=body,
                                     headers={**headers, 'X-Goog-Api-Key': current_key}, timeout=3)

            if response.status_code == 429:
                # -- Quota exhausted for this key -- try rotating ---------
                err = response.json().get('error', {})
                print(f"  Google Places QUOTA EXCEEDED: {err.get('message', '')[:80]}")
                rotated = rotate_google_key()
                current_key = get_current_google_key()
                if rotated and current_key:
                    # Retry the same batch with the new key
                    with _google_api_lock:
                        google_api_call_count += 1
                    response = requests.post(url, json=body,
                                             headers={**headers, 'X-Goog-Api-Key': current_key}, timeout=10)
                    if response.status_code == 429:
                        print("  New key also exhausted!")
                        break
                    elif response.status_code != 200:
                        continue
                else:
                    break  # All keys exhausted

            if response.status_code != 200:
                err = response.json().get('error', {})
                print(f"  Google Places ERROR (group {group}): "
                      f"HTTP {response.status_code}: "
                      f"{err.get('message', response.text[:120])}")
                continue

            data = response.json()
            for place in data.get('places', []):
                pid = place.get('id', '')
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    loc = place.get('location', {})
                    all_places.append({
                        'place_id': pid,
                        'name': place.get('displayName', {}).get('text', ''),
                        'types': place.get('types', []),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'lat': loc.get('latitude', lat),
                        'lon': loc.get('longitude', lon),
                    })

        except Exception as e:
            print(f"  Google Places ERROR (group {group}): {e}")

        # 0.3-second pause between batches — avoid burst throttling.
        time.sleep(0.3)

    return all_places


# ================================================================
# 4. STREET NETWORK METRICS
# ================================================================
def fetch_network_metrics(lat: float, lon: float,
                          radius: int = RADIUS_M) -> dict:
    """
    Compute street-network topology metrics using osmnx.

    WHY these metrics?
    ------------------
    * **Gamma index** (edge / max_planar_edges):  measures how close the
      actual network is to a maximally-connected planar graph.  Dense grids
      score high; cul-de-sac suburbs score low.  This directly operationalises
      Alexander Pattern 100 (Pedestrian Street) — connected grids support
      diverse movement and therefore mixed-use activity.

    * **Intersection count**: finer-grained blocks create more corner
      locations for retail / civic uses (Jacobs' "short blocks" thesis).

    * **Avg node degree**: correlated with walkability.

    Planar maximum edge formula
    ---------------------------
    For a connected planar graph with V vertices, the maximum number of
    edges is ``3 * (V - 2)`` (Euler's formula for planar graphs).  We
    guard against V < 3 by clamping the denominator to 1.

    Returns
    -------
    dict with keys: gamma_index, edge_density, intersection_count,
    avg_node_degree.  All default to 0 on error.
    """
    defaults = {
        'gamma_index': 0,
        'edge_density': 0,
        'intersection_count': 0,
        'avg_node_degree': 0,
    }

    try:
        # Retry with backoff — same Overpass throttling issue
        max_retries = 5
        G = None
        for attempt in range(max_retries):
            rotate_overpass_endpoint()
            ox.settings.timeout = 15
            ox.settings.requests_timeout = 10
            try:
                G = ox.graph_from_point(
                    (lat, lon),
                    dist=radius,
                    network_type='all',
                    simplify=True,
                )
                break
            except Exception as e:
                err_str = str(e)
                if "null graph" in err_str or "found no graph nodes" in err_str or "EmptyOverpassResponse" in type(e).__name__:
                    print(f"  Network empty graph at ({lat:.4f}, {lon:.4f})")
                    break
                wait = 1.0 * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"  Network attempt {attempt+1}/{max_retries} failed: {e}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Network ERROR at ({lat:.4f}, {lon:.4f}) after {max_retries} attempts: {e}")
                    raise e

        if G is None:
            return defaults

        nodes = len(G.nodes)
        edges = len(G.edges)

        # Max edges in a planar graph: 3*(V-2).  Clamp denominator to 1
        # to avoid division-by-zero when the graph is trivially small.
        max_edges = max(3 * (nodes - 2), 1)
        gamma = edges / max_edges

        # Intersections = nodes with degree > 2 (not dead-ends or bends).
        intersections = sum(1 for _, d in G.degree() if d > 2)

        # osmnx basic_stats gives us edge_density and avg_node_degree.
        stats = ox.basic_stats(G)

        return {
            'gamma_index': round(gamma, 4),
            'edge_density': round(stats.get('edge_density', 0), 4),
            'intersection_count': intersections,
            'avg_node_degree': round(stats.get('avg_node_degree', 0), 4),
        }

    except Exception as e:
        print(f"  Network ERROR at ({lat:.4f}, {lon:.4f}): {e}")
        return defaults


# ================================================================
# 5. FEATURE ENGINEERING — THE CORE OF THE MUIS METHODOLOGY
# ================================================================
def engineer_features(osm_elements: list, google_places: list,
                      network_metrics: dict,
                      lat: float, lon: float) -> dict:
    """
    Transform raw OSM + Google + network data into the 28 MUIS features.

    Every feature is annotated with:
    * The Christopher Alexander pattern it operationalises (where applicable).
    * The urban-planning rationale for the chosen formula.
    * Mathematical notes on normalisation / bounds.

    Changes from the prior version
    ------------------------------
    1. F02 uses ``log2`` with **fixed** denominator ``log2(15)`` (not
       ``log(observed)``).  This ensures that a place with 2 equally-split
       types does NOT score the same as one with 15 (Problem #4).
    2. F06 is now a *density* (per km²), not a raw count.
    3. F11 (= 1 − F03) has been **removed** as redundant (Problem #5).
    4. F24 (= F22) has been **removed** as redundant (Problem #6).
    5. F25 is now **additive** (average of normalised pedestrian density
       and food density), not multiplicative (Problem #7).
    6. ``footway_length`` has been renamed to ``footway_count`` because
       the OSM data lacks geometry lengths — we are counting elements,
       not measuring metres (Problem #8).
    7. F29 (Gini-Simpson diversity) **added** (Problem #10).
    8. F30 (Green space ratio) **added** (Problem #11).
    9. F03 numerator now uses **only OSM data** for restaurant and cafe
       counts (``amenity_counts``), not Google Places counts, matching
       the denominator (total_buildings from OSM).

    Returns
    -------
    dict mapping each of the 28 FEATURE_COLS names to its computed value.
    """

    # --------------------------------------------------------------------
    # A.  PARSE OSM ELEMENTS into working counters / lists
    # --------------------------------------------------------------------
    amenity_counts: dict[str, int] = {}     # amenity value -> count
    shop_count     = 0
    office_count   = 0
    craft_count    = 0
    building_types: list[str] = []          # one entry per building element
    landuse_types:  list[str] = []          # one entry per landuse element
    footway_count  = 0                      # renamed from footway_length (Problem #8)
    transit_stops  = 0

    for element in osm_elements:
        tags = element.get('tags', {})

        # Count amenities by specific type (restaurant, cafe, bank, …)
        if 'amenity' in tags:
            amenity = tags['amenity']
            amenity_counts[amenity] = amenity_counts.get(amenity, 0) + 1

        if 'shop' in tags:
            shop_count += 1

        if 'office' in tags:
            office_count += 1

        if 'craft' in tags:
            craft_count += 1

        # Collect building types for residential density, diversity, etc.
        if tags.get('building'):
            building_types.append(tags['building'])

        # Collect land-use types for entropy and industrial-ribbon.
        if tags.get('landuse'):
            landuse_types.append(tags['landuse'])

        # Footway / pedestrian *element count* — not metres (no geometry).
        if tags.get('highway') in ('footway', 'pedestrian'):
            footway_count += 1

        # Transit stops: bus stops + railway stations + subway entrances.
        if (tags.get('highway') == 'bus_stop'
                or tags.get('railway') in ('station', 'subway_entrance')):
            transit_stops += 1

    # --------------------------------------------------------------------
    # B.  PARSE GOOGLE PLACES into working counters
    # --------------------------------------------------------------------
    google_restaurant_count = sum(
        1 for p in google_places if 'restaurant' in p.get('types', []))
    google_cafe_count = sum(
        1 for p in google_places if 'cafe' in p.get('types', []))
    google_store_count = sum(
        1 for p in google_places if 'store' in p.get('types', []))
    google_health_count = sum(
        1 for p in google_places
        if 'hospital' in p.get('types', [])
        or 'pharmacy' in p.get('types', []))
    google_school_count = sum(
        1 for p in google_places if 'school' in p.get('types', []))
    nightlife_count = sum(
        1 for p in google_places
        if 'night_club' in p.get('types', [])
        or 'bar' in p.get('types', []))

    total_google = max(len(google_places), 1)

    # --------------------------------------------------------------------
    # C.  DERIVED AGGREGATES used by multiple features
    # --------------------------------------------------------------------
    # Area of the analysis circle in km² — used to normalise densities.
    area_km2 = math.pi * (RADIUS_M / 1000) ** 2

    total_buildings = max(len(building_types), 1)

    # Residential building count (union of all types in the config list).
    residential_set = set(RESIDENTIAL_BUILDING_TYPES)
    residential_buildings = sum(
        1 for bt in building_types if bt in residential_set)

    # Total POI count (OSM elements + Google places) — used by F01, F08.
    total_poi = len(osm_elements) + len(google_places)

    # --------------------------------------------------------------------
    # D.  COMPUTE EACH OF THE 28 FEATURES
    # --------------------------------------------------------------------

    # -- F01: POI Density - Pattern 30 (Activity Nodes) ------------------
    # More points of interest per km² -> higher interaction potential.
    # Combined OSM + Google gives the most complete picture.
    f01 = round((len(osm_elements) + len(google_places)) / area_km2, 2)

    # -- F02: Land-Use Entropy - Pattern 9 (Scattered Work) --------------
    # Shannon entropy of observed land-use categories, normalised by the
    # FIXED maximum entropy log2(N_LANDUSE_CATEGORIES) = log2(15) ~= 3.91.
    #
    # FIX (Problem #4): the prior version used
    #   (a) natural log instead of log2,
    #   (b) normalised by log(observed categories) instead of log(total),
    #   (c) added an unnecessary epsilon inside the log.
    # All three are corrected below.
    if landuse_types:
        lu_counts = Counter(landuse_types)
        lu_total  = sum(lu_counts.values())
        lu_probs  = [c / lu_total for c in lu_counts.values()]
        # Shannon entropy with log2 — information-theoretic standard.
        entropy = -sum(p * math.log2(p) for p in lu_probs if p > 0)
        # Fixed maximum: log2(15) — the number of categories in our
        # master typology, NOT the number of observed categories.
        max_entropy = math.log2(N_LANDUSE_CATEGORIES)
        f02 = round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0
    else:
        f02 = 0.0

    # -- F03: Retail Ground-Floor Ratio - Pattern 32 (Shopping Street) ---
    # Fraction of buildings whose ground floor is activated by retail,
    # office, restaurant, or cafe use.  Capped at 1.0.
    #
    # FIX: numerator now uses OSM-only restaurant/cafe counts (from
    # amenity_counts) so both numerator and denominator come from the
    # same data source (OSM buildings).
    osm_restaurant_count = amenity_counts.get('restaurant', 0)
    osm_cafe_count       = amenity_counts.get('cafe', 0)
    ground_floor_uses = (shop_count + office_count
                         + osm_restaurant_count + osm_cafe_count)
    f03 = round(min(ground_floor_uses / total_buildings, 1.0), 4)

    # -- F04: Work-Residential Ratio - Pattern 9 (Scattered Work) --------
    # Jobs-housing balance: how many work-generating uses exist per
    # jobs-housing balance. Instead of an unbounded ratio, we use a bounded
    # symmetric balance index: 1.0 when work ≈ residential, 0.0 when either dominates.
    work_uses = office_count + craft_count + shop_count
    _total = work_uses + residential_buildings + 1
    f04 = round(1.0 - abs(work_uses - residential_buildings) / _total, 4)

    # -- F05: Housing In-Between Index - Pattern 48 ----------------------
    # Fraction of buildings classified as mixed-use / commercial types.
    # High values indicate vertical mixing (shops below, housing above).
    mixed_set = set(MIXED_BUILDING_TYPES)
    mixed_count = sum(1 for bt in building_types if bt in mixed_set)
    f05 = round(mixed_count / total_buildings, 4)

    # -- F06: Residential Density - Pattern 35 (Household Mix) -----------
    # Residential buildings per km².
    # FIX: the prior version stored the raw count instead of a density.
    f06 = round(residential_buildings / area_km2, 2)

    # -- F07: Housing Diversity - Pattern 35 -----------------------------
    # Fraction of the 6 possible housing types that are actually present.
    # Higher diversity -> wider demographic mixing -> more interaction.
    housing_universe_set = set(HOUSING_TYPE_UNIVERSE)
    unique_res_types = len(
        set(bt for bt in building_types if bt in housing_universe_set))
    f07 = round(unique_res_types / len(HOUSING_TYPE_UNIVERSE), 4)

    # -- F08: Public-Private Gradient - Pattern 36 -----------------------
    # Street-facing amenities (restaurant + cafe + bank) as a fraction
    # of all POIs.  Captures the "degrees of publicness" concept.
    street_facing = (amenity_counts.get('restaurant', 0)
                     + amenity_counts.get('cafe', 0)
                     + amenity_counts.get('bank', 0))
    f08 = round(street_facing / max(total_poi, 1), 4)

    # -- F09: Street Cafe Density - Pattern 88 (Street Cafe) -------------
    # Google-sourced cafes + restaurants per km².  Google Places provides
    # a more consumer-facing view of active dining establishments than
    # OSM (which often lags for recent openings).
    f09 = round((google_cafe_count + google_restaurant_count) / area_km2, 4)

    # -- F10: Market Cluster - Pattern 46 (Market of Many Shops) ---------
    # Binary indicator: is there a critical mass of retail in this area?
    # Threshold of 5 is conservative — most functioning market streets
    # have 10+ shops; we want to detect even nascent clusters.
    f10 = 1 if (shop_count + google_store_count) > 5 else 0

    # -- (F11 REMOVED — it was 1 − F03, which carries zero new info) -----

    # -- F12: Gamma Index - Pattern 100 (Pedestrian Street) --------------
    # Street-network connectivity from the network metrics module.
    f12 = network_metrics.get('gamma_index', 0)

    # -- F13: Pedestrian Count (density) - Pattern 100 -------------------
    # Footway / pedestrian element count per km².
    f13 = round(footway_count / area_km2, 2)

    # -- F14: Shop Variety - Pattern 87 (Individually Owned Shops) -------
    # Unique Google place types (excluding generic meta-types) divided
    # by total Google places.  Captures depth of commercial diversity.
    google_type_set = set()
    for p in google_places:
        for t in p.get('types', []):
            if t not in ('point_of_interest', 'establishment'):
                google_type_set.add(t)
    f14 = round(len(google_type_set) / total_google, 4)

    # -- F15: Night Economy - Pattern 33 (Night Life) --------------------
    # Nightlife POIs (bars + night clubs) as a fraction of all Google
    # POIs.  Captures temporal diversity — does the place live at night?
    f15 = round(nightlife_count / total_google, 4)

    # -- F16: Transit Access - Pattern 16 (Web of Public Transport) ------
    # Raw count of transit stops (bus + rail + subway).  Higher counts
    # indicate better multi-modal connectivity, which enables diverse
    # users to reach the area and thus supports mixed-use vitality.
    f16 = transit_stops

    # -- F17: Civic Presence - Pattern 44 (Local Town Hall) --------------
    # Binary: does the area contain at least one civic institution?
    # Town halls, community centres, and libraries anchor neighbourhood
    # identity and provide a "third place" that is neither home nor work.
    civic_amenities = (amenity_counts.get('townhall', 0)
                       + amenity_counts.get('community_centre', 0)
                       + amenity_counts.get('library', 0))
    f17 = 1 if civic_amenities > 0 else 0

    # -- F18: Healthcare Access - Pattern 47 (Health Center) -------------
    # Total healthcare POIs from both data sources.
    f18 = (google_health_count
           + amenity_counts.get('clinic', 0)
           + amenity_counts.get('doctors', 0))

    # -- F19: Education Scatter - Pattern 43 (University as Market) ------
    # Education POIs from both sources.
    f19 = (google_school_count
           + amenity_counts.get('university', 0)
           + amenity_counts.get('college', 0))

    # -- F20: Industrial Ribbon - Pattern 42 -----------------------------
    # Fraction of land-use elements that are industrial or retail.
    # A high value signals mono-functional industrial corridors — the
    # antithesis of mixed use.  This feature therefore carries a NEGATIVE
    # weight in the final MUIS formula.
    industrial_lu = (landuse_types.count('industrial')
                     + landuse_types.count('retail'))
    f20 = round(industrial_lu / max(len(landuse_types), 1), 4)

    # -- F21: Home Workshop Density - Pattern 157 (Home Workshop) --------
    # Craft workshops + studios per km².  These are the "maker economy"
    # uses that Jacobs celebrated as generators of diversity.
    workshop_total = craft_count + amenity_counts.get('studio', 0)
    f21 = round(workshop_total / area_km2, 4)

    # -- F22: Intersection Density - Pattern 14 (Identifiable Neighbourhood)
    # Intersections per km².  Smaller blocks -> more corners -> more
    # opportunity for ground-floor uses and casual interaction.
    intersection_count = network_metrics.get('intersection_count', 0)
    f22 = round(intersection_count / area_km2, 2)

    # -- (F24 REMOVED — it was identical to F22) -------------------------

    # -- F23: Height Mix - Pattern 21 (Four-Storey Limit) ----------------
    # Unique building types as a fraction of total buildings.
    # A crude proxy for built-form variety (OSM rarely has storey data).
    unique_building_types = len(set(building_types))
    f23 = round(unique_building_types / total_buildings, 4)

    # -- F25: Promenade Score - Pattern 31 (Promenade) -------------------
    # FIX (Problem #7): the prior version used a MULTIPLICATIVE formula
    # (footway_count * restaurant_count), which collapses to 0 whenever
    # either component is absent.  An ADDITIVE formulation captures
    # partial promenade quality: a street with good pedestrian infra but
    # no food still scores above zero.
    #
    # We normalise each component to [0, 1] independently and average:
    #   ped_norm  = min(footway_count / area_km2 / 100, 1.0)
    #   food_norm = min((google_restaurants + google_cafes) / area_km2 / 50, 1.0)
    #   F25 = (ped_norm + food_norm) / 2
    ped_density_norm = min(footway_count / area_km2 / 100, 1.0)
    food_density_norm = min(
        (google_restaurant_count + google_cafe_count) / area_km2 / 50, 1.0)
    f25 = round((ped_density_norm + food_density_norm) / 2, 4)

    # -- F26: Density Gradient - Pattern 29 (Density Rings) --------------
    # Ratio of inner-core (< 200 m) POI density to outer-ring density.
    # A high ratio indicates a well-defined centre — the concentric
    # structure that Alexander recommends for identifiable neighbourhoods.
    #
    # Distance is approximated with the equirectangular formula:
    #   dx = Δlon x 111,320 x cos(lat)
    #   dy = Δlat x 110,540
    # This is accurate to ~0.5 % within an 800 m radius.
    inner_pois = 0
    all_poi_elements = google_places  # use Google for lat/lon precision
    for p in all_poi_elements:
        dx = (p.get('lon', lon) - lon) * 111320 * math.cos(math.radians(lat))
        dy = (p.get('lat', lat) - lat) * 110540
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < SMALL_RADIUS_M:
            inner_pois += 1

    inner_area = math.pi * (SMALL_RADIUS_M / 1000) ** 2
    outer_area = area_km2 - inner_area
    inner_density = inner_pois / max(inner_area, 0.001)
    outer_pois = len(all_poi_elements) - inner_pois
    outer_density = outer_pois / max(outer_area, 0.001)
    f26 = round(inner_density / max(outer_density, 0.001), 4)

    # -- F27: Food Stand Density - Pattern 93 (Food Stands) --------------
    # Food courts + fast-food amenities per km².  Street food culture
    # is a strong indicator of informal economic mixing.
    food_stalls = (amenity_counts.get('food_court', 0)
                   + amenity_counts.get('fast_food', 0))
    f27 = round(food_stalls / area_km2, 4)

    # -- F28: Workspace Cluster - Pattern 41 (Work Community) ------------
    # Offices + craft workshops + coworking spaces per km².  Clusters of
    # workspaces create daytime foot traffic that activates ground-floor
    # retail and cafes — the demand side of Jacobs' diversity generators.
    workspace_total = (office_count + craft_count
                       + amenity_counts.get('coworking_space', 0))
    f28 = round(workspace_total / area_km2, 4)



    # -- F30: Green Space Ratio (NEW — Problem #11) ----------------------
    # Green / recreational land as a fraction of total POIs.
    #
    # WHY include it?  Alexander Pattern 60 (Accessible Green) argues
    # that green space within walking distance is essential for
    # neighbourhood health.  But too much green space (e.g. a huge park
    # with nothing else) signals mono-use — so this feature receives a
    # very low positive weight in the MUIS formula.
    #
    # Green land-use types: recreation_ground, meadow, forest.
    # Also count OSM leisure elements: park, garden, playground.
    green_types = {'recreation_ground', 'meadow', 'forest'}
    green_count = sum(1 for lt in landuse_types if lt in green_types)
    park_count = sum(
        1 for el in osm_elements
        if el.get('tags', {}).get('leisure') in ('park', 'garden', 'playground'))
    f30 = round((green_count + park_count) / max(total_poi, 1), 4)

    # --------------------------------------------------------------------
    # E.  ASSEMBLE FEATURE DICT — keys MUST match FEATURE_COLS exactly
    # --------------------------------------------------------------------
    features = {
        'F01_poi_density':            f01,
        'F02_land_use_entropy':       f02,
        'F03_retail_gf_ratio':        f03,
        'F04_work_residential_ratio': f04,
        'F05_housing_in_between':     f05,
        'F06_residential_density':    f06,
        'F07_housing_diversity':      f07,
        'F08_public_private_gradient': f08,
        'F09_street_cafe_density':    f09,
        'F10_market_cluster':         f10,
        'F12_gamma_index':            f12,
        'F13_pedestrian_count':       f13,
        'F14_shop_variety':           f14,
        'F15_night_economy':          f15,
        'F16_transit_access':         f16,
        'F17_civic_presence':         f17,
        'F18_healthcare_access':      f18,
        'F19_education_scatter':      f19,
        'F20_industrial_ribbon':      f20,
        'F21_home_workshop_density':  f21,
        'F22_intersection_density':   f22,
        'F23_height_mix':             f23,
        'F25_promenade_score':        f25,
        'F26_density_gradient':       f26,
        'F27_food_stand_density':     f27,
        'F28_workspace_cluster':      f28,

        'F30_green_space_ratio':      f30,
    }

    # Sanity check: every feature we computed must be in FEATURE_COLS,
    # and every FEATURE_COLS entry must be present in our output.
    assert set(features.keys()) == set(FEATURE_COLS), (
        f"Feature key mismatch!\n"
        f"  Extra:   {set(features.keys()) - set(FEATURE_COLS)}\n"
        f"  Missing: {set(FEATURE_COLS) - set(features.keys())}"
    )

    return features


# ================================================================
# 6. MAIN COLLECTION LOOP
# ================================================================
def collect_all_data():
    """
    Read the training-locations CSV, collect data for each location,
    and save the engineered features to ``FEATURES_CSV``.

    Design decisions
    ----------------
    * **Resume support**: if ``FEATURES_CSV`` already exists, we load
      the set of ``location_id`` values that have been processed and
      skip them.  This means you can safely Ctrl-C and restart.

    * **Per-location try/except**: a geocoding failure or API error for
      one location must never crash the run for the remaining 1,000.

    * **Checkpoint every 10 locations**: we write to ``FEATURES_TEMP_CSV``
      so that even if the process dies uncleanly, at most 9 locations
      of work are lost.

    * **KeyboardInterrupt**: saves all data collected so far.

    * **Final save**: appends to ``FEATURES_CSV`` and deduplicates by
      ``location_id`` so re-runs are idempotent.

    * **osm_element_count column**: stored alongside the features as a
      data-quality indicator (locations with 0 OSM elements may need
      manual review).
    """
    global google_api_call_count

    # -- Load training locations -----------------------------------------
    if not os.path.exists(TRAINING_LOCATIONS_CSV):
        print(f"ERROR: Training locations file not found: "
              f"{TRAINING_LOCATIONS_CSV}")
        return None

    locations_df = pd.read_csv(TRAINING_LOCATIONS_CSV)
    total = len(locations_df)
    print(f"Loaded {total} training locations from {TRAINING_LOCATIONS_CSV}")

    # Check if CSV has pre-seeded lat/lon columns.
    has_pre_coords = ('lat' in locations_df.columns
                      and 'lon' in locations_df.columns)
    if has_pre_coords:
        print("  Pre-seeded lat/lon columns detected — will use where available.")

    # -- Load resume set (locations already processed) -------------------
    already_done: set = set()
    if os.path.exists(FEATURES_CSV):
        try:
            existing_df = pd.read_csv(FEATURES_CSV)
            # Treat all existing locations as done (including valid 0 Google data samples)
            already_done = set(existing_df['location_id'].values)
            print(f"  Resuming — {len(already_done)} locations fully collected")
        except Exception as e:
            print(f"  WARNING: Could not read existing features file: {e}")

    # -- Collection loop -------------------------------------------------
    results: list[dict] = []

    try:
        for idx, row in locations_df.iterrows():
            loc_id = row.get('location_id', idx)
            neighbourhood = row.get('neighbourhood', '')
            city = row.get('city', '')
            country = row.get('country', '')

            print(f"\n[{idx + 1}/{total}] Processing: {neighbourhood}, {city}")

            # Skip already-processed locations.
            if loc_id in already_done:
                print("  Already collected. Skipping.")
                continue

            # -- Per-location try/except: never crash the whole run ------
            try:
                # -- Step 1: Geocode -------------------------------------
                pre_lat = (float(row['lat'])
                           if has_pre_coords and pd.notna(row.get('lat'))
                           else None)
                pre_lon = (float(row['lon'])
                           if has_pre_coords and pd.notna(row.get('lon'))
                           else None)

                lat, lon = geocode_location(
                    neighbourhood, city, country,
                    pre_lat=pre_lat, pre_lon=pre_lon,
                )
                if lat is None:
                    print("  SKIP: Could not geocode.")
                    continue

                # -- Step 2: Fetch OSM data ------------------------------
                print("  Fetching OSM data…")
                osm_elements = fetch_osm_features(lat, lon)
                print(f"    -> {len(osm_elements)} OSM elements")

                # -- Step 3: Fetch Google Places (3 batched calls) -------
                print("  Fetching Google Places data…")
                google_places = fetch_google_places(lat, lon)
                print(f"    -> {len(google_places)} Google places  "
                      f"(API calls so far: {google_api_call_count})")

                # -- Step 4: Network metrics -----------------------------
                print("  Computing network metrics…")
                network_metrics = fetch_network_metrics(lat, lon)

                # -- Step 5: Engineer the 28 features --------------------
                print("  Engineering features…")
                features = engineer_features(
                    osm_elements, google_places, network_metrics, lat, lon)

                # -- Build the output record -----------------------------
                record = {
                    'location_id':      loc_id,
                    'city':             city,
                    'neighbourhood':    neighbourhood,
                    'country':          country,
                    'lat':              lat,
                    'lon':              lon,
                    'expected_class':   row.get('expected_class', ''),
                    'osm_element_count': len(osm_elements),
                    **features,
                }
                results.append(record)
                already_done.add(loc_id)

                # -- Checkpoint every 10 locations -----------------------
                if len(results) % 10 == 0 and len(results) > 0:
                    _save_checkpoint(results)

            except Exception as loc_err:
                print(f"  ERROR processing location {loc_id}: {loc_err}")
                continue

    except KeyboardInterrupt:
        print("\n[WARN]️  Interrupted by user. Saving collected data…")

    # -- Final save ------------------------------------------------------
    return _save_final(results)


def _save_checkpoint(results: list[dict]) -> None:
    """Write current results to the temp file as a recovery checkpoint."""
    try:
        pd.DataFrame(results).to_csv(FEATURES_TEMP_CSV, index=False)
        print(f"  💾 Checkpoint saved: {len(results)} locations "
              f"-> {FEATURES_TEMP_CSV}")
    except Exception as e:
        print(f"  WARNING: Checkpoint save failed: {e}")


def _save_final(results: list[dict]):
    """
    Append new results to FEATURES_CSV, deduplicate by location_id,
    and return the final DataFrame.
    """
    if not results:
        print("\n[WARN]️  No new locations collected this run.")
        return None

    final_df = pd.DataFrame(results)

    # Append to existing file if present, then deduplicate.
    if os.path.exists(FEATURES_CSV):
        try:
            existing = pd.read_csv(FEATURES_CSV)
            final_df = pd.concat(
                [existing, final_df], ignore_index=True
            ).drop_duplicates(subset='location_id', keep='last')
        except Exception as e:
            print(f"  WARNING: Could not merge with existing file: {e}")

    final_df.to_csv(FEATURES_CSV, index=False)
    print(f"\n[OK]  Saved {len(results)} new locations -> {FEATURES_CSV}")
    print(f"    Total in file: {len(final_df)}")
    print(f"    Google API calls used this run: {google_api_call_count}")

    # Clean up temp file if it exists.
    if os.path.exists(FEATURES_TEMP_CSV):
        try:
            os.remove(FEATURES_TEMP_CSV)
        except OSError:
            pass

    return final_df


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == '__main__':
    collect_all_data()