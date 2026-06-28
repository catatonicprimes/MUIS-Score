# MUIS E2E Test Infrastructure & Feature Mapping

This document describes the test infrastructure for the Mixed-Use Interaction Score (MUIS) project and maps all 27 features.

## 1. Feature Mapping Table

The MUIS model leverages 27 distinct urban features computed from OpenStreetMap (OSM) and Google Places (New) APIs. The table below lists all 27 features, their urban planning rationale, their Christopher Alexander Pattern Reference, and their normalization strategy.

| Feature ID & Name | Description | Urban Planning Rationale | Christopher Alexander Pattern Reference | Normalization Strategy | Expert Weight |
|---|---|---|---|---|---|
| `F01_poi_density` | POIs per km² (OSM + Google) | Activity intensity: density of facilities indicates social interaction potential. | Pattern 30: Activity Nodes | `log_robust` | 0.11 |
| `F02_land_use_entropy` | Shannon land use entropy | Diversity of land uses prevents single-use zoning and keeps areas active. | Pattern 9: Scattered Work | `passthrough` | 0.18 |
| `F03_retail_gf_ratio` | Ground-floor retail building ratio | Active frontages activate streets and encourage pedestrian traffic. | Pattern 32: Shopping Street | `passthrough` | 0.06 |
| `F04_work_residential_ratio` | Work-residential balance index | Bounded jobs-housing balance to measure mixed-use completeness. | Pattern 9: Scattered Work | `passthrough` | 0.09 |
| `F05_housing_in_between` | Mixed-use building ratio | Vertical mixing (retail downstairs, apartments upstairs) builds vibrancy. | Pattern 48: Housing In-Between | `passthrough` | 0.02 |
| `F06_residential_density` | Residential buildings per km² | Basic population density threshold required to support local businesses. | Pattern 35: Household Mix | `log_robust` | 0.01 |
| `F07_housing_diversity` | Unique housing type variety ratio | Demographic mixing of households via diverse residential typologies. | Pattern 35: Household Mix | `passthrough` | 0.04 |
| `F08_public_private_gradient` | Street-facing amenities / total POI | Transitions from public street frontages to private residential domains. | Pattern 36: Degrees of Publicness | `passthrough` | 0.02 |
| `F09_street_cafe_density` | Restaurants + cafes per km² (Google) | Third places that support informal social contacts and street vitality. | Pattern 88: Street Cafe | `log_robust` | 0.03 |
| `F10_market_cluster` | Binary indicator for shop clusters | Agglomeration economies and retail clusters generate high footfalls. | Pattern 46: Market of Many Shops | `passthrough` | 0.02 |
| `F12_gamma_index` | Street network connectivity ratio | Higher connectivity in street layouts enables pedestrian routing choices. | Pattern 100: Pedestrian Street | `robust` | 0.08 |
| `F13_pedestrian_count` | Pedestrian footway elements per km² | Presence of pedestrian infrastructure dictates walkability. | Pattern 100: Pedestrian Street | `log_robust` | 0.05 |
| `F14_shop_variety` | Unique shop types / total shops (Google)| Depth of retail commercial diversity and options for local residents. | Pattern 87: Individually Owned Shops| `passthrough` | 0.04 |
| `F15_night_economy` | Nightlife POIs / total Google POIs | Temporal diversity: ensures neighborhoods remain active after dark. | Pattern 33: Night Life | `passthrough` | 0.04 |
| `F16_transit_access` | Transit stops count | Public transit connectivity connects the neighborhood to the region. | Pattern 16: Web of Public Transport | `log_robust` | 0.05 |
| `F17_civic_presence` | Binary indicator for civic amenities | Civic buildings (libraries, town halls) anchor neighborhood identity. | Pattern 44: Local Town Hall | `passthrough` | 0.01 |
| `F18_healthcare_access` | Healthcare POIs count | Essential local medical services support local livability. | Pattern 47: Health Center | `log_robust` | 0.01 |
| `F19_education_scatter` | Education POIs count | Local schools and educational facilities anchor community networks. | Pattern 43: University as Market | `log_robust` | 0.01 |
| `F20_industrial_ribbon` | Industrial land use fraction | Negative indicator: high industrial presence represents mono-use zoning. | Pattern 42: Industrial Ribbon (Negative) | `passthrough` | -0.02 |
| `F21_home_workshop_density`| Craft/workshop POIs per km² | Maker economies and artisanal jobs integrate work into the community. | Pattern 157: Home Workshop | `log_robust` | 0.01 |
| `F22_intersection_density` | Intersections per km² | Finer block granularity provides more street corners for interactions. | Pattern 14: Identifiable Neighbourhood| `log_robust` | 0.04 |
| `F23_height_mix` | Unique building type ratio | Diverse heights and building typologies represent architectural variety.| Pattern 21: Four-Storey Limit | `passthrough` | 0.01 |
| `F25_promenade_score` | Pedestrian count + food density average | Walks that are activated by physical paths and retail dining spots. | Pattern 31: Promenade | `log_robust` | 0.02 |
| `F26_density_gradient` | Inner/outer POI density ratio | Structural concentration towards the center creates clear destinations.| Pattern 29: Density Rings | `robust` | 0.03 |
| `F27_food_stand_density` | Fast food + food courts per km² | Informal food stands and street vendors keep sidewalks animated. | Pattern 93: Food Stands | `log_robust` | 0.01 |
| `F28_workspace_cluster` | Offices + coworking spaces per km² | Workspace clusters create daytime foot traffic supporting local retail. | Pattern 41: Work Community | `log_robust` | 0.02 |
| `F30_green_space_ratio` | Parks + leisure / total POI | Green infrastructure within walking distance is vital for health. | Pattern 60: Accessible Green | `passthrough` | 0.01 |

---

## 2. Test Suite Architecture

The end-to-end (E2E) testing suite for the MUIS project is structured in 4 tiers:

### Tier 1: Feature Coverage (>= 135 cases)
Parameterizes tests for all 27 features with at least 5 different valid inputs/scenarios each to verify feature engineering correctness.

### Tier 2: Boundary & Corner Cases (>= 135 cases)
Parameterizes tests for all 27 features with at least 5 boundary/edge/invalid/extreme inputs each (e.g. empty lists, negative values, very large values, NaN/None handling, extreme ratios).

### Tier 3: Cross-Feature Combinations (>= 27 cases)
Verifies complex inter-dependencies by parameterizing tests across 27 distinct combinations of features, confirming normalization scaling behaves as expected.

### Tier 4: Real-World Application Scenarios (>= 14 cases)
Defines 14 distinct real-world neighborhood/city profiles (e.g., Manhattan CBD, Suburbia, Industrial Corridors, mixed-use nodes) and validates the pipeline end-to-end.

---

## 3. Mocking & Network Restrictions

To run reliably in CI/CD without hitting external Overpass (OSM), Nominatim, or Google Places APIs, all external HTTP requests and spatial analysis functions are fully mocked.
- **FastAPI `/api/predict`**: Tested via FastAPI's `TestClient` or HTTPX requests.
- **API Key Rotation**: Verified via mock status code returns (`429` for primary key, rotating to backup key).
