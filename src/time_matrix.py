

import math
import logging
from datetime import date, timedelta
from typing import Optional
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Constants
EARTH_RADIUS_KM = 6371.0 
NYC_AVG_SPEED_KMH = 5.0
MIN_TRAVEL_MINS = 4

# Google Maps day encoding : 0=Sunday ... 6=Saturday
PYTHON_TO_GOOGLE_DAY = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:0}

def TravelTime(parsed_query,rag_pois):
    trip_duration = parsed_query.get('trip_duration_days',3)
    start_date = parsed_query.get('start_date', None)
    pref_zones = parsed_query.get('preferred_zones', [])
    must_pois = parsed_query['hard_constraints'].get('must_include_pois',[])
    avoid_days = parsed_query['hard_constraints'].get('avoid_days',[])

    # Finding trip days indices
    trip_days_set = set(trip_days(trip_duration, start_date, avoid_days))
    logger.info("Trip days indices: %s", trip_days_set)

    # Building depot node
    depot = build_depot(rag_pois, pref_zones)

    # Filtering - hard constraint (closed all trip days)
    selected, removed = valid_pois(rag_pois, must_pois, trip_days_set)
    logger.info(
        "Filtering result: %s kept / %s removed",
        len(selected), len(removed)
    )

    # Travel time matrix (haversine)
    all_locations = [depot] + selected
    matrix_raw = build_haversine_matrix(all_locations)
    locations_id = [n['id'] for n in all_locations ]

    travel_time_matrix = {
        locations_id[i] : { locations_id[j] : matrix_raw[i][j] for j in range(len(locations_id))}
        for i in range(len(locations_id))
    }

    poi_index_map = {nid: idx for idx, nid in enumerate(locations_id)}

    return {
        'parsed_query' : parsed_query,
        'selected_pois' : selected,
        'removed_pois' : removed,
        'travel_time_matrix' : travel_time_matrix,
        'matrix_raw' : matrix_raw,
        'location_ids' : locations_id,
        'depot' : depot,
        'trip_day_indices' : list(trip_days_set),
        'poi_index_map' : poi_index_map,
    }

ZONE_FALLBACK_CENTROIDS = {
    "lower_manhattan":  (40.7128, -74.0060),
    "midtown":          (40.7549, -73.9840),
    "upper_east_side":  (40.7735, -73.9565),
    "upper_west_side":  (40.7870, -73.9754),
    "brooklyn":         (40.6782, -73.9442),
    "bronx":            (40.8448, -73.8648),
    "queens":           (40.7282, -73.7949),
    "staten_island":    (40.5795, -74.1502),
}



def centroid(coords):
    'Mean lat and lon of a list of (lat, lon) tuples'
    n = len(coords)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,  
    )

NYC_FALLBACK_CENTROID = centroid(list(ZONE_FALLBACK_CENTROIDS.values()))


def build_depot(rag_pois, pref_zones):
    '''
     Depot priority:
      1. Centroid of candidates in preferred_zones  
      2. Centroid of ALL candidates 
    '''

    # Centroid of candidates in preferred_zones 
    if pref_zones:
        pref_z = set(z.lower().strip() for z in pref_zones)
        pref_z_coords = [ 
            (p['location']['latitude'], p['location']['longitude'])
            for p in rag_pois
            if p.get('location') and 
            p.get('zone','').lower() in pref_z
        ]
    if pref_zones and pref_z_coords:
        lat, lon = centroid(pref_z_coords)
        label = f"Depot - centroid of zones : {', '.join(pref_z)}"
        logger.info(
            "Depot set to centroid of preferred zones %s : (%.4f, %.4f)",
            pref_z, lat,lon
        )
    else:
        # Centroid of ALL candidates
        all_coords =[
            (p['location']['latitude'], p['location']['longitude'])
            for p in rag_pois
            if p.get('location')
        ]
        if all_coords:
            lat, lon = centroid(all_coords)
            label = "Depot - centroid of all candidates"
            logger.info(
                "Depot set to centroid of all candidates : (%.4f, %.4f)",
                lat, lon
            )
        else:
            lat, lon = NYC_FALLBACK_CENTROID
            label = "Depot — centroid of all NYC zones (no candidates available)"
            logger.warning("No candidate locations found. Using  NYC centroid depot.")

    return {
        'id' : 'depot',
        'location' : {
            'latitude' : lat,
            'longitude' : lon
        }
    }

def trip_days(duration, start_date, avoid_days):
    """
    Returns list of Google weekday indices (0=Sun) for each trip day.
    Skips avoided days and picks the next available day instead.
    Falls back to Monday start when start_date is absent.
    """
    avoid = set(avoid_days or [])

    if start_date:
        try:
            start = date.fromisoformat(start_date)
        except ValueError:
            logger.warning("Invalid start date '%s' — falling back to Monday", start_date)
            start = None
    else:
        start = None

    if start is None:
        logger.warning("start_date not provided — default trip days start from Mon")

    result = []
    day_offset = 0

    while len(result) < duration:
        if start:
            python_day = (start + timedelta(days=day_offset)).weekday()
            google_day = PYTHON_TO_GOOGLE_DAY[python_day]
        else:
            # default: start from Monday (google=1), increment
            google_day = (1 + day_offset) % 7

        if google_day not in avoid:
            result.append(google_day)

        day_offset += 1

        # safety: avoid infinite loop if all 7 days are avoided
        if day_offset > duration + 7:
            logger.error("Cannot find %d valid trip days — avoid_days too restrictive", duration)
            break

    return result   

def poi_open_days(poi):
    '''
    Extract set of Google weekday indices when this POI is open.
    Returns full set {0..6} when hours data is missing (safe default = keep POI).
    '''
    hours = poi.get("regularOpeningHours")
    if not hours:
        return set(range(7))
    periods = hours.get("periods", [])
    if not periods:
        return set(range(7))
    return {p["open"]["day"] for p in periods if "open" in p}

def must_pois_check(poi, trip_days_set):
    '''
    Check if must-include POI is operational or open on trip duration
    Returns flags and status of POI
    '''
    flags = list(poi.get('flags',[]))
    status = True
    if poi.get('businessStatus', 'OPERATIONAL') != 'OPERATIONAL':
        flags.append("Not Operational")
        status = False
    if not trip_days_set & poi_open_days(poi):
        flags.append("Not open on trip duration")
        status = False
    return flags, status


def _opens_outside_touring_day(poi: dict, trip_days_set: set) -> bool:
    """
    Returns True if ALL opening periods on trip days start at or after
    DAY_START_HOUR + DAY_CAPACITY_MIN (i.e. after 10PM for 780 min cap).
    These POIs are night-only venues that cannot be visited during touring hours.
    """
    DAY_START_HOUR   = 9
    DAY_CAPACITY_MIN = 780
    cutoff_hour      = DAY_START_HOUR + DAY_CAPACITY_MIN // 60  # = 22 (10PM)

    periods = (poi.get('regularOpeningHours') or {}).get('periods', [])
    relevant = [
        p for p in periods
        if p.get('open', {}).get('day') in trip_days_set
    ]

    if not relevant:
        return False  # no hours data — keep POI

    # True only if EVERY relevant period opens at or after cutoff
    return all(
        p['open'].get('hour', 0) >= cutoff_hour
        for p in relevant
    )

    
def valid_pois(rag_pois, must_pois,trip_days_set):
    '''
    Hard filter: remove POIs closed on ALL trip days.
    Must-include POIs survive with a flag, never dropped.
    Returns (filtered: list, removed: list)
    '''
    selected = []
    removed = []
    for poi in rag_pois:
        name = poi.get('displayName', {}).get('text', '').lower()
        # Check must_include_pois, add flags if needed
        if name in must_pois:
            flags, status = must_pois_check(poi, trip_days_set)
            poi = dict(poi)
            poi['flags'] = flags
            selected.append(poi)
            if not status:
                logger.warning(
                    "Must-include POI '%s' failed check '%s' - keeping with flag",
                    name, ' | '.join(flags)
                )
            continue
        # Check business status
        if poi.get('businessStatus', 'OPERATIONAL') != 'OPERATIONAL':
            logger.warning("POI '%s' business not operational", name)
            removed.append(poi)
            continue

        # Open on at least one trip day check
        if not (set(trip_days_set) & poi_open_days(poi)):
            logger.warning("POI '%s' closed all trip days", name)
            removed.append(poi)
            continue
        
        if _opens_outside_touring_day(poi, trip_days_set):
            logger.warning(
                "POI '%s' only opens outside touring hours — removed",
                name
            )
            removed.append(poi)
            continue
        selected.append(poi)
    return selected, removed


# Haversine 

def haversine_km(lat1, lon1, lat2, lon2):
    'Great-circle distance in kilometres.'
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def km_to_min(km):
    'Convert km → integer minutes using NYC average speed.'
    return max(MIN_TRAVEL_MINS, round((km / NYC_AVG_SPEED_KMH) * 60))


def build_haversine_matrix(all_locations):
    '''
    Returns symmetric list[list[int]] of travel times in minutes.
    nodes[0] is the depot; diagonal is 0.
    '''

    n = len(all_locations)
    matrix = [[0] * n for _ in range(n)]

    lats = [p['location']['latitude'] for p in all_locations]
    lons = [p['location']['longitude'] for p in all_locations]

    for i in range(n):
        for j in range(i+1, n):
            t = km_to_min(haversine_km(lats[i], lons[i], lats[j], lons[j]))
            matrix[i][j] = t
            matrix[j][i] = t
    return matrix

