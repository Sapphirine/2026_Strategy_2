

from config import API_KEY
import time
from typing import List, Dict
from math import ceil
from collections import defaultdict
import math
import requests
import json


SEARCH_URL = 'https://places.googleapis.com/v1/places:searchNearby'
DETAILS_URL = 'https://places.googleapis.com/v1/places/'

SEARCH_HEADERS = {
    'Content-Type': 'application/json',
    'X-Goog-Api-Key': API_KEY,
    'X-Goog-FieldMask' : (
        'places.id,'
        'places.displayName,'
        'places.primaryType,'
        'places.location,'
        'places.rating,'
        'places.userRatingCount,'
        'places.types'
    ) 
}

DETAILS_HEADERS = {
    'Content-Type': 'application/json',
    'X-Goog-Api-Key': API_KEY,
    'X-Goog-FieldMask' : (
        'id,'
        'displayName,'
        'businessStatus,'
        'formattedAddress,'
        'photos,'
        'reviews,'
        'regularOpeningHours'
    ) 
}

ZONES = {
    "lower_manhattan": {
        "low": {"latitude": 40.7000, "longitude": -74.0200}, 
        "high": {"latitude": 40.7400, "longitude": -73.9700}
    },
    "midtown": {
        "low": {"latitude": 40.7400, "longitude": -74.0100},
        "high": {"latitude": 40.7800, "longitude": -73.9500}
    },
    "upper_west_side": {
        "low": {"latitude": 40.7700, "longitude": -74.0000},
        "high": {"latitude": 40.8000, "longitude": -73.9500}
    },
    "upper_east_side": {
        "low": {"latitude": 40.7600, "longitude": -73.9800},
        "high": {"latitude": 40.7900, "longitude": -73.9400}
    },
    "brooklyn": {
        "low": {"latitude": 40.6000, "longitude": -74.0500},
        "high": {"latitude": 40.7400, "longitude": -73.8500}
    },
    "bronx": {
        "low": {"latitude": 40.7900, "longitude": -73.9300},
        "high": {"latitude": 40.9200, "longitude": -73.7500}
    },
    "queens": {
        "low": {"latitude": 40.6500, "longitude": -73.9600},
        "high": {"latitude": 40.8000, "longitude": -73.7000}
    },
    "staten_island": {
        "low": {"latitude": 40.4900, "longitude": -74.2600},
        "high": {"latitude": 40.6500, "longitude": -74.0500}
    }
}


CATEGORIES = {
    "tourism_attractions": [
        "tourist_attraction",
        "zoo",
        "aquarium",
        "amusement_park",
        "botanical_garden",
        "planetarium",
        "amusement_center"
    ],
    "history_culture_art": [
        "historical_place",
        "museum",
        "cultural_landmark",
        "art_gallery",
        "art_studio",
        "monument",
        "cultural_center",
        "performing_arts_theater"
    ],
    "parks_nature_beach": [
        "park",
        "national_park",
        "hiking_area",
        "garden",
        "wildlife_park",
        "beach"
    ],
    "nightlife": [
        "night_club",
        "casino"
    ],
    "recreation_active": [
        "karaoke",
        "bowling_alley",
        "adventure_sports_center"
    ]
}



ZONE_CATEGORY_ALLOCATION = {

    "lower_manhattan": {
    "target_pois": 200,
    "total_searches": 11,
    "category_weights": {
        "history_culture_art": 0.40,
        "tourism_attractions": 0.30,
        "parks_nature_beach": 0.15,
        "nightlife": 0.10,
        "recreation_active": 0.05
    }    
},
    "midtown": {
    "target_pois": 250,
    "total_searches": 13,
    "category_weights": {
        "history_culture_art": 0.35,
        "tourism_attractions": 0.30,
        "parks_nature_beach": 0.10,
        "nightlife": 0.15,
        "recreation_active": 0.10
    }
},
    "upper_east_side": {
    "target_pois": 130,
    "total_searches": 7,
    "category_weights": {
        "history_culture_art": 0.45,
        "parks_nature_beach": 0.25,
        "tourism_attractions": 0.15,
        "nightlife": 0.05,
        "recreation_active": 0.10
    }
},
    "upper_west_side": {
    "target_pois": 130,
    "total_searches": 7,
    "category_weights": {
        "history_culture_art": 0.40,
        "parks_nature_beach": 0.35,
        "tourism_attractions": 0.10,
        "nightlife": 0.10,
        "recreation_active": 0.05
    }
},
    "brooklyn": {
    "target_pois": 170,
    "total_searches": 8,
    "category_weights": {
        "parks_nature_beach": 0.30,
        "nightlife": 0.25,
        "history_culture_art": 0.20,
        "tourism_attractions": 0.15,
        "recreation_active": 0.10
    }
},
    "bronx": {
    "target_pois": 100,
    "total_searches": 6,
    "category_weights": {
        "tourism_attractions": 0.40,
        "parks_nature_beach": 0.35,
        "history_culture_art": 0.15,
        "nightlife": 0.05,
        "recreation_active": 0.05
    }
},
    "queens": {
    "target_pois": 100,
    "total_searches": 6,
    "category_weights": {
        "parks_nature_beach": 0.40,
        "tourism_attractions": 0.30,
        "history_culture_art": 0.15,
        "nightlife": 0.05,
        "recreation_active": 0.10
    }
},
    "staten_island": {
    "target_pois": 60,
    "total_searches": 4,
    "category_weights": {
        "tourism_attractions": 0.50,
        "parks_nature_beach": 0.30,
        "history_culture_art": 0.10,
        "nightlife": 0.05,
        "recreation_active": 0.05
    }
}
}

def split_rectangle(low, high, n):
    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)

    lat_step = (high["latitude"] - low["latitude"]) / rows
    lng_step = (high["longitude"] - low["longitude"]) / cols

    rects = []
    for i in range(rows):
        for j in range(cols):
            if len(rects) >= n:
                break
            rects.append({
                "low": {
                    "latitude": low["latitude"] + i * lat_step,
                    "longitude": low["longitude"] + j * lng_step
                },
                "high": {
                    "latitude": low["latitude"] + (i + 1) * lat_step,
                    "longitude": low["longitude"] + (j + 1) * lng_step
                }
            })
    return rects


import math

def split_rectangle_into_circles(low, high, n):
    """
    Split a rectangle into n circular subzones suitable for Nearby Search.
    Returns: list of {center: {lat,lng}, radius}
    """

    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)

    lat_step = (high["latitude"] - low["latitude"]) / rows
    lng_step = (high["longitude"] - low["longitude"]) / cols

    circles = []

    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break

            center_lat = low["latitude"] + (i + 0.5) * lat_step
            center_lng = low["longitude"] + (j + 0.5) * lng_step

            # Approx meters (NYC latitude-safe approximation)
            lat_m = lat_step * 111_000
            lng_m = lng_step * 85_000

            radius = int(max(lat_m, lng_m) / 2)

            circles.append({
                "center": {
                    "latitude": center_lat,
                    "longitude": center_lng
                },
                "radius": max(radius, 200)  # Google rejects very small radii
            })

    return circles


def nearby_search(circle,category):
    '''
    Search for POIs in a zone and category
    '''

    payload = {

        "includedPrimaryTypes": CATEGORIES[category],
        "locationRestriction": {
            "circle": circle
        },

        "maxResultCount": 20,
        "languageCode": "en",
        "rankPreference": "POPULARITY"
    }

    try:
        response = requests.post(SEARCH_URL, headers=SEARCH_HEADERS, json = payload)
        response.raise_for_status()
        return response.json().get('places',[])
    except requests.exceptions.RequestException as e:
        print(f'Error for {category}:{e}')
        return []



def fetch_place_details(place_id):
    '''
    Fetch detailed info for a POI
    '''
    url = f'{DETAILS_URL}{place_id}'

    try:
        response = requests.get(url, headers=DETAILS_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Details error for {place_id}: {e}")
        return None


def fetch_places_ids():
    '''
    Collect POI IDs from all zones and categories
    Returns: dict of place_id and basic POI data
    '''
    all_pois={}
    min_user_ratings = 100
    for zone, bounds in ZONES.items():
        print(f'\nCollecting POIs for {zone.upper()}')
        zone_pois=set()
        zone_config = ZONE_CATEGORY_ALLOCATION[zone]
        zone_target = zone_config["target_pois"]
        zone_searches = zone_config["total_searches"]
        subzones = split_rectangle_into_circles(bounds["low"], bounds["high"], zone_searches)
        category_counts = {cat: 0 for cat in zone_config["category_weights"].keys()}
        category_target = {cat: int(zone_target*weight) for cat,weight in zone_config["category_weights"].items()}
        category_list = list(zone_config["category_weights"].items())
        for subzone_idx,circle in enumerate(subzones):
            print(f'\nSubzone {subzone_idx+1}/{len(subzones)}')
            rotated_categories = (
                category_list[subzone_idx % len(category_list):] +
                category_list[:subzone_idx % len(category_list)]
            )           
            if all(count >= category_target[cat] 
               for cat, count in category_counts.items()):
                print('All categories full')
                break
            for category, weight in rotated_categories:
                if category_counts[category] >= category_target[category]: continue
                places = nearby_search(circle,category)
                for p in places:
                    place_id = p['id']
                    if place_id in zone_pois:
                        all_pois[place_id]['categories'].add(category)
                        continue  
                    if p.get("userRatingCount", 0) < min_user_ratings:
                        continue
                    if p.get("rating", 0) < 3.3:
                        continue
                    category_counts[category] +=1
                    zone_pois.add(place_id)
                    all_pois[place_id] = {
                            **p,  
                            'zone': zone,
                            'categories': {category}
                        }
                    if category_counts[category] >= category_target[category]: break
                time.sleep(0.2)
           
            for cat,count in category_counts.items():
                print(f'{cat}: {count}/{category_target[cat]}')
            if len(zone_pois) >= zone_target: break
        print(f'Total POIs for {zone} are {len(zone_pois)}\n')
        print(f'{zone} Overall category vs counts')
        for cat,count in category_counts.items():
            print(f'{cat}: {count}/{category_target[cat]}')
        time.sleep(0.5)
    print(f'TOTAL: {len(all_pois)} unique POIs')
    return all_pois

def build_places_database(pois_basic):
    '''
    Fetch detailed information for all POIs
    Returns: organized database by category
    '''
    db=defaultdict(list)
    min_reviews = 100

    for place_id, basic_data in pois_basic.items():


        details = fetch_place_details(place_id)

        if details is None:
           
            print(f"Failed to fetch details for {basic_data['displayName'].get('text', 'Unknown')}")

            continue

        business_status = details.get('businessStatus', 'OPERATIONAL')
        user_rating_count = basic_data.get('userRatingCount', 0)

        if business_status in ['CLOSED_PERMANENTLY','CLOSED_TEMPORARILY']:
            print(f"Removed {business_status} for {basic_data['displayName'].get('text', 'Unknown')}")
            continue

        if user_rating_count < min_reviews:
            print(f"Removed {user_rating_count} for {basic_data['displayName'].get('text', 'Unknown')}")
            continue


        full_details = {
            **basic_data,
            **details,
            'categories': list(basic_data['categories'])
        }

        for cat in basic_data['categories']:
            db[cat].append({**full_details})
        time.sleep(0.1)

    for category, data in db.items():
        print(f'{category} {len(data)} POIs')
   
    return dict(db)

def save_database(db, filename='nyc_pois_database.json'):
    '''
    Save database to JSON file
    '''

    flat_db = []
    for category, pois in db.items():
        flat_db.extend(pois)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(flat_db, f, indent=2, ensure_ascii=False)

    print(f"\n Database saved to {filename}")
    print(f"  Total POIs: {len(flat_db)}")

    # save category-organized version
    category_filename = 'nyc_pois_by_category.json'
    with open(category_filename, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f" Category-organized database saved to {category_filename}")



def main():
    """Main execution flow"""

    print("Starting NYC POI Database Collection")
   

    # Step 1: Collect POI IDs and basic info
    print("\nSTEP 1: Collecting POI IDs from Nearby Search...")
    pois_basic = fetch_places_ids()

    # Step 2: Fetch detailed information
    print("\nSTEP 2: Fetching detailed information...")
    database = build_places_database(pois_basic)

    # Step 3: Save to files
    print("\nSTEP 3: Saving database...")
    save_database(database)

    # print("\n" + "="*60)
    print("COLLECTION COMPLETE!")
    # print("="*60)

    # Print summary statistics
    print("\nFINAL STATISTICS:")
    total_pois = sum(len(pois) for pois in database.values())
    print(f"Total POIs in database: {total_pois}")
    print(f"Categories covered: {len(database)}")
    print(f"Zones covered: {len(ZONES)}")

    # Show sample POI
    if database:
        sample_category = list(database.keys())[0]
        sample_poi = database[sample_category][0]
        print(f"\nSample POI structure:")
        print(json.dumps(sample_poi, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
