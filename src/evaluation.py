
import csv
import json
import math
import pandas as pd
import folium
from folium import plugins
import logging
from collections import defaultdict
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)



EARTH_RADIUS_KM   = 6371.0
NYC_AVG_SPEED_KMH = 5.0
MIN_TRAVEL_MINS   = 4

DAY_START_HOUR  = 9          # touring day starts 9 AM
DAY_CAPACITY_MIN  = 780        # 9 AM → 10 PM  (780 min window)

# Python weekday → Google weekday  (Mon=0 → 1, … Sun=6 → 0)
PYTHON_TO_GOOGLE_DAY = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
GOOGLE_DAY_NAMES     = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

# Time-slot bounds in minutes from 9 AM 
TIME_SLOT_BOUNDS = {
    'morning':   (0,   180),
    'afternoon': (180, 480),
    'evening':   (480, DAY_CAPACITY_MIN),
}

# Zone centroids used as depot fallback
ZONE_FALLBACK_CENTROIDS = {
    'lower_manhattan':  (40.7128, -74.0060),
    'midtown':          (40.7549, -73.9840),
    'upper_east_side':  (40.7735, -73.9565),
    'upper_west_side':  (40.7870, -73.9754),
    'brooklyn':         (40.6782, -73.9442),
    'bronx':            (40.8448, -73.8648),
    'queens':           (40.7282, -73.7949),
    'staten_island':    (40.5795, -74.1502),
}

DAY_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue']


# Distance meaurement
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def km_to_min(km):
    """km → integer minutes at NYC walking/transit average """
    return max(MIN_TRAVEL_MINS, round((km / NYC_AVG_SPEED_KMH) * 60))


def travel_min_between(loc_a, loc_b):
    """Travel time (minutes) between two {'latitude', 'longitude'} dicts."""
    return km_to_min(haversine_km(
        loc_a['latitude'],  loc_a['longitude'],
        loc_b['latitude'],  loc_b['longitude'],
    ))

def hhmm_to_abs(hhmm):
    """'09:30' → 570  (minutes from midnight)."""
    h, m = map(int, hhmm.strip().split(':'))
    return h * 60 + m


def abs_to_hhmm(mins):
    """570 → '09:30'."""
    return f"{mins // 60:02d}:{mins % 60:02d}"

def abs_to_from9(mins):
    """Absolute minutes-from-midnight → minutes from 9 AM ."""
    return mins - DAY_START_HOUR * 60


def sort_day_keys(keys):
    """Sort 'day_1', 'day_2', … in numeric order."""
    return sorted(keys, key=lambda k: int((k.split('_'))[1]))


# Dataloader
def load_poi_db(path):
    """Load nyc_pois_database.json → {poi_id: poi_dict}."""
    with open(path, 'r') as f:
        pois = json.load(f)
    return {p['id']: p for p in pois}


def load_session(path):
    """Load session_memory.json → full session dict."""
    with open(path, 'r') as f:
        return json.load(f)


# build itinerary from CSV
def build_itinerary(csv_path, poi_db) :
    """
    Reads CSV and constructs a per-day list of enriched stop dicts.

    CSV columns: day_key, google_day, poi_id, arrival_start, arrival_end
    Stop dict keys:
      poi_id, poi_name, poi (full dict), zone, primaryType,
      arrival_start (HH:MM), arrival_end (HH:MM),
      arrival_abs_min, end_abs_min, 
      google_day,    ← for opening-hours checks


    """
    raw = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip().strip("'").strip('"') for name in reader.fieldnames]
        for row in reader:
            raw[row['day_key'].strip()].append(row)

    itinerary = {}

    for dk in sort_day_keys(raw.keys()):

        stops = []

        for row in raw[dk]:
            
            pid = row['poi_id'].strip().strip('"')
            arrival_start = row['arrival_start'].strip().strip('"')
            arrival_end = row['arrival_end'].strip().strip('"')
            google_day = int(row['google_day'].strip().strip('"'))
           
            poi = poi_db.get(pid)
            if poi is None:
                logger.warning("[%s] POI '%s' not found in database — skipping", dk, pid)
                continue

            arr_abs = hhmm_to_abs(arrival_start)
            end_abs = hhmm_to_abs(arrival_end)

            if end_abs <= arr_abs:
                logger.warning(
                    "[%s] '%s': end_time (%s) ≤ arrival_start (%s) — check CSV",
                    dk, pid, arrival_end, arrival_start
                )

            stops.append({
                # Identifiers
                'poi_id':  pid,
                'poi_name':  poi.get('displayName', {}).get('text', pid),
                'poi':  poi,
                'zone':  poi.get('zone', ''),
                'primaryType':  poi.get('primaryType', ''),

                # Times 
                'arrival_start':  arrival_start,
                'arrival_end':  arrival_end,
                'arrival_from9_min':  abs_to_from9(arr_abs),  # from 9 AM

                # Times in min
                'arrival_abs_min':  arr_abs,           # from midnight
                'end_abs_min': end_abs,

                # Weekday
                'google_day':  google_day,


            })

        itinerary[dk] = stops

    total_stops = sum(len(s) for s in itinerary.values())
    logger.info("Itinerary loaded: %d days, %d stops", len(itinerary), total_stops)
    return itinerary

# Depot
def build_depot(itinerary):
    """
    Depot = zone-centroid of the FIRST stop on day_1  (consistent rule).
    Falls back to ZONE_FALLBACK_CENTROIDS → midtown if zone unknown.

    The SAME depot location is used for ALL days.
    """
    first_day  = sort_day_keys(itinerary.keys())[0]
    first_stops = itinerary[first_day]

    if not first_stops:
        zone = 'midtown'
        logger.warning("No stops on %s — using midtown as depot", first_day)
    else:
        zone = first_stops[0]['zone'].lower()

    lat, lon = ZONE_FALLBACK_CENTROIDS.get(zone, ZONE_FALLBACK_CENTROIDS['midtown'])
    logger.info("Depot: zone '%s' centroid → (%.4f, %.4f)", zone, lat, lon)

    return {
        'id':  'depot',
        'zone':     zone,
        'location' : {'latitude': lat, 'longitude': lon},
    }



# travel time
def build_travel_times(itinerary, depot):
    """
    Returns
      travel_per_day: {day_key: [t0, t1, …, tN]}

    Where for a day with k stops:
      t0       = depot → stop_1        (leg from depot)
      t1…t(k-1) = stop_i → stop_{i+1} (internal legs)
      t_k  = stop_k → depot        (return leg)

    """
    depot_loc  = depot['location']
    travel_per_day   = {}

    for dk, stops in itinerary.items():
        if not stops:
            travel_per_day[dk] = []
            continue

        times    = []
        prev_loc = depot_loc

        for stop in stops:
            loc = stop['poi'].get('location', {})
            t   = travel_min_between(prev_loc, loc)
            times.append(t)

            prev_loc = loc
            logger.info('id %s, time from previous %s', stop['poi_id'], t)

        # Return leg: last stop → depot
        t_back = travel_min_between(prev_loc, depot_loc)
        times.append(t_back)

        travel_per_day[dk] = times   # length = len(stops) + 1

    return travel_per_day

# opening hours and open days check
def check_opening_(itinerary):
    """
    For each stop on its assigned google_day, checks:
      (a) arrival_start ≥ POI open time
      (b) arrival_end   ≤ POI close time

    Checks each stop is assigned to a day when the POI is actually open.

    Returns list of violation dicts.
    If no periods exist for that day → skip (assume open).
    """
    violations = []

    for dk, stops in itinerary.items():
        for stop in stops:
            poi   = stop['poi']
            google_day = stop['google_day']
            arr_abs  = stop['arrival_abs_min']
            end_abs  = stop['end_abs_min']
            periods  = (poi.get('regularOpeningHours') or {}).get('periods', [])
            closed_day = False

            # Periods for this specific weekday
            day_period = [
                p for p in periods
                if p.get('open', {}).get('day') == google_day
            ]

            if not day_period:
                continue    # no hours data → assume open 

            open_days = {p.get('open', {}).get('day') for p in periods if 'open' in p}


            issues = []
            if google_day not in open_days:
                closed_day = True
            # Use the open window on that day

            o   = day_period[0].get('open', {})
            c  = day_period[0].get('close', {})
            open_abs  = o.get('hour', DAY_START_HOUR) * 60 + o.get('minute', 0)
            # Overnight close → cap at midnight of same day
            if c.get('day', google_day) != google_day:
                close_abs = 24 * 60
            else:
                close_abs = c.get('hour', 22) * 60 + c.get('minute', 0)

            if arr_abs < open_abs:
                issues.append(
                    f"arrives {stop['arrival_start']} "
                    f"before open {abs_to_hhmm(open_abs)}"
                )
            if end_abs > close_abs:
                issues.append(
                    f"departs {stop['arrival_end']} "
                    f"after close {abs_to_hhmm(close_abs)}"
                )

            if issues:
                violations.append({
                    'day_key':      dk,
                    'poi_id':       stop['poi_id'],
                    'poi_name':     stop['poi_name'],
                    'google_day':   google_day,
                    'issues':       issues,
                    'assigned on closed day' : closed_day,
                })
                stop['violations'] = {
                    'issues':       issues,
                    'assigned on closed day' : closed_day,
                }

    return violations


# Preference adherance
def time_pref_adherence(itinerary, time_prefs):
    """
    Measures how well the itinerary respects parsed_query time preferences.
    """
    pref_total,  pref_met  = 0, 0
    avoid_total, avoid_met  = 0, 0

    for stops in itinerary.values():
        for stop in stops:
            poi_type = stop['primaryType']
            poi_id = stop['poi_id']
            arrival  = stop['arrival_from9_min']

            for slot, prefs in time_prefs.items():
                if slot not in TIME_SLOT_BOUNDS:
                    continue   # skip 'night' key if present
                s_start, s_end = TIME_SLOT_BOUNDS[slot]

                if poi_type in prefs.get('prefer', []):
                    pref_total += 1
                    if s_start <= arrival <= s_end:
                        pref_met += 1
                    else:

                        logger.info('Preference criterion not met for id %s',poi_id)

                if poi_type in prefs.get('avoid', []):
                    avoid_total += 1
                    if not (s_start <= arrival <= s_end):
                        avoid_met += 1
                    else:
                        logger.info('Avoid criterion not met for id %s',poi_id)

    total = pref_total + avoid_total
    return {
        'prefer_adherence_%':   round((pref_met  / pref_total  * 100) if pref_total  else 100.0, 2),
        'avoid_adherence_%':  round((avoid_met / avoid_total * 100) if avoid_total else 100.0, 2),
        'time_pref_combined_%': round(((pref_met + avoid_met) / total * 100) if total else 100.0, 2),
    }


# metrics
def compute_metrics(itinerary, travel_per_day, time_prefs):
    """
    Computes:
      - Total & per-day travel time 
      - Backtracking score
      - Cross-zone change rate
      - Day-balance std-dev
      - Preference adherence
      - Opening hours check
      - Assigned on open days check

    """
    #  Travel per day & total
    travel_per_day_min = {dk: sum(times) for dk, times in travel_per_day.items()}
    total_travel  = sum(travel_per_day_min.values())

    #  Backtracking score 
    # Only inter-POI legs (not depot-return)
    inter_legs = []
    for dk, times in travel_per_day.items():
        # times = [depot→s1, s1→s2, …, sN→depot]
        # inter-POI legs = all except the final return leg
        if len(times) > 1:
            inter_legs.extend(times[:-1])

    avg_leg  = sum(inter_legs) / len(inter_legs) if inter_legs else 0
    backtracks   = sum(1 for t in inter_legs if t > avg_leg * 2)
    backtrack_score = (backtracks / len(inter_legs) * 100) if inter_legs else 0.0

    #  Cross-zone changes 
    zone_changes = 0
    total_legs_z = 0
    for stops in itinerary.values():
        for i in range(1, len(stops)):
            total_legs_z += 1
            if stops[i]['zone'] != stops[i - 1]['zone']:
                zone_changes += 1
                logger.info('Zone change: Prev id %s, zone %s | Current id %s, zone %s',
                           stops[i-1]['poi_id'],stops[i - 1]['zone'],
                            stops[i]['poi_id'], stops[i]['zone'])
    cross_zone_rate = (zone_changes / total_legs_z * 100) if total_legs_z else 0.0

    #  Day balance (std-dev of POI counts) 
    day_counts  = [len(stops) for stops in itinerary.values()]
    mean_count  = sum(day_counts) / len(day_counts) if day_counts else 0
    balance_std = math.sqrt(
        sum((c - mean_count) ** 2 for c in day_counts) / len(day_counts)
    ) if day_counts else 0.0

    #  Preference adherence 
    pref_metrics = time_pref_adherence(itinerary, time_prefs)

    # open days and open hours check
    violations = check_opening_(itinerary)
    invalid_days = 0
    total_pois  = 0
    poi_constraints = 0
    for v in violations:
        total_pois += 1
        if v['assigned on closed day']: invalid_days += 1
        if v['issues']: poi_constraints += 1
    invalid_days_assignment = round(((invalid_days / total_pois)*100) if total_pois else 0, 2)
    pois_hours_adherance = round(((poi_constraints/ total_pois)*100) if total_pois else 100, 2)

    metrics= {
        'total_travel_min':   total_travel,
        'travel_per_day_min':   travel_per_day_min,
        'backtracking_score_%':   round(backtrack_score, 2),
        'cross_zone_changes':   zone_changes,
        'cross_zone_rate_%':   round(cross_zone_rate, 2),
        'day_balance_std':        round(balance_std, 2),
        'pois_per_day':   {dk: len(stops) for dk, stops in itinerary.items()},
        'preference_metrics':     pref_metrics,
        'invalid_days_assignment_%' : invalid_days_assignment,
        'pois_hours_adherance_%' : pois_hours_adherance,
    }
    return metrics


# visualize
def plot_eval_map(itinerary, depot, out_path = 'evaluation_map.html'):
    """
    Folium map showing:
      • Depot marker (black home icon)
      • Numbered circle markers per stop, colour-coded by day
      • Directed polyline (arrows) connecting stops in order, starting/ending at depot
      • Popup: POI name · stop # · time range · zone · type · violations 
      • Legend: day colours
    """

    depot_loc  = depot['location']
    map_center = [depot_loc['latitude'], depot_loc['longitude']]

    fmap = folium.Map(location=map_center, zoom_start=13, tiles='CartoDB positron')

    # Depot
    folium.Marker(
        location=map_center,
        popup=folium.Popup(
            f"<b>🏨 Depot</b><br>Zone: {depot.get('zone', '')}", max_width=200
        ),
        icon=folium.Icon(color='black', icon='home', prefix='fa'),
    ).add_to(fmap)

    for day_idx, dk in enumerate(sort_day_keys(itinerary.keys())):
        stops      = itinerary[dk]
        color      = DAY_COLORS[day_idx % len(DAY_COLORS)]
        day_label  = dk.replace('_', ' ').title()
        day_coords = []

        for seq, stop in enumerate(stops, start=1):
            loc = stop['poi'].get('location', {})
            lat, lon = loc.get('latitude'), loc.get('longitude')
            if lat is None or lon is None:
                logger.warning("[%s] No coordinates for '%s' — skipping marker", dk, stop['poi_name'])
                continue

            day_coords.append([lat, lon])
            viol = stop.get('violations', {})
            issues = viol.get('issues', [])
            closed_day = viol.get('assigned on closed day', False)

            popup_html = (
                f"<b>{stop['poi_name']}</b><br>"
                f"📅 {day_label} · Stop {seq}<br>"
                f"🕐 {stop['arrival_start']} – {stop['arrival_end']}<br>"
                f"📍 {stop['zone']} | {stop['primaryType']}<br>"
                + (f"<br><span style='color:red'>⚠️ {", ".join(issues)}</span><br>" if viol else "")
                + (f"<br><span style='color:red'>⚠️ Closed</span><br>" if closed_day else "")
            )

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{day_label} · #{seq} · {stop['poi_name']}",
                icon=folium.DivIcon(
                    html=(
                        f"<div style='background:{color};color:white;"
                        f"border-radius:50%;width:26px;height:26px;"
                        f"text-align:center;line-height:26px;"
                        f"font-weight:bold;font-size:12px;"
                        f"border:2px solid white;"
                        f"box-shadow:0 1px 3px rgba(0,0,0,.4);'>{seq}</div>"
                    ),
                    icon_size=(26, 26),
                    icon_anchor=(13, 13),
                ),
            ).add_to(fmap)

        # Directed polyline: depot → stops → depot
        if day_coords:
            full_path = [map_center] + day_coords + [map_center]
            line = folium.PolyLine(
                locations=full_path,
                color=color, weight=2.5, opacity=0.75,
                tooltip=day_label,
            )
            line.add_to(fmap)
            plugins.PolyLineTextPath(
                line, '     ➤     ', repeat=True, offset=8,
                attributes={'fill': color, 'font-size': '14', 'font-weight': 'bold'},
            ).add_to(fmap)

    # Legend
    legend_items = ''.join(
        f"<li><span style='color:{DAY_COLORS[i % len(DAY_COLORS)]}'>■</span> "
        f"{dk.replace('_', ' ').title()}</li>"
        for i, dk in enumerate(sort_day_keys(itinerary.keys()))
    )
    fmap.get_root().html.add_child(folium.Element(f"""
    <div style='position:fixed;top:90px;left:10px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;min-width:120px;'>
        <b>Trip Days</b>
        <ul style='margin:5px 0;padding-left:0;list-style:none'>{legend_items}</ul>
    </div>
    """))

    fmap.save(out_path)
    logger.info("Map saved → %s", out_path)
    return out_path


def comparison(csv_path, session_path,poi_db_path):
    'Comparison performance metrics of Agent and modern LLMs(ChatGPT)'
    #  Load 
    logger.info("Loading POI database …")
    poi_db = load_poi_db(poi_db_path)

    logger.info("Loading session memory …")
    session    = load_session(session_path)
    pq   = session['parsed_query']

    time_prefs = pq.get('time_preferences', {})

    # build itinerary
    itinerary = build_itinerary(csv_path, poi_db)

    #  Depot 
    depot = build_depot(itinerary)

    #  Travel times (per-day arrays) 
    logger.info("Computing haversine travel times …")
    travel_per_day = build_travel_times(itinerary, depot)

    #  Metrics 
    logger.info("Computing metrics …")
    metrics = compute_metrics(itinerary, travel_per_day, time_prefs)

    #  Map 
    logger.info("Generating evaluation map …")
    plot_eval_map(itinerary, depot)
    itinerary_evaluation = session['itinerary_evaluation']


    comparison_table = {
        'Metric' : ['total_travel_min', 'backtracking_score_%', 'cross_zone_rate_%',
                   'day_balance_std','preference_metrics_%', 'invalid_days_assignment_%', 
                   'pois_hours_adherance_%'],
        'Agent' : [itinerary_evaluation['total_travel_min'],
                  itinerary_evaluation['backtracking_score_%'],
                  itinerary_evaluation['cross_zone_rate'],
                  itinerary_evaluation['day_balance_std'],
                  itinerary_evaluation['preference_metrics'].get('time_pref_combined_%'),
                  itinerary_evaluation['invalid_day_assignment_rate'],
                  itinerary_evaluation['constraint_satisfaction_rate_%']],
        'ChatGPT' : [metrics['total_travel_min'],
                  metrics['backtracking_score_%'],
                  metrics['cross_zone_rate_%'],
                  metrics['day_balance_std'],
                  metrics['preference_metrics'].get('time_pref_combined_%'),
                  metrics['invalid_days_assignment_%'],
                  metrics['pois_hours_adherance_%']],
    }

    return pd.DataFrame(comparison_table)



