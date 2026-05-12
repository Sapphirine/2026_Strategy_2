
import logging, json, uuid
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

travel_data = None
total_pois = None
ID_to_POI = None

def initialize(_travel_data):
    global travel_data, total_pois, ID_to_POI
    travel_data = _travel_data
    total_pois = travel_data['selected_pois']
    ID_to_POI = {
    poi['id'] : poi
    for poi in total_pois}


DAY_START_HOUR = 9
DAY_CAPACITY_MIN = 780
MEAL_BREAK_MIN = 60
DAY_START_MIN = DAY_START_HOUR * 60
MEAL_AFTER_MIN = DAY_START_MIN + 180


def get_exact_hours(poi, day):
    '''
    Returns (open_min, close_min) in minutes from midnight for the specific day.
    Falls back to full day if no data.
    '''

    periods = (poi.get('regularOpeningHours') or {}).get('periods', [])
    day_p   = [p for p in periods if p.get('open', {}).get('day') == day]
    if not day_p :
        return (DAY_START_MIN, DAY_START_MIN + DAY_CAPACITY_MIN)

    p = day_p[0]
    o = p.get('open',{})
    c = p.get('close',{})
    # Overnight closing 
    if c.get('day', o.get('day')) != o.get('day'):
        close_min = DAY_START_MIN + DAY_CAPACITY_MIN
    else:
        close_min = min(DAY_START_MIN + DAY_CAPACITY_MIN, c.get('hour', 21) * 60 + c.get('minute', 0)) 
    # night-only period
    if close_min <=0 : # skip this period
        return (DAY_START_MIN, DAY_START_MIN)
    
    open_min = o.get('hour', DAY_START_HOUR) * 60 + o.get('minute', 0) 
    
    return (open_min, close_min)

def mins_to_hhmm(mins):
    'Convert absolute minutes from midnight to HH:MM string.'
    h, m = divmod(mins, 60)
    return f'{h % 24:02d}:{m:02d}'

def make_meal_break(start_min):
    return {
        'poi_id' : 'meal_break',
        'poi_name' : 'Meal Break',
        'zone': '',
        'primaryType': 'meal_break',
        'start_time': mins_to_hhmm(start_min),
        'end_time' : mins_to_hhmm(start_min + MEAL_BREAK_MIN),
        'start_min' : start_min,
        'end_min' : start_min + MEAL_BREAK_MIN,
        'visit_duration_min' : MEAL_BREAK_MIN,
        'travel_to_next_min' : 0,
        'flags' : [],
    }

# SCHEDULE ONE DAY
def schedule_day(route, day):
    '''
    Walk through ordered POI sequence for one day.
    Assigns exact clock times, inserts meal break after first POI
    that ends past 12PM, verifies opening hours.

    Returns (scheduled_route, violations, summary)
    '''
    if not route:
        return [], [], {
        'total_pois' : 0,
        'total_travel_min' : 0,
        'total_visit_min' : 0,
        'day_start_time' : '-',
        'day_end_time' : '-',
        'violations': []
        }
    scheduled = []
    violations = []
    current_min = DAY_START_MIN
    meal_inserted = False
    total_travel = 0
    total_visit = 0

    for i, stop in enumerate(route):
        poi_id = stop['poi_id']
        poi_name = stop["poi_name"]
        poi = ID_to_POI[poi_id]
        visit_dur = stop["visit_duration_min"]
        travel_prev = stop["travel_from_prev_min"]
        flags = list(stop.get('flags', []))

        # Arrival after travel
        arrival_min = current_min + travel_prev
        total_travel += travel_prev

        # Exact opening hours for this specific day
        opens_min, closes_min = get_exact_hours(poi, day)

        # Wait if arriving before opening
        if arrival_min < opens_min:
            arrival_min = opens_min


        # Meal Break : insert before this POI if past 12PM and not yet inserted
        if not meal_inserted and arrival_min >= MEAL_AFTER_MIN:
            # insert meal break starting at current_min (before travel and arrival)
            meal_start = current_min
            scheduled.append(make_meal_break(meal_start))
            current_min = meal_start + MEAL_BREAK_MIN
            # recalculate arrival after meal break
            arrival_min  = current_min + travel_prev
            meal_inserted = True
            if arrival_min < opens_min:
                arrival_min = opens_min

        end_min = arrival_min + visit_dur
        total_visit += visit_dur

        # Violation checks

        # POI assigned on closed day
        open_days = {
            p['open']['day']
            for p in (poi.get('regularOpeningHours') or {}).get('periods', [])
            if 'open' in p
        } or set(range(7))

        if day not in open_days:
            flags.append('assigned on closed day')
            violations.append({
                'poi_id': poi_id, 'poi_name': poi_name,
                'issue': 'assigned on closed day',
                'detail': f'POI closed on day={day}'
            })

        # Visit starts before opening
        if arrival_min < opens_min - 60:
            flags.append('arrives before opening')
            violations.append({
                'poi_id': poi_id, 'poi_name': poi_name,
                'issue': 'arrives before opening',
                'detail': f'arrival {mins_to_hhmm(arrival_min)} < opens {mins_to_hhmm(opens_min)}'
            })

        # Visit ends after closing
        if end_min > closes_min:
            flags.append('visit ends after closing-visit on another day')
            violations.append({
                'poi_id': poi_id, 'poi_name': poi_name,
                'issue': 'visit ends after closing-visit on another day',
                'detail': f'end {mins_to_hhmm(end_min)} > closes {mins_to_hhmm(closes_min)}'
            })

        # Day capacity exceeded
        if end_min > DAY_START_MIN + DAY_CAPACITY_MIN:
            flags.append('exceeds day capacity')
            violations.append({
                'poi_id': poi_id, 'poi_name': poi_name,
                'issue': 'exceeds day capacity',
                'detail': f'end {mins_to_hhmm(end_min)} exceeds day limit'
            })

        # Travel to next POI
        is_last = (i==len(route)-1)
        if not is_last:
            next_stop = route[i+1]
            travel_next = next_stop["travel_from_prev_min"]
        else:
            travel_next = 0

        scheduled.append({
            'poi_id':             poi_id,
            'poi_name':           poi_name,
            'poi' :               poi,
            'start_time':         mins_to_hhmm(arrival_min),
            'end_time':           mins_to_hhmm(end_min),
            'start_min':          arrival_min,
            'end_min':            end_min,
            'visit_duration_min': visit_dur,
            'travel_to_next_min': travel_next,
            'flags':              flags,

        })

        current_min = end_min

    # End-of-day meal break if never inserted (all POIs before noon)
    if not meal_inserted and len(route)>0:
        meal_start = current_min
        scheduled.append(make_meal_break(meal_start))
        current_min += meal_start + MEAL_BREAK_MIN

    summary = {
        'total_pois' : len(route),
        'total_travel_min' : total_travel,
        'total_visit_min' : total_visit,
        'day_start_time' : scheduled[0]['start_time'] if scheduled else '-',
        'day_end_time' : mins_to_hhmm(current_min),
        'violations': violations
    }
    return scheduled, violations, summary


def build_itinerary_text (parsed_query, itinerary, trip_days, score):
    group = parsed_query.get('group_type', 'solo')
    days = parsed_query.get('trip_duration_days', 3)
    trip_day_indices = ', '.join(str(i) for i in trip_days)
    pace = parsed_query.get('pace', 'moderate')
    zones = ', '.join(parsed_query.get('preferred_zones', [])) or 'any'
    categories = ', '.join(parsed_query.get('preferred_categories', [])) or 'any'
    types = ', '.join(parsed_query.get('preferred_types', [])) or 'any'
    implicit_needs = ', '.join(parsed_query.get('implicit_needs', [])) or 'none'
    exclude_types = ', '.join(parsed_query['hard_constraints'].get("must_exclude_types", [])) or 'none'
    must_include_pois = ', '.join(parsed_query['hard_constraints'].get("must_include_pois", [])) or 'none'
    soft_constraints = parsed_query.get('soft_constraints', {})
    crowd  = "less crowded" if soft_constraints.get("prefer_less_crowded") else "any crowd"
    time_prefs =[]
    for slot, prefs in parsed_query.get('time_preferences',{}).items():
        pref, avoid = ', '.join(prefs.get('prefer', [])), ', '.join(prefs.get('avoid', []))
        if pref: time_prefs.append(f'prefer {pref} in {slot}')
        if avoid : time_prefs.append(f'avoid {avoid} in {slot}')
    time_prefs = '; '.join(time_prefs) or 'none'

    # Extract POI names day wise
    poi_names = []
    for day_idx, (day_key, route) in enumerate(itinerary.items()):
        day_index = trip_day_indices[day_idx]
        poi_names.append(f'Day index {day_index} -')
        for stop in route:
            poi_names.append(stop.get('poi_name', '') + ',')
        poi_names.append(';')
    poi_names = ' '.join(poi_names)

    return (
        f"Group : {group} trip; "
        f"Duration : {days} days; "
        f"Pace : {pace}; "
        f"Day indices : {trip_day_indices}; "
        f"Zone preference : {zones}; "
        f"Category preference : {categories}; "
        f"Type preference : {types}; "
        f"Implicit needs : {implicit_needs}; "
        f"Crowd preference : {crowd}; "
        f"Types excluded : {exclude_types}; "
        f"Must include POIs : {must_include_pois}; "
        f"Time windows preferences : {time_prefs}; "
        f"Places visited on each day index : {poi_names} "
        f"Satisfaction Score : {score} "
    )

def store_completed_itinerary(parsed_query, itinerary, trip_days, metrics, store_path):
    'Stores real, constraint-satisfying itinerary in knowledge base useful for RAG'
    try:
        with open(store_path, 'r') as f:
            kb = json.load(f)
    except FileNotFoundError:
        kb = []

    record = {
        "itinerary_id" : str(uuid.uuid4())[:8],
        'query_profile' : parsed_query,
        'trip_day_indices' : trip_days,
        'itinerary' : itinerary,
        'evaluation_score' : metrics,
        'itinerary_text' : build_itinerary_text(parsed_query, itinerary, trip_days, metrics['satisfaction_score'])
    }
    kb.append(record)

    with open(store_path, 'w') as f:
        json.dump(kb, f, indent = 2)

    print(f'Stored itinerary #{len(kb)} in knowledge base')


def temporalScheduling(sequence_output, store_path = 'itineraries_history.json'):
    schedule = sequence_output['schedule']
    trip_days = sequence_output['trip_days']
    matrix = sequence_output['matrix_raw']
    location_ids = sequence_output['location_ids']
    metrics = sequence_output['metrics']
    pq = sequence_output['parsed_query']
    depot = sequence_output['depot']


    itinerary = {}
    day_summaries = {}
    all_violations = []

    for day_idx, (day_key, route) in enumerate(schedule.items()):
        day = trip_days[day_idx]

        scheduled, violations, summary = schedule_day(route, day)

        itinerary[day_key] = scheduled
        day_summaries[day_key] = summary
        all_violations.extend(violations)

        if violations:
            logger.warning('%s: %d violation(s)', day_key, len(violations))
        else:
            logger.info('%s: clean — %d POIs | travel %d min | visit %d min',
                        day_key, summary['total_pois'],
                        summary['total_travel_min'], summary['total_visit_min'])

    feasible = len(violations)==0
    store_completed_itinerary(pq, itinerary, trip_days, metrics, store_path)

    return {
        'parsed_query':  pq,
        'trip_day_indices' : trip_days,
        'depot' :  depot,
        'itinerary':     itinerary,
        'day_summaries': day_summaries,
        'all_violations': all_violations,
        'total_violations': len(all_violations),
        'metrics' :  metrics,
        'feasible':      feasible,
        'solver_status': sequence_output.get('solver_status', 'FEASIBLE'),
    }

