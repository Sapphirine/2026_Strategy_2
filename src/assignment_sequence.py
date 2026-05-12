
import json, logging, random
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DAY_START_HOUR = 9
DAY_CAPACITY_MIN = 780
MEAL_BREAK_MIN = 60
USABLE_MINS = DAY_CAPACITY_MIN - MEAL_BREAK_MIN
MAX_SLACK_MIN = 30
SOLVE_TIME_SEC = 60
MUST_PENALTY = 10000000
VISIT_DURATIONS_MIN = {
    'museum': 90,              'zoo': 120,              'aquarium': 90,
    'amusement_park': 90,     'botanical_garden': 60,  'planetarium': 60,
    'amusement_center': 90,    'historical_landmark': 45,'historical_place': 45,
    'monument': 30,            'cultural_landmark': 45, 'art_gallery': 60,
    'art_studio': 45,          'cultural_center': 60,   'performing_arts_theater': 90,
    'tourist_attraction': 60,  'observation_deck': 60,  'park': 45,
    'national_park': 90,       'garden': 45,            'night_club': 90,
    'casino': 90,              'hiking_area' : 120,     'beach' : 45,
    'karaoke' : 60 ,           'bowling_alley': 60 ,    'adventure_sports_center' : 90,
}

DEFAULT_VISIT_MIN = 60
PACE_MAX_POIS =  {'relaxed': 3, 'moderate': 5, 'packed': 7}
DEFAULT_MAX_POIS = 5

# minutes from DAY_START_HOUR (9 AM = 0)
TIME_SLOT_BOUNDS = {
    'morning':   (0,   180),
    'afternoon': (180, 480),
    'evening':   (480, DAY_CAPACITY_MIN)
}


def visit_duration(poi, max_pois):
    '''
    Returns visit duration of each poi
    Adjusts visit duration if pace is relaxed
    '''

    return (VISIT_DURATIONS_MIN.get(poi.get('primaryType',''), DEFAULT_VISIT_MIN) + (30 if max_pois == 3 else 0))



def time_window(poi, trip_days, visit_dur, time_prefs):
    '''
    Check tw_start and tw_end of a poi for a specified day with user's time preferences
    Returns (tw_start, tw_end)
    '''
    periods = (poi.get('regularOpeningHours') or {}).get('periods', [])
    relevant = [p for p in periods if p.get('open', {}).get('day') in set(trip_days)]

    if not relevant:
        tw_start, tw_end = 0, DAY_CAPACITY_MIN  - visit_dur
    else:
        opens, closes = [], []

        for p in relevant:
            o = p.get('open', {})
            c = p.get('close', {})
            # Overnight closing 
            if c.get('day', o.get('day')) != o.get('day'):
                close_min = DAY_CAPACITY_MIN
            else:
                close_min = min(DAY_CAPACITY_MIN, c.get('hour', 21) * 60 + c.get('minute', 0) - DAY_START_HOUR *60) 
            # night-only period
            if close_min <=0 : continue # skip this period
            
            open_min = max(0, o.get('hour', DAY_START_HOUR) * 60 + o.get('minute', 0) - DAY_START_HOUR * 60)

            opens.append(open_min)
            closes.append(close_min)
        # all periods night-only
        if not closes: return None
            
        tw_start = max(0, min(opens))
        tw_end = min(DAY_CAPACITY_MIN, max(closes))-visit_dur

    # apply time-of-day preferences
    poi_type = poi.get('primaryType', '')
    for slot, prefs in time_prefs.items():
        s_start, s_end = TIME_SLOT_BOUNDS[slot]

        if poi_type in prefs.get('prefer', []):
            if tw_start <= s_end:
                tw_start = max(s_start, tw_start)

            else:
                logger.warning("POI '%s' preferred in '%s' but opens after slot.",
                               poi.get('displayName', {}).get('text', ''), slot)
        if poi_type in prefs.get('avoid', []):
            before_slot = max(0, min(s_start, tw_end) - tw_start)
            after_slot  = max(0, tw_end - max(s_end, tw_start))


            if before_slot == 0 and after_slot == 0:
                # avoided slot covers entire window — cannot honour preference
                logger.warning(
                    "POI '%s' cannot avoid '%s' — avoided slot covers entire window.",
                    poi.get('displayName', {}).get('text', ''), slot
                )
            elif before_slot >= after_slot:
                # larger window is before the avoided slot
                tw_end = min(tw_end, s_start)

            else:
                # larger window is after the avoided slot
                tw_start = max(tw_start, s_end)

    if tw_end < 0:
        tw_end = DAY_CAPACITY_MIN - visit_dur  
        
    if tw_end <= tw_start:
        tw_start = 0
        tw_end   = DAY_CAPACITY_MIN - visit_dur
        logger.warning(
            "POI '%s' invalid window after preference logic — reset to full day",
            poi.get('displayName', {}).get('text', ''), 
        )


    return (int(tw_start), int(tw_end))


def build_model_data(travel_data):
    '''
    Build time_window for each POI for each of the trip days
    Build travel_time matrix with values for each open day of POI
    Returns dict containing matrix, visit durations, time windows for all POIs for all open days
    '''
    pq = travel_data['parsed_query']
    selected_pois = travel_data['selected_pois']
    matrix_raw = travel_data['matrix_raw']
    location_ids = travel_data['location_ids']
    trip_days = travel_data['trip_day_indices']
    time_prefs = pq.get('time_preferences', {})
    pace = pq.get('pace', 'moderate')
    must_names = {
        m.lower().strip()
        for m in pq['hard_constraints'].get('must_include_pois',[])
    }


    must_poi_ids = {
        poi['id'] for poi in selected_pois
        if any(m in poi.get('displayName', {}).get('text', '').lower() for m in must_names)
    }

    poi_allowed_days = {}   # poi_id -> list of vehicle indices
    for poi in selected_pois:
        open_days = {
            p['open']['day']
            for p in (poi.get('regularOpeningHours') or {}).get('periods', [])
            if 'open' in p
        } or set(range(7))   # no hours data -> open all days

        allowed = [
            day for day_idx, day in enumerate(trip_days)
            if day in open_days
        ]
        poi_allowed_days[poi['id']] = allowed

    max_pois = PACE_MAX_POIS.get(pace, DEFAULT_MAX_POIS)
    min_pois = max(1, max_pois-1)


    # visit durations: index 0 = depot = 0, then one per POI
    visit_durations = [0] + [visit_duration(p, max_pois) for p in selected_pois]

    # time windows: index 0 = depot (full day), then one per POI
    time_windows = [(0, DAY_CAPACITY_MIN)]
    for poi, vd in zip(selected_pois, visit_durations[1:]):
        time_windows.append(time_window(poi, trip_days, vd, time_prefs))

    logger.info(
        "Model: %s POIs | %s days | %s must_poi_ids",
        len(selected_pois),  len(trip_days), must_poi_ids
    )

    return {
        'num_nodes':        len(location_ids),   
        'num_days':         pq['trip_duration_days'],
        'matrix':           matrix_raw,
        'visit_durations':  visit_durations,
        'time_windows':     time_windows,
        'max_pois_per_day': max_pois,
        'min_pois_per_day': min_pois,
        'location_ids':     location_ids,
        'selected_pois':    selected_pois,
        'must_poi_ids':     must_poi_ids,
        'poi_by_node':      {i + 1: p for i, p in enumerate(selected_pois)},
        'poi_allowed_days' : poi_allowed_days
    }



# OR-TOOLS CALLBACKS
class Callbacks:
    def __init__(self, matrix, visit_durations, manager):
        self.m = matrix
        self.vd = visit_durations
        self.mgr = manager

    def time_transit(self, from_idx, to_idx):
        'Arc cost = service time at from_node + travel to to_node.'
        f = self.mgr.IndexToNode(from_idx)
        t = self.mgr.IndexToNode(to_idx)
        return self.m[f][t] + self.vd[f] # travel time + visit duration

    def unit_count(self, from_idx):
        'Accumulates to 1 per POI visited (depot = 0).'
        return 0 if self.mgr.IndexToNode(from_idx)==0 else 1



# BUILD AND SOLVE OR-TOOLS MODELS

def solve(data, mode = 'enforce_must', VehicleVar = True):
    num_nodes = data['num_nodes']
    num_days = data['num_days']
    matrix = data['matrix']
    vd = data['visit_durations']
    tws = data['time_windows']
    poi_allowed_days = data['poi_allowed_days']
    poi_by_nd = data['poi_by_node']
    must_poi_ids = data['must_poi_ids']
    max_p = data['max_pois_per_day']
    min_p = data['min_pois_per_day']

    # Define nodes
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_days, 0) # node 0: depot

    # Create graph of nodes -> builds edges between nodes, possible routes, constraints
    model = pywrapcp.RoutingModel(manager)

    # Define cost rule -> sum of all edge costs
    cb = Callbacks(matrix, vd, manager)
    transit_idx = model.RegisterTransitCallback(cb.time_transit)
    model.SetArcCostEvaluatorOfAllVehicles(transit_idx)

   
    model.AddDimension(
        transit_idx,
        MAX_SLACK_MIN,
        DAY_CAPACITY_MIN,
        True,
        'Time'
    )
    time_dim = model.GetDimensionOrDie('Time')

    # time windows per POI (node 0 = depot, skip)
    # Time window Constraint maintained
    for node in range(1,num_nodes):
        tw = tws[node]
        if tw is not None:
            tw_s, tw_e = tws[node]
            time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(tw_s, tw_e)
            
     # Each vehicle returns to depot by DAY_CAPACITY_MIN
    for v in range(num_days):
        time_dim.CumulVar(model.End(v)).SetRange(0, DAY_CAPACITY_MIN)

    # ── Count dimension: pace + balance 
    # Max POIs per day enforces pace.
    # Add 1 if POI visited (from->to)
    # Constraint  POIs per day <= max_p
    # Slack = 0, no waiting needed for counting
    count_idx = model.RegisterUnaryTransitCallback(cb.unit_count)
    model.AddDimension(count_idx, 0, max_p, True, 'Count')
    count_dim = model.GetDimensionOrDie('Count')

    # ── Disjunctions
    # All nodes are optional; penalty controls priority.
    # if enforce_must = False, all nodes - score-weighted
    for node in range(1, num_nodes):

        poi   = data['poi_by_node'][node]

        score = poi.get('combined_score', 0.5)
        index = manager.NodeToIndex(node)

        # must_nodes: MUST_PENALTY (10M >> max travel cost ~800 min) = never dropped if enforce_must = True
        if poi['id'] in must_poi_ids and mode == 'enforce_must':
            if tws[node] is None:
                logger.warning(
                    "Must-include POI '%s' is night-only — cannot be scheduled. "
                    "Consider relaxing constraints.",
                    poi.get('displayName', {}).get('text', '')
                    )
                penalty = int(score * 10000) + 1000
            else:
                penalty = MUST_PENALTY
        else:
            # others : score-weighted (higher combined_score = higher penalty = preferred)
            penalty = int(score * 10000) + 1000

        # cost of visiting or penalty for skipping
        model.AddDisjunction([index], penalty)
        # Hard-deactivate: OR-Tools will never visit this node
        if tws[node] is None:                       
            model.ActiveVar(index).SetMax(0)

    if VehicleVar:
        for node in range(1, num_nodes):
            poi_id  = poi_by_nd[node]['id']
            allowed = poi_allowed_days.get(poi_id, list(range(num_days)))
            if len(allowed) < num_days:
                # only restrict if not open all days — avoids unnecessary constraint
                model.VehicleVar(manager.NodeToIndex(node)).SetValues(allowed)

    
    # Search parameters
    sp = pywrapcp.DefaultRoutingSearchParameters()

    sp.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    # explore alternatives, rearrange POIs
    sp.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # stop searching
    sp.time_limit.seconds = SOLVE_TIME_SEC


    solution = model.SolveWithParameters(sp)
    feasible = solution is not None

    if feasible:
        logger.info('Solver FEASIBLE | Objective: %d', solution.ObjectiveValue())
    else:
        logger.error('Solver INFEASIBLE - check time windows and day capacity |  mode=%s', mode)

    return {
        'solution' : solution,
        'model' : model,
        'manager' : manager,
        'time_dim' : time_dim,
        'count_dim' : count_dim,
        'feasible' : feasible,
        'mode' : mode
    }

def time_pref_adherence(schedule, time_prefs, trip_days):
    '''
    Measures how well scheduled POIs respect user time preferences.
    prefer: POI should arrive within preferred slot
    avoid:  POI should not arrive within avoided slot
    '''

    pref_total, pref_met = 0, 0
    avoid_total , avoid_met = 0, 0

    for route in schedule.values():
        for stop in route:
            poi_type = stop['primaryType']
            arrival  = stop['arrival_time_min']

            for slot, prefs in time_prefs.items():
                s_start, s_end = TIME_SLOT_BOUNDS[slot]
                if poi_type in prefs.get('prefer',[]):
                    pref_total += 1
                    if s_start <= arrival <= s_end:
                        pref_met += 1
                if poi_type in prefs.get('avoid',[]):
                    avoid_total += 1
                    if not (s_start <= arrival <= s_end):
                        avoid_met += 1

    prefer_rate = (pref_met / pref_total * 100) if pref_total else 100.0
    avoid_rate  = (avoid_met  / avoid_total  * 100) if avoid_total  else 100.0

    total = pref_total + avoid_total
    combined = (
        (pref_met + avoid_met) / total * 100
        if total else 100.0
    )

    return {
        'prefer_adherence_%': round(prefer_rate, 2),
        'avoid_adherence_%':  round(avoid_rate,  2),
        'time_pref_combined_%': round(combined,  2),
    }



def naive_travel_time(all_indices, naive_pois_per_day, trip_days, visited, matrix):
    'Unoptimized travel time'
    total_trial_travel = []
    for _ in range(10): # 10 trails
        # random picking of POIs
        picked = random.sample(all_indices, min(visited+5, len(all_indices)))
        random.shuffle(picked)
        trial_travel = 0
        ptr = 0
        for _ in range(len(trip_days)):
            day_route = picked[ptr : ptr + naive_pois_per_day]
            ptr += naive_pois_per_day
            if not day_route:
                continue
            # depot-> first POI
            trial_travel += matrix[0][day_route[0]]

            # internal travel
            for i in range(len(day_route)-1):
                trial_travel += matrix[day_route[i]][day_route[i+1]]

            # last POI-> depot
            trial_travel += matrix[day_route[-1]][0]
        total_trial_travel.append(trial_travel)

    avg = sum(total_trial_travel)/ len(total_trial_travel)

    return avg, max(total_trial_travel)



def compute_satisfaction_score(metrics):
    '''
    Weighted satisfaction score in [0, 1].
    Hard constraints weighted higher than soft preferences.
    '''
    pref = metrics.get('preference_metrics', {})
    components = {
        # Hard constraints 
        'constraint_satisfaction_rate_%': (metrics.get('constraint_satisfaction_rate_%', 100), 0.20),
        'must_include_satisfaction_%': (metrics.get('must_include_satisfaction_%', 100), 0.20),
        'unique_visit_rate':  (metrics.get('unique_visit_rate', 100), 0.10),
        'invalid_day_assignment_rate':  (100 - metrics.get('invalid_day_assignment_rate', 0), 0.10),

        # Soft optimization quality
        'feasibility_rate_%':  (metrics.get('feasibility_rate_%', 100), 0.10),
        'poi_coverage_rate_%':  (metrics.get('poi_coverage_rate_%', 80), 0.05),
        'distance_reduction_%':  (min(metrics.get('distance_reduction_%', 0), 100), 0.05),
        'cross_zone_rate':  (100 - metrics.get('cross_zone_rate', 0), 0.05),
        'day_balance_std':   (max(0, 100 - metrics.get('day_balance_std', 0)),0.05),

        # User preferences 
        'prefer_adherence_%': (pref.get('prefer_adherence_%', 100),  0.05),
        'avoid_adherence_%': (pref.get('avoid_adherence_%',  100), 0.05),
    }


    score = sum(val * w for val, w in components.values()) / 100
    return round(min(max(score, 0.0), 1.0), 4)


def evaluate(schedule, data, travel_data):
    trip_days = travel_data['trip_day_indices']
    selected_pois = data['selected_pois']
    total_pois = len(selected_pois)
    visited = sum(len(r) for r in schedule.values())
    must_poi_ids = data['must_poi_ids']
    matrix = data['matrix']
    location_ids = travel_data['location_ids']
    poi_allowed_days = data['poi_allowed_days']
    poi_index_map = travel_data['poi_index_map']
    time_prefs = travel_data['parsed_query'].get('time_preferences', {})

    # Constraint Satisfaction Rate
    violations = 0
    total_checks = 0

    for day_idx , (day_key, route) in enumerate(schedule.items()):
        day = trip_days[day_idx]
        for stop in route:
            tw_s, tw_e = stop['time_window']
            arrival = stop['arrival_time_min']
            visit_dur = stop["visit_duration_min"]

            total_checks += 3

            # time window start
            if arrival < tw_s: violations += 1

            # time window end
            if arrival + visit_dur > tw_e + MAX_SLACK_MIN: violations += 1

            # day capacity
            if arrival + visit_dur > DAY_CAPACITY_MIN: violations += 1

    csr = (1 - violations/total_checks) * 100 if total_checks else 0

    # Feasiblility Rate
    feasible_days = sum(1 for r in schedule.values() if r)
    feasibility_rate = (feasible_days/len(trip_days)) * 100

    # POI Coverage Rate
    coverage_rate = (visited / total_pois) * 100

    # Must_include Satisfaction Rate
    visited_ids = {s['poi_id'] for r in schedule.values() for s in r}
    must_covered = sum(1 for m in must_poi_ids if m in visited_ids)
    must_rate = (must_covered / len(must_poi_ids)) * 100 if must_poi_ids else 100.0

    # POI visited once
    unique_visit_rate = (len(visited_ids) / visited * 100
                     if visited else 100.0)

    # POIs assigned to open days
    invalid_assignment = 0
    for day_idx , (day_key, route) in enumerate(schedule.items()):
        day = trip_days[day_idx]
        for stop in route:
            stop_id = stop['poi_id']
            if day not in poi_allowed_days.get(stop_id, set(range(7))): 
                invalid_assignment += 1
    invalid_day_assignment_rate = invalid_assignment / visited * 100 if visited else 0



    # Total Distance Reduction
    all_indices = list(range(1, len(location_ids))) # exclude depot index 0
    naive_pois_per_day = max(1, round(visited/ len(trip_days)))
    naive_travel, max_naive_travel = naive_travel_time(all_indices, naive_pois_per_day, 
                                                       trip_days, visited, matrix)


    optimized_travel = sum(
        stop['travel_from_prev_min']
        for r in schedule.values()
        for stop in r
    ) + sum(
        matrix[poi_index_map[r[-1]['poi_id']]][0]
        for r in schedule.values()
        if r
    )
    distance_reduction = ((naive_travel - optimized_travel) / naive_travel * 100 if naive_travel else 0)
    max_distance_reduction = ((max_naive_travel - optimized_travel) / max_naive_travel * 100 if max_naive_travel else 0)


    # Backtracking Score
    all_travels = [
        stop['travel_from_prev_min']
        for r in schedule.values()
        for stop in r
    ]

    avg_travel = sum(all_travels) / len(all_travels) if all_travels else 0
    backtrack = sum(1 for t in all_travels if t > avg_travel * 2)
    backtrack_score = (backtrack / len(all_travels) * 100) if all_travels else 0

    # Cross-Zone changes
    zone_changes = 0
    total_legs   = 0

    for route in schedule.values():
        for i in range(1, len(route)):
            total_legs += 1
            if route[i]['zone'] != route[i-1]['zone']:
                zone_changes += 1

    cross_zone_rate = (zone_changes / total_legs * 100) if total_legs else 0

    # Day Balance Score
    day_counts = [len(r) for r in schedule.values()]
    mean_count = sum(day_counts) / len(day_counts)
    variance = sum((c - mean_count) ** 2 for c in day_counts) / len(day_counts)
    balance_std = variance ** 0.5

    # Preference metric
    preference_metrics = time_pref_adherence(schedule, time_prefs, trip_days)

    metrics = {
        'constraint_satisfaction_rate_%': round(csr, 2),
        'feasibility_rate_%':             round(feasibility_rate, 2),
        'poi_coverage_rate_%':            round(coverage_rate, 2),
        'must_include_satisfaction_%':    round(must_rate, 2),
        'total_travel_min':               optimized_travel,
        'distance_reduction_%':           round(distance_reduction, 2),
        'maximum_distance_reduction_%':   round(max_distance_reduction, 2),
        'backtracking_score_%':           round(backtrack_score, 2),
        'day_balance_std':                round(balance_std, 2),
        'cross_zone_rate' :               round(cross_zone_rate,2),
        'unique_visit_rate' :             round(unique_visit_rate, 2),
        'invalid_day_assignment_rate' :   round(invalid_day_assignment_rate, 2),
        'pois_per_day':                   dict(zip(schedule.keys(), day_counts)),
        'preference_metrics' :            preference_metrics,
       
    }
    satisfaction_score = compute_satisfaction_score(metrics)
    metrics['satisfaction_score'] = satisfaction_score

    logger.info("\n=== Stage 5 Metrics ===")
    for k, v in metrics.items():
        logger.info("  %-40s %s", k, v)

    return metrics


def build_output(data, sol_data, travel_data):
    '''
    Input: solution data from solver
    Build sequence of POIs for each day
    Flags if must_include_pois give infeasible solution
    Returns: schedule, status, total travel time
    '''
    passthrough = {
        'parsed_query' : travel_data['parsed_query'],
        'depot' : travel_data['depot'],
        'trip_days' : list(travel_data['trip_day_indices']),
        'location_ids' : travel_data['location_ids'],
        'poi_index_map' : travel_data['poi_index_map'],
        'matrix_raw' : travel_data['matrix_raw'],
        'must_poi_ids' : list(data['must_poi_ids']),
        'max_pois_per_day' : data['max_pois_per_day'],
        'time_windows' : data['time_windows']
    }

    # if no FEASIBLE solution
    if not sol_data['feasible']:
        return {
            **passthrough,
            'schedule' : {},
            'dropped_pois' : data['selected_pois'],
            'forced_unscheduled' : [],
            'total_travel_time' : 0,
            'solver_status' : 'INFEASIBLE',
            'feasible' : False,
        }

    solution = sol_data['solution']
    model = sol_data['model']
    manager = sol_data['manager']
    time_dim = sol_data['time_dim']
    count_dim = sol_data['count_dim']
    matrix = data['matrix']
    vd = data['visit_durations'] # index 0 = depot
    tws = data['time_windows']
    loc_ids   = data['location_ids']
    num_days  = data['num_days']
    must_poi_ids = data['must_poi_ids']
    selected_pois = data['selected_pois']
    poi_by_nd = data['poi_by_node']
    mode      = sol_data['mode']

    schedule = {}
    visited_poi_ids = set() # track by poi_id
    total_travel = 0

    for day in range(num_days):
        route = []
        idx = model.Start(day)
        prev_node = 0 # depot

        while not model.IsEnd(idx):
            node = manager.IndexToNode(idx) # node 0: depot
            if node != 0:
                poi = poi_by_nd[node]
                arrival = solution.Value(time_dim.CumulVar(idx))
                travel_prev = matrix[prev_node][node]
                total_travel += travel_prev
                prev_node_info = loc_ids[prev_node]

                route.append({
                    'poi_id' : loc_ids[node],
                    # 'poi' : poi,
                    'poi_name' : poi.get('displayName', {}).get('text', ''),
                    'zone' : poi.get('zone', ''),
                    'primaryType' : poi.get('primaryType', ''),
                    'arrival_time_min' : arrival,
                    'travel_from_prev_min' : travel_prev,
                    'visit_duration_min' : vd[node],
                    'time_window' : tws[node],
                    'flags' : list(poi.get('flags', [])),
                    'prev_node' : prev_node_info
                })

                visited_poi_ids.add(loc_ids[node])
                prev_node = node

            idx = solution.Value(model.NextVar(idx))
        total_travel += matrix[prev_node][0]
        schedule[f'day_{day+1}'] = route

    # must_include_pois as optional
    poi_by_id = {p['id']: p for p in selected_pois}
    forced_unscheduled = []
    if mode == 'score_weighted_must':
        for poi_id in must_poi_ids:
            if poi_id not in visited_poi_ids:
                poi = poi_by_id[poi_id]
                name = poi.get('displayName', {}).get('text', '')
                forced_unscheduled.append({
                    'poi_id':      poi_id,
                    'poi_name':    name,
                    'zone':        poi.get('zone', ''),
                    'primaryType': poi.get('primaryType', ''),
                    'flags':       ['must_include_not_feasible'],
                    'reason': (
                        'Requested as must-include but could not be scheduled '
                        'within opening hours, day capacity, and pace constraints.'
                    ),
                })
                logger.warning(
                    "Must-include '%s' dropped — constraint conflict.",
                    name
                )

    forced_ids = {e['poi_id'] for e in forced_unscheduled}

    dropped = [
        p for p in selected_pois
        if p['id'] not in visited_poi_ids
        and p['id'] not in forced_ids
    ]

    status = 'FEASIBLE' if mode=='enforce_must' else 'FEASIBLE_WITH_MUST_OPTIONAL'

    logger.info(
        'Done: %d visited | %d dropped | %d must-include unschedulable | travel %d min',
        len(visited_poi_ids), len(dropped), len(forced_unscheduled), total_travel
    )
    metrics = evaluate(schedule, data, travel_data)

    return {
        **passthrough,
        'schedule':           schedule,
        'dropped_pois':       dropped,
        'forced_unscheduled': forced_unscheduled,
        'total_travel_min':   total_travel,
        'solver_status':      status,
        'feasible':           True,
        'metrics' : metrics
    }

def build_sequence(travel_data):

    data = build_model_data(travel_data)
    # Case 1: must_include enforced with max penalty
    if data['must_poi_ids']:
        sol_data = solve(data, mode = 'enforce_must')
        if sol_data['feasible']:
            sol_data['mode'] = 'enforce_must'

        else:
            logger.warning(
                'Case 1 infeasible (must_include enforced). '
                'Retrying with must_include as score-weighted optional.'
            )
            # Case 2: must_include treated as regular scored POI
            sol_data = solve(data, mode='score_weighted_must')
            sol_data['mode'] = 'score_weighted_must'

    else:
        # No must-include POIs — skip straight to Case 2 (score_weighted)
        sol_data = solve(data, mode='score_weighted_must')
        sol_data['mode'] = 'score_weighted_must'
            
    if not sol_data['feasible']:                                     
        logger.warning("Case 2 infeasible. Retrying without VehicleVar day constraints.")
        # Case 3: must_include treated as regular scored POI, VehicleVar =  False
        sol_data = solve(data, mode='score_weighted_must', VehicleVar =  False)
        sol_data['mode'] = 'score_weighted_must'

        if not sol_data['feasible']:
            # Case 4: no feasible solution at all
            logger.error('Case 4: infeasible even without must_include enforcement.')
            sol_data['mode'] = 'infeasible'

    
    return build_output(data, sol_data, travel_data)

