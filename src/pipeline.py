

import json, os, logging
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

import query_parser, ABSA, RAG, time_matrix, assignment_sequence, TemporalScheduling

pois_database = None
pois_by_category = None
def initialize(_pois_database, _pois_by_category):
    global pois_database, pois_by_category
    pois_database = _pois_database
    pois_by_category = _pois_by_category
    query_parser.initialize(_pois_database, _pois_by_category)
    ABSA.initialize(_pois_database, _pois_by_category)
    RAG.initialize_db(_pois_database, _pois_by_category)


class SessionMemory:
    """
    Lightweight structured memory for one pipeline session.
    Stores:
      - Itinerary Q&A       → itinerary, day_summaries, violations
      - POI recommendations → dropped_pois, trip_days
      - Conflict explanation → model_constraints, day_summaries
    """
    def __init__(self):
        self.store = {}

    def save(self, sequence_output, final_itinerary, travel_data, retrieved_itineraries):
        pq = sequence_output['parsed_query']
        self.store = {
            'session_time' : datetime.now().isoformat(),
            'parsed_query' : pq,

            # ---Itinerary Q&A
            'itinerary' : final_itinerary['itinerary'],
            'day_summaries' : final_itinerary['day_summaries'],
            'all_violations' : final_itinerary['all_violations'],
            'total_violations' : final_itinerary['total_violations'],
            'feasibility_status' : final_itinerary['solver_status'],
            'itinerary_evaluation' : final_itinerary['metrics'],


            # ---POI recommendations
            'starting_ending_point' : sequence_output['depot'],
            'dropped_pois' : sequence_output.get('dropped_pois', []),
            'forced_pois_unscheduled' : sequence_output['forced_unscheduled'],
            'trip_day_indices' : final_itinerary['trip_day_indices'],

            # ---Constraint conflict explanation
            'model_constraints' : {
            'max_pois_per_day' : sequence_output['max_pois_per_day'],
            'must_poi_ids' : list(sequence_output.get('must_poi_ids', set())),
            'pace' : pq.get('pace', 'moderate'),
            'num_days' : pq.get('trip_duration_days', 3),
            'must_exclude_types' : pq['hard_constraints'].get('must_exclude_types', []),
            'days_avoided' : pq['hard_constraints'].get('avoid_days', [])
            },

            # ---Dropped POIs explanation
            'time_windows' : 
            {l_id : tw for l_id, tw in zip(sequence_output.get('location_ids', []), sequence_output.get('time_windows', []))},
            'location_ids' : sequence_output.get('location_ids', []),
            'removed_pois' : travel_data.get('removed_pois', []),

            # Past itineraries information
            'rag_context':{
                'rag_active' : True if retrieved_itineraries else False,
                'retrieved_itineraries' :[
                    {
                        'similarity_score' : r['similarity_score'],
                        'satisfaction_score': r['evaluation_score']['satisfaction_score'],
                        'group_type' : r['query_profile']['group_type'],
                        'days':   r['query_profile']['trip_duration_days'],
                        'pace': r['query_profile']['pace'],
                        'preferred_zones':    r['query_profile'].get('preferred_zones', []),
                        'preferred_categories':   r['query_profile'].get('preferred_categories', []),
                        'preferred_types':   r['query_profile'].get('preferred_types', []),
                        'poi_names':    [                          # first POI per day only
                                stops[0]['poi_name']
                                for stops in list(r['itinerary'].values())
                                if stops and stops[0].get('poi_id') != 'meal_break'
                            ],
                    }
                    for r in retrieved_itineraries
                ]
            }
        }

        logger.info('Session memory saved — %d days | %d violations',
                    pq.get('trip_duration_days'), final_itinerary['total_violations'])

    def get(self, key, default = None):
        return self.store.get(key, default)

    def to_dict(self):
        return dict(self.store)

    def dump(self, path = 'session_memory.json'):
        with open(path, 'w') as f:
            json.dump(self.store, f, indent = 2)
        logger.info('Session memory written to %s', path)

    @classmethod
    def load(cls, path = 'session_memory.json'):
        mem = cls()
        with open(path) as f:
            mem.store = json.load(f)
        return mem

def run_pipeline(user_query, memory = None):
    if memory is None:
        memory = SessionMemory()

    logger.info('Pipeline start | query: "%s"', user_query)

    # Stage 1: NLP Perception
    logger.info('Stage 1 — NLP parsing')
    query_parser.initialize(pois_database, pois_by_category)
    parser = query_parser.QueryParser()
    pq = parser.parse(user_query)
    parsed_query = pq.to_dict()
    print(parsed_query)
    save_json(parsed_query, 'parsed_query.json')


    # Stage 2: ABSA
    logger.info('Stage 2 — ABSA')
    ABSA.initialize(pois_database,pois_by_category)
    analyzer = ABSA.ABSA_analyzer(parsed_query)
    absa_results = analyzer.analyze_all()
    absa_output = {'parsed_query': parsed_query,
                  'absa_results' :  absa_results}
    save_json(absa_output, 'absa_output.json')

    # Stage 3: RAG
    logger.info('Stage 3 — RAG retrieval')
    RAG.initialize(parsed_query, absa_results)
    rag_output = RAG.rag(parsed_query, absa_results)
    save_json(rag_output, 'rag_output.json')

    # Stage 4: Pre-processing + Travel Matrix
    logger.info('Stage 4 — Filtering + matrix')
    travel_data = time_matrix.TravelTime(parsed_query = parsed_query, rag_pois = rag_output['candidates'])
    save_json(travel_data, 'travel_data.json')

    if not travel_data['selected_pois']:
        logger.error('Stage 4: no POIs survived filtering — aborting pipeline')
        return empty_result(parsed_query, memory, 'STAGE4_NO_POIS')

    # Stage 5: VRP Solver
    logger.info('Stage 5 — VRP solving')
    sequence_output = assignment_sequence.build_sequence(travel_data)
    save_json(dict(sequence_output), 'sequence_output.json')

    if not sequence_output['feasible']:
        logger.error('Stage 5: infeasible — no schedule produced')
        return empty_result(parsed_query, memory, 'STAGE5_INFEASIBLE')


    # Stage 6: Temporal Scheduling
    logger.info('Stage 6 — Temporal scheduling')
    TemporalScheduling.initialize(travel_data)
    final_itinerary = TemporalScheduling.temporalScheduling(sequence_output, store_path = 'itineraries_history.json')
    save_json(final_itinerary, 'final_itinerary.json')

    # Memory save
    memory.save(
        sequence_output = sequence_output,
        final_itinerary = final_itinerary,
        travel_data = travel_data,
        retrieved_itineraries = rag_output['retrieved_itineraries'],
    )
    memory.dump('session_memory.json')

    logger.info('Pipeline complete')

    return {
        'final_itinerary' : final_itinerary,
        'memory' : memory,
        'metrics' : final_itinerary['metrics']
    }

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    logger.info('Saved %s', path)

def empty_result(parsed_query, memory, reason = ''):
    return {
        'final_itinerary': {
            'parsed_query':    parsed_query,
            'itinerary':   {},
            'day_summaries':   {},
            'all_violations':  [],
            'total_violations': 0,
            'feasible':        False,
            'solver_status':   reason,
        },
        'metrics':      {},
        'memory':   memory,
        'map_path':      None,
        'timeline_path': None,
    }

