
import json, logging, faiss
import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


pois_database = None
pois_by_category = None
parsed_query = None
absa_results = None

def initialize_db(_pois_database, _pois_by_category):
    global pois_database, pois_by_category, C_GLOBAL
    pois_database = _pois_database
    pois_by_category = _pois_by_category
    C_GLOBAL = round(
    sum(p.get("rating", 0.0) for p in _pois_database if p.get("rating", 0.0) > 0)
    / sum(1 for p in _pois_database if p.get("rating", 0.0) > 0),
    2)
    

def initialize(_parsed_query, _absa_results):
    global parsed_query, absa_results
    parsed_query = _parsed_query
    absa_results = _absa_results


def rag(parsed_query, absa_results, stored_itinerary_path = 'itineraries_history.json'):
    '''
    Passthrough function to build 10 itineraries for RAG 
    Full RAG — activated once knowledge base has 10 real itineraries.
    '''
    try:
        with open(stored_itinerary_path , 'r') as f:
            kb = json.load(f)
    except FileNotFoundError:
        print('Itinerary knowledge base not found - using passthrough function')
        return passthrough(parsed_query, absa_results)
    if len(kb) < 10:
        print(f'Only {len(kb)} itineraries found - using passthrough function')
        return passthrough(parsed_query, absa_results)
    else:
        rag_pipeline = RAGPipeline(
            parsed_query = parsed_query,
            absa_results = absa_results,
            itineraries = kb
        )
        return rag_pipeline.run()

def weighted_rating(poi):
    'Takes account of Rating and User rating count'
    r = poi.get("rating", 0.0)
    n = poi.get("userRatingCount", 0)
    m = 100 # minimum votes
    C = C_GLOBAL #  mean rating of POI database

    # Bayesian average — standard approach
    return (n / (n + m)) * r + (m / (n + m)) * C

def combined(poi, absa_w = 0.6, wr_w = 0.4):
    absa_norm = (poi.get('overall_absa_score', 0.0) + 1.0) /2.0 # [-1,1]-> [0,1]
    wr = weighted_rating(poi) / 5.0 # Normalize [0,5] -> [0,1]
    return round(absa_w * absa_norm + wr_w * wr, 4)

def passthrough(parsed_query,absa_results):
    'Building Itinerary knowledge base for RAG'
    days = parsed_query.get("trip_duration_days", 3)
    top_k = min(20* days, len(absa_results))
    must_include_pois = parsed_query['hard_constraints'].get('must_include_pois',[])

    must_pois = [p for p in absa_results if p.get('displayName',{}).get('text','').lower() in must_include_pois]
    other_pois = [p for p in absa_results if p.get('displayName',{}).get('text','').lower() not in must_include_pois]

    other_pois.sort(key = combined, reverse = True)

    candidates = must_pois + other_pois[:top_k - len(must_pois)]

    for i, p in enumerate(candidates, 1):
        p['combined_score'] = combined(p)
        p['retrieval_rank'] = i
    print(f'{len(candidates)} candidates retrieved')

    return {
        'parsed_query' : parsed_query,
        'candidates'  : candidates,
        'retrieved_itineraries' : None,
        'top_k' : len(candidates),
        'rag_active': False
    }


EMBED_MODEL_ID = "all-MiniLM-L6-v2"
print(f'Loading Sentence-BERT...')
embedder = SentenceTransformer(EMBED_MODEL_ID)
print('Sentence-BERT loaded')


def build_faiss_index(itineraries):
    '''
    Embed itinerary_text for each  itinerary.
    Returns (index, embeddings).
    '''
    texts = [it['itinerary_text'] for it in itineraries]

    print(f'Embeddings {len(texts)} itineraries')
    embeddings = embedder.encode(
        texts,
        batch_size = 32,
        show_progress_bar = True,
        convert_to_numpy = True,
        normalize_embeddings = True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    logger.info('FAISS index %s itineraries | dim = %s',index.ntotal, dim)
    return index, embeddings


def build_query_text(parsed_query):
    '''
    Convert ParsedQuery to text — mirrors itinerary_text structure
    for meaningful cosine similarity.
    '''
    group = parsed_query.get('group_type', 'solo')
    days = parsed_query.get('trip_duration_days', 3)
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
        if pref: 
            time_prefs.append(f'prefer {pref} in {slot}')
        if avoid : 
            time_prefs.append(f'avoid {avoid} in {slot}')
    time_prefs = '; '.join(time_prefs) or 'none'

    return (
        f"Group : {group} trip; "
        f"Duration : {days} days; "
        f"Pace : {pace}; "
        f"Zone preference : {zones}; "
        f"Category preference : {categories}; "
        f"Type preference : {types}; "
        f"Implicit needs : {implicit_needs}; "
        f"Crowd preference : {crowd}; "
        f"Types excluded : {exclude_types}; "
        f"Must include POIs : {must_include_pois}; "
        f"Time windows preferences : {time_prefs} ;"
    )



def retrieve_similar_itineraries(parsed_query, itineraries, faiss_index, top_k = 3):

    '''
    Retrieve top-K past itineraries most similar to current query.
    Returns retrieved itineraries with similarity scores.
    '''
    query_text = build_query_text(parsed_query)
    query_emb = embedder.encode(
        [query_text],
        convert_to_numpy = True,
        normalize_embeddings = True,
    ).astype(np.float32)

    sims, idxs = faiss_index.search(query_emb, top_k)

    retrieved = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx == -1: continue
        itin = itineraries[idx].copy()
        itin['similarity_score'] = round(float(sim), 4)
        retrieved.append(itin)

    logger.info('Retrieved %s similar past itineraries', len(retrieved))

    for r in retrieved:
        pois = []
        for day_stops in r['itinerary'].values():
            if not day_stops:
                continue
            p = day_stops[0]['poi_name']
            pois.append(p)

        logger.info(
            'sim: %s | outcome: %s | group: %s | days: %s | pois: %s ',
            r['similarity_score'], r['evaluation_score']['satisfaction_score'], 
            r['query_profile']['group_type'],
            r['query_profile']['trip_duration_days'], ', '.join(pois)
        )

    return retrieved


def get_poi_ids(itinerary):
    'Returns dict of POI IDs and POIs of visited POIs in an itinerary'
    pois = {}
    for day_route in itinerary.values():
        for stop in day_route:
            poi_id = stop['poi_id']
            if poi_id != 'meal_break':
                poi = stop['poi']
                pois[poi_id] = poi
    return pois


def apply_flag_penalty(poi, parsed_query):
    flags = poi.get('flags', [])
    penalty = 0.0
    implicit = set(parsed_query.get('implicit_needs', []))
    soft = parsed_query['soft_constraints']
    for flag in flags:
        fname = flag.strip()
        if fname == 'very_crowded' and soft['prefer_less_crowded']:
            penalty += 0.15
        if fname == "not_kid_friendly" and "kid_friendly" in implicit:
            penalty += 0.10
        if fname == "not_romantic" and "romantic" in implicit:
            penalty += 0.15
        if fname == "expensive" and "budget_conscious" in implicit:
            penalty += 0.10
    return penalty

def build_candidate_pool(parsed_query, retrieved_itineraries, absa_results):
    must_include_pois = set(parsed_query['hard_constraints'].get('must_include_pois', []))

    # Step 1: collect POI IDs from retrieved itineraries weighted by similarity
    poi_retrieval_scores = {}
    poi_retrieval_lookup = {}

    for itin in retrieved_itineraries:
        sim = itin['similarity_score']
        outcome = itin['evaluation_score']['satisfaction_score']
        weight = sim * outcome # high sim + high outcome = best signal

        for poi_id, poi in get_poi_ids(itin['itinerary']).items():
            poi_retrieval_scores[poi_id] = poi_retrieval_scores.get(poi_id, 0.0) + weight
            poi_retrieval_lookup[poi_id] = poi
    logger.info(
        'Retrieved POIs %s', len(poi_retrieval_scores)
    )

    if poi_retrieval_scores:
        max_retrieval = max(poi_retrieval_scores.values())
        if max_retrieval > 0:
            poi_retrieval_scores = {
                k: v/ max_retrieval
                for k,v in poi_retrieval_scores.items()
            }
        
    # Step 2: cross reference with ABSA results
    # Only keep POIs that are in ABSA results
    absa_by_id = {r['id'] : r for r in absa_results}

    candidates = []
    seen_ids = set()

    # Step 3: must_include POIs always first regardless
    for poi in absa_results:
        poi_name = poi.get('displayName', {}).get('text', '').lower()
        if poi_name in must_include_pois:
            retrieval = poi_retrieval_scores.get(poi['id'], 0.0)
            combined_score = round(0.35 * retrieval + combined(poi, absa_w = 0.45, wr_w = 0.20), 4)
            poi['combined_score'] = combined_score
            poi['retrieval_weight'] = round(retrieval, 4)
            candidates.append(poi)
            seen_ids.add(poi['id'])

    # Step 4: POIs from retrieved itineraries + ABSA cross-reference
    for poi_id, retrieval_w in sorted(
        poi_retrieval_scores.items(),
        key = lambda x: x[1], reverse = True
    ):

        if poi_id in seen_ids:
            continue
        if poi_id in absa_by_id:
            poi = absa_by_id[poi_id].copy()
            combined_score = round(0.45 * retrieval_w + combined(poi, absa_w = 0.40, wr_w = 0.15), 4)
        else:
            poi = poi_retrieval_lookup.get(poi_id, {}).copy()
            poi['overall_absa_score'] = 0 # no ABSA - neutral sentiment
            poi['aspect_sentiments'] = {}
            poi['flags'] = []
            combined_score = round(0.60 * retrieval_w + combined(poi, absa_w = 0.0, wr_w = 0.40), 4)


         # Flag penalty
        penalty = apply_flag_penalty(poi, parsed_query)
        poi['combined_score'] = round(max(0.0, combined_score - penalty), 4)
        poi['retrieval_weight'] = round(retrieval_w, 4)
        candidates.append(poi)
        seen_ids.add(poi_id)

    # Step 5: fill remaining from ABSA results not yet in candidates
    for poi in absa_results:
        if poi['id'] in seen_ids:
            continue
        penalty = apply_flag_penalty(poi, parsed_query)
        poi['combined_score'] = combined(poi, absa_w = 0.75, wr_w = 0.25)
        poi['retrieval_weight'] = 0.0
        candidates.append(poi)
        seen_ids.add(poi['id'])

    # Sort by combined score
    must_pois =[c for c in candidates 
               if c.get('displayName', {}).get('text', '').lower() in must_include_pois]

    other_pois =[c for c in candidates 
               if c.get('displayName', {}).get('text', '').lower() not in must_include_pois]
    other_pois.sort(key = lambda x: x['combined_score'], reverse = True)

    # Trim to top_k
    days = parsed_query.get("trip_duration_days", 3)
    top_k = min(20* days, len(candidates))
    final = must_pois + other_pois[: top_k - len(must_pois)]

    for i, c in enumerate(final, 1):
        c['retrieval_rank'] = i

    logger.info('Candidate pool before RAG: %s, after RAG: %s POIs', len(absa_results), len(final))

    for c in final[:5]:
        c_name = c.get('displayName', {}).get('text', '').lower()
        logger.info(
            'Rank: %s | POI name: %s | Retrieval score: %s | absa: %s | Combined score: %s | flags : %s',
            c['retrieval_rank'], c_name , c['retrieval_weight'], c['overall_absa_score'],
            c['combined_score'], c.get('flags', []) or 'none'
        )
    return final

class RAGPipeline:
    '''
    Past itineraries: encode POI combinations that worked together
    Current query :  retrieve similar past trips
    Adaptation :  extract POI patterns + cross-reference ABSA scores
    Output :  ranked candidate pool
    '''

    def __init__(self, parsed_query, absa_results, itineraries):
        self.parsed_query = parsed_query
        self.absa_results = absa_results
        self.itineraries = itineraries
        self.faiss_index, _ = build_faiss_index(self.itineraries)

    def run(self):
        # Retrieve similar past itineraries
        retrieved = retrieve_similar_itineraries(self.parsed_query, self.itineraries, self.faiss_index, top_k = 3)

        # Build candidate pool (RAG + ABSA cross reference)
        candidates = build_candidate_pool(self.parsed_query, retrieved, self.absa_results)

        return {
            'parsed_query' : self.parsed_query,
            'candidates'  : candidates,
            'retrieved_itineraries' : retrieved,
            'top_k' : len(candidates),
            'rag_active': True
        }
