
from dataset_new import ZONES, CATEGORIES
import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional
import torch
from transformers import pipeline

DEVICE = 'mps'

_pois_database = None
_pois_by_category = None
POI_NAME_LOOKUP = {}

def initialize(pois_database, pois_by_category):
    global _pois_database, _pois_by_category, POI_NAME_LOOKUP
    _pois_database = pois_database
    _pois_by_category = pois_by_category
    POI_NAME_LOOKUP = {
        poi['displayName']['text'].lower() : poi
        for poi in pois_database
        if 'displayName' in poi and 'text' in poi['displayName']
    }


# Fixed travel aspects with keyword triggers

ASPECTS = {
    "crowd"         : ["crowded","crowd","busy","packed","queue","line","wait",
                       "tourist","mob","rush","peak","empty","quiet","peaceful","full","space"],
    "family"        : ["kid", "child", "family", "toddler", "stroller", "baby", "young", "interactive", 
                       "fun for", "great for kids", "kid-friendly", "child-friendly",
                       "family-friendly", "children will", "kids will", "kids love", "kids enjoy",
                       "good for kids", "perfect for kids", "bring kids", "all ages", "suitable for children","enjoy"],
    "solo"          : ["solo","alone","self-guided","audio guide","independent",
                       "self-paced","own pace","by yourself","single visitor"],
    "couple"        : ["romantic","romance","intimate","couples","date","together",
                       "atmosphere","ambiance","cozy","quiet","beautiful setting"],
    "friends"       : ["group","friends","social","lively","fun","energetic",
                       "group activity","hang out","together","crew","enjoy"],
    "value"         : ["worth","value","price","cost","fee","ticket","expensive",
                       "cheap","affordable","free","money","admission"],
    "exhibits"      : ["exhibit","collection","display","artifact","gallery","unique",
                       "artwork","painting","sculpture","installation","piece","art"],
    "accessibility" : ["wheelchair","accessible","lift","ramp","disability",
                       "mobility","stair","step","handicap"],
    "staff"         : ["staff","guide","guard","employee","worker","service",
                       "helpful","rude","friendly","knowledgeable"],
    "location"      : ["location","transport","subway","bus","parking","walk",
                       "distance","nearby","central","accessible"],
    "overall"       : ["amazing","incredible","beautiful","awesome","unique","high rated","great",
                       "must","recommend","visit","experience","loved","exceeded","worth","fantastic",
                       "hated","best","worst", "terrible","awful","disappointed","boring"],
}



# Category-specific aspects

CATEGORY_ASPECTS = {
    "history_culture_art" : ["exhibits","staff","crowd","value"],
    "parks_nature_beach"  : ["crowd","accessibility"],
    "tourism_attractions"  : ["crowd","value","overall","family"],
    "nightlife"           : ["crowd","staff"],
    "recreation_active"   : ["value","staff"],
}

# Implicit_needs aspect to  upweight
IMPLICIT_NEEDS_ASPECT_WEIGHTS = {
    "kid_friendly"     : {"family"       : 2.0, "crowd": 1.5},
    "avoid_crowds"     : {"crowd"        : 2.0},
    "budget_conscious" : {"value"        : 2.0},
    "romantic"         : {"couple"       : 2.0, "crowd": 1.5, "overall": 1.5},
    "accessibility"    : {"accessibility": 2.0},
    "outdoor_activities": {"crowd"       : 1.5, "overall": 1.5},
}

# Activated directly from parsed_query["group_type"]
GROUP_TYPE_ASPECT_WEIGHTS = {
    "solo"    : {"solo"   : 2.0, "value"  : 1.5},
    "couple"  : {"couple" : 2.0, "crowd"  : 1.5},
    "friends" : {"friends": 2.0, "crowd"  : 1.5, "value": 1.3},
    "family"  : {"family" : 2.0, "crowd"  : 1.5, "accessibility": 1.3},
}


MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"

print(f'Loading model {MODEL_ID}...')

sentiment_pipeline = pipeline(
    'text-classification',
    model = MODEL_ID,
    tokenizer = MODEL_ID,
    device = DEVICE,
)
print('DeBERTa-ABSA loaded')


def build_categories(type_):
    'Build categories list from a type'
    cat = []
    for c, t in CATEGORIES.items():
        if type_ in t and c not in cat:
            cat.append(c)
    return cat



def filter_relevant_pois(parsed_query):
    '''
    Filter POIs from database relevant to ParsedQuery.
    Only analyze reviews of POIs the user might actually visit.

    Filter logic:
    - Must match preferred_categories OR preferred_types
    - Must match preferred_zones (if specified)
    - Must be OPERATIONAL
    - must_include_pois always included regardless
    '''
    preferred_categories = set(parsed_query.get('preferred_categories', []))
    preferred_types      = set(parsed_query.get("preferred_types", []))
    preferred_zones      = set(parsed_query.get("preferred_zones", []))
    must_include_pois    = set(parsed_query["hard_constraints"].get("must_include_pois", []))
    must_exclude_types   = set(parsed_query["hard_constraints"].get("must_exclude_types", []))

    relevant = []
    seen_categories = set()
    seen_poi_ids = set()

     # Always include must_include_pois
    for poi_name in must_include_pois:
        poi = POI_NAME_LOOKUP.get(poi_name)
        if poi and poi.get('id') not in seen_poi_ids:
            if poi.get("businessStatus") == "OPERATIONAL":
                relevant.append(poi)
                seen_poi_ids.add(poi.get('id'))



    for t in preferred_types:

        # Skip excluded types
        if t in must_exclude_types:
            continue

        cat = build_categories(t)
        for c in cat:
            if c not in seen_categories:
                seen_categories.add(c)
                for poi in _pois_by_category[c]:
                    if poi.get("businessStatus") != "OPERATIONAL":
                        continue
                    poi_name = poi.get('displayName',{}).get('text', '').lower()
                    poi_types = set(poi.get('types', []))
                    poi_zone = poi.get('zone', '').lower()
                    poi_id    = poi.get("id")

                    if poi_id in seen_poi_ids:
                        continue

                     # Zone filter (only if zones specified)
                    if preferred_zones and poi_zone not in preferred_zones:
                        continue

                    # Skip excluded types
                    if poi_types & must_exclude_types:
                        continue

                    relevant.append(poi)
                    seen_poi_ids.add(poi_id)

    print(f'Filtered {len(relevant)} relevant POIs from {len(_pois_database)} total')
    return relevant


# ASPECTS
# Detect aspects
def split_sentences(text):
    # Split on punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    result = []
    for s in sentences:
        # if sentence still too long,split on comma/semicolon
        if len(s.split()) > 60:
            sub = re.split(r'(?<=[;,])\s+', s)
            result.extend([x.strip() for x in sub if len(x.split())>=5])
        elif len(s.split()) >=5:
            result.append(s)
    return result


def get_active_aspects(parsed_query):
    '''
    Determine which aspects to track based on ParsedQuery.
    Always include 'overall'. Add category-specific, group_type specific and implicit_needs aspects.
    '''
    active = {'overall'}

    # Build categories from preferred_types
    pref_categories = list({
        c
        for t in parsed_query.get('preferred_types',[])
        for c in build_categories(t)                   
    })

    # Add category specific aspects
    for cat in pref_categories:
        active.update(CATEGORY_ASPECTS.get(cat,[]))

    # Add implicit_needs aspects
    for need in parsed_query.get('implicit_needs',[]):

        asp = IMPLICIT_NEEDS_ASPECT_WEIGHTS.get(need,{})
        active.update(asp.keys())

    # Add group_type aspects
    grp = parsed_query.get('group_type', 'solo')
    active.update(GROUP_TYPE_ASPECT_WEIGHTS.get(grp,{}).keys())

    # Add soft_constraints aspects
    if parsed_query.get('soft_constraints', {}).get('prefer_less_crowded'): active.update({'crowd'})
    if parsed_query.get('soft_constraints', {}).get('minimize_travel'): active.update({'location'})

    return list(active)

def detect_aspects(sentence, active_aspects):
    '''
    Detect which aspects a sentence mentions.
    Returns list of matching aspect names.
    '''
    sentence_lower = sentence.lower()
    matched = []
    for aspect in active_aspects:
        keywords = ASPECTS[aspect]
        if any(kw in sentence_lower for kw in keywords):
            matched.append(aspect)

    return matched

def build_aspect_weights(active_aspects, parsed_query):
    'Build aspect weight map from implicit_needs + group_type'

    weights = {aspect : 1.0 for aspect in active_aspects}

    # From implicit needs
    for need in parsed_query.get('implicit_needs' , []):
        for aspect, w in IMPLICIT_NEEDS_ASPECT_WEIGHTS.get(need,{}).items():
            weights[aspect] = max(w, weights.get(aspect,1.0))

    # From group_type
    grp = parsed_query.get('group_type','solo')
    for aspect, w in GROUP_TYPE_ASPECT_WEIGHTS[grp].items():
        weights[aspect] = max(w, weights.get(aspect,1.0))

    # From soft_constraints
    if parsed_query['soft_constraints'].get('prefer_high_rated'):
        weights['overall'] = max(weights.get('overall', 1.0), 1.5)

    if parsed_query['soft_constraints'].get('prefer_less_crowded'):
        weights['crowd'] = max(weights.get('crowd', 1.0), 1.5)

    if parsed_query['soft_constraints'].get('minimize_travel'):
        weights['location'] = max(weights.get('location', 1.0), 1.5)

    return weights

# Map model labels to numeric scores

LABEL_TO_SCORE = {
    'positive' : 1.0,
    'neutral' : 0.0,
    'negative' : -1.0
}

def score_pairs_batch(pairs, batch_size=32):
    '''
    Batch-score a list of (sentence, aspect) pairs using the pipeline.
    Args:
        pairs : list of (sentence, aspect) tuples
        batch_size : number of pairs per pass

    Returns:
        list of (label, score) in same order as input pairs
    '''
    if not pairs:
        return []

    inputs = [{"text": sentence, "text_pair": aspect} for sentence, aspect in pairs]

    try:
        batch_results = sentiment_pipeline(
            inputs,
            batch_size  = batch_size,
            truncation  = True,
            max_length  = 512,
            top_k       = None,
        )
    except Exception:
        return [('neutral', 0.0)] * len(pairs)

    outputs = []
    for results in batch_results:
        try:
            label_probs = {r["label"].lower(): r["score"] for r in results}
            for l, prob in label_probs.items():
                if (l == 'neutral' and prob < 0.6) or (l == 'negative' and prob < 0.6):
                    label = results[1]['label'].lower()
                    conf  = results[1]['score']
                else:
                    label = results[0]['label'].lower()
                    conf  = results[0]['score']
                break
            numeric = LABEL_TO_SCORE.get(label, 0.0) * conf
            outputs.append((label, round(numeric, 4)))
        except Exception:
            outputs.append(('neutral', 0.0))

    return outputs

def aggregate_aspect_sentiments(sentence_aspects, aspect_weights):
    '''
    Aggregate sentence-level scores into per-aspect summary.

    Args:
        sentences_aspects: list of (sentence, aspects, label, score)
        aspect_weights: {aspect: weight} from implicit_needs and group_type

    Returns:
        {aspect: {score, label, mentions, weighted_score}}
    '''
    aspect_scores = {}
    aspect_counts = {}
    result = {}
    for _, asp, label, score in sentence_aspects:
        if asp not in aspect_scores:
            aspect_scores[asp] = []
            aspect_counts[asp] = 0
        aspect_scores[asp].append(score)
        aspect_counts[asp]+=1

    for asp, scores in aspect_scores.items():
        asp_avg_score = sum(scores)/ len(scores)
        asp_weight = aspect_weights.get(asp, 1.0)
        # Normalize weighted score to [-1, 1]
        asp_weighted_score = round(max(-1.0, min(1.0, asp_avg_score * asp_weight)), 4)
        result[asp] = {
            'aspect_raw_score' : round(asp_avg_score, 4),
            'aspect_weighted_score' : asp_weighted_score,
            'label' : 'positive' if asp_weighted_score > 0.1 
                        else 'negative' if asp_weighted_score < -0.1
                        else 'neutral',
            'mentions' : aspect_counts[asp]
        }

    return result


def compute_overall_score(aspects_sentiments, aspect_weights):

    'Weighted average across all aspects'
    if not aspects_sentiments:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for aspect, data in aspects_sentiments.items():
        w = aspect_weights.get(aspect, 1.0)
        weighted_sum += data['aspect_weighted_score']
        total_weight += w
    return round(weighted_sum/total_weight, 4) if total_weight else 0.0


def generate_flags(aspects_sentiments, parsed_query):

    'Flags are raised when sentiment strongly contradicts user preferences'

    flags = []
    implicit = parsed_query.get('implicit_needs', [])
    soft = parsed_query.get('soft_constraints', {})
    grp = parsed_query.get('group_type', 'solo')

    # Crowd flag
    crowd = aspects_sentiments.get('crowd' , {})
    if soft.get('prefer_less_crowded') and crowd.get('label')=='negative':
        flags.append('very_crowded')

    # Couple flag
    if grp == 'couple' or 'romantic' in implicit:
        couple = aspects_sentiments.get('couple', {})
        if couple.get('label') == 'negative':
            flags.append('not_romantic')
    # Family flag
    if grp == 'family' or 'kid_friendly' in implicit:
        family = aspects_sentiments.get('family', {})
        if family.get('label') == 'negative':
            flags.append('not_kid_friendly')
    # Friends flag
    if grp == 'friends':
        friends = aspects_sentiments.get('friends', {})
        if friends.get('label') == 'negative':
            flags.append('not_group_friendly')
    # Solo flag
    if grp == 'solo':
        solo = aspects_sentiments.get('solo', {})
        if solo.get('label') == 'negative':
            flags.append('not_solo_friendly')

    # Value flag
    if "budget_conscious" in implicit:
        value = aspect_sentiments.get("value", {})
        if value.get("label") == "negative":
            flags.append("expensive")

    # Accessibility flag
    if "accessibility_required" in implicit:
        access = aspect_sentiments.get("accessibility", {})
        if access.get("label") == "negative":
            flags.append("accessibility_issues")

    return flags

class ABSA_analyzer:
    '''
    Aspect-Based Sentiment Analysis.

    Input : ParsedQuery + pois_database
    Output: List of ABSAResult per relevant POI
            → saved to absa_output.json

    Flow per POI:
      reviews → sentences → aspect detection → sentiment scoring
      → aggregation → flags → ABSAResult
    '''
    def __init__(self, parsed_query):
        self.parsed_query = parsed_query
        self.active_aspects = get_active_aspects(self.parsed_query)
        self.aspect_weights = build_aspect_weights(self.active_aspects, self.parsed_query)
        print(f'Active aspects : {self.active_aspects}')
        print(f'Aspect weights : {self.aspect_weights}')

    def analyze_reviews(self,reviews):
        '''
        Analyze all reviews for a POI.
        Returns (sentences_aspects, aspect_sentiments).
        Batches all (sentence, aspect) pairs across all reviews into a single pipeline
        '''
        # sentences_aspects = []
        pair_meta = []   # [(sentence, aspect), ...]
        for review in reviews:
            text = review.get('text', {})
            if isinstance(text,dict):
                text = text.get('text', '')
            if not text or not isinstance(text, str):
                continue
            sentences = split_sentences(text)
            for sentence in sentences:
                matched_aspects = detect_aspects(sentence, self.active_aspects)
                if not matched_aspects:
                    matched_aspects = ['overall']

                for asp in matched_aspects:
                    pair_meta.append((sentence, asp))

       
        scored = score_pairs_batch(pair_meta)
        sentences_aspects = [
            (sentence, asp, label, score)
            for (sentence, asp), (label, score) in zip(pair_meta, scored)
        ]

        aspects_sentiments = aggregate_aspect_sentiments(sentences_aspects, self.aspect_weights)
        return sentences_aspects, aspects_sentiments

    def analyze_poi(self, poi):
        "Analyze a single POI's reviews"
        reviews = poi.get('reviews', [])
        if not reviews:
            return None

        poi_name = poi.get('displayName', {}).get('text', 'Unknown')
        _, aspects_sentiments = self.analyze_reviews(reviews)
        overall_score = compute_overall_score(aspects_sentiments, self.aspect_weights)
        flags = generate_flags(aspects_sentiments, self.parsed_query)

        return {
            **poi,
            'aspects_sentiments' : aspects_sentiments,
            'overall_absa_score' : overall_score,
            'flags' : flags,
        }


    def analyze_all(self):
        '''
        Run ABSA on all relevant POIs.
        Returns list of ABSAResult dicts.
        '''

        relevant_pois = filter_relevant_pois(self.parsed_query)
        results = []
        print(f'Running ABSA on {len(relevant_pois)} POIs...')

        for poi in relevant_pois:
            result = self.analyze_poi(poi)
            if result: results.append(result)

        # Sort by overall score descending
        results.sort(key = lambda x: x['overall_absa_score'], reverse = True)
        print(f' ABSA complete - {len(results)} POIs analyzed')
        return results
