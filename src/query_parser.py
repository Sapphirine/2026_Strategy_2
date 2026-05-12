
import json
import re
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from dataset_new import ZONES, CATEGORIES
import mlx.core as mx


pois_database = None
pois_by_category = None
POI_NAME_TO_CATEGORIES = {}
KNOWN_POI_NAMES = None

def initialize(_pois_database, _pois_by_category):
    global pois_database, pois_by_category, POI_NAME_TO_CATEGORIES, KNOWN_POI_NAMES
    pois_database = _pois_database
    pois_by_category = _pois_by_category

    # Build dict of name and category

    POI_NAME_TO_CATEGORIES = {
        poi['displayName']['text'].lower() : poi.get('categories', [])
        for poi in _pois_database
        if 'displayName' in poi and 'text' in poi['displayName']
    }
    # POI display names
    KNOWN_POI_NAMES = list(POI_NAME_TO_CATEGORIES.keys())


from llama_ import hf_key
from huggingface_hub import login
login(token=hf_key) 

device = ('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)



ALLOWED_ZONES = list(ZONES.keys())

ALLOWED_CATEGORIES = list(CATEGORIES.keys())

ALLOWED_PACE = ['relaxed', 'moderate', 'packed']
ALLOWED_GROUP = ['solo', 'couple', 'family', 'friends']


# # Build dict of name and category

print(f'ZONES : {ALLOWED_ZONES}')
print(f'\nCATEGORIES : {ALLOWED_CATEGORIES}')

def build_preferred_types(preferred_categories):
    '''
    Returns preferred_types from CATEGORIED dict
    Only types belonging to preferred categories are included
    '''
    types = []
    for cat in preferred_categories:
        types.extend(CATEGORIES.get(cat,[]))
    return list(set(types))

# All unique poi types in database
ALLOWED_TYPES = build_preferred_types(CATEGORIES.keys())


# ParsedQuery Dataclass
@dataclass
class ParsedQuery:
    raw_query : str = ''
    trip_duration_days : int = 3
    start_date : Optional[str] = None
    group_type : str = 'solo'
    pace : str = 'moderate'
    preferred_zones : list = field(default_factory = list)
    preferred_categories : list = field(default_factory = list)
    preferred_types : list = field(default_factory = list)
    time_preferences: dict = field(default_factory=lambda: {
        "morning"  : {"prefer": [], "avoid": []},
        "afternoon": {"prefer": [], "avoid": []},
        "evening"  : {"prefer": [], "avoid": []}
    })
    hard_constraints : dict = field(default_factory = lambda : {
        'must_include_pois' : [],
        'must_exclude_types' : [],
        'max_hours_per_day' : 12,
        'avoid_days' : [],
    })
    soft_constraints : dict = field(default_factory = lambda : {
        'prefer_high_rated' : False,
        'prefer_less_crowded' : False,
        'need_meal_breaks' : True,
        'minimize_travel' : True,
        'balance_days' : True,
    })
    implicit_needs : list = field(default_factory = list)
    confidence : float = 0.0
    warnings : list = field(default_factory = list)

    def to_dict(self):
        return asdict(self)

    def summary(self):
        lines = [
            f'Duration : {self.trip_duration_days} day(s)',
            f'Group : {self.group_type}',
            f'Pace : {self.pace}',
            f"Zones : {self.preferred_zones or 'any'}",
            f"Categories : {self.preferred_categories or 'any'}",
            f"Types : {self.preferred_types or 'any'}",
            f"Exclude types : {self.hard_constraints['must_exclude_types']}",
            f"Include POIs : {self.hard_constraints['must_include_pois']}",
            f"Avoid days : {self.hard_constraints['avoid_days']}",
            f"Rating pref : {self.soft_constraints['prefer_high_rated']}",
            f"Min travel pref : {self.soft_constraints['minimize_travel']}",
            f"Crowd pref : {self.soft_constraints['prefer_less_crowded']}",
            f"Breaks pref : {self.soft_constraints['need_meal_breaks']}",
            f"Time prefs : {self.time_preferences}",
            f"Implicit : {self.implicit_needs}",
            f"Confidence : {self.confidence:.2f}",
        ]
        if self.warnings :
            lines.append(f"Warnings : {self.warnings}")
        return '\n'.join(lines)

from mlx_lm import load, generate

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
print(f"Loading {MODEL_ID}...")
model, tokenizer = load(MODEL_ID)
print("Llama-3.2-3B-Instruct (4bit MLX) loaded")
print("Model and tokenizer loaded successfully.")


def build_prompt(query):
    return f"""
    Extract travel info from the user query. Return ONLY valid JSON exactly matching the schema below.No explanations.
    
    STRICT RULES:
    - preferred_zones: only from {ALLOWED_ZONES} or []
    - preferred_categories: only from {ALLOWED_CATEGORIES} or []
    - preferred_types: only from {ALLOWED_TYPES} or []
    - group_type: only from {ALLOWED_GROUP}, default "solo"
    - pace: only from {ALLOWED_PACE}, default "moderate"
    - must_include_pois: only from {KNOWN_POI_NAMES}. Extract exact name of X from the list, if user says "X is a must" or "must visit X" or "don't want to miss X"
    - time_preferences: MUST use exact nested format:
  "morning": {{"prefer": [...], "avoid": [...]}}
  NEVER use "morning": [...] — always use the prefer/avoid dict structure
    - must_exclude_types: only from {ALLOWED_TYPES} or []
    - ALWAYS Extract EXACT names as mentioned in the lists above, if needed
    - time_preferences values: only from {ALLOWED_TYPES}. Only populate if user EXPLICITLY mentions a time slot (morning/afternoon/evening). Otherwise keep [] ,  Extract exact name from the list
    - start_date: if date mentioned, convert to YYYY-MM-DD. "April 4 2026" → "2026-04-04". "May 10" → "2026-05-10". If no date mentioned → null
    - implicit_needs: only ["kid_friendly"] if family/kids mentioned, ["senior_friendly"] if seniors mentioned, ["romantic"] if couple mentioned, else []
    - avoid_days values: use Google Maps day encoding.
  Sunday=0, Monday=1, Tuesday=2, Wednesday=3, Thursday=4, Friday=5, Saturday=6
  Example: "avoid Wednesday" → 3, "skip Sunday" → 0
    - If information not explicitly stated, use default value. Do NOT infer or guess.
    
    JSON schema (use exact keys, replace values only):
    {{
      "trip_duration_days": 3,
      "start_date": null,
      "group_type": "solo",
      "pace": "moderate",
      "preferred_zones": [],
      "preferred_categories": [],
      "preferred_types": [],
      "time_preferences": {{
        "morning": {{"prefer": [], "avoid": []}},
        "afternoon": {{"prefer": [], "avoid": []}},
        "evening": {{"prefer": [], "avoid": []}}
      }},
      "hard_constraints": {{
        "must_include_pois": [],
        "must_exclude_types": [],
        "max_hours_per_day": 12,
        "avoid_days": []
      }},
      "soft_constraints": {{
        "prefer_high_rated": false,
        "prefer_less_crowded": false
      }},
      "implicit_needs": []
    }}
    
    User Query: {query}
    JSON:
    """.strip()

def extract_json(text):
    """
    Robust JSON extraction: find the first {...} block and parse it.
    """
    text=re.sub(r'```(json)?','',text)
    brace_stack = []
    start_idx = None
    end_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack: start_idx = i
            brace_stack.append(char)
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:
                    end_idx = i+1
                    break

    try:
        match = text[start_idx : end_idx]
        return json.loads(match)
    except:
        raise ValueError('No JSON object found')

def run_llm(prompt: str, max_new_tokens: int = 800) -> str:
    mx.clear_cache()
    messages = [
        {"role": "system", "content": "You are a travel query parser. Return only valid JSON. No extra text."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    response = generate(
        model, tokenizer,
        prompt=input_text,
        max_tokens=max_new_tokens,
        verbose=False
    )
    return response.strip()


class QueryValidator:
    '''
    Post-generation validator.
    Every field LLM outputs is checked against the whitelist.
    Invalid values are dropped and replaced with safe defaults.
    '''

    @staticmethod
    def validate_list_field(values,whitelist,field_name,warnings):
        '''
        Keep only values present in whitelist.
        Warn on dropped values
        '''
        if not isinstance(values,list):
            warnings.append(f'Field {field_name} was not a list -  reset to []')
            return []
        def normalize(v):
            return v.lower().strip().replace(' ', '_')   

        def find_match(value, whitelist):
            for w in whitelist:
                w = w.lower()
                value = normalize(value)
                if value in w or w in value: 
                    return w
            return None
        valid   = [find_match(v, whitelist) for v in values
                   if isinstance(v, str) and find_match(v, whitelist)]
        dropped = [v for v in values
                   if not isinstance(v, str) or find_match(v, whitelist) is None]
        if dropped:
            warnings.append(f'Hallucinated values dropped from {field_name} : {dropped}')
        return valid

    @staticmethod
    def validate_scalar(value,allowed,default,field_name,warnings):
        '''
        Validate scalar against allowed values
        '''
        if value not in allowed:
            warnings.append(f'Invalid value {value} for {field_name} -  reset to {default}')
            return default
        return value

    @staticmethod
    def validate_int(value,default, min_val,max_val,field_name,warnings):
        '''
        Validate integer is within expected range.
        '''
        try:
            v = int(value)
            if not (min_val <= v <= max_val):
                raise ValueError
            return v
        except (ValueError,TypeError):
            warnings.append(f'Invalid int {value} for {field_name} - reset to {default}')
            return default

    @staticmethod
    def validate_bool(value, default,field_name, warnings):
        if not isinstance(value,bool):
            warnings.append(f'Invalid bool for {field_name} - reset to {default}')
            return default
        return value

    def validate(self,raw_dict,query):
        '''
        Validate all fields in LLM's output dict.
        Returns a clean, safe ParsedQuery.
        '''
        pq = ParsedQuery(raw_query = query)
        w = pq.warnings

        #Scalar fields
        pq.trip_duration_days = self.validate_int(
            raw_dict.get('trip_duration_days', 3), 3, 1, 30, 'trip_duration_days', w
        )
        pq.group_type = self.validate_scalar(
            raw_dict.get('group_type', 'solo'), ALLOWED_GROUP,'solo' ,'group_type', w
        )
        pq.pace = self.validate_scalar(
            raw_dict.get('pace', 'moderate'), ALLOWED_PACE, 'moderate', 'pace', w
        )

        pq.start_date = raw_dict.get('start_date', 'null')

        # List fields - whitelisted
        pq.preferred_zones = self.validate_list_field(
            raw_dict.get('preferred_zones', []), ALLOWED_ZONES, 'preferred_zones', w
        )
        # Creating preferred categories field
        pq.preferred_categories = self.validate_list_field(
            raw_dict.get('preferred_categories', []), ALLOWED_CATEGORIES, 'preferred_categories', w
        )

        # Adding categories from types
        types_ = self.validate_list_field(
            raw_dict.get('preferred_categories', []), ALLOWED_TYPES, 'preferred_categories_types', w
        )

        for t in types_:
            for cat, types in CATEGORIES.items():
                if t in types and cat not in pq.preferred_categories :
                    pq.preferred_categories.append(cat)


        # Time preferences

        ALLOWED_TIME_VALUES = ALLOWED_CATEGORIES + ALLOWED_TYPES
        raw_time = raw_dict.get('time_preferences',{})
        for window in ["morning", "afternoon", "evening"]:
            pref = raw_time.get(window, {"prefer": [], "avoid": []})
            if not isinstance(pref, dict):
                pref = {"prefer": [], "avoid": []}
                w.append(f'time_preferences.{window} was not a dict — reset to default')
    
            for p in ['prefer', 'avoid']:
                types = []
                cat_type = self.validate_list_field(
                pref.get(p, []), ALLOWED_TIME_VALUES, f'time_preferences.{window}.{p}', w
            )
                for c in cat_type:
                    if c in ALLOWED_CATEGORIES:
                        types += build_preferred_types([c])
                    else:
                        types.append(c)
                pq.time_preferences[window][p] = list(set(types))

        # Hard constraints
        raw_hard = raw_dict.get('hard_constraints', {})
        
        # Validating POIs from specific_pois
        raw_pois = raw_hard.get('must_include_pois', [])
        validated_pois = []
        for poi in raw_pois:
            poi_l = poi.lower().strip()
            if poi_l in KNOWN_POI_NAMES:
                validated_pois.append(poi_l)
            else:
                # Partial match fallback
                match = next(
                    (name for name in KNOWN_POI_NAMES if poi_l in name or name in poi_l),
                    None
                )
                if match:
                    validated_pois.append(match)
                else:
                    w.append(f"POI not found in database, dropped: '{poi}'")

        pq.hard_constraints["must_include_pois"] = validated_pois

        # Preferred categories + category of specific PIOs
        for poi in pq.hard_constraints["must_include_pois"]:
            for cat in POI_NAME_TO_CATEGORIES.get(poi, []):
                if cat not in pq.preferred_categories:
                    pq.preferred_categories.append(cat)


       
        

        # Creating preferred types field
        pq.preferred_types = self.validate_list_field(
            raw_dict.get('preferred_types', []), ALLOWED_TYPES, 'preferred_types', w
        )

       

        pq.hard_constraints["must_exclude_types"] = self.validate_list_field(
            raw_hard.get("must_exclude_types",[]), ALLOWED_TYPES, "must_exclude_types" , w
        )

        # Preferred types + types from preferred categories
        pq.preferred_types = list(set(pq.preferred_types + build_preferred_types(pq.preferred_categories)))


        pq.preferred_types = [c for c in pq.preferred_types 
                                   if c not in pq.hard_constraints["must_exclude_types"]]



        raw_avoid = raw_hard.get('avoid_days', [])
        validated_avoid = []
        for v in raw_avoid if isinstance(raw_avoid, list) else []:
            try:
                day = int(v)   # handles both "3" and 3
                if 0 <= day <= 6:
                    validated_avoid.append(day)
                else:
                    w.append(f'avoid_days value {v} out of range 0-6 — dropped')
            except (ValueError, TypeError):
                w.append(f'avoid_days invalid value {v} — dropped')
        pq.hard_constraints['avoid_days'] = validated_avoid

        # Soft constraints
        raw_soft = raw_dict.get('soft_constraints', {})

        for bf in raw_soft.keys():
            pq.soft_constraints[bf] = self.validate_bool(
                raw_soft.get(bf, pq.soft_constraints[bf]), pq.soft_constraints[bf], bf, w
            )
        pq.soft_constraints["minimize_travel"] = True
        pq.soft_constraints["balance_days"] = True
        pq.soft_constraints['need_meal_breaks'] = True

        # Implicit needs
        raw_implicit = raw_dict.get('implicit_needs', [])
        pq.implicit_needs = [
            item.strip()[:30]
            for item in raw_implicit
            if isinstance(item, str) and item.strip()
        ][:5]

        # Confidence
        filled_fields = sum([
            bool(pq.preferred_categories),
            bool(pq.preferred_zones),
            bool(pq.preferred_types),
            bool(pq.hard_constraints["must_include_pois"]),
            bool(pq.hard_constraints["must_exclude_types"]),
        ])
        pq.confidence = round(filled_fields / 5, 2)
        return pq



class QueryParser:
    """
    NLP Query Parser using Llama-3.2-3B + post-generation validation.

    Flow:
      raw query
        → build_prompt() injects whitelist into prompt
        → Llama-3.2-3B → raw JSON string
        → extract_json() parses JSON safely
        → QueryValidator.validate() checks every field against whitelist
        → ParsedQuery (clean, hallucination-free)
    """
    def __init__(self):
        self.validator = QueryValidator()


    def parse(self,query):
        """
        Parse a natural language query into a validated ParsedQuery.

        Args:
            query: Raw user input string

        Returns:
            ParsedQuery
        """

        if not query or not query.strip():
            pq = ParsedQuery()
            pq.warnings.append('Empty query - all defaults used')
            return pq

        query = query.strip()

        # Build prompt for LLM
        prompt = build_prompt(query)

        # LLM inference
        raw_text = run_llm(prompt)

        # Parse JSON from output
        raw_dict = extract_json(raw_text)
       
        if not raw_dict:
            pq = ParsedQuery(raw_query = query)
            pq.warnings.append("LLM did not return valid JSON — all defaults used")
            return pq


        # Validate every field against whitelist
        pq = self.validator.validate(raw_dict,query)
        return pq
