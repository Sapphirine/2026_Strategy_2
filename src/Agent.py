
import json, os, logging, re
from collections import deque
from visualize import plot_map, plot_timeline
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
import pipeline
from config import API_KEY


# Load JSON databases
with open('nyc_pois_database.json' , 'r') as f:
    pois_database = json.load(f)

with open('nyc_pois_by_category.json', 'r') as f:
    pois_by_category = json.load(f)
POI_DB_DICT = {p['id']: p for p in pois_database}
DAY_LOOKUP ={1 : 'Monday', 2 : 'Tuesday', 3 : 'Wednesday', 4: 'Thursday', 5 : 'Friday', 6 : 'Saturday', 0 : 'Sunday'}
DAY_START_HOUR = 9

try:
    from mlx_lm import load, generate
    MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    model, tokenizer = load(MODEL_ID)
    LLM_AVAILABLE = True
    logger.info('LLM loaded: Llama-3.2-3B-Instruct-4bit')
except Exception as e:
    LLM_AVAILABLE = False
    logger.warning('LLM not available (%s) — responses will be template-based', e)



def format_itinerary_text(final_itinerary):

    itinerary     = final_itinerary.get('itinerary', {})
    day_summaries = final_itinerary.get('day_summaries', {})

    def maps_link(lat, lon, name):
        # Google maps link
        query = f"{lat},{lon}"
        url   = f"https://www.google.com/maps/search/?api=1&query={query}"
        return f"<a href='{url}' target='_blank'>📍 Open in Google Maps</a>"

    day_blocks = []

    for day_key, stops in itinerary.items():
        summary   = day_summaries.get(day_key, {})
        day_label = day_key.replace('_', ' ').title()

        stop_cards = []
        for seq, stop in enumerate(stops, 1):
            if stop['poi_id'] == 'meal_break':
                stop_cards.append(
                    f"<div style='padding:6px 0;color:#888'>🍽️ Meal Break — "
                    f"{stop['start_time']} – {stop['end_time']}</div>"
                )
                continue

            poi     = stop.get('poi', {})
            loc     = poi.get('location', {})
            lat     = loc.get('latitude', '')
            lon     = loc.get('longitude', '')
            address = poi.get('formattedAddress', 'Address not available')
            rating  = poi.get('rating', 'N/A')
            n_ratings = poi.get('userRatingCount', 0)
            photos  = poi.get('photos', [])[:3]

            link_html = maps_link(lat, lon, stop['poi_name']) if lat and lon else ''

            card = f"""
            <div style='border:1px solid #ddd;border-radius:8px;padding:10px 14px;
                        margin-bottom:10px;background:#fafafa;'>
              <div style='font-size:15px;font-weight:bold;color:#1a1a2e;'>
                {seq}. {stop['poi_name']}
              </div>
              <div style='color:#555;font-size:13px;margin-top:3px'>
                🕐 {stop['start_time']} – {stop['end_time']} 
                &nbsp;|&nbsp; ⭐ {rating} ({n_ratings:,} ratings)
              </div>
              <div style='color:#666;font-size:12px;margin-top:3px'>
                📫 {address}
              </div>
              <div style='font-size:12px;margin-top:3px'>{link_html}</div>
              {photo_html}
            </div>"""
            stop_cards.append(card)

        block = f"""
        <div style='margin-bottom:20px'>
          <div style='font-size:17px;font-weight:bold;color:#1a3a5c;
                      border-bottom:2px solid #4a90d9;padding-bottom:4px;
                      margin-bottom:10px;'>
            📅 {day_label}
            <span style='font-size:13px;font-weight:normal;color:#555;'>
              &nbsp; {summary.get('day_start_time','?')} – {summary.get('day_end_time','?')}
              &nbsp;|&nbsp; {summary.get('total_pois',0)} stops
              &nbsp;|&nbsp; {summary.get('total_travel_min',0)} min travel
            </span>
          </div>
          {''.join(stop_cards)}
        </div>"""
        day_blocks.append(block)

    return f"""
        <div style='font-family:Arial,sans-serif;max-width:700px;
                    color:#222222;background:#ffffff;
                    padding:8px 4px;border-radius:8px;'>
          <h2 style='color:#2c3e50;margin-bottom:16px;'>🗽 Your NYC Itinerary</h2>
          {''.join(day_blocks)}
</div>"""

def format_itinerary_text1(itinerary):
    print('itinerary text function')

def format_rag_text(memory_dict):
    parsed_query = memory_dict.get('parsed_query')
    group  = parsed_query.get("group_type")
    days   = parsed_query.get("trip_duration_days")
    zones  = parsed_query.get("preferred_zones") or ["NYC"]
    cat = parsed_query.get("preferred_categories", [])
    retrieved_itineraries = memory_dict.get('rag_context', {}).get('retrieved_itineraries',[])

    # Summarise retrieved past trips
    past_summaries = []
    top_pois = []
    for r in retrieved_itineraries[:2]:
        poi_names= ', '.join(r["poi_names"][:3])
        top_pois.extend(r["poi_names"][:3])
        past_summaries.append(
            f"A {r['days']}-day {r['group_type']} trip "
            f"to {', '.join(r['preferred_zones'] or ['NYC'])} with preference on {', '.join(r['preferred_categories'])} "
            f"visiting {', '.join(poi_names)} (similarity:{r['similarity_score']*100}% and rated : {r['satisfaction_score']*100}%)"
        )
    past_text = ". ".join(past_summaries)


    system_prompt = '''
        You are a travel advisor. Based on similar past trips, 
        Write 2-3 sentences explaining why these places suit this traveler.'''

    user_prompt = f"""
            CURRENT TRIP: {days}-day {group} trip to {', '.join(zones)} with preference on {', '.join(cat)}
            SIMILAR PAST TRIPS: {past_text}
            RECOMMENDED PLACES: {', '.join(top_pois)}"""

    return clean_response(llm_generate(system_prompt, user_prompt))


def generate_itinerary(user_message):
    '''
    Generates fresh itinerary, maps and timeline, and suggests relevant past itineraries 
    '''
    print('Generating fresh itinerary...\n')
    pipeline.initialize(pois_database, pois_by_category)
    result = pipeline.run_pipeline(user_query = user_message)
    memory_dict = result['memory'].to_dict()
   
    response = {
        'type': 'itinerary',
        'itinerary' : result['final_itinerary'],
        'text' : format_itinerary_text(result['final_itinerary']),
        'map_data' : plot_map(result['final_itinerary']),
        'timeline' : plot_timeline(result['final_itinerary']),
        'rag_text' : format_rag_text(memory_dict)

    }
    return response, memory_dict



 # Itinerary Q&A
def handle_itinerary_qa(user_message, memory_dict, history = None):
    'Agent answers user question regarding current itinerary'
    print('Handling itinerary Q&A...\n')
    itinerary     = memory_dict.get('itinerary', {})
    day_summaries = memory_dict.get('day_summaries', {})
    violations    = memory_dict.get('all_violations', [])

    itin_lines = []
    for day_key, stops in itinerary.items():
        summary = day_summaries.get(day_key, {})
        itin_lines.append(
            f"{day_key.replace('_',' ').upper()} ({summary.get('day_start_time','?')}–"
            f"{summary.get('day_end_time','?')} | "
            f"{summary.get('total_pois',0)} POIs | "
            f"travel {summary.get('total_travel_min',0)} min | "
            f"POIs visit {summary.get('total_visit_min',0)} min):")
        for stop in stops:
            poi = stop.get('poi', {})
            if stop['poi_id'] != 'meal_break':
                flag_str = f" [⚠️  {','.join(stop['flags'])}]" if stop['flags'] else ''
                itin_lines.append(
                    f"  {stop['start_time']}–{stop['end_time']}  "
                    f"{stop['poi_name']} ({poi['primaryType']})"
                    f"  → {stop['travel_to_next_min']} min to next"
                    f"{flag_str}"
                )
    violation_text = (
        '\n'.join(f"  ⚠️  {v['poi_name']}: {v['issue']} — {v['detail']}"
                  for v in violations)
        if violations else 'None'
    )

    system_prompt = (
        'You are a helpful NYC trip assistant. '
        'Answer questions strictly using the itinerary data provided. '
        'Reply directly to the user in 3-5 sentences. '
        'Do NOT analyse or narrate your reasoning. ' 
        'Do not invent information. Be concise.'
    )
    user_prompt = (
        f'ITINERARY:\n{chr(10).join(itin_lines)}\n\n'
        f'VIOLATIONS: {violation_text}\n\n'
        f'USER QUESTION: {user_message}'
    )    
    return llm_generate(system_prompt, user_prompt, history = history)

# Contextual POI Recommendations
def handle_recommendations(user_message, memory_dict, history = None):
    'Reads dropped POIs'
    print('Handling recommendations...\n')
    dropped_pois = memory_dict.get('dropped_pois', [])
    trip_days    = memory_dict.get('trip_day_indices', [])
    itinerary    = memory_dict.get('itinerary', {})

    if not dropped_pois:
        return "All retrieved POIs are already in your itinerary. No alternatives available."

    # Detect if user mentions a specific day
    target_day_idx = None
    msg = user_message.lower()
    for i in range(1, len(trip_days) + 1):
        if f'day {i}' in msg:
            target_day_idx = i - 1
            break

     #  Filter dropped POIs by open day if day specified
    def is_open_on_day(poi, google_day):
        open_days = {
            p['open']['day']
            for p in (poi.get('regularOpeningHours') or {}).get('periods', [])
            if 'open' in p
        } or set(range(7))
        return google_day in open_days

    if target_day_idx is not None and target_day_idx < len(trip_days):
        google_day = trip_days[target_day_idx]
        candidates = [p for p in dropped_pois if is_open_on_day(p, google_day)]
        day_label  = f'day {target_day_idx + 1}'
    else:
        candidates = list(dropped_pois)
        day_label  = 'any day'

    #  Sort by combined_score descending 
    candidates.sort(key=lambda p: p.get('combined_score', 0), reverse=True)
    top3 = candidates[:3]

    if not top3:
        return f"No alternative POIs found that are open on {day_label}."

    #  Build context for LLM
    poi_lines = []
    for p in top3:
        name  = p.get('displayName', {}).get('text', p.get('id', ''))
        ptype = p.get('primaryType', '')
        zone  = p.get('zone', '')
        score = p.get('combined_score', 0)
        poi_lines.append(f"  - {name} ({ptype}, {zone}, score={score:.2f})")

    system_prompt = (
        'You are a helpful NYC trip assistant. '
        'Recommend POIs from the provided list only. '
        'Reply directly to the user in 3-5 sentences. '
        'Do NOT analyse or narrate your reasoning. '         
        'Be friendly and concise. Do not invent POIs.'
    )
    user_prompt = (
        f'USER REQUEST: {user_message}\n\n'
        f'AVAILABLE ALTERNATIVES (open on {day_label}, ranked by your preferences and overall score):\n'
        f'{chr(10).join(poi_lines)}\n\n'
        f'Suggest these alternatives with a brief reason for each.'
    )

    return llm_generate(system_prompt, user_prompt, history = history)


# Dropped POI Explanation

def infer_drop_reason(poi, itinerary, model_constraints, time_windows):
    '''
    Infers why a dropped POI was not scheduled.
    Checks capacity, time window, and score.
    '''
    max_pois   = model_constraints.get('max_pois_per_day', 5)
    day_counts = {k: len(v) for k, v in itinerary.items()}
    reasons = []

    if all(c >= max_pois for c in day_counts.values()):
        reasons.append('all days at pace capacity')

    poi_id = poi.get('id', '')
    if poi_id in time_windows:
        tw_s, tw_e = time_windows.get(poi_id, [])
        if tw_e - tw_s < 60:
            reasons.append('very narrow time window — conflicted with other POIs')
    score = poi.get('combined_score', 0)
    reasons.append(f'lower priority score ({score:.2f}) — dropped to fit constraints')

    return ', '.join(reasons)



def handle_drop_explanation(user_message, memory_dict, history = None):
    '''
    Reads removed pois(hard filter reasons) + 
    dropped_pois (inferred reasons) + 
    forced_unscheduled (must-include failures).
    '''
    print('Handling drop explanation...\n')
    removed_pois      = memory_dict.get('removed_pois', [])       
    dropped_pois      = memory_dict.get('dropped_pois', [])      
    forced_unsched    = memory_dict.get('forced_pois_unscheduled', []) 
    itinerary  = memory_dict.get('itinerary', {})
    constraints = memory_dict.get('model_constraints', {})
    time_windows      = memory_dict.get('time_windows', [])

    msg = user_message.lower()
    words = [word for word in msg.split() if len(word) > 4]
    print('words', words,'\n')

     #  Try to identify which POI user is asking about 
    def name_matches(poi, query):
        name = (poi.get('displayName', {}).get('text', '')).lower()
        return name if any(w in name for w in words) else ''



    explanation_lines = []
    # Stage 4 hard-filtered
    for poi in removed_pois:
        name = name_matches(poi, msg)
        print('name', name)
        if name:
            explanation_lines.append(f"'{name}' removed in pre-processing")

     # Stage 5 dropped
    for poi in dropped_pois:
        name = name_matches(poi, msg)
        print('name', name)
        if name:
            reason = infer_drop_reason(poi, itinerary, constraints, time_windows)
            explanation_lines.append(f"'{name}' not scheduled for travel optimization: {reason}")

     # Stage 5 forced_unscheduled (must-include failures)
    for poi in forced_unsched:
        name = name_matches(poi, msg)
        print('name', name)
        if name:
            reason = infer_drop_reason(poi, itinerary, constraints, time_windows)
            explanation_lines.append(
                    f"'{name}' (must-include) could not be scheduled : {reason}"
            )
    if not explanation_lines:
        return handle_conflict_explanation(user_message = user_message, memory_dict = memory_dict, history = history)
        


    exclude_str    = f" | Must exclude: {', '.join(constraints['must_exclude_types'])}" if constraints['must_exclude_types'] else ''
    days_avoid_str = f" | Days avoided: {', '.join(DAY_LOOKUP[d] for d in constraints['days_avoided'])}" if constraints['days_avoided'] else ''

    # Build text for user constraints
    user_const = (
        f"Max POIs per day : {constraints['max_pois_per_day']} "
        f"{constraints['pace']} pace; total {constraints['num_days']} trip days "
        f"{exclude_str}"
        f"{days_avoid_str}")

    system_prompt = (
        'You are a helpful NYC trip assistant. '
        'Explain clearly why the mentioned POI was not included in the itinerary. '
        'Reply directly to the user in 3-5 sentences. '
        'NEVER analyse or narrate your reasoning. '         
        'Use only the provided reasons. Be concise and empathetic.'
    )
    user_prompt = (
        f'USER QUESTION: {user_message}\n\n'
        f"USER CONSTRAINTS : {user_const}\n"
        f'EXCLUSION REASONS:\n'
        + '\n'.join(f'  - {line}' for line in explanation_lines)
        + '\n\nExplain these exclusions in a friendly, clear way.'
    )

    return llm_generate(system_prompt, user_prompt, history = history)

#  Constraint Conflict Explanation 
def handle_conflict_explanation(user_message, memory_dict, history = None):
    '''
    Constraint Conflict Explanation
    - LLM explains why a request conflicts with the constraints 
    - Prior Q&A turns passed for LLM context
    '''
    print('Handling conflict explanation...\n')
    constraints = memory_dict.get('model_constraints',{})
    day_summaries = memory_dict.get('day_summaries',{})

    day_loads = []
    for day_key, summary in day_summaries.items():
        day_loads.append(
            f"{day_key} : Total {summary.get('total_pois', 0)} POIs | "
            f"{summary.get('total_travel_min', 0)} min travel | "
            f"{summary.get('total_visit_min', 0)} min visits | "
            f"ends {summary.get('day_end_time', '?')}"
        )
    visited_pois = {}
    itinerary = memory_dict.get('itinerary', {})
    for day, stops in itinerary.items():
        if stops:
            for stop in stops:
                pid = stop['poi_id']
                if pid != 'meal_break': 
                    visited_pois[pid]= stop.get('poi_name','')
    must_names = [v_n for vid,v_n in visited_pois.items() if vid in constraints.get('must_poi_ids', [])]
    constraint_text = (
        f"Max POIs per day : {constraints.get('max_pois_per_day', 'N/A')} | "
        f"Trip pace : {constraints.get('pace', 'N/A')} | "
        f"Trip duration : {constraints.get('num_days', 'N/A')} days | " 
        f"Excluded types : {', '.join(constraints.get('must_exclude_types', [])) or 'none'} | "
        f"Days avoided : {', '.join(DAY_LOOKUP[d] for d in constraints.get('days_avoided', [])) or 'none' } | "
        f"Must-include POIs : {', '.join(must_names)}"
    )
    system_prompt = (
       'You are a helpful NYC trip assistant. '
       'Explain scheduling conflicts using only the provided constraint data. '
       'Use conversation history to understand what the user is referring to. '
       'Reply directly to the user in 3-5 sentences. '
       'Do NOT analyse or narrate your reasoning. '         
       'Do NOT repeat the constraint data back. '    
       'Suggest one actionable resolution. Do not invent data.'
   )
    user_prompt = (
        f"User request : {user_message} \n"
        f"Current constraints : \n{constraint_text}\n"
        f"Current day loads:\n{'; '.join(day_loads) or 'No day data'}\n\n"
        'Explain why this request conflicts and suggest one resolution'
    )
    return llm_generate(system_prompt, user_prompt, history = history)


#Router schema
def make_router_prompt(query, history = None):
    '''
    Define router schema for intent classifier
    Uses last 1-2 messages for reference 
    '''
    context_block = ''
    if history:
        prior_user = [entry['content'] for entry in history if entry['role']=='user']
        if prior_user:
            lines = '\n'.join(f'-{p}' for p in prior_user[-2:])
            context_block = f'\nRECENT USER QUESTIONS (for context only) : \n{lines}\n'
    return f"""
    You are an intent router for a Trip Planner Q&A Agent.
    Return ONLY valid JSON. No explanation, no markdown.
    Schema:
    {{
    "plan" : [
    {{"handler" : "<handler_name>", "args" :{{"user_message" : "<user_message>"}}}}
    ]
    }}

    Available handlers and when to use them : 
    1) "generate_itinerary" -  user wants a new itinerary/trip generated with preferences written in user_message
    2) "recommendations" - user asks for suggestions, alternatives, or additional POI options
    3) "drop_explanation" - user mentions a specific POI and asks why the POI was excluded, missed, or dropped
    4) "conflict_explanation" - user_message conflicts with the constraints, capacity, or feasibility of current itinerary
    5)"itinerary_qa" - questions about current itinerary: times, days, stops, violations, schedule, flags



    Routing rules (follow strictly):
     - First read all the above handlers and then pick the BEST handler 
     - If ambigous, prefer "itinerary_qa"

     {context_block}
     Current question : 
     {query}
     JSON:
    """.strip()

def llm_generate(system_prompt, user_prompt, history = None, max_new_tokens = 400):
    '''
    Generate response using Llama-3.2-3B-Instruct-4bit via mlx_lm.
    Falls back to returning user_prompt content directly if LLM unavailable.
    '''
    if not LLM_AVAILABLE:
        return user_prompt
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    response = generate(
        model, tokenizer,
        prompt = prompt,
        max_tokens = max_new_tokens,
        verbose = False
    )
    return response.strip()




# Extract valid JSON from the router prompt
def extract_json(text):
    'Extract valid JSON from the router prompt'
    text=re.sub(r'```(?:json)?','',text).strip()
    
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError('No JSON object found')
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                match = text[start_idx : i+1]
                try:
                    return json.loads(match)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSON from LLM: {exc}') from exc
    raise ValueError('Unbalanced braces - no complete JSON object found')

def clean_response(text):
    '''
    Strip LLM chain-of-thought preamble from user-facing responses.
    '''
    # Remove common preamble patterns the model emits before the real answer
    preamble_patterns = [
        r"^(Analyzing|Let me|I will|I'm going to|Based on the|Looking at|"
        r"The (user|constraint|data)|According to|To (answer|resolve|address))"
    ]
    import re
    lines = text.split('\n')
    # Find where actual answer starts — skip leading reasoning lines
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.match(p, stripped, re.IGNORECASE) for p in preamble_patterns):
            start = i + 1
        else:
            break
    cleaned = '\n'.join(lines[start:]).strip()
    return cleaned if cleaned else text.strip()

def route_question(question, history = None):
    '''
    Implement the router that classifies user intent and outputs structured JSON
    '''
    router_text = make_router_prompt(question, history)

    system_prompt = "Output strictly valid JSON only. No extra text, no explanation, no markdown, no preamble."
    raw = llm_generate(system_prompt = system_prompt, user_prompt = router_text, max_new_tokens = 250)
    logger.debug('Router raw output: %s', raw)

    try:
        plan = extract_json(raw)
        if 'plan' not in plan or not isinstance(plan['plan'], list) or len(plan['plan'])==0:
            raise ValueError('Invalid plan schema')
        return plan
    except Exception:
        return {'plan' : [
            {'handler' : 'itinerary_qa', 'args' : {'user_message' : question}}
        ]}



class ConversationAgent:
    '''
    Multi-turn conversation agent for trip itinerary Q&A.

    Uses:
    - history kept for context
    - SessionMemory dict for structured pipeline data
    - Auto-resets history after AUTO_RESET_TURNS exchanges
    - Llama-3.2-3B-Instruct-4bit for response generation

    '''
    HANDLERS = {
        'generate_itinerary' : generate_itinerary,
        'itinerary_qa':         handle_itinerary_qa,
        'recommendations':      handle_recommendations,
        'conflict_explanation': handle_conflict_explanation,
        'drop_explanation':     handle_drop_explanation,
    }
    WINDOW = 4 # last 4 exchanges kept
    AUTO_RESET_TURNS = 10

    def __init__(self):
        self.memory_dict = {}
        self.history = deque(maxlen = self.WINDOW * 2)
        self.turn_count = 0
        logger.info('ConversationAgent initialised')

    def chat(self, user_message):
        if self.turn_count >= self.AUTO_RESET_TURNS:
            logger.info('Auto-reset after %s exchanges', self.turn_count)
            self.reset()
        plan = route_question(user_message, self.history)
        response = ''
        for step in plan.get('plan', []):
            handler_name = step.get('handler','')
            msg=step['args'].get('user_message', user_message)

            if handler_name == 'generate_itinerary':
                response, self.memory_dict = generate_itinerary(user_message = msg)


            else:
                if not self.memory_dict:
                    response = (
                        "Please generate an itinerary first before asking questions about it. "
                    )
                    break
                args = {
                    'user_message': msg,
                    'memory_dict': self.memory_dict,
                    'history' : list(self.history)
                }


                if handler_name not in self.HANDLERS:
                    # unknown intent — general response using itinerary context
                    logger.warning('Unknown handler %s - falling back to itinerary_qa', handler_name)
                    response = handle_itinerary_qa(**args)
                else:   
                    handler_func = self.HANDLERS[handler_name]
                    response = handler_func(**args)
                    response = clean_response(response)
                if response:
                    self.history.append({'role': 'user', 'content': user_message})
                    self.history.append({'role' : 'assistant', 'content' : response})
                    self.turn_count += 1 

        if not response:
            response = "I'm sorry, I couldn't understand that. Please try rephrasing."

        return response

    def reset(self, keep_memory = True):
        '''
        keep_memory = True -> only session memory kept, history deleted
        keep_memory = False -> deletes history and session memory (new trip session)
        '''
        self.history.clear()
        self.turn_count = 0
        if not keep_memory:
            self.memory_dict = {}
            logger.info('Full session reset')
        else:
            logger.info('History cleared - session memory preserved')


