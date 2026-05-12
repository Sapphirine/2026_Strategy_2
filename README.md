# 🗽 TripEase - Sentiment-Aware AI Agent for Intelligent Travel Itinerary Planning



TripEase is an end-to-end AI travel agent that takes a plain-English trip query and produces a fully scheduled, time-validated, multi-day itinerary. It combines NLP query parsing, aspect-based sentiment analysis of visitor reviews, retrieval-augmented generation from past trips, VRP-based tour optimisation, exact temporal scheduling, and a conversational Gradio UI — all running locally on Apple Silicon.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Module Reference](#module-reference)
4. [Data Sources](#data-sources)
5. [Models Used](#models-used)
6. [Installation](#installation)
7. [Running the Project](#running-the-project)
8. [Usage Guide](#usage-guide)
9. [Output Files](#output-files)
10. [Repository Structure](#repository-structure)
11. [Constraints & Design Decisions](#constraints--design-decisions)
12. [Known Limitations](#known-limitations)

---

## Project Overview

| Property | Detail |
|---|---|
| **Domain** | NYC tourism — Points of Interest (POIs) across 8 boroughs/zones |
| **Input** | Free-text trip query (group type, duration, dates, preferences, must-includes, pace) |
| **Output** | Day-by-day scheduled itinerary + interactive map + timeline + conversational Q&A |
| **Hardware target** | Apple M1 Pro · 16 GB RAM · MPS acceleration |
| **LLM (generation)** | `mlx-community/Llama-3.2-3B-Instruct-4bit` via `mlx_lm` |
| **ABSA model** | `yangheng/deberta-v3-base-absa-v1.1` via HuggingFace Transformers |
| **Optimiser** | Google OR-Tools CVRPTW solver |
| **UI** | Gradio Blocks |

---

## Pipeline Architecture

```
User natural-language query
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — NLP Perception  (query_parser.py)            │
│  Llama-3.2-3B-Instruct-4bit (4-bit quantised, MLX)      │
│  Extracts: group_type, duration, dates, pace,           │
│  preferred zones/categories, must-include POIs,         │
│  hard + soft constraints, implicit needs                │
└───────────────────────┬─────────────────────────────────┘
                        │ parsed_query (dict)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — ABSA  (ABSA.py)                              │
│  DeBERTa-v3-base-absa-v1.1 (HuggingFace, MPS)           │
│  Runs aspect-level sentiment on every candidate POI's   │
│  Google reviews; aspects: crowd, value, family,         │
│  exhibits, staff, location, solo, couple, friends,      │
│  accessibility                                          │
│  Scores each POI → combined_score (ABSA + rating)       │
└───────────────────────┬─────────────────────────────────┘
                        │ absa_results (scored POI list)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3 — RAG  (RAG.py)                                │
│  SentenceTransformer + FAISS                            │
│  Retrieves semantically similar past itineraries from   │
│  itineraries_history.json; surfaces them in UI as       │
│  "trips similar to yours" to inform the current plan    │
└───────────────────────┬─────────────────────────────────┘
                        │ rag_output (candidates + retrieved)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4 — Pre-processing + Travel Matrix (time_matrix) │
│  Hard-filter POIs closed on all trip days               │
│  Enforce must-include survival                          │
│  Compute travel-time matrix (Haversine distance)        │
│  Output: filtered POI list + travel matrix + hotel depot│
└───────────────────────┬─────────────────────────────────┘
                        │ travel_data (filtered POIs, matrix)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5 — VRP Solver  (assignment_sequence.py)         │
│  Google OR-Tools CVRPTW                                 │
│  Simultaneously solves: which POIs on which day +       │
│  sequence within each day + global travel minimisation  │
│  Hard: time windows, day capacity (12 hr), max POIs,    │
│        time-of-day prefs, must-include, visit-once      │
│  Soft: day balance, preferred zones                     │
│  Output: {day_1: [ordered POIs], day_2: [...]}          │
└───────────────────────┬─────────────────────────────────┘
                        │ sequence_output
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 6 — Temporal Scheduling  (TemporalScheduling.py) │
│  Walks each day sequentially:                           │
│  Assigns exact clock start/end times                    │
│  Inserts travel slots + meal break (after 12:00)        │
│  Verifies against opening hours → flags violations      │
│  Appends trip to itineraries_history.json               │
│  Output: {day_1: [{poi, start_time, end_time, ...}]}    │
└───────────────────────┬─────────────────────────────────┘
                        │ final_itinerary
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 7 — Conversational Agent  (Agent.py)             │
│  ConversationAgent (deque windowed history, 4 turns)    │
│  LLM router → classifies intent → dispatches handler    │
│                                                         │
│  Handlers:                                              │
│  • generate_itinerary   — triggers Stages 1-6           │
│  • itinerary_qa         — Q&A on current schedule       │
│  • recommendations      — suggest dropped/alt POIs      │
│  • conflict_explanation — explain scheduling conflicts  │
│  • drop_explanation     — explain why a POI was removed │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 8 — UI  (UI.py)                                  │
│  Gradio Blocks                                          │
│  Left:  chat panel + example chips                      │
│  Right: tabbed panel —                                  │
│    🗺️  Map (Folium HTML),  📅 Timeline (Plotly),         │
│    ⭐  Place Insights (ABSA cards), 📊 Trip Stats,       │
│    📋  Full Itinerary cards  & Suggestions               │
│  Visualisations rendered ONCE on itinerary generation;  │
│  follow-up chat turns do NOT re-render them             │
└─────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `query_parser.py` — Stage 1: NLP Perception
**Purpose:** Parse a free-text user query into a structured `ParsedQuery` dataclass.

**Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (4-bit quantised, runs via `mlx_lm` on MPS, ~2 GB VRAM)

**Key outputs (all fields of `parsed_query` dict):**
| Field | Type | Example |
|---|---|---|
| `group_type` | str | `"family"` |
| `trip_duration_days` | int | `3` |
| `start_date` | str | `"2026-06-06"` |
| `pace` | str | `"moderate"` |
| `preferred_zones` | list[str] | `["midtown"]` |
| `preferred_categories` | list[str] | `["tourism_attraction"]` |
| `must_include_pois` | list[str] | `["Empire State Building"]` |
| `hard_constraints` | dict | avoid_days, must_exclude_types |
| `soft_constraints` | dict | prefer_less_crowded, time_of_day_prefs |
| `implicit_needs` | list[str] | `["kid_friendly"]` |

**Public API:**
```python
query_parser.initialize(pois_database, pois_by_category)
parser = query_parser.QueryParser()
pq = parser.parse("3-day family trip, must visit Empire State")
parsed_query = pq.to_dict()
```

---

### `ABSA.py` — Stage 2: Aspect-Based Sentiment Analysis
**Purpose:** Score every candidate POI by analysing Google review text across 10 tourism-relevant aspects.

**Model:** `yangheng/deberta-v3-base-absa-v1.1` (DeBERTa-v3, ~180 MB, loaded once on MPS)

**Aspects analysed:** `crowd`, `value`, `family`, `exhibits`, `staff`, `location`, `solo`, `couple`, `friends`, `accessibility`

**Scoring formula:**
```
combined_score = α × absa_weighted_score + β × bayesian_rating_score
```
where the Bayesian rating score shrinks small-sample ratings toward the global mean.

**Per-POI output (`aspects_sentiments` dict):**
```json
{
  "crowd": {
    "label": "positive",
    "aspect_weighted_score": 0.72,
    "mentions": 14,
    "positive": 10, "neutral": 3, "negative": 1
  }
}
```

**Public API:**
```python
ABSA.initialize(pois_database, pois_by_category)
analyzer = ABSA.ABSA_analyzer(parsed_query)
absa_results = analyzer.analyze_all()   # list of scored POI dicts
```

---

### `RAG.py` — Stage 3: Retrieval-Augmented Generation
**Purpose:** Semantic retrieval of past itineraries most similar to the current query, to inform the current plan and surface them in the UI.

**Model:** `all-MiniLM-L6-v2` SentenceTransformer + FAISS flat index (CPU)

**Index built over:** `itineraries_history.json` — each entry is encoded as a feature vector from `(group_type, days, pace, zones, categories, satisfaction_score)`.

**Output (`rag_output` dict):**
```json
{
  "candidates": [...],          // scored POI list carried forward to Stage 4
  "retrieved_itineraries": [...] // top-k similar past trips
}
```

**Public API:**
```python
RAG.initialize_db(pois_database, pois_by_category)
RAG.initialize(parsed_query, absa_results)
rag_output = RAG.rag(parsed_query, absa_results)
```

---

### `time_matrix.py` — Stage 4: Pre-processing + Travel Matrix
**Purpose:** Filter POIs that cannot be visited on any trip day; compute Haversine travel-time matrix between all surviving POIs + hotel depot.

**Hard filters applied:**
- POI closed on **all** trip days → removed
- POI `businessStatus != OPERATIONAL` → removed
- `must_include` POIs bypass soft filters but are still subject to hard ones

**Travel time formula:** Haversine distance ÷ 5 km/h average NYC walking/transit speed, minimum 4 minutes.

**Output (`travel_data` dict):**
| Key | Description |
|---|---|
| `selected_pois` | POIs surviving all filters (with ABSA enrichment) |
| `removed_pois` | Hard-filtered POIs + reason strings |
| `travel_time_matrix` | N×N matrix in minutes |
| `depot` | Hotel/start location coordinates |
| `trip_day_indices` | Google weekday integers for each trip day |

**Public API:**
```python
travel_data = time_matrix.TravelTime(parsed_query, rag_pois=rag_output['candidates'])
```

---

### `assignment_sequence.py` — Stage 5: VRP Solver
**Purpose:** Globally optimise which POIs go on which day **and** in what sequence, simultaneously, using Google OR-Tools CVRPTW.

**Solver configuration:**
- Arc cost: travel time between POIs (minutes × 100 for integer precision)
- Time-window constraints: from `regularOpeningHours` per POI
- Day capacity: 780 min − 60 min meal break = 720 min usable
- Max POIs per day: pace-dependent (relaxed → 6, moderate → 5, fast → 7)
- Must-include penalty: 10,000,000 (effectively hard constraint)
- Solve time limit: 60 seconds

**Soft constraints as penalties:**
- Day imbalance penalty
- Zone clustering reward (nearby POIs grouped)

**Output (`sequence_output` dict):**
| Key | Description |
|---|---|
| `day_sequences` | `{day_key: [ordered poi_ids]}` |
| `dropped_pois` | POIs not scheduled (capacity/time conflicts) |
| `forced_unscheduled` | Must-include POIs solver could not fit |
| `feasible` | bool — whether a valid schedule was found |
| `model_constraints` | Solver parameters used |
| `time_windows` | `{poi_id: [open_min, close_min]}` |

**Public API:**
```python
sequence_output = assignment_sequence.build_sequence(travel_data)
```

---

### `TemporalScheduling.py` — Stage 6: Temporal Scheduling
**Purpose:** Convert the ordered POI sequences from Stage 5 into a precise clock-time schedule, inserting travel slots and a meal break, and flagging any opening-hour violations.

**Walk-through per day:**
1. Start from `DAY_START_HOUR = 9:00 AM`
2. For each POI: `start_time = arrival_time`, `end_time = start_time + visit_duration`
3. Insert `travel_to_next` gap
4. Insert 60-min meal break after 12:00
5. Flag if `start_time < open` or `end_time > close`

**Appends to `itineraries_history.json`** for future RAG retrieval.

**Output (`final_itinerary` dict):**
```json
{
  "itinerary": {
    "day_1": [
      {"poi_id": "...", "poi_name": "...", "start_time": "09:00",
       "end_time": "10:30", "travel_to_next_min": 12, "flags": [], "poi": {...}}
    ]
  },
  "day_summaries": {"day_1": {"total_pois": 4, "total_travel_min": 38, ...}},
  "all_violations": [],
  "total_violations": 0,
  "solver_status": "FEASIBLE",
  "metrics": {...},
  "trip_day_indices": [1, 2, 3]
}
```

**Public API:**
```python
TemporalScheduling.initialize(travel_data)
final_itinerary = TemporalScheduling.temporalScheduling(
    sequence_output, store_path='itineraries_history.json'
)
```

---

### `Agent.py` — Stage 7: Conversational Agent
**Purpose:** Multi-turn chat interface orchestrating the full pipeline and all Q&A handlers.

**Architecture:**
- `ConversationAgent` class with `deque(maxlen=8)` windowed history (last 4 turns)
- Auto-reset after 10 exchanges (history cleared, session memory preserved)
- `route_question()` — LLM-powered intent router → JSON plan
- `SessionMemory` (in `pipeline.py`) — structured dict carrying all pipeline state

**Intent routing (5 handlers):**
| Handler | Trigger |
|---|---|
| `generate_itinerary` | User describes a new trip |
| `itinerary_qa` | Questions about current schedule |
| `recommendations` | Requests for alternatives / extra POIs |
| `conflict_explanation` | Request conflicts with schedule capacity |
| `drop_explanation` | User asks why a specific POI was excluded |

**Anti-hallucination design:**
- All handlers receive **only pre-computed pipeline data** — no LLM inference on factual details
- LLM is used for **natural language generation only**, never for fact retrieval
- System prompts explicitly instruct: "Do not invent information"
- Router falls back to `itinerary_qa` on any parse failure

**Public API:**
```python
agent = ConversationAgent()
response = agent.chat("Generate a 3-day family trip...")
response = agent.chat("Why was the zoo not included?")
agent.reset(keep_memory=True)   # clear history, keep itinerary data
agent.reset(keep_memory=False)  # full new session
```

---

### `visualize.py` — Visualisations
**Purpose:** Render the itinerary as an interactive Folium map and a Plotly Gantt-style timeline.

**`plot_map(final_itinerary)`**
- One Folium marker per POI, colour-coded by day
- Polyline route connecting stops in sequence
- Rich popup HTML: name, type, rating, address, review snippets, visit time
- Photo thumbnails via Google Places API (graceful `onerror` fallback)
- Saves to `itinerary_map.html` → served via Gradio file endpoint

**`plot_timeline(final_itinerary)`**
- Plotly horizontal bar chart (Gantt) — one row per day
- Bars colour-coded by `primaryType`
- Hover tooltip: POI name, times, rating, combined score
- Travel gaps shown as light-grey bars
- Saves to `itinerary_timeline.html`

---

### `UI.py` — Stage 8: Gradio Interface
**Purpose:** Web UI bridging the chat agent and visualisations.

**Layout (two columns):**
| Left (chat) | Right (tabs) |
|---|---|
| Chatbot panel | 🗺️ Map (Folium iframe) |
| Text input + Send/Reset | 📅 Timeline (Plotly iframe) |
| Quick-action chips | ⭐ Place Insights (ABSA cards) |
| Example query accordion | 📊 Trip Stats & Metrics |
| | 📋 Full Itinerary cards & Suggestions|

**Rendering rule:** Map, timeline, ABSA, stats, and itinerary panels update **only** when `generate_itinerary` fires. Follow-up Q&A turns update only the chatbot.

**`handle_message(user_message, history, memory_dict)`** — main Gradio callback:
```
user message → agent.chat() → parse response type
  if type == 'itinerary' → update all panels
  else                   → update chat only
```

---

### `pipeline.py` — Pipeline Orchestrator
**Purpose:** Sequence Stages 1-6, manage `SessionMemory`, handle abort conditions.

**Abort paths:**
- Stage 4 returns empty `selected_pois` → `STAGE4_NO_POIS`
- Stage 5 returns `feasible=False` → `STAGE5_INFEASIBLE`

Both return a structured `empty_result` dict so the agent can surface a graceful explanation to the user.

**`SessionMemory` class:**
- Single source of truth for all handlers in Stage 7
- Carries: itinerary, day summaries, violations, dropped POIs, constraints, time windows, RAG context
- `.to_dict()` → `memory_dict` passed through all handlers
- `.dump('session_memory.json')` → persists session for debugging

---

### `evaluation.py` — Metrics
**Purpose:** Compute quality metrics for a generated itinerary.

**Metrics computed:**
- `constraint_satisfaction_rate` — % of hard constraints met
- `cross_zone_rate` — zone change rate
- `day_balance_std` — standard deviation of POI counts across days
- `backtracking_score` — travel time optimization rate
- `total_travel_min` — total travel time
- `preference_metrics` - % of preferences met/avoided
- `invalid_day_assignment_rate` - % of POIs assigned on a closed day

---

## Data Sources

### `nyc_pois_database.json`
Full POI records fetched from Google Places API (New). Each entry contains:
```
id, displayName, primaryType, types, categories,
location {latitude, longitude},
zone, rating, userRatingCount, businessStatus,
formattedAddress, photos, reviews, regularOpeningHours
```

### `nyc_pois_by_category.json`
Same POIs indexed by category for fast lookup during ABSA and filtering.

### `itineraries_history.json`
Growing log of completed itineraries. Each entry is a full `final_itinerary` dict with its `query_profile` and `evaluation_score`. Used as the RAG retrieval corpus. Appended automatically at the end of every successful pipeline run.

**POI zones:** `lower_manhattan`, `midtown`, `upper_east_side`, `upper_west_side`, `brooklyn`, `bronx`, `queens`, `staten_island`

**POI categories:**
```
tourism_attraction  ·  history_culture_art  ·  parks_nature_beach
nightlife           ·  recreation_active
```

---

## Models Used

| Stage | Model | Library | Device | Approx. Size |
|---|---|---|---|---|
| 1 — Query parsing | `mlx-community/Llama-3.2-3B-Instruct-4bit` | `mlx_lm` | MPS (unified) | ~1.8 GB |
| 2 — ABSA | `yangheng/deberta-v3-base-absa-v1.1` | `transformers` | MPS | ~180 MB |
| 3 — RAG embeddings | `all-MiniLM-L6-v2` | `sentence-transformers` | CPU | ~80 MB |
| 7 — Agent / router / generation | Same Llama model (reused from Stage 1) | `mlx_lm` | MPS | shared |

> All models are downloaded automatically on first run via Hugging Face Hub. A Hugging Face token is required for the Llama model (gated repo).

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- Jupyter Notebook or JupyterLab
- Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct`

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/itinerary-planner-ai.git
cd itinerary-planner-ai
```

### 2. Create a virtual environment
```bash
python -m venv tripease-env
source tripease-env/bin/activate
```

### 3. Install dependencies

**Core ML / LLM stack (Apple Silicon only):**
```bash
pip install mlx mlx-lm
```

**ABSA + embeddings:**
```bash
pip install transformers accelerate sentence-transformers
pip install torch          # CPU/MPS wheels — no CUDA needed on Apple Silicon
```

**Vector search:**
```bash
pip install faiss-cpu
```

**VRP solver:**
```bash
pip install ortools
```

**Visualisation:**
```bash
pip install folium plotly
```

**UI:**
```bash
pip install gradio
```

**Utilities:**
```bash
pip install numpy pandas
```

**One-liner (all at once):**
```bash
pip install mlx mlx-lm transformers accelerate torch \
            sentence-transformers faiss-cpu ortools \
            folium plotly gradio numpy pandas
```

### 4. Hugging Face authentication
The Llama model is gated. Create a file `llama_.py` (already gitignored) in the project root:
```python
hf_key = YOUR_HF_TOKEN_HERE
```
Then request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### 5. Google Places API key
Create a file `config.py` in the project root:
```python
API_KEY = YOUR_GOOGLE_API_TOKEN
```
This key is used to build photo URLs in itinerary cards. The pipeline runs without it — photo thumbnails will simply be hidden.

### 6. Data files
Place the following files in the project root:
```
nyc_pois_database.json
nyc_pois_by_category.json
```

---

## Running the Project

### Option A — Gradio UI (recommended)
```bash
# From the project root, with the virtual environment active:
jupyter notebook UI.ipynb
# Run all cells — the Gradio interface launches at http://127.0.0.1:7860
```

Or if using the converted `.py` script:
```bash
python UI.py
```

### Option B — Notebook pipeline (stage by stage)
Open and run cells in sequence:
```
query_parser.ipynb   →  ABSA.ipynb   →  RAG.ipynb
time_matrix.ipynb    →  assignment&sequence.ipynb
TemporalScheduling.ipynb  →  Agent.ipynb  →  UI.ipynb
```
Each notebook saves its output as a `.json` file that the next stage reads.

### Option C — Programmatic (Python script)
```python
import json
import pipeline
from Agent import ConversationAgent

# Load databases once
with open('nyc_pois_database.json') as f:
    pois_database = json.load(f)
with open('nyc_pois_by_category.json') as f:
    pois_by_category = json.load(f)

# Initialise pipeline
pipeline.initialize(pois_database, pois_by_category)

# Start agent
agent = ConversationAgent()

# Generate itinerary
response = agent.chat(
    "Generate a 3-day family trip starting 6 June 2026. "
    "Kids love zoos and aquariums. Empire State is a must. "
    "Moderate pace. Staying in midtown."
)

# Follow-up Q&A
print(agent.chat("What time does day 2 start?"))
print(agent.chat("Why was Central Park Zoo not included?"))
print(agent.chat("What else can I visit on day 3?"))
```

---

## Usage Guide

### Writing trip queries
The agent understands natural language. Include any combination of:

| Intent | Example phrasing |
|---|---|
| Duration | `"3-day trip"`, `"weekend trip"` |
| Dates | `"starting June 6"`, `"May 12–14"` |
| Group | `"family with kids"`, `"solo"`, `"couple"`, `"with friends"` |
| Zone | `"staying in midtown"`, `"Brooklyn waterfront"` |
| Categories | `"museums and art galleries"`, `"parks and nature"` |
| Pace | `"relaxed"`, `"moderate"`, `"fast-paced"` |
| Must-includes | `"Empire State is a must"`, `"I must see the MET"` |
| Avoid | `"avoid Wednesdays"`, `"no nightlife"` |
| Implicit | `"kids love zoos"` → infers `kid_friendly` need |
| Crowd preference | `"avoid crowds"` → weights low-crowd ABSA scores |

### Conversation examples
```
You:   "Generate a 3-day family trip starting 6 June. Kids love zoos.
        Empire State must. Moderate pace. Midtown."

Agent: [Itinerary + map + timeline rendered]

You:   "Why does day 1 end so early?"
You:   "What other places can I visit on day 2?"
You:   "Why was Central Park Zoo not included?"
You:   "Can I add one more stop without going over 12 hours?"
```

### Quick-action chips (UI)
| Chip | What it triggers |
|---|---|
| 🤔 Why was X dropped? | `drop_explanation` handler |
| 🌐 What else can I visit on day 2? | `recommendations` handler |
| ⚠️ Explain the conflict, why no X? | `conflict_explanation` handler |
| 📋 Summarise day 1 | `itinerary_qa` handler |
| 💯 What's my trip score? | `itinerary_qa` handler |

---

## Output Files

| File | Stage | Description |
|---|---|---|
| `parsed_query.json` | 1 | Structured constraints from NLP parsing |
| `absa_output.json` | 2 | ABSA scores per POI + parsed_query |
| `rag_output.json` | 3 | Candidate POIs + retrieved past trips |
| `travel_data.json` | 4 | Filtered POIs + travel-time matrix |
| `sequence_output.json` | 5 | Day-by-day ordered POI sequences |
| `final_itinerary.json` | 6 | Full scheduled itinerary with clock times |
| `session_memory.json` | 7 | Agent's structured session state |
| `itineraries_history.json` | 6→3 | Cumulative trip log (RAG corpus) |
| `itinerary_map.html` | 7 | Interactive Folium map |
| `itinerary_timeline.html` | 7 | Plotly Gantt timeline |

---

## Project Structure

```
.
├── src/
│   ├── query_parser.py    # Stage 1 — NLP query parsing (Llama)
│   ├── ABSA.py   # Stage 2 — Aspect-based sentiment analysis (DeBERTa)
│   ├── RAG.py    # Stage 3 — Semantic retrieval (FAISS + SentenceTransformer)
│   ├── time_matrix.py           # Stage 4 — POI filtering + Haversine travel matrix
│   ├── assignment_sequence.py   # Stage 5 — CVRPTW solver (OR-Tools)
│   ├── TemporalScheduling.py    # Stage 6 — Clock-time scheduling
│   ├── Agent.py    # Stage 7 — Conversational agent + all handlers
│   ├── visualize.py   # Folium map + Plotly timeline rendering
│   ├── pipeline.py       # Pipeline orchestrator + SessionMemory
│   ├── UI.py      # Stage 8 — Gradio Blocks interface
│   ├── evaluation.py    # Itinerary quality metrics
│   ├── config.py      # Google API key (gitignored)
│   └── llama_.py     # Hugging Face token (gitignored)
├──data/
│   ├── nyc_pois_database.json   # Primary POI database (Google Places data)
│   └── nyc_pois_by_category.json # POIs indexed by category
├──results/
│   ├── parsed_query.json  # Stage 1 output (auto-generated)
│   ├── absa_output.json  # Stage 2 output (auto-generated)
│   ├── rag_output.json   # Stage 3 output (auto-generated)
│   ├── travel_data.json    # Stage 4 output (auto-generated)
│   ├── sequence_output.json     # Stage 5 output (auto-generated)
│   ├── final_itinerary.json     # Stage 6 output (auto-generated)
│   ├── session_memory.json    # Stage 7 session state (auto-generated)
│   ├── itinerary_map.png # Folium map (auto-generated)
    └── itinerary_timeline.png   # Plotly timeline (auto-generated)
```

---

## Constraints & Design Decisions

### Why MLX + 4-bit Llama?
`mlx_lm` uses Apple's MLX framework which runs on the M1's unified memory GPU. The 4-bit quantised Llama-3.2-3B fits comfortably in ~2 GB, leaving plenty of headroom for the DeBERTa ABSA model and OR-Tools. This avoids the need for any cloud API and keeps the full pipeline offline.

### Why DeBERTa for ABSA, not the LLM?
Aspect-based sentiment on short review snippets is a classification task — DeBERTa fine-tuned on ABSA benchmarks is both faster and more accurate than prompting a 3B generative model for this. It processes one POI's reviews in ~2 seconds on MPS.

### Why OR-Tools CVRPTW, not a greedy heuristic?
A greedy day-by-day assignment produces local optima — a POI that seems good on day 1 might block a more compatible cluster on day 2. OR-Tools solves the full multi-day assignment and sequence **simultaneously**, guaranteeing a globally better result within the 60-second solve budget.

### Why FAISS, not a vector database?
For a course-project-scale RAG corpus (hundreds of past itineraries), FAISS in-memory is sufficient, zero-overhead, and requires no external service. A production system would replace this with a persistent store like Chroma or Qdrant.

### Why Gradio, not Streamlit?
Gradio `gr.State` makes it straightforward to carry `memory_dict` between callbacks without session management complexity. The `gr.Chatbot` component with `sanitize_html=False` allows rendering rich HTML itinerary cards directly in the chat panel.

### Anti-hallucination measures
1. Every handler reads only from `memory_dict` (pre-computed pipeline data)
2. LLM is used exclusively for **phrasing** — not for factual retrieval
3. System prompts carry hard instructions: "Do not invent information", "Use only the provided data"
4. Router falls back to `itinerary_qa` on any malformed JSON output
5. `clean_response()` strips chain-of-thought preamble before the user sees a response

---

## Known Limitations

- **NYC only** — POI databases are NYC-specific; extending to other cities requires a new data collection pipeline.
- **Haversine travel times** — Straight-line distance with a fixed 5 km/h speed. Real transit times (subway, walking, taxi) vary significantly; OSRM or Google Directions API would improve accuracy in production.
- **Static databases** — POI opening hours and ratings are snapshots. A production system would refresh via the Google Places API periodically.
- **Single-user session** — The Gradio UI runs one `ConversationAgent` instance globally; concurrent users would share state. Multi-user support requires per-session agent instantiation.
- **Llama 3B reasoning limits** — The 3B model occasionally produces imprecise ABSA target identification on very short or non-English review snippets; the DeBERTa model handles these cases more reliably.

---

## Acknowledgements

- [Google OR-Tools](https://developers.google.com/optimization) — CVRPTW solver
- [MLX Community](https://huggingface.co/mlx-community) — Apple Silicon optimised model weights
- [yangheng/deberta-v3-base-absa-v1.1](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1) — ABSA model
- [sentence-transformers](https://www.sbert.net/) — Semantic embeddings for RAG
- [Folium](https://python-visualization.github.io/folium/) — Interactive maps
- [Plotly](https://plotly.com/python/) — Gantt timeline visualisation
- [Gradio](https://gradio.app/) — Web UI framework

---

*TripEase · Course Project · Built for M1 Pro · All pipeline stages run fully offline after initial model downloads.*
