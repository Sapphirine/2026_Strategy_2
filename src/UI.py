
import gradio as gr
import json, re
import Agent
from Agent import ConversationAgent, clean_response
from IPython.display import HTML, display
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

agent = ConversationAgent()
logger.info('UI ready — ConversationAgent initialised')


DAY_HEX = ['#1565C0', '#6A1B9A', '#00695C', '#AD1457', '#E65100', '#00838F', '#558B2F']
DAY_LIGHT = ['#E3F2FD', '#F3E5F5', '#E0F2F1', '#FCE4EC', '#FFF3E0', '#E0F7FA', '#F1F8E9']

ASPECT_META = {
    'crowd':  ('👥', 'Crowd Level'),
    'value': ('💰', 'Value for Money'),
    'family':  ('👨\u200d👩\u200d👧', 'Family Friendly'),
    'exhibits':  ('🏛️', 'Exhibits / Attractions'),
    'staff':  ('🙋', 'Staff & Guides'),
    'location': ('📍', 'Location / Access'),
    'solo': ('🚶', 'Solo Friendly'),
    'couple':  ('💑', 'Couple Friendly'),
    'friends':  ('🧑\u200d🤝\u200d🧑', 'With Friends'),
    'accessibility': ('♿', 'Accessibility'),
}

SENT_COLOR = {'positive': '#2E7D32', 'neutral': '#E65100', 'negative': '#C62828'}
SENT_BG  = {'positive': '#E8F5E9', 'neutral': '#FFF8E1', 'negative': '#FFEBEE'}
BAR_COLOR  = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#EF5350'}

def ph(emoji, title, sub):
    s = ('display:flex;align-items:center;justify-content:center;'
         'height:460px;flex-direction:column;gap:14px;'
         'background:linear-gradient(135deg,#f5f7fa,#c3cfe2);'
         'border-radius:12px;color:#555;font-family:Inter,sans-serif;')
    return (
        f"<div style='{s}'>"
        f"<div style='font-size:52px'>{emoji}</div>"
        f"<div style='font-size:15px;font-weight:600;color:#333'>{title}</div>"
        f"<div style='font-size:12px;color:#777;text-align:center;max-width:280px'>{sub}</div>"
        "</div>"
    )

MAP_PH   = ph('🗺️', 'Map will appear here',
                    'Generate your itinerary to see the interactive route map')
TIMELINE_PH = ph('📅', 'Timeline will appear here',
                    'Day-by-day schedule, colour-coded by POI type')
ABSA_PH  = ph('⭐', 'POI Insights will appear here',
                    'Sentiment analysis from verified visitor reviews')
STATS_PH  = ph('📊', 'Trip Stats will appear here',
                    'Optimisation quality, constraint satisfaction, and metrics')
ITINERARY_PH = ph('📋', 'Itinerary will appear here',
                   'Your full day-by-day schedule with timings, ratings and map links')

import os

def embed_html(source, height='492px'):
    
    if not source:
        return None

    if isinstance(source, str) and os.path.exists(source):
        abs_path = os.path.abspath(source)
        return (
            f'<iframe src="/gradio_api/file={abs_path}" '
            f'width="100%" height="{height}" '
            'style="border:none;border-radius:10px;" '
            'allowfullscreen></iframe>'
        )

    b64 = base64.b64encode(source.encode('utf-8')).decode('ascii')
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        f'width="100%" height="{height}" '
        'style="border:none;border-radius:10px;" '
        'allowfullscreen></iframe>'
    )

def score_bar(score, label):
    "Render a bar for aspects' scores"
    pct = max(0, min(100, int((score + 1)/2 * 100)))
    color = BAR_COLOR.get(label, "#FFC107")
    return (
        "<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
        "<div style='flex:1;background:#EEEEEE;border-radius:4px;height:6px'>"
        f"<div style='width:{pct}%;background:{color};height:6px;border-radius:4px'></div>"
        "</div>"
        f"<span style='font-size:10px;color:#9E9E9E;min-width:34px;text-align:right'>"
        f"{score:+.2f}</span></div>"
    )

def poi_card(stop, pq):
    'Build one card for one POI with aspect bars'
    poi = stop.get('poi', {})
    aspects = poi.get('aspects_sentiments', {})
    combined = poi.get('combined_score')
    rating   = poi.get('rating')
    name = stop['poi_name']
    ptype    = poi.get('primaryType', '')
    
    time_s   = (
        f"{stop.get('start_time', '')}\u2013{stop.get('end_time', '')}"
        if stop.get('start_time') else ''
    )

     # Prefer aspects present in user's implicit_needs / preferences first
    implicit   = set(pq.get('implicit_needs', []))
    pref_crowd   = pq.get('soft_constraints', {}).get('prefer_less_crowded', False)
    group_type  = pq.get('group_type', 'solo')

    # Priority aspects 
    priority = []
    if 'kid_friendly' in implicit and 'family' in aspects:
        priority.append('family')
    if pref_crowd and 'crowd' in aspects:
        priority.append('crowd')
    if group_type in ASPECT_META and group_type in aspects:
        priority.append(group_type)
    ordered = priority + [k for k in ASPECT_META if k not in priority]

    # Aspect bars
    bars = []
    for key in ordered:
        if key not in aspects:
            continue
        asp = aspects[key]
        if asp.get("mentions", 0) == 0:
            continue
        icon, label_text = ASPECT_META[key]
        lbl = asp.get("label", "neutral")
        score = asp.get("aspect_weighted_score", 0.0)
        is_prio  = key in priority
        row_bg   = f"background:{SENT_BG[lbl]};" if is_prio else ''
        bars.append(
            f"<div style='margin:3px 0;padding:2px 5px;border-radius:4px;{row_bg}'>"
            f"<div style='display:flex;justify-content:space-between;'>"
            f"<span style='font-size:11px;color:{SENT_COLOR[lbl]};"
            f"font-weight:{'600' if is_prio else '400'}'>"
            f"{icon} {label_text}</span>"
            + score_bar(score, lbl) + "</div>"  + "</div>" 
        )
    bars_html = ''.join(bars) if bars else (
        "<span style='font-size:11px;color:#BDBDBD'>No aspect data</span>"
    )


     # Header badges
    badges = ''
    if rating is not None:
        badges += (f"<span style='background:#FFF8E1;color:#F57F17;"
                   f"border-radius:10px;padding:1px 7px;font-size:10px'>⭐ {rating}</span> ")
    if combined is not None:
        badges += (f"<span style='background:#E3F2FD;color:#1565C0;"
                   f"border-radius:10px;padding:1px 7px;font-size:10px'>"
                   f"🎯 {int(combined*100)}% match</span>")


    return (
        "<div style='background:#FFF;border-radius:8px;padding:10px 12px;"
        "margin-bottom:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);"
        "border-left:3px solid #90CAF9;'>"
        # Header
        "<div style='display:flex;justify-content:space-between;"
        "align-items:flex-start;margin-bottom:4px;'>"
        f"<div><span style='font-weight:700;font-size:12px;color:#1565C0'>{name}</span>"
        f"<br><span style='background:#F5F5F5;color:#757575;border-radius:3px;"
        f"padding:1px 5px;font-size:10px'>{ptype.replace('_',' ').title()}</span></div>"
        f"<div style='text-align:right'>"
        f"<span style='font-size:10px;color:#9E9E9E'>{time_s}</span>"
        f"<br><div style='margin-top:2px'>{badges}</div></div></div>"
        # Aspect bars
        + f"<div style='margin-top:3px'>{bars_html}</div></div>"
    )

def build_absa_html(itinerary, parsed_query):
    """
    Build the full POI Insights HTML panel.
    """

    pq = parsed_query 
    parts = [
        "<div style='font-family:Inter,sans-serif;padding:6px 2px;'>"
        "<div style='background:linear-gradient(90deg,#E3F2FD,#F3E5F5);"
        "border-radius:8px;padding:8px 12px;margin-bottom:10px;'>"
        "<span style='font-size:12px;font-weight:700;color:#1565C0;'>"
        "🧠 Visitor Sentiment Analysis</span>"
        "<br><span style='font-size:11px;color:#555;'>"
        "Aspect scores from verified reviews · Highlighted rows match your preferences"
        "</span></div>"
    ]

    for i, (day_key, stops) in enumerate(itinerary.items()):
        day_label = day_key.replace("_", " ").title()
        dc  = DAY_HEX[i % len(DAY_HEX)]
        parts.append(
            f"<div style='font-size:12px;font-weight:700;color:{dc};"
            f"margin:10px 0 5px;padding:4px 8px;"
            f"background:linear-gradient(90deg,{dc}22,transparent);"
            f"border-left:3px solid {dc};border-radius:0 4px 4px 0;'>"
            f"📅 {day_label}</div>"
        )
        any_poi = False
        for stop in stops:
            if stop.get("poi_id") == "meal_break":
                parts.append(
                        "<div style='font-size:11px;color:#BDBDBD;margin:1px 0 3px 8px'>"
                        "🍽️ Meal Break</div>"
                    )
                continue
            if not stop.get("poi", {}).get("aspects_sentiments"):
                # No ABSA data — show plain name only
                parts.append(
                    f"<div style='font-size:11px;color:#bbb;margin:2px 0 6px 4px;'>"
                    f"· {stop.get('poi_name','?')} — no review data</div>"
                )
                continue
            parts.append(poi_card(stop, pq))
            any_poi = True
        if not any_poi:
            parts.append(
                "<div style='font-size:11px;color:#BDBDBD;padding:5px 8px;"
                "background:#FAFAFA;border-radius:5px;margin-bottom:5px'>"
                "No POI data for this day.</div>"
            )

    parts.append("</div>")
    return "".join(parts)

def build_stats_html(memory_dict):
    """Trip Stats panel from session memory_dict."""
    metrics  = memory_dict.get('itinerary_evaluation',
                               memory_dict.get('metrics', {}))
    pq   = memory_dict.get('parsed_query', {})
    if isinstance(pq, str):
        try: pq = json.loads(pq)
        except Exception: pq = {}
    status  = memory_dict.get('feasibility_status', 'FEASIBLE')
    pois_per = metrics.get('pois_per_day', {})
    pref_m = metrics.get('preference_metrics', {})

    sat  = metrics.get('satisfaction_score', 0)
    sat_pct  = int(sat * 100)
    sat_col  = '#4CAF50' if sat_pct >= 80 else '#FFC107' if sat_pct >= 60 else '#EF5350'
    st_col  = '#4CAF50' if status == 'FEASIBLE' else '#FFC107'
    st_lbl = ' Fully Optimized' if status == 'FEASIBLE' else '⚡ Best Effort'

    def bar(val, color='#42A5F5', mx=100):
        pct = min(100, max(0, val / mx * 100)) if mx else 0
        return (
            "<div style='flex:1;background:#EEE;border-radius:4px;height:8px'>"
            f"<div style='width:{pct:.0f}%;background:{color};"
            "height:8px;border-radius:4px'></div></div>"
        )

    def row(icon, label, val_s, val, color):
        return (
            "<div style='display:flex;align-items:center;gap:7px;margin:5px 0'>"
            f"<span style='font-size:13px;width:20px;text-align:center'>{icon}</span>"
            f"<span style='font-size:11px;color:#555;width:155px;flex-shrink:0'>{label}</span>"
            + bar(val, color)
            + f"<span style='font-size:10px;color:#777;width:42px;"
            f"text-align:right'>{val_s}</span></div>"
        )

    pace = pq.get('pace', 'moderate')
    group = pq.get('group_type', 'solo')
    days = pq.get('trip_duration_days', '3')
    zones = ', '.join(pq.get('preferred_zones', [])) or 'NYC'

    meta = ''.join(
        f"<span style='background:#F5F5F5;color:#555;border-radius:4px;"
        f"padding:2px 8px;font-size:11px'>{t}</span> "
        for t in [f'📅 {days} days', f'👥 {group}', f'⚡ {pace}', f'📍 {zones}']
    )

    q_block = (
        "<div style='margin-bottom:10px'>"
        "<div style='font-size:11px;font-weight:600;color:#333;"
        "margin-bottom:5px'>📈 Quality Metrics</div>"
        + row('✅', 'Constraint Satisfaction',
              f"{metrics.get('constraint_satisfaction_rate_%', 0):.0f}%",
              metrics.get('constraint_satisfaction_rate_%', 0), '#4CAF50')
        + row('🗺️', 'Travel Optimized',
              f"{metrics.get('distance_reduction_%', 0):.0f}%",
              metrics.get('distance_reduction_%', 0), '#00ACC1')
        + row('⚖️', 'Day Balance',
              f"{max(0, 100 - metrics.get('day_balance_std', 0) * 20):.0f}%",
              max(0, 100 - metrics.get('day_balance_std', 0) * 20), '#7CB342')
        + row('🏙️', 'Zone Efficiency',
              f"{100 - metrics.get('cross_zone_rate', 0):.0f}%",
              100 - metrics.get('cross_zone_rate', 0), '#F57C00')
        + "</div>"
    )

    pref_block = ''
    if pref_m:
        pref_block = (
            "<div style='margin-bottom:10px'>"
            "<div style='font-size:11px;font-weight:600;color:#333;"
            "margin-bottom:5px'>⏰ Time Preference Adherence</div>"
            + row('✨', 'Preferred Slots Met',
                  f"{pref_m.get('prefer_adherence_%', 100):.0f}%",
                  pref_m.get('prefer_adherence_%', 100), '#AB47BC')
            + row('🚫', 'Avoided Slots Respected',
                  f"{pref_m.get('avoid_adherence_%', 100):.0f}%",
                  pref_m.get('avoid_adherence_%', 100), '#66BB6A')
            + "</div>"
        )

    day_bars = ''
    if pois_per:
        mx = max(pois_per.values()) or 1
        day_bars = (
            "<div><div style='font-size:11px;font-weight:600;color:#333;"
            "margin-bottom:5px'>📍 Stops per Day</div>"
            + ''.join(
                f"<div style='display:flex;align-items:center;gap:7px;margin:3px 0'>"
                f"<span style='font-size:10px;color:#555;width:55px;flex-shrink:0'>"
                f"{k.replace('_',' ').title()}</span>"
                f"<div style='flex:1;background:#EEE;border-radius:4px;height:16px'>"
                f"<div style='width:{int(v/mx*100)}%;"
                f"background:{DAY_HEX[idx % len(DAY_HEX)]};"
                f"height:16px;border-radius:4px;"
                f"display:flex;align-items:center;padding-left:6px'>"
                f"<span style='font-size:9px;color:white;font-weight:600'>"
                f"{v} stops</span></div></div></div>"
                for idx, (k, v) in enumerate(pois_per.items())
            ) + "</div>"
        )

    return (
        "<div style='font-family:Inter,sans-serif;padding:8px 4px;'>"
        f"<div style='text-align:center;padding:14px 0 10px;"
        f"border-bottom:1px solid #EEE;margin-bottom:10px;'>"
        f"<div style='font-size:44px;font-weight:700;color:{sat_col};line-height:1'>"
        f"{sat_pct}%</div>"
        f"<div style='font-size:12px;color:#757575;margin-top:3px'>"
        f"Overall Trip Quality Score</div>"
        f"<div style='margin-top:7px'>"
        f"<span style='background:{st_col}22;color:{st_col};"
        f"border-radius:10px;padding:2px 12px;"
        f"font-size:11px;font-weight:600'>{st_lbl}</span></div></div>"
        + f"<div style='display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px'>{meta}</div>"
        + q_block + pref_block + day_bars
        + "</div>"
    )

def handle_message(user_msg, chat_history, memory_dict):
    """
    Route user message:
    - Empty memory_dict  →  generate_itinerary  (new trip query)
    - Populated   →    (follow-up Q&A)
    """
    user_msg = user_msg.strip()
    if not user_msg:
        return chat_history, memory_dict, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""

    NO_CHANGE = gr.update()  
    try:
        result = agent.chat(user_msg)
    except Exception as e:
        logger.error('agent.chat error: %s', e, exc_info=True)
        err_msgs = [{"role": "user",  "content": user_msg},
                    {"role": "assistant", "content": f'⚠️ Agent error: {e}'}]
        return (chat_history + err_msgs,
                memory_dict, NO_CHANGE, NO_CHANGE, NO_CHANGE, NO_CHANGE, NO_CHANGE, '')

     #  New itinerary 
    if isinstance(result, dict) and result.get('type') == 'itinerary':
        itinerary_data = result.get('itinerary', {})


        bot_text = """✅ **Itinerary generated!** See the **📋 Itinerary** tab for your 
        full schedule, and the map/timeline tabs for visuals."""
        itinerary_html = result.get('text') or ''
        rag_text = result.get('rag_text', '')
   
        if rag_text and rag_text.strip():
            itinerary_html += (
                "<hr style='border:none;border-top:1px solid #ddd;margin:14px 0'>"
                "<div style='background:#F3E5F5;border-radius:6px;padding:10px 14px;"
                "border-left:3px solid #9C27B0;'>"
                "<span style='font-size:12px;font-weight:700;color:#6A1B9A;'>"
                "💡 Similar past trips suggest:</span>"
                f"<p style='font-size:12px;color:#333333;font-style:italic;"
                f"margin:5px 0 0;line-height:1.6;'>{rag_text.strip()}</p>"
            "</div>"
        )

        # map_data / timeline
        map_html  = embed_html(result.get('map_data')) or MAP_PH
        n_days = len(result.get('itinerary', {}).get('itinerary', {}))
        tl_height = f"{max(380, 95 * n_days + 320)}px"
        timeline_html = embed_html(result.get('timeline'), height=tl_height) or TIMELINE_PH
        absa_html  = build_absa_html(
            itinerary_data.get('itinerary', {}),
            agent.memory_dict.get('parsed_query', {}),
        )
        stats_html = build_stats_html(agent.memory_dict)

        new_msgs = [{"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": bot_text}]
        return (
            chat_history + new_msgs,
            agent.memory_dict,
            map_html, timeline_html, absa_html, stats_html, itinerary_html, '',
        )

    #  Q&A 
    bot_text = clean_response(result) if isinstance(result, str) else str(result)
    new_msgs = [{"role": "user",  "content": user_msg},
                {"role": "assistant", "content": bot_text or "I couldn't process that."}]

    return (chat_history + new_msgs,
            memory_dict, NO_CHANGE, NO_CHANGE, NO_CHANGE, NO_CHANGE, NO_CHANGE, '')

def reset_session():
    """Reset agent + all Gradio UI state to fresh."""
    global agent
    agent = ConversationAgent()
    logger.info('Session reset')
    return [], {}, MAP_PH, TIMELINE_PH, ABSA_PH, STATS_PH,ITINERARY_PH,''


_CSS = """
body, .gradio-container { font-family: 'Inter', sans-serif !important; }

/*  Header */
.nyc-header {
    background: linear-gradient(135deg, #1565C0 0%, #6A1B9A 100%);
    border-radius: 12px; padding: 14px 22px; margin-bottom: 10px; color: white;
}
.nyc-header h1 { margin:0; font-size:20px; font-weight:700; }
.nyc-header p { margin:3px 0 0; font-size:12px; opacity:0.85; }

/*  Chatbot  */
.chatbot-ui .bubble-wrap { background: transparent !important; }
.chatbot-ui [data-testid="bot"] .prose {
    background: #F0F4FF !important; color: #1a1a2e !important;
    border-radius: 8px !important; padding: 8px 12px !important;
}
.chatbot-ui [data-testid="user"] .prose {
    background: #E8F5E9 !important; color: #1a2e1a !important;
    border-radius: 8px !important; padding: 8px 12px !important;
}
.chatbot-ui .avatar-container { display: none !important; }
.chatbot-ui .message-btn-row { display: none !important; }
.chatbot-ui .message-buttons { display: none !important; }
.chatbot-ui .panel { border-left: none !important; }
.chatbot-ui .message.panel::before,
.chatbot-ui .message.panel::after { display: none !important; }

/*  Input + Buttons  */
.input-row textarea {
    border-radius:8px !important; font-size:13px !important;
    border:1.5px solid #90CAF9 !important; resize:none !important;
}
.input-row textarea:focus { border-color:#1565C0 !important; box-shadow:none !important; }
#send-btn { background:linear-gradient(135deg,#1565C0,#1E88E5) !important;
             color:white !important; border-radius:8px !important; font-weight:600 !important; }
#reset-btn { background:linear-gradient(135deg,#B71C1C,#E53935) !important;
             color:white !important; border-radius:8px !important; font-weight:600 !important; }
#send-btn:hover, #reset-btn:hover { opacity:0.88 !important; }

/*  Chips  */
.chip button {
    background:#E8EAF6 !important; color:#3949AB !important;
    border:1px solid #C5CAE9 !important; border-radius:16px !important;
    font-size:11px !important; padding:3px 10px !important;
    font-weight:500 !important; white-space:nowrap !important;
}
.chip button:hover { background:#C5CAE9 !important; }

/*  Right column */
.viz-tabs {
    position: sticky !important;
    top: 10px !important;
    align-self: flex-start !important;
}

/*  Tab nav bar  */
.viz-tabs > div > div.tabs > div.tab-nav {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
}
.viz-tabs > div > div.tabs > div.tab-nav > button {
    font-size: 11px !important; font-weight: 500 !important;
    padding: 5px 9px !important; white-space: nowrap !important;
    flex-shrink: 1 !important; min-width: 0 !important;
}
.viz-tabs > div > div.tabs > div.tab-nav > button.selected {
    background: #1565C0 !important; color: white !important;
    border-radius: 6px !important;
}

/*  Tab panel*/
/* No overflow here — overflow:hidden would block child scrollbars   */
.viz-tabs .tabitem {
    height: 520px !important;
}

/*  Scrollable panels (Itinerary, Insights, Stats) */
#itin-panel, #absa-panel, #stats-panel {
    height: 500px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: #90CAF9 #F5F5F5 !important;
}

/*  Footer  */
footer { display: none !important; }
.built-with { display: none !important; }
"""

CHIPS = [
    '🤔 Why was X dropped?',
    '🌐 What else can I visit on day 2?',
    '⚠️ Explain the conflict, why no X?',
    '📋 Summarise day 1',
    '💯 What\'s my trip score?',
]

EXAMPLES = [
    'Generate a 3-day family trip starting 6 June 2026. Kids love zoos and aquariums. Empire State is a must. Moderate pace. Midtown.',
    'Generate a 2-day solo trip. Art galleries and museums. Upper West Side. Relaxed. Avoid crowds. Museums in the morning.',
    'Generate a weekend couple trip. Brooklyn waterfront, cultural spots, evening out.',
    'Generate a 5-day friends trip starting 19 July 2026. Packed. History, parks and nightlife. Lower Manhattan. Avoid Wednesday',
    'Generate 3 day  trip with friends starting May 11, 2026.  Avoid Tuesday. Fast pace. Staying in lower manhattan. Avoid crowds.',
]


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue='blue', secondary_hue='indigo',
        font=gr.themes.GoogleFont('Inter'),
    ),
    css=_CSS, title='TripEase',
) as demo:

    memory_dict = gr.State({})

    gr.HTML("""
    <div class='nyc-header'>
      <h1>🗽 TripEase &nbsp;&middot;&nbsp; NYC Itinerary AI Agent</h1>
      <p>Describe your trip in plain language &mdash; then ask follow-up questions
         to explore, explain, or refine your schedule.</p>
    </div>""")

    with gr.Row(equal_height=False):

        #  LEFT — Chat 
        with gr.Column(scale=5, min_width=400):

            chatbot = gr.Chatbot(
                value=[], height=490,
                show_label=False,
                elem_classes='chatbot-ui',
                layout='panel',             
                avatar_images=None,
                render_markdown=True,
                sanitize_html=False,  
            )

            with gr.Row(elem_classes='input-row'):
                msg_box = gr.Textbox(
                    placeholder=('e.g.  3-day family trip, kids love zoos, '
                                 'Empire State is a must, moderate pace, midtown…'),
                    show_label=False, lines=2, max_lines=4, scale=6, container=False,
                )
                with gr.Column(scale=1, min_width=90):
                    send_btn  = gr.Button('Send ▶',  elem_id='send-btn',  size='sm')
                    reset_btn = gr.Button('🔄 New Trip',  elem_id='reset-btn', size='sm')

            with gr.Row():
                for chip in CHIPS:
                    gr.Button(chip, size='sm', elem_classes='chip').click(
                        fn=lambda m=chip: m, outputs=msg_box,
                    )

            with gr.Accordion('💡  Example trip queries — click to load', open=False):
                for ex in EXAMPLES:
                    gr.Button(ex, size='sm', elem_classes='chip').click(
                        fn=lambda m=ex: m, outputs=msg_box,
                    )

        #  RIGHT — Visualisations 
        with gr.Column(scale=7, min_width=500, elem_classes='viz-tabs'):
            with gr.Tabs():

                with gr.Tab('🗺️  Map'):
                    map_out = gr.HTML(value=MAP_PH)

                with gr.Tab('📅  Timeline'):
                    timeline_out = gr.HTML(value=TIMELINE_PH)

                with gr.Tab('⭐  POI Insights'):
                    absa_out = gr.HTML(value=ABSA_PH, elem_id='absa-panel')

                with gr.Tab('📊  Trip Stats'):
                    stats_out = gr.HTML(value=STATS_PH, elem_id='stats-panel')

                with gr.Tab('📋  Itinerary'):
                    itinerary_out = gr.HTML(value=ITINERARY_PH, elem_id='itin-panel')

    OUT = [chatbot, memory_dict, map_out, timeline_out, absa_out, stats_out, itinerary_out,  msg_box]

    send_btn.click(fn=handle_message,
                   inputs=[msg_box, chatbot, memory_dict], outputs=OUT)
    msg_box.submit(fn=handle_message,
                   inputs=[msg_box, chatbot, memory_dict], outputs=OUT)
    reset_btn.click(fn=reset_session, inputs=[], outputs=OUT)
try:
    demo.close()
except Exception:
    pass
demo.launch(share=False, server_name='127.0.0.1', allowed_paths = [os.getcwd()])

