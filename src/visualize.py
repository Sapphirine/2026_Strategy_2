
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

import folium
import plotly.graph_objects as go
from folium import plugins


#  Colour palettes 
DAY_HEX    = ['#1565C0', '#6A1B9A', '#00695C', '#AD1457', '#E65100',
              '#00838F', '#558B2F']
DAY_FOLIUM = ['blue', 'red', 'green', 'purple', 'orange', 'cadetblue', 'darkred']

TYPE_COLORS = {
    'museum':   '#4e79a7',
    'zoo':    '#59a14f',
    'aquarium':    '#17becf',
    'historical_landmark':     '#b07aa1',
    'historical_place':   '#b07aa1',
    'art_gallery':  '#f28e2b',
    'art_studio':     '#f0a400',
    'performing_arts_theater': '#e15759',
    'tourist_attraction':  '#76b7b2',
    'cultural_landmark': '#edc948',
    'monument':   '#ff9da7',
    'cultural_center':   '#9c755f',
    'park':   '#bab0ac',
    'national_park': '#86b97c',
    'garden':   '#8bc34a',
    'botanical_garden':  '#8bc34a',
    'observation_deck': '#4dc9bf',
    'amusement_park':  '#e91e63',
    'amusement_center':  '#f06292',
    'night_club':  '#7c4dff',
    'casino':  '#ff6d00',
    'hiking_area':'#795548',
    'beach':  '#ffd54f',
    'meal_break':  '#e0e0e0',
}
DEFAULT_COLOR = '#aec7e8'

def plot_map(itinerary_info, out_path = 'itinerary_map.html'):
    '''
    Returns an out_path string of interactive map showing
      - Color-coded markers per day
      - Polyline connecting stops in order
      - Popup: POI name, time, duration, flags
      - Depot marker
    '''
    itinerary = itinerary_info['itinerary']
    depot = itinerary_info['depot']
    violations = {v['poi_id'] for v in itinerary_info.get('all_violations',[])}

    # Center map on depot or mean of all POI coords
    depot_loc = depot.get('location', {})
    map_center = [
        depot_loc.get('latitude', 40.7549),
        depot_loc.get('longitude', -73.9840)
    ]

    fmap = folium.Map(location=map_center, zoom_start=13, tiles=None)
    folium.TileLayer('CartoDB positron', control=False).add_to(fmap)

    # Depot marker
    folium.Marker(
        location = map_center,
        popup = folium.Popup('🏨 Depot', max_width = 200),
        icon = folium.Icon(color = 'black', icon = 'home', prefix = 'fa'),
    ).add_to(fmap)

    for day_idx, (day_key, stops) in enumerate(itinerary.items()):
        hex_col = DAY_HEX[day_idx % len(DAY_HEX)]
        fol_col  = DAY_FOLIUM[day_idx % len(DAY_FOLIUM)]
        day_label = day_key.replace('_', ' ').title()
        grp   = folium.FeatureGroup(name=day_label, show=True)
        day_coords    = []
        seq   = 0

        for stop in (stops):
            poi_id = stop['poi_id']

            # Meal break (no fixed location)
            if poi_id == 'meal_break':
                continue

            # Get coords
            poi = stop['poi']
            loc = poi.get('location', {})
            lat  = loc.get('latitude')
            long = loc.get('longitude')
            ptype   = poi.get('primaryType', '')
            t_color = TYPE_COLORS.get(ptype, DEFAULT_COLOR)

            if lat is None or long is None:
                logger.warning('No coordinates for POI %s - skipping marker', poi_id)
                continue

            day_coords.append([lat, long])
            seq += 1
            # violation flag on popup
            flag_html = ''
            if stop['flags']:
                flag_html = (
                    "<br><span style='color:#E53935;font-size:11px;'>"
                    f"⚠️ {', '.join(stop['flags'])}</span>"
                )
            if poi_id in violations:
                flag_html += (
                    "<br><span style='color:#E53935;font-size:11px;'>"
                    "⚠️ scheduling violation</span>"
                )

            popup_html = (
                "<div style='font-family:Inter,sans-serif;min-width:190px;'>"
                f"<b style='font-size:13px;color:{hex_col};'>{stop['poi_name']}</b>"
                f"<br><span style='background:{t_color};color:white;border-radius:3px;"
                f"padding:1px 5px;font-size:10px;'>{ptype.replace('_',' ').title()}</span>"
                f"<br><br>🕐 <b>{stop['start_time']}–{stop['end_time']}</b>"
                f"<br>⏱ {stop['visit_duration_min']} min visit"
                f"<br>🚶 {stop.get('travel_to_next_min', 0)} min to next"
                + flag_html + "</div>"
            )
            marker_bg = '#E53935' if poi_id in violations else hex_col

            folium.Marker(
                location = [lat, long],
                popup = folium.Popup(popup_html, max_width = 250),
                tooltip = f"{day_label} · #{seq} · {stop['poi_name']}",
                icon=folium.DivIcon(
                    html=(
                        f"<div style='background:{marker_bg};color:white;"
                        "border-radius:50%;width:28px;height:28px;"
                        "text-align:center;line-height:28px;"
                        "font-weight:bold;font-size:12px;"
                        "border:2px solid white;"
                        "box-shadow:0 2px 5px rgba(0,0,0,.3);'>"
                        f"{'⚠️' if poi_id in violations else seq}"
                        f"</div>"
                    ),
                    icon_size=(26, 26),
                    icon_anchor=(13, 13),
                ),
            ).add_to(grp)

        # Polyline connecting day's stops
        if len(day_coords) > 1:
            line = folium.PolyLine(
                locations = [map_center] + day_coords + [map_center],
                color = hex_col,
                weight = 2.5,
                opacity = 0.7,
                tooltip = day_label,
            )
            line.add_to(grp)
            plugins.PolyLineTextPath(        # arrows along the line
                line,
                '     ➤     ',
                repeat=True,
                offset= 8,
                attributes={
                    'fill': hex_col,
                    'font-size': '14',
                    'font-weight': 'bold',
                },
            ).add_to(grp)

        grp.add_to(fmap)

    folium.LayerControl(position='topright', collapsed=False).add_to(fmap)
    # Legend
    legend_items = ''.join(
        f"<li style='margin:3px 0;'>"
        f"<span style='background:{DAY_HEX[i]};color:white;border-radius:3px;"
        f"padding:1px 7px;font-size:11px;'>{dk.replace('_',' ').title()}</span></li>"
        for i, dk in enumerate(itinerary.keys())
)

    legend_html = f"""
    <div style='position:fixed;top:90px;left:10px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ddd;font-family:Inter,sans-serif;
                box-shadow:0 2px 8px rgba(0,0,0,.1);'>
        <b style='font-size:12px;'>🗺️ Trip Days</b><ul style='margin:5px 0;padding-left:0;list-style:none'>
        {legend_items}</ul>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(out_path)
    logger.info('Map saved to %s', out_path)
    return out_path


def plot_timeline(itinerary_info, out_path='itinerary_timeline.html'):
    '''
    Plotly horizontal Gantt chart showing:
      - One row per day
      - Bars: POI visits (colored by type) + meal breaks (grey)
      - Travel gaps visible as whitespace between bars
      - Hover: name, time, duration
    '''
    itinerary   = itinerary_info['itinerary']
    day_summaries = itinerary_info['day_summaries']

    fig   = go.Figure()
    y_days = list(itinerary.keys())
    seen_leg = set()
    n = len(y_days)

    for day_key, stops in itinerary.items():
        y_label  = day_key.replace('_', ' ').title()
        prev_end = 0

        for stop in stops:
            poi = stop.get('poi', {})
            x_start  = stop['start_time']
            x_end  = stop['end_time']
            start  = stop['start_min'] - 540
            end  = stop['end_min']   - 540
            duration = max(end - start, 1)

            # Travel gap before this stop
            gap = start - prev_end
            if gap > 2 and prev_end > 0:
                show_t = 'Travel' not in seen_leg
                fig.add_trace(go.Bar(
                    name='Travel', x=[gap], y=[y_label], base=[prev_end],
                    orientation='h',
                    marker_color='#F5F5F5',
                    marker_line_color='#E0E0E0', marker_line_width=0.5,
                    hovertemplate=f'🚶 Travel: {gap} min<extra></extra>',
                    showlegend=show_t, opacity=0.8,
                    ))
                seen_leg.add('Travel')
            prev_end = end

            is_meal  = stop['poi_id'] == 'meal_break'
            has_flag   = bool(stop['flags'])
            ptype    = 'meal_break' if is_meal else poi.get('primaryType', '')
            bar_color  = TYPE_COLORS.get(ptype, DEFAULT_COLOR)
            type_label = 'Meal Break' if is_meal else ptype.replace('_', ' ').title()
            rating   = poi.get('rating', '')
            rating_str = f"⭐ {rating}" if rating else ''
            flag_str   = f"⚠️ {', '.join(stop['flags'])}" if has_flag else ''

            hover = (
                f"<span style='font-size:11px'><b>{stop['poi_name']}</b></span><br>"
                f"<span style='font-size:10px'>"
                f"⏱ {x_start} – {x_end}<br>"
                f"🕐 {stop['visit_duration_min']} min<br>"
                f"📍 {ptype} · {poi.get('zone', '')}"
                + (f" · {rating_str}" if rating_str else '')
                + (f"<br>{flag_str}"   if flag_str   else '')
            )

            show_leg = type_label not in seen_leg
            if show_leg:
                seen_leg.add(type_label)

            fig.add_trace(go.Bar(
                name=type_label,
                x=[(stop['end_min'] - stop['start_min'])],
                y=[y_label],
                base=[stop['start_min'] - 540],
                orientation='h',
                marker_color=bar_color,
                marker_line_color='red' if has_flag else 'rgba(255,255,255,0.4)',
                marker_line_width=2 if has_flag else 0.5,
                hovertemplate=hover + '<extra></extra>',
                showlegend=show_leg,
                opacity=0.6 if is_meal else 0.9,     
            ))

    max_end = max(
        stop['end_min'] - 540
        for stops in itinerary.values()
        for stop in stops
    )
    x_max = min(795, max_end + 30)   # 30 min padding after last stop, cap at 22:00
    
    tick_vals  = list(range(0, x_max + 1, 120))   # every 2 hours instead of 1
    tick_texts = [f'{9 + t // 60:02d}:00' for t in tick_vals]

    
    fig.update_layout(
        title=dict(
            text='Trip Itinerary — Day-by-Day Timeline',
            font=dict(size=13, family='Inter,sans-serif'), x=0.5,
        ),
        barmode='stack',
        bargap=0.45,
        bargroupgap=0.1,
        xaxis=dict(
            title='Time of Day',
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickangle=-45,          
            range=[0, x_max],       
            showgrid=True, gridcolor='#e0e0e0', zeroline=False,
        ),
        yaxis=dict(
            categoryorder='array',
            categoryarray=list(reversed(y_days)),
            tickfont=dict(size=11, family='Inter,sans-serif'),
        ),
        autosize=True,
        height=max(260, 95 * n + 180),
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(size=11, family='Inter,sans-serif'),
        margin=dict(l=80, r=40, t=50, b=120),  
                                               
        legend=dict(
            orientation='h', yanchor='top', y=-0.28,
            xanchor='center', x=0.5, font=dict(size=9),
        ),
    )
    # Time-slot shading
    for x0, x1, lbl, bg in [
        (0, 180, 'Morning',   'rgba(255,235,59,0.06)'),
        (180, 480, 'Afternoon', 'rgba(30,136,229,0.04)'),
        (480, 780, 'Evening',   'rgba(156,39,176,0.05)'),
    ]:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=bg, line_width=0,
            annotation_text=lbl, annotation_position='top left',
            annotation_font_size=9, annotation_font_color='#9E9E9E',
        )

    fig.add_vline(
        x=180, line_dash='dash', line_color='grey',
        annotation_text='12:00', annotation_position='top',
        annotation_font_size=9, annotation_font_color='#757575',
    )

    
    fig.write_html(out_path, include_plotlyjs=True)
    logger.info('Timeline saved to %s', out_path)
    return out_path




