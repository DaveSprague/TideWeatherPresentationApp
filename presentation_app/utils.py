"""
Standalone utilities: arrow geometry and map creation
"""
import math
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from config import (
    MAP_STYLE, DEFAULT_ZOOM, MAP_HEIGHT,
    MIN_WIND_SPEED_DISPLAY, WIND_ARROW_SCALE,
    WIND_HISTORY_LENGTH, WIND_FADE_OPACITY_MIN, WIND_FADE_OPACITY_MAX
)
from data.processor import SurgeProcessor


def calculate_arrow_geometry(center_lat: float, center_lon: float, direction_from: float, magnitude: float, scale: float = 0.002, arrowhead_size: float = 0.25, arrowhead_angle: float = 25) -> Dict:
    direction_to = (direction_from + 180) % 360
    angle_rad = math.radians(direction_to)
    lon_factor = math.cos(math.radians(center_lat)) or 1e-6
    def offset(lat: float, lon: float, dlat: float, dlon: float):
        return lat + dlat, lon + (dlon / lon_factor)
    arrow_length = magnitude * scale
    end_lat, end_lon = offset(center_lat, center_lon, arrow_length * math.cos(angle_rad), arrow_length * math.sin(angle_rad))
    arrowhead_len = arrow_length * arrowhead_size
    base_lat, base_lon = offset(end_lat, end_lon, -arrowhead_len * math.cos(angle_rad), -arrowhead_len * math.sin(angle_rad))
    wing_width = arrowhead_len * math.tan(math.radians(arrowhead_angle))
    perp = angle_rad + math.pi / 2
    left_lat, left_lon = offset(base_lat, base_lon, wing_width * math.cos(perp), wing_width * math.sin(perp))
    right_lat, right_lon = offset(base_lat, base_lon, -wing_width * math.cos(perp), -wing_width * math.sin(perp))
    return {
        'arrow_lats': [center_lat, end_lat],
        'arrow_lons': [center_lon, end_lon],
        'arrowhead_lats': [left_lat, end_lat, right_lat, left_lat],
        'arrowhead_lons': [left_lon, end_lon, right_lon, left_lon]
    }


def create_presentation_map(df: pd.DataFrame, center_lat: float, center_lon: float, current_time: Optional[pd.Timestamp] = None, station_name: str = "Belfast Harbor", wind_history_mode: str = 'arrows', history_length: int = 6) -> go.Figure:
    if current_time is None:
        current_time = df.index[0]
    if current_time in df.index:
        current_data = df.loc[current_time]
        current_idx = df.index.get_loc(current_time)
    else:
        current_data = df.iloc[0]
        current_time = df.index[0]
        current_idx = 0
    fig = go.Figure()
    surge_color = SurgeProcessor.get_surge_color(current_data.get('surge', 0))
    surge_value = current_data.get('surge', 0)
    marker_size = 20 + abs(surge_value) * 8
    fig.add_trace(go.Scattermap(lat=[center_lat], lon=[center_lon], mode='markers', marker=dict(size=marker_size, color=surge_color, opacity=0.7, symbol='circle'), text=f"Surge: {surge_value:+.2f} ft", hovertemplate='<b>%{text}</b><br>Time: %{x}<extra></extra>', name='Surge'))
    wind_speed = current_data.get('wind_speed', 0)
    wind_dir = current_data.get('wind_dir_from', 0)
    if wind_speed >= MIN_WIND_SPEED_DISPLAY:
        arrow = calculate_arrow_geometry(center_lat, center_lon, wind_dir, wind_speed, scale=WIND_ARROW_SCALE, arrowhead_size=0.25, arrowhead_angle=25)
        fig.add_trace(go.Scattermap(lat=arrow['arrow_lats'], lon=arrow['arrow_lons'], mode='lines', line=dict(color='black', width=4), opacity=1.0, hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scattermap(lat=arrow['arrowhead_lats'], lon=arrow['arrowhead_lons'], mode='lines', fill='toself', fillcolor='black', line=dict(color='black', width=3), opacity=1.0, hoverinfo='skip', showlegend=False))
    else:
        fig.add_trace(go.Scattermap(lat=[], lon=[], mode='lines', hoverinfo='skip', showlegend=False, visible=False))
        fig.add_trace(go.Scattermap(lat=[], lon=[], mode='lines', hoverinfo='skip', showlegend=False, visible=False))
    fig.add_trace(go.Scattermap(lat=[center_lat], lon=[center_lon], mode='markers', marker=dict(size=10, color='white', symbol='circle', opacity=0.8), text=station_name, hovertemplate='<b>%{text}</b><extra></extra>', name='Station', showlegend=False))
    if wind_history_mode != 'off' and current_idx > 0:
        start_idx = max(0, current_idx - history_length)
        history_data = df.iloc[start_idx:current_idx]
        for hist_idx, (_, row) in enumerate(history_data.iterrows()):
            spd = row.get('wind_speed', 0)
            direc = row.get('wind_dir_from', 0)
            if spd >= MIN_WIND_SPEED_DISPLAY:
                age_offset = len(history_data) - hist_idx - 1
                geom = calculate_arrow_geometry(center_lat, center_lon, direc, spd, scale=WIND_ARROW_SCALE, arrowhead_size=0.25, arrowhead_angle=25)
                opacity_factor = 1 - (age_offset / len(history_data))
                opacity = WIND_FADE_OPACITY_MIN + (WIND_FADE_OPACITY_MAX - WIND_FADE_OPACITY_MIN) * opacity_factor
                fig.add_trace(go.Scattermap(lat=geom['arrow_lats'], lon=geom['arrow_lons'], mode='lines', line=dict(color=f'rgba(100,100,100,{opacity})', width=3), hoverinfo='skip', showlegend=False))
                fig.add_trace(go.Scattermap(lat=geom['arrowhead_lats'], lon=geom['arrowhead_lons'], mode='lines', fill='toself', fillcolor=f'rgba(80,80,80,{opacity*0.8})', line=dict(color=f'rgba(100,100,100,{opacity})', width=2), hoverinfo='skip', showlegend=False))
    fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=center_lat, lon=center_lon), zoom=DEFAULT_ZOOM), height=MAP_HEIGHT, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision='map-constant')
    return fig
