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


def build_wind_rose_traces(df: pd.DataFrame, center_lat: float, center_lon: float, current_idx: Optional[int]) -> list:
    """Return scattermap traces for a wind rose built from history up to current_idx."""
    traces: list = []
    if df is None or df.empty:
        return traces
    history = df if current_idx is None else df.iloc[:current_idx + 1]
    if history.empty:
        return traces
    direction_bins = {i: {'energy': 0.0, 'count': 0} for i in range(8)}
    compass_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    compass_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for _, row in history.iterrows():
        wind_speed = row.get('wind_speed', 0) or 0
        wind_dir_from = row.get('wind_dir_from', 0) or 0
        if pd.isna(wind_speed) or pd.isna(wind_dir_from):
            continue
        wind_to = (wind_dir_from + 180) % 360
        sector = int((wind_to + 22.5) // 45) % 8
        direction_bins[sector]['energy'] += max(float(wind_speed), 0.0)
        direction_bins[sector]['count'] += 1
    max_energy = max([bin_data['energy'] for bin_data in direction_bins.values()] + [1])
    lon_factor = math.cos(math.radians(center_lat)) or 1e-6
    inner_radius_deg = 0.008
    max_outer_radius_deg = 0.022
    for sector_idx in range(8):
        bin_data = direction_bins[sector_idx]
        energy_factor = bin_data['energy'] / max_energy if max_energy > 0 else 0
        outer_radius_deg = inner_radius_deg + (max_outer_radius_deg - inner_radius_deg) * energy_factor
        segment_start_angle = compass_angles[sector_idx] - 22.5
        segment_end_angle = compass_angles[sector_idx] + 22.5
        angles = [segment_start_angle + (segment_end_angle - segment_start_angle) * i / 19 for i in range(20)]
        outer_lats, outer_lons = [], []
        inner_lats, inner_lons = [], []
        for angle in angles:
            angle_rad = math.radians(angle)
            outer_lats.append(center_lat + (outer_radius_deg * math.cos(angle_rad)))
            outer_lons.append(center_lon + (outer_radius_deg * math.sin(angle_rad)) / lon_factor)
        for angle in reversed(angles):
            angle_rad = math.radians(angle)
            inner_lats.append(center_lat + (inner_radius_deg * math.cos(angle_rad)))
            inner_lons.append(center_lon + (inner_radius_deg * math.sin(angle_rad)) / lon_factor)
        segment_lats = outer_lats + inner_lats + [outer_lats[0]]
        segment_lons = outer_lons + inner_lons + [outer_lons[0]]
        color_intensity = int(120 + 135 * energy_factor)
        fill_color = f'rgba(50, {color_intensity}, {255 - color_intensity//3}, 0.7)'
        line_color = 'rgba(40, 40, 80, 0.8)'
        traces.append(go.Scattermap(lat=segment_lats, lon=segment_lons, mode='lines', fill='toself', fillcolor=fill_color, line=dict(color=line_color, width=2), showlegend=False, hoverinfo='text', text=f"{compass_names[sector_idx]}: {bin_data['energy']:.1f} kt·hrs"))
        ring_thickness = (outer_radius_deg - inner_radius_deg)
        arrow_center_radius = inner_radius_deg + ring_thickness * 0.5
        arrow_length = ring_thickness * 0.35
        base_angle_rad = math.radians(compass_angles[sector_idx])
        start_lat = center_lat + arrow_center_radius * math.cos(base_angle_rad)
        start_lon = center_lon + (arrow_center_radius * math.sin(base_angle_rad)) / lon_factor
        end_lat = start_lat + arrow_length * math.cos(base_angle_rad)
        end_lon = start_lon + (arrow_length * math.sin(base_angle_rad)) / lon_factor
        head_len = arrow_length * 0.3
        head_angle = math.radians(32)
        left_angle = base_angle_rad + math.pi - head_angle
        right_angle = base_angle_rad + math.pi + head_angle
        left_lat = end_lat + head_len * math.cos(left_angle)
        left_lon = end_lon + (head_len * math.sin(left_angle)) / lon_factor
        right_lat = end_lat + head_len * math.cos(right_angle)
        right_lon = end_lon + (head_len * math.sin(right_angle)) / lon_factor
        traces.append(go.Scattermap(lat=[start_lat, end_lat], lon=[start_lon, end_lon], mode='lines', line=dict(color='rgba(255,255,255,0.95)', width=4), hoverinfo='skip', showlegend=False))
        traces.append(go.Scattermap(lat=[left_lat, end_lat, right_lat], lon=[left_lon, end_lon, right_lon], mode='lines', line=dict(color='rgba(255,255,255,0.95)', width=4), hoverinfo='skip', showlegend=False))
    return traces


def create_presentation_map(df: pd.DataFrame, center_lat: float, center_lon: float, current_time: Optional[pd.Timestamp] = None, station_name: str = "Belfast Harbor", wind_history_mode: str = 'arrows', history_length: int = 6, wind_rose_overlay: bool = False) -> go.Figure:
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
    if wind_rose_overlay:
        for trace in build_wind_rose_traces(df, center_lat, center_lon, current_idx):
            fig.add_trace(trace)
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
