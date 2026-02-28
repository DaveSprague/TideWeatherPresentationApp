"""
Standalone utilities: arrow geometry and map creation
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
from .config import (
    MAP_STYLE, DEFAULT_ZOOM, MAP_HEIGHT,
    MIN_WIND_SPEED_DISPLAY, WIND_ARROW_SCALE,
    WIND_FADE_OPACITY_MIN, WIND_FADE_OPACITY_MAX,
    SURGE_RING_RADIUS_DEG_PER_FT, SURGE_RING_LEVELS,
)
from .data.models import StationData
from .data.processor import SurgeProcessor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wind_rose_color(energy_factor: float) -> str:
    """Map energy_factor [0, 1] to a meteorological warm-ramp RGBA color.

    0 % → sky blue, 50 % → yellow, 100 % → deep red.
    """
    f = max(0.0, min(1.0, energy_factor))
    if f <= 0.5:
        t = f * 2.0
        r = int(30 + t * (255 - 30))
        g = int(180 + t * (210 - 180))
        b = int(230 + t * (0 - 230))
    else:
        t = (f - 0.5) * 2.0
        r = int(255 + t * (180 - 255))
        g = int(210 + t * (20 - 210))
        b = 0
    return f'rgba({r}, {g}, {b}, 0.75)'


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string (#rrggbb) to an rgba() string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def _circle_lats_lons(center_lat: float, center_lon: float, radius_deg: float,
                      n_pts: int = 48):
    """Return (lats, lons) for a closed circle polygon."""
    lon_factor = math.cos(math.radians(center_lat)) or 1e-6
    angles = [2 * math.pi * i / n_pts for i in range(n_pts + 1)]
    lats = [center_lat + radius_deg * math.cos(a) for a in angles]
    lons = [center_lon + (radius_deg * math.sin(a)) / lon_factor for a in angles]
    return lats, lons


# ---------------------------------------------------------------------------
# Arrow geometry
# ---------------------------------------------------------------------------

def calculate_arrow_geometry(center_lat: float, center_lon: float,
                              direction_from: float, magnitude: float,
                              scale: float = 0.002, arrowhead_size: float = 0.25,
                              arrowhead_angle: float = 25) -> Dict:
    direction_to = (direction_from + 180) % 360
    angle_rad = math.radians(direction_to)
    lon_factor = math.cos(math.radians(center_lat)) or 1e-6

    def offset(lat, lon, dlat, dlon):
        return lat + dlat, lon + (dlon / lon_factor)

    arrow_length = magnitude * scale
    end_lat, end_lon = offset(center_lat, center_lon,
                              arrow_length * math.cos(angle_rad),
                              arrow_length * math.sin(angle_rad))
    arrowhead_len = arrow_length * arrowhead_size
    base_lat, base_lon = offset(end_lat, end_lon,
                                -arrowhead_len * math.cos(angle_rad),
                                -arrowhead_len * math.sin(angle_rad))
    wing_width = arrowhead_len * math.tan(math.radians(arrowhead_angle))
    perp = angle_rad + math.pi / 2
    left_lat, left_lon = offset(base_lat, base_lon,
                                wing_width * math.cos(perp),
                                wing_width * math.sin(perp))
    right_lat, right_lon = offset(base_lat, base_lon,
                                  -wing_width * math.cos(perp),
                                  -wing_width * math.sin(perp))
    return {
        'arrow_lats': [center_lat, end_lat],
        'arrow_lons': [center_lon, end_lon],
        'arrowhead_lats': [left_lat, end_lat, right_lat, left_lat],
        'arrowhead_lons': [left_lon, end_lon, right_lon, left_lon],
    }


# ---------------------------------------------------------------------------
# Wind rose
# ---------------------------------------------------------------------------

def build_wind_rose_traces(df: pd.DataFrame, center_lat: float, center_lon: float,
                           current_idx: Optional[int]) -> list:
    """Return Scattermap traces for a wind rose built from history up to current_idx."""
    traces: list = []
    if df is None or df.empty:
        return traces
    history = df if current_idx is None else df.iloc[:current_idx + 1]
    if history.empty:
        return traces

    # --- Vectorised energy binning (replaces Python row loop) ---
    speeds = history['wind_speed'].fillna(0).clip(lower=0).to_numpy(dtype=float)
    dirs_from = history['wind_dir_from'].fillna(0).to_numpy(dtype=float)
    dirs_to = (dirs_from + 180.0) % 360.0
    sectors = ((dirs_to + 22.5) // 45).astype(int) % 8
    sector_energies = np.bincount(sectors, weights=speeds, minlength=8)
    max_energy = max(float(sector_energies.max()), 1.0)

    compass_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    compass_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    lon_factor = math.cos(math.radians(center_lat)) or 1e-6
    inner_radius_deg = 0.008
    max_outer_radius_deg = 0.022

    for sector_idx in range(8):
        energy_factor = sector_energies[sector_idx] / max_energy
        outer_radius_deg = inner_radius_deg + (max_outer_radius_deg - inner_radius_deg) * energy_factor
        segment_start_angle = compass_angles[sector_idx] - 22.5
        segment_end_angle = compass_angles[sector_idx] + 22.5
        angles = [segment_start_angle + (segment_end_angle - segment_start_angle) * i / 19
                  for i in range(20)]

        outer_lats = [center_lat + outer_radius_deg * math.cos(math.radians(a)) for a in angles]
        outer_lons = [center_lon + (outer_radius_deg * math.sin(math.radians(a))) / lon_factor
                      for a in angles]
        inner_lats = [center_lat + inner_radius_deg * math.cos(math.radians(a))
                      for a in reversed(angles)]
        inner_lons = [center_lon + (inner_radius_deg * math.sin(math.radians(a))) / lon_factor
                      for a in reversed(angles)]

        seg_lats = outer_lats + inner_lats + [outer_lats[0]]
        seg_lons = outer_lons + inner_lons + [outer_lons[0]]

        fill_color = _wind_rose_color(energy_factor)
        traces.append(go.Scattermap(
            lat=seg_lats, lon=seg_lons,
            mode='lines', fill='toself', fillcolor=fill_color,
            line=dict(color='rgba(60, 60, 60, 0.6)', width=1),
            showlegend=False, hoverinfo='text',
            text=f"{compass_names[sector_idx]}: {sector_energies[sector_idx]:.1f} kt·hrs",
        ))

        # Direction arrow inside each petal
        ring_thickness = outer_radius_deg - inner_radius_deg
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
        traces.append(go.Scattermap(
            lat=[start_lat, end_lat], lon=[start_lon, end_lon],
            mode='lines', line=dict(color='rgba(255,255,255,0.95)', width=4),
            hoverinfo='skip', showlegend=False,
        ))
        traces.append(go.Scattermap(
            lat=[left_lat, end_lat, right_lat], lon=[left_lon, end_lon, right_lon],
            mode='lines', line=dict(color='rgba(255,255,255,0.95)', width=4),
            hoverinfo='skip', showlegend=False,
        ))
    return traces


# ---------------------------------------------------------------------------
# Surge visualisation helpers
# ---------------------------------------------------------------------------

def build_surge_reference_circles(center_lat: float, center_lon: float) -> list:
    """Dashed reference rings at 1 ft, 2 ft, 3 ft surge levels inside the wind rose hole."""
    traces: list = []
    lon_factor = math.cos(math.radians(center_lat)) or 1e-6
    n_dashes = 20
    dash_fraction = 0.6  # fraction of each sector that is drawn (rest is gap)

    for level in SURGE_RING_LEVELS:
        radius_deg = level * SURGE_RING_RADIUS_DEG_PER_FT
        seg_lats: list = []
        seg_lons: list = []
        for i in range(n_dashes):
            sector = 2 * math.pi / n_dashes
            start_a = i * sector
            end_a = start_a + sector * dash_fraction
            for j in range(8):
                a = start_a + (end_a - start_a) * j / 7
                seg_lats.append(center_lat + radius_deg * math.cos(a))
                seg_lons.append(center_lon + (radius_deg * math.sin(a)) / lon_factor)
            seg_lats.append(None)
            seg_lons.append(None)

        traces.append(go.Scattermap(
            lat=seg_lats, lon=seg_lons,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.55)', width=1),
            hoverinfo='skip', showlegend=False,
        ))
        # Label at top (north) of each ring
        label_lat = center_lat + radius_deg + 0.0002
        traces.append(go.Scattermap(
            lat=[label_lat], lon=[center_lon],
            mode='text',
            text=[f'{level}ft'],
            textfont=dict(size=9, color='rgba(255, 255, 255, 0.8)'),
            hoverinfo='skip', showlegend=False,
        ))
    return traces


def build_surge_indicator_traces(center_lat: float, center_lon: float,
                                 surge_value: float, surge_color: str) -> list:
    """Filled polygon whose radius scales with surge magnitude.

    Replaces the old fixed-size circle marker so the surge circle grows to
    touch the matching reference ring (1 ft surge → 1-ft ring, etc.).
    """
    surge_abs = min(abs(surge_value), 3.5)
    # Minimum visible radius even when surge is near zero
    radius_deg = max(surge_abs * SURGE_RING_RADIUS_DEG_PER_FT,
                     SURGE_RING_RADIUS_DEG_PER_FT * 0.2)
    lats, lons = _circle_lats_lons(center_lat, center_lon, radius_deg)
    fill_color = _hex_to_rgba(surge_color, 0.55)
    edge_color = _hex_to_rgba(surge_color, 0.9)
    return [go.Scattermap(
        lat=lats, lon=lons,
        mode='lines', fill='toself',
        fillcolor=fill_color,
        line=dict(color=edge_color, width=2),
        text=f"Surge: {surge_value:+.2f} ft",
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Surge',
    )]


# ---------------------------------------------------------------------------
# Per-station map rendering
# ---------------------------------------------------------------------------

def _render_station_traces(
    fig: go.Figure,
    station: StationData,
    time_idx: int,
    wind_history_mode: str = 'arrows',
    history_length: int = 6,
    wind_rose_overlay: bool = False,
) -> None:
    """Add all per-station traces to *fig* in-place.

    Accepts a fully-populated StationData (anim_df must not be None) and the
    current animation frame index.  To render multiple stations on the same
    map, call this function once per station before setting the map layout.
    """
    df = station.anim_df
    center_lat = station.lat
    center_lon = station.lon
    station_name = station.name
    current_idx = min(time_idx, len(df) - 1)
    current_data = df.iloc[current_idx]

    if wind_rose_overlay:
        for trace in build_wind_rose_traces(df, center_lat, center_lon, current_idx):
            fig.add_trace(trace)
        for trace in build_surge_reference_circles(center_lat, center_lon):
            fig.add_trace(trace)

    surge_color = SurgeProcessor.get_surge_color(current_data.get('surge', 0))
    surge_value = current_data.get('surge', 0)
    for trace in build_surge_indicator_traces(center_lat, center_lon, surge_value, surge_color):
        fig.add_trace(trace)

    wind_speed = current_data.get('wind_speed', 0)
    wind_dir = current_data.get('wind_dir_from', 0)
    if wind_speed >= MIN_WIND_SPEED_DISPLAY:
        arrow = calculate_arrow_geometry(center_lat, center_lon, wind_dir, wind_speed,
                                        scale=WIND_ARROW_SCALE, arrowhead_size=0.25,
                                        arrowhead_angle=25)
        fig.add_trace(go.Scattermap(
            lat=arrow['arrow_lats'], lon=arrow['arrow_lons'],
            mode='lines', line=dict(color='black', width=4),
            opacity=1.0, hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scattermap(
            lat=arrow['arrowhead_lats'], lon=arrow['arrowhead_lons'],
            mode='lines', fill='toself', fillcolor='black',
            line=dict(color='black', width=3),
            opacity=1.0, hoverinfo='skip', showlegend=False,
        ))
    else:
        fig.add_trace(go.Scattermap(lat=[], lon=[], mode='lines',
                                    hoverinfo='skip', showlegend=False, visible=False))
        fig.add_trace(go.Scattermap(lat=[], lon=[], mode='lines',
                                    hoverinfo='skip', showlegend=False, visible=False))

    # Station label — white dot rendered on top of the surge polygon
    fig.add_trace(go.Scattermap(
        lat=[center_lat], lon=[center_lon], mode='markers',
        marker=dict(size=8, color='white', symbol='circle', opacity=0.9),
        text=station_name,
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Station', showlegend=False,
    ))

    if wind_history_mode != 'off' and current_idx > 0:
        start_idx = max(0, current_idx - history_length)
        history_data = df.iloc[start_idx:current_idx]
        for hist_idx, (_, row) in enumerate(history_data.iterrows()):
            spd = row.get('wind_speed', 0)
            direc = row.get('wind_dir_from', 0)
            if spd >= MIN_WIND_SPEED_DISPLAY:
                age_offset = len(history_data) - hist_idx - 1
                geom = calculate_arrow_geometry(center_lat, center_lon, direc, spd,
                                               scale=WIND_ARROW_SCALE, arrowhead_size=0.25,
                                               arrowhead_angle=25)
                opacity_factor = 1 - (age_offset / len(history_data))
                opacity = (WIND_FADE_OPACITY_MIN
                           + (WIND_FADE_OPACITY_MAX - WIND_FADE_OPACITY_MIN) * opacity_factor)
                fig.add_trace(go.Scattermap(
                    lat=geom['arrow_lats'], lon=geom['arrow_lons'],
                    mode='lines',
                    line=dict(color=f'rgba(100,100,100,{opacity})', width=3),
                    hoverinfo='skip', showlegend=False,
                ))
                fig.add_trace(go.Scattermap(
                    lat=geom['arrowhead_lats'], lon=geom['arrowhead_lons'],
                    mode='lines', fill='toself',
                    fillcolor=f'rgba(80,80,80,{opacity * 0.8})',
                    line=dict(color=f'rgba(100,100,100,{opacity})', width=2),
                    hoverinfo='skip', showlegend=False,
                ))


def create_presentation_map(
    stations: List[StationData],
    time_indices: List[int],
    wind_history_mode: str = 'arrows',
    history_length: int = 6,
    wind_rose_overlay: bool = False,
) -> go.Figure:
    """Build a Plotly map figure for one or more stations.

    Each StationData in *stations* must have ``anim_df`` populated.
    *time_indices* provides the current animation frame index for each station.
    When multiple stations are passed the map centres on their geographic centroid.
    """
    fig = go.Figure()
    for station, time_idx in zip(stations, time_indices):
        _render_station_traces(fig, station, time_idx,
                               wind_history_mode, history_length, wind_rose_overlay)

    center_lat = sum(s.lat for s in stations) / len(stations)
    center_lon = sum(s.lon for s in stations) / len(stations)
    fig.update_layout(
        map=dict(style=MAP_STYLE, center=dict(lat=center_lat, lon=center_lon), zoom=DEFAULT_ZOOM),
        height=MAP_HEIGHT, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, uirevision='map-constant',
    )
    return fig
