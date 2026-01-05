"""
Standalone Presentation-Mode Storm Surge Visualization
Run: python -m presentation_app.app
"""
import dash
from dash import dcc, html, Input, Output, State, Patch
import dash_bootstrap_components as dbc
import pandas as pd
import logging
import os
import math
import plotly.graph_objects as go

from utils import calculate_arrow_geometry, create_presentation_map, build_wind_rose_traces
from components.overlay_panel import create_overlay_panel
from config import (
    MIN_WIND_SPEED_DISPLAY,
    WIND_ARROW_SCALE,
    WIND_HISTORY_LENGTH,
    MAP_STYLE,
    DEFAULT_ZOOM
)
from data.loader import DataLoader
from data.noaa_api import NOAAClient
from data.processor import SurgeProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_water_level_chart(df, current_time, current_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['water_level'], mode='lines', name='Observed', line=dict(color='#3498db', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['predicted'], mode='lines', name='Predicted', line=dict(color='#95a5a6', width=1, dash='dash')))
    y_min = df['water_level'].min() - 1
    y_max = df['water_level'].max() + 1
    fig.add_trace(go.Scatter(x=[current_time, current_time], y=[y_min, y_max], mode='lines', name='Current', line=dict(color='red', width=2), hoverinfo='skip'))
    surge_val = current_data.get('surge', 0)
    water_val = current_data.get('water_level', 0)
    fig.add_trace(go.Scatter(x=[current_time], y=[water_val], mode='markers+text', marker=dict(size=12, color='red', symbol='circle'), text=[f"Water: {water_val:.1f} ft<br>Surge: {surge_val:+.1f} ft"], textposition='top center', hoverinfo='skip', showlegend=False))
    fig.update_layout(title='Water Level', xaxis=dict(title=''), yaxis=dict(title='Feet', range=[y_min, y_max]), height=280, margin=dict(l=50, r=20, t=40, b=40), hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def create_wind_speed_chart(df, current_time, current_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['wind_speed'], mode='lines', name='Wind Speed', line=dict(color='#2ecc71', width=2), fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'))
    y_max = df['wind_speed'].max() + 5
    fig.add_trace(go.Scatter(x=[current_time, current_time], y=[0, y_max], mode='lines', name='Current', line=dict(color='red', width=2), hoverinfo='skip'))
    wind_spd = current_data.get('wind_speed', 0)
    wind_dir = current_data.get('wind_dir_from', 0)
    fig.add_trace(go.Scatter(x=[current_time], y=[wind_spd], mode='markers+text', marker=dict(size=12, color='red', symbol='circle'), text=[f"{wind_spd:.1f} kts @ {wind_dir:.0f}°"], textposition='top center', hoverinfo='skip', showlegend=False))
    fig.update_layout(title='Wind Speed', xaxis=dict(title=''), yaxis=dict(title='Knots', range=[0, y_max]), height=280, margin=dict(l=50, r=20, t=40, b=40), hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def create_wind_rose_map(df, center_lat, center_lon, current_idx):
    fig = go.Figure()
    for trace in build_wind_rose_traces(df, center_lat, center_lon, current_idx):
        fig.add_trace(trace)
    if not fig.data:
        return fig
    fig.add_trace(go.Scattermap(lat=[center_lat], lon=[center_lon], mode='markers', marker=dict(size=10, color='white'), hoverinfo='skip', showlegend=False))
    fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=center_lat, lon=center_lon), zoom=DEFAULT_ZOOM - 0.5), height=220, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision='wind-rose')
    return fig


def create_recent_wind_arrows(df, center_lat, center_lon, current_idx, history_length=10):
    fig = go.Figure()
    if df is None or df.empty:
        return fig
    start_idx = max(0, current_idx - history_length + 1)
    history = df.iloc[start_idx:current_idx + 1]
    if history.empty:
        return fig
    total = len(history)
    for idx, (_, row) in enumerate(history.iterrows()):
        wind_speed = row.get('wind_speed', 0) or 0
        wind_dir = row.get('wind_dir_from', 0) or 0
        if pd.isna(wind_speed) or pd.isna(wind_dir) or wind_speed < MIN_WIND_SPEED_DISPLAY:
            continue
        age_factor = idx / max(total - 1, 1)
        opacity = 0.25 + 0.7 * age_factor
        line_width = 2 + 2 * age_factor
        arrow_geom = calculate_arrow_geometry(center_lat=center_lat, center_lon=center_lon, direction_from=wind_dir, magnitude=wind_speed, scale=WIND_ARROW_SCALE, arrowhead_size=0.22, arrowhead_angle=25)
        fig.add_trace(go.Scattermap(lat=arrow_geom['arrow_lats'], lon=arrow_geom['arrow_lons'], mode='lines', line=dict(color=f'rgba(60, 60, 60, {opacity})', width=line_width), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scattermap(lat=arrow_geom['arrowhead_lats'], lon=arrow_geom['arrowhead_lons'], mode='lines', fill='toself', fillcolor=f'rgba(40, 40, 40, {opacity})', line=dict(color=f'rgba(60, 60, 60, {opacity})', width=line_width * 0.7), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scattermap(lat=[center_lat], lon=[center_lon], mode='markers', marker=dict(size=9, color='white'), hoverinfo='skip', showlegend=False))
    fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=center_lat, lon=center_lon), zoom=DEFAULT_ZOOM - 0.3), height=200, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision='wind-samples')
    return fig

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Storm Surge Visualization - Presentation Mode"
app_data = {'tide_df': None,'weather_df': None,'processed_df': None,'station_id': '8415191'}

try:
    tide_df = pd.read_csv('tide_belfast.csv', comment='#', header=None, names=['timestamp', 'timestring', 'water_level'])
    cols_wx = ['timestamp','timestring','solar','rain','wind_speed','wind_dir_from','gust','air_temp','vapor','pressure','humidity','temp_humid']
    weather_df = pd.read_csv('weather_belfast.csv', comment='#', header=None, names=cols_wx)
    app_data['tide_df'] = tide_df
    app_data['weather_df'] = weather_df
    logger.info(f"Loaded default data: {len(tide_df)} tide rows, {len(weather_df)} weather rows")
except Exception as e:
    logger.warning(f"Could not load default data: {e}")


def get_initial_dates():
    if app_data['tide_df'] is not None and app_data['weather_df'] is not None:
        try:
            tide_df = app_data['tide_df'].copy()
            weather_df = app_data['weather_df'].copy()
            tide_df['dt'] = pd.to_datetime(tide_df['timestring'], errors='coerce')
            weather_df['dt'] = pd.to_datetime(weather_df['timestring'], errors='coerce')
            min_date_overlap = max(tide_df['dt'].min(), weather_df['dt'].min())
            max_date_overlap = min(tide_df['dt'].max(), weather_df['dt'].max())
            forced_center = pd.Timestamp('2024-01-10')
            center_date = forced_center
            min_date = min(forced_center, min_date_overlap)
            max_date = max_date_overlap
            return min_date, max_date, center_date
        except:
            pass
    return None, None, None

initial_min, initial_max, initial_center = get_initial_dates()

app.layout = html.Div([
    html.Div([dcc.Graph(id='surge-map', config={'displayModeBar': False}, style={'width': '100%', 'height': '100%'})], style={'height': '70vh', 'width': '100%'}),
    html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(id='water-level-chart', config={'displayModeBar': False}, style={'height': '100%'})], width=6, style={'padding': '0 5px'}),
            dbc.Col([dcc.Graph(id='wind-speed-chart', config={'displayModeBar': False}, style={'height': '100%'})], width=6, style={'padding': '0 5px'}),
        ], style={'height': '100%', 'margin': 0})
    ], style={'height': '30vh', 'width': '100%', 'background': '#f8f9fa', 'padding': '10px 0'}),
    create_overlay_panel(min_date=initial_min, max_date=initial_max, center_date=initial_center),
], style={'margin': 0, 'padding': 0, 'overflow': 'hidden', 'height': '100vh'})


@app.callback(
    Output('center-date-picker', 'min_date_allowed'),
    Output('center-date-picker', 'max_date_allowed'),
    Output('center-date-picker', 'date'),
    Input('load-sample-data', 'n_clicks'),
    prevent_initial_call=True
)
def load_sample_data(n_clicks):
    if app_data['tide_df'] is None or app_data['weather_df'] is None:
        return dash.no_update
    try:
        tide_df = app_data['tide_df'].copy()
        weather_df = app_data['weather_df'].copy()
        tide_df['dt'] = pd.to_datetime(tide_df['timestring'], errors='coerce')
        weather_df['dt'] = pd.to_datetime(weather_df['timestring'], errors='coerce')
        min_date = max(tide_df['dt'].min(), weather_df['dt'].min())
        max_date = min(tide_df['dt'].max(), weather_df['dt'].max())
        center_date = min_date + (max_date - min_date) / 2
        return min_date, max_date, center_date
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return dash.no_update


@app.callback(
    [Output('surge-map', 'figure'),
     Output('water-level-chart', 'figure'),
     Output('wind-speed-chart', 'figure'),
     Output('station-name-display', 'children'),
     Output('time-slider', 'max'),
     Output('time-slider', 'marks'),
     Output('animation-data-store', 'data')],
    Input('center-date-picker', 'date'),
    State('data-store', 'data')
)
def process_data(center_date, stored_data):
    if app_data['tide_df'] is None or app_data['weather_df'] is None:
        empty_fig = {'data': [], 'layout': {'title': 'Upload data to begin'}}
        return empty_fig, empty_fig, empty_fig, "No data", 0, {}, None
    try:
        center_dt = pd.to_datetime(center_date)
        start_dt = center_dt - pd.Timedelta(days=2)
        end_dt = center_dt + pd.Timedelta(days=2)
        loader = DataLoader()
        tide_raw = app_data['tide_df'].copy()
        weather_raw = app_data['weather_df'].copy()
        tide_raw['dt'] = pd.to_datetime(tide_raw['timestring'], errors='coerce')
        weather_raw['dt'] = pd.to_datetime(weather_raw['timestring'], errors='coerce')
        available_min = max(tide_raw['dt'].min(), weather_raw['dt'].min())
        available_max = min(tide_raw['dt'].max(), weather_raw['dt'].max())
        def build_filtered_frames(start_ts, end_ts):
            tide_df_local = tide_raw.set_index('dt')[['water_level']]
            tide_df_local['water_level'] = pd.to_numeric(tide_df_local['water_level'], errors='coerce')
            tide_df_local = tide_df_local[(tide_df_local.index >= start_ts) & (tide_df_local.index <= end_ts)].dropna()
            weather_df_local = weather_raw.set_index('dt')
            weather_df_local = weather_df_local[(weather_df_local.index >= start_ts) & (weather_df_local.index <= end_ts)]
            for col in ['wind_speed', 'wind_dir_from', 'pressure']:
                weather_df_local[col] = pd.to_numeric(weather_df_local[col], errors='coerce')
            weather_df_local = weather_df_local[['wind_speed', 'wind_dir_from', 'pressure']].dropna()
            return tide_df_local, weather_df_local
        tide_df, weather_df = build_filtered_frames(start_dt, end_dt)
        if tide_df.empty or weather_df.empty:
            center_dt = min(max(center_dt, available_min), available_max)
            start_dt = center_dt - pd.Timedelta(days=2)
            end_dt = center_dt + pd.Timedelta(days=2)
            tide_df, weather_df = build_filtered_frames(start_dt, end_dt)
        merged_df = loader.merge_datasets(tide_df, weather_df)
        if merged_df.empty:
            raise ValueError("No data after merging")
        station_id = app_data['station_id']
        noaa_client = NOAAClient(station_id)
        predictions = noaa_client.fetch_predictions(start_dt, end_dt, use_hilo=True)
        if predictions is None:
            predictions = noaa_client.try_fallback_stations(start_dt, end_dt)
        processor = SurgeProcessor()
        processed_df = processor.calculate_surge_from_predictions(merged_df, predictions, method='pchip')
        anim_df = processor.resample_data(processed_df, interval='15min')
        app_data['processed_df'] = anim_df
        from config import STATION_INFO
        station_info = STATION_INFO.get(station_id, STATION_INFO['8415191'])
        center_lat = station_info['lat']
        center_lon = station_info['lon']
        station_name = station_info['name']
        animation_frames = []
        for idx, (ts, row) in enumerate(anim_df.iterrows()):
            surge_val = row.get('surge', 0)
            wind_spd = row.get('wind_speed', 0)
            wind_dir = row.get('wind_dir_from', 0)
            arrow_geom = calculate_arrow_geometry(center_lat=center_lat, center_lon=center_lon, direction_from=wind_dir, magnitude=wind_spd, scale=0.002, arrowhead_size=0.25, arrowhead_angle=25)
            animation_frames.append({'timestamp': ts.isoformat(), 'timestamp_str': ts.strftime('%B %d, %Y %H:%M'), 'surge': surge_val, 'surge_color': SurgeProcessor.get_surge_color(surge_val), 'surge_size': 20 + abs(surge_val) * 8, 'wind_speed': wind_spd, 'wind_dir': wind_dir, 'arrow_lats': arrow_geom['arrow_lats'], 'arrow_lons': arrow_geom['arrow_lons'], 'arrowhead_lats': arrow_geom['arrowhead_lats'], 'arrowhead_lons': arrow_geom['arrowhead_lons'], 'water_level': row.get('water_level', 0), 'predicted': row.get('predicted', 0)})
        current_time = anim_df.index[0]
        current_data = anim_df.iloc[0]
        wind_mode = 'arrows'
        map_fig = create_presentation_map(anim_df, center_lat, center_lon, current_time, station_name, wind_history_mode=wind_mode, wind_rose_overlay=True)
        water_chart = create_water_level_chart(anim_df, current_time, current_data)
        wind_chart = create_wind_speed_chart(anim_df, current_time, current_data)
        marks = {}
        midnight_idxs = [i for i, ts in enumerate(anim_df.index) if ts.hour == 0 and ts.minute == 0]
        if midnight_idxs:
            stride = max(1, len(midnight_idxs) // 8)
            for pos in midnight_idxs[::stride]:
                marks[pos] = anim_df.index[pos].strftime('%b %d')
        if not marks:
            stride = max(1, len(anim_df) // 8)
            for i in range(0, len(anim_df), stride):
                marks[i] = anim_df.index[i].strftime('%m/%d %H:%M')
        marks[len(anim_df) - 1] = anim_df.index[-1].strftime('%b %d')
        animation_store = {'frames': animation_frames, 'center_lat': center_lat, 'center_lon': center_lon, 'water_level_min': float(anim_df['water_level'].min()) - 1, 'water_level_max': float(anim_df['water_level'].max()) + 1}
        return (map_fig, water_chart, wind_chart, station_name, len(anim_df) - 1, marks, animation_store)
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        empty_fig = {'data': [], 'layout': {'title': f'Error: {str(e)}'}}
        return empty_fig, empty_fig, empty_fig, "Error", 0, {}, None


@app.callback(
    [Output('surge-map', 'figure', allow_duplicate=True),
     Output('water-level-chart', 'figure', allow_duplicate=True),
     Output('wind-speed-chart', 'figure', allow_duplicate=True),
     Output('current-time-display', 'children'),
     Output('current-surge-value', 'children'),
     Output('current-wind-value', 'children')],
    Input('time-slider', 'value'),
    State('animation-data-store', 'data'),
    prevent_initial_call=True
)
def update_time_position(time_idx, animation_data):
    if not animation_data or time_idx is None or app_data.get('processed_df') is None:
        return (dash.no_update,) * 6
    frames = animation_data['frames']
    if time_idx >= len(frames):
        return (dash.no_update,) * 6
    frame = frames[time_idx]
    try:
        anim_df = app_data['processed_df']
        current_time = anim_df.index[time_idx]
        from config import STATION_INFO
        station_id = app_data['station_id']
        station_info = STATION_INFO.get(station_id, STATION_INFO['8415191'])
        map_fig = create_presentation_map(anim_df, station_info['lat'], station_info['lon'], current_time, station_info['name'], wind_history_mode='arrows', wind_rose_overlay=True)
        map_fig.update_layout(uirevision='map-constant')
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        map_fig = dash.no_update
    water_chart_patch = Patch()
    wind_chart_patch = Patch()
    water_chart_patch['data'][2]['x'] = [frame['timestamp'], frame['timestamp']]
    water_chart_patch['data'][3]['x'] = [frame['timestamp']]
    water_chart_patch['data'][3]['y'] = [frame['water_level']]
    water_chart_patch['data'][3]['text'] = [f"Water: {frame['water_level']:.1f} ft<br>Surge: {frame['surge']:+.1f} ft"]
    wind_chart_patch['data'][1]['x'] = [frame['timestamp'], frame['timestamp']]
    wind_chart_patch['data'][2]['x'] = [frame['timestamp']]
    wind_chart_patch['data'][2]['y'] = [frame['wind_speed']]
    wind_chart_patch['data'][2]['text'] = [f"{frame['wind_speed']:.1f} kts @ {frame['wind_dir']:.0f}°"]
    time_str = frame['timestamp_str']
    surge_str = f"{frame['surge']:+.2f} ft"
    wind_str = f"{frame['wind_speed']:.1f} kts @ {frame['wind_dir']:.0f}°"
    return (map_fig, water_chart_patch, wind_chart_patch, time_str, surge_str, wind_str)


@app.callback(Output('time-slider', 'value'), Output('animation-interval', 'disabled'), Input('play-button', 'n_clicks'), Input('pause-button', 'n_clicks'), Input('reset-button', 'n_clicks'), Input('animation-interval', 'n_intervals'), State('time-slider', 'value'), State('time-slider', 'max'), State('step-size-selector', 'value'), prevent_initial_call=True)
def control_animation(play_clicks, pause_clicks, reset_clicks, n_intervals, current_value, max_value, step_size):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, True
    trigger = ctx.triggered[0]['prop_id']
    if 'play-button' in trigger:
        return current_value, False
    elif 'pause-button' in trigger:
        return current_value, True
    elif 'reset-button' in trigger:
        return 0, True
    elif 'animation-interval' in trigger:
        new_value = current_value + step_size
        if new_value >= max_value:
            return max_value, True
        return new_value, False
    return dash.no_update, True


@app.callback(Output('animation-interval', 'interval'), Input('speed-slider', 'value'))
def update_speed(speed_ms):
    return speed_ms


if __name__ == '__main__':
    logger.info("Starting Belfast Harbor Storm Surge & Wind Visualization - PRESENTATION MODE (Standalone)")
    port = int(os.getenv('PORT', '8052'))
    logger.info(f"Server running on http://localhost:{port}")
    app.run(debug=True, port=port)
