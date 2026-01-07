"""
Standalone Presentation-Mode Storm Surge Visualization
Run: python app.py
"""
import dash
from dash import dcc, html, Input, Output, State, Patch
import dash_bootstrap_components as dbc
import pandas as pd
import logging
import os
import uuid
import plotly.graph_objects as go

from presentation_app.utils import create_presentation_map
from presentation_app.components.overlay_panel import create_overlay_panel
from presentation_app.data.loader import DataLoader
from presentation_app.data.noaa_api import NOAAClient
from presentation_app.data.processor import SurgeProcessor
from presentation_app.cache import LRUCacheTTL
from presentation_app.config import STATION_INFO, CACHE_ENABLED, CACHE_MAX_SIZE, CACHE_TTL_SECONDS, DATA_WINDOW_HOURS, SLIDER_MARK_STRIDE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Cache wrapper functions
def cache_get(key):
    """Unified cache get supporting both dict and LRU cache."""
    return session_cache.get(key) if hasattr(session_cache, 'get') else None


def cache_set(key, value):
    """Unified cache set supporting both dict and LRU cache."""
    if isinstance(session_cache, dict):
        session_cache[key] = value
    else:
        session_cache.set(key, value)


def create_empty_figure(title: str = "No data"):
    """Factory for empty/error figure."""
    return {'data': [], 'layout': {'title': title}}


def generate_slider_marks(datetimes: pd.DatetimeIndex) -> dict:
    """Generate slider marks from DatetimeIndex, showing midnights with stride."""
    marks = {}
    midnight_idxs = [i for i, ts in enumerate(datetimes) if ts.hour == 0 and ts.minute == 0]
    if midnight_idxs:
        stride = max(1, len(midnight_idxs) // SLIDER_MARK_STRIDE)
        for pos in midnight_idxs[::stride]:
            marks[pos] = datetimes[pos].strftime('%b %d')
    if not marks:
        stride = max(1, len(datetimes) // SLIDER_MARK_STRIDE)
        for i in range(0, len(datetimes), stride):
            marks[i] = datetimes[i].strftime('%m/%d %H:%M')
    marks[len(datetimes) - 1] = datetimes[-1].strftime('%b %d')
    return marks


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


def build_frame_patches(frame):
    """Create chart patches and display strings for a single animation frame."""
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
    return water_chart_patch, wind_chart_patch, time_str, surge_str, wind_str


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Storm Surge Visualization - Presentation Mode"
server = app.server  # Expose Flask server for gunicorn
app_data = {'tide_df': None,'weather_df': None,'station_id': '8415191'}
# In-memory cache: LRU + TTL when enabled, plain dict otherwise
session_cache = LRUCacheTTL(max_size=CACHE_MAX_SIZE, ttl_seconds=CACHE_TTL_SECONDS) if CACHE_ENABLED else {}

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
            loader = DataLoader()
            tide_df, weather_df = loader.parse_raw_dataframes(app_data['tide_df'], app_data['weather_df'])
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
    html.Div([dcc.Graph(id='surge-map', config={'displayModeBar': False}, animate=False, style={'width': '100%', 'height': '100%'})], style={'height': '70vh', 'width': '100%'}),
    html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(id='water-level-chart', config={'displayModeBar': False}, animate=False, style={'height': '100%'})], width=6, style={'padding': '0 5px'}),
            dbc.Col([dcc.Graph(id='wind-speed-chart', config={'displayModeBar': False}, animate=False, style={'height': '100%'})], width=6, style={'padding': '0 5px'}),
        ], style={'height': '100%', 'margin': 0})
    ], style={'height': '30vh', 'width': '100%', 'background': '#f8f9fa', 'padding': '10px 0'}),
    create_overlay_panel(min_date=initial_min, max_date=initial_max, center_date=initial_center),
], style={'margin': 0, 'padding': 0, 'overflow': 'hidden', 'height': '100vh'})


@app.callback(Output('session-id', 'data'), Input('time-slider', 'id'), State('session-id', 'data'))
def ensure_session_id(_, existing):
    if existing:
        return existing
    return str(uuid.uuid4())


@app.callback(
    Output('data-version', 'data'),
    Input('upload-tide-data', 'contents'),
    Input('upload-weather-data', 'contents'),
    State('data-version', 'data'),
    prevent_initial_call=True
)
def invalidate_cache_on_upload(tide_contents, weather_contents, current_version):
    """Increment data version when new files are uploaded to invalidate the cache."""
    return (current_version or 0) + 1


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
        loader = DataLoader()
        tide_df, weather_df = loader.parse_raw_dataframes(app_data['tide_df'], app_data['weather_df'])
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
    State('session-id', 'data'),
    State('data-version', 'data')
)
def process_data(center_date, session_id, data_version):
    if app_data['tide_df'] is None or app_data['weather_df'] is None:
        empty_fig = create_empty_figure('Upload data to begin')
        return empty_fig, empty_fig, empty_fig, "No data", 0, {}, None
    try:
        session_token = session_id or str(uuid.uuid4())
        cache_key = (session_token, str(center_date), data_version or 0)
        cached = cache_get(cache_key)
        if cached:
            return cached
        center_dt = pd.to_datetime(center_date)
        start_dt = center_dt - pd.Timedelta(hours=DATA_WINDOW_HOURS)
        end_dt = center_dt + pd.Timedelta(hours=DATA_WINDOW_HOURS)
        loader = DataLoader()
        tide_raw, weather_raw = loader.parse_raw_dataframes(app_data['tide_df'], app_data['weather_df'])
        available_min = max(tide_raw['dt'].min(), weather_raw['dt'].min())
        available_max = min(tide_raw['dt'].max(), weather_raw['dt'].max())
        tide_df, weather_df = loader.filter_by_window(tide_raw, weather_raw, start_dt, end_dt)
        if tide_df.empty or weather_df.empty:
            center_dt = min(max(center_dt, available_min), available_max)
            start_dt = center_dt - pd.Timedelta(hours=DATA_WINDOW_HOURS)
            end_dt = center_dt + pd.Timedelta(hours=DATA_WINDOW_HOURS)
            tide_df, weather_df = loader.filter_by_window(tide_raw, weather_raw, start_dt, end_dt)
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
        station_info = STATION_INFO.get(station_id, STATION_INFO['8415191'])
        center_lat = station_info['lat']
        center_lon = station_info['lon']
        station_name = station_info['name']
        animation_frames = []
        for idx, (ts, row) in enumerate(anim_df.iterrows()):
            surge_val = row.get('surge', 0)
            wind_spd = row.get('wind_speed', 0)
            wind_dir = row.get('wind_dir_from', 0)
            animation_frames.append({'timestamp': ts.isoformat(), 'timestamp_str': ts.strftime('%B %d, %Y %H:%M'), 'surge': surge_val, 'wind_speed': wind_spd, 'wind_dir': wind_dir, 'water_level': row.get('water_level', 0)})
        current_time = anim_df.index[0]
        current_data = anim_df.iloc[0]
        wind_mode = 'arrows'
        map_fig = create_presentation_map(anim_df, center_lat, center_lon, current_time, station_name, wind_history_mode=wind_mode, wind_rose_overlay=True)
        water_chart = create_water_level_chart(anim_df, current_time, current_data)
        wind_chart = create_wind_speed_chart(anim_df, current_time, current_data)
        marks = generate_slider_marks(anim_df.index)
        anim_records = anim_df.reset_index().rename(columns={'index': 'dt'})
        anim_records['dt'] = anim_records['dt'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        animation_store = {
            'frames': animation_frames,
            'records': anim_records.to_dict('records'),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'station_name': station_name,
            'station_id': station_id,
            'water_level_min': float(anim_df['water_level'].min()) - 1,
            'water_level_max': float(anim_df['water_level'].max()) + 1,
            'cache_key': str(cache_key)
        }
        result = (map_fig, water_chart, wind_chart, station_name, len(anim_df) - 1, marks, animation_store)
        cache_set(cache_key, result)
        cache_set(f"{cache_key}_df", anim_df)
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        empty_fig = create_empty_figure(f'Error: {str(e)}')
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
    if not animation_data or time_idx is None:
        return (dash.no_update,) * 6
    frames = animation_data['frames']
    if time_idx >= len(frames):
        return (dash.no_update,) * 6
    frame = frames[time_idx]
    
    # Rebuild DataFrame from cache for map updates
    try:
        # Get the cache key from animation data to ensure we're using the right date's data
        stored_cache_key = animation_data.get('cache_key')
        df_cache_key = None
        if stored_cache_key:
            import ast
            cache_key_tuple = ast.literal_eval(stored_cache_key)
            df_cache_key = f"{cache_key_tuple}_df"

        anim_df = None
        if df_cache_key:
            anim_df = cache_get(df_cache_key)

        if anim_df is not None:
            # Make sure time_idx is within bounds
            if time_idx >= len(anim_df):
                time_idx = len(anim_df) - 1
            current_time = anim_df.index[time_idx]
            current_data = anim_df.iloc[time_idx]
            station_id = animation_data.get('station_id', '8415191')
            station_info = STATION_INFO.get(station_id, STATION_INFO['8415191'])
            station_name = animation_data.get('station_name', station_info['name'])
            map_fig = create_presentation_map(anim_df, station_info['lat'], station_info['lon'], current_time, station_name, wind_history_mode='arrows', wind_rose_overlay=True, current_idx=time_idx, current_data=current_data)
            map_fig.update_layout(uirevision='map-constant', transition={'duration': 0})
        else:
            map_fig = dash.no_update
    except Exception as e:
        logger.error(f"Error creating map: {e}")
        map_fig = dash.no_update
    
    water_chart_patch, wind_chart_patch, time_str, surge_str, wind_str = build_frame_patches(frame)
    return (map_fig, water_chart_patch, wind_chart_patch, time_str, surge_str, wind_str)


@app.callback(
    Output('time-slider', 'value'),
    Output('animation-interval', 'disabled'),
    Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('animation-interval', 'n_intervals'),
    Input('time-slider', 'value'),  # Pause when user moves the slider
    State('time-slider', 'value'),
    State('time-slider', 'max'),
    State('step-size-selector', 'value'),
    prevent_initial_call=True
)
def control_animation(play_clicks, pause_clicks, reset_clicks, n_intervals, slider_value, current_value, max_value, step_size):
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
    elif 'time-slider' in trigger:
        # User manually adjusted the slider; pause animation and keep slider value
        return slider_value, True
    return dash.no_update, True


@app.callback(Output('animation-interval', 'interval'), Input('speed-slider', 'value'))
def update_speed(speed_ms):
    return speed_ms


@app.server.route('/health')
def health_check():
    return {'status': 'ok'}, 200


if __name__ == '__main__':
    logger.info("Starting Belfast Harbor Storm Surge & Wind Visualization - PRESENTATION MODE (Standalone)")
    port = int(os.getenv('PORT', '8052'))
    logger.info(f"Server running on http://localhost:{port}")
    app.run(debug=True, port=port)
