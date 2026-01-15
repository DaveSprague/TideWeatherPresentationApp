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
import sys
import uuid
import webbrowser
import threading
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from presentation_app.utils import create_presentation_map
from presentation_app.components.overlay_panel import create_overlay_panel
from presentation_app.data.loader import DataLoader
from presentation_app.data.noaa_api import NOAAClient
from presentation_app.data.processor import SurgeProcessor
from presentation_app.cache import LRUCacheTTL
from presentation_app.config import STATION_INFO, CACHE_ENABLED, CACHE_MAX_SIZE, CACHE_TTL_SECONDS, DATA_WINDOW_HOURS, SLIDER_MARK_STRIDE, WIND_SPEED_UNIT, KNOTS_TO_MPH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Get the base path for bundled resources (PyInstaller compatibility)
def get_base_path():
    """Get the base path for resources - handles both dev and PyInstaller bundled app."""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).parent


BASE_PATH = get_base_path()


# Cache wrapper functions
def cache_get(key):
    """Retrieve a value from cache supporting both dict and LRU cache implementations."""
    return session_cache.get(key) if hasattr(session_cache, 'get') else None


def cache_set(key, value):
    """Store a value in cache supporting both dict and LRU cache implementations."""
    if isinstance(session_cache, dict):
        session_cache[key] = value
    else:
        session_cache.set(key, value)


def format_wind_speed(speed_kts: float) -> str:
    """Format wind speed value with correct unit based on config setting."""
    if WIND_SPEED_UNIT == 'mph':
        return f"{speed_kts * KNOTS_TO_MPH:.1f} mph"
    return f"{speed_kts:.1f} kts"


def create_empty_figure(title: str = "No data"):
    """Create an empty Plotly figure with a title message for error states."""
    return {'data': [], 'layout': {'title': title}}


def generate_slider_marks(datetimes: pd.DatetimeIndex) -> dict:
    """Generate slider marks from datetime index, showing midnight timestamps with stride."""
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
    """Create water level chart with observed, predicted, and current time marker."""
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
    """Create wind speed chart with area fill and current time marker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['wind_speed'], mode='lines', name='Wind Speed', line=dict(color='#2ecc71', width=2), fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'))
    y_max = df['wind_speed'].max() + 5
    fig.add_trace(go.Scatter(x=[current_time, current_time], y=[0, y_max], mode='lines', name='Current', line=dict(color='red', width=2), hoverinfo='skip'))
    wind_spd = current_data.get('wind_speed', 0)
    wind_dir = current_data.get('wind_dir_from', 0)
    fig.add_trace(go.Scatter(x=[current_time], y=[wind_spd], mode='markers+text', marker=dict(size=12, color='red', symbol='circle'), text=[f"{format_wind_speed(wind_spd)} @ {wind_dir:.0f}°"], textposition='top center', hoverinfo='skip', showlegend=False))
    y_axis_label = 'MPH' if WIND_SPEED_UNIT == 'mph' else 'Knots'
    fig.update_layout(title='Wind Speed', xaxis=dict(title=''), yaxis=dict(title=y_axis_label, range=[0, y_max]), height=280, margin=dict(l=50, r=20, t=40, b=40), hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def create_full_range_combined_chart(tide_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """Create a combined full-range chart with tide, surge, and wind as subplots with shared x-axis."""
    
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Tide: Observed vs Predicted (Full Range)', 
                       'Storm Surge (Full Range)', 
                       'Wind Speed (Full Range)'),
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Row 1: Tide chart (observed vs predicted)
    fig.add_trace(go.Scatter(
        x=tide_df.index, 
        y=tide_df['water_level'], 
        mode='lines', 
        name='Observed Water Level',
        line=dict(color='#3498db', width=1.5),
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Water Level: %{y:.2f} ft<extra></extra>'
    ), row=1, col=1)
    
    # Row 2 & 3: Will be populated with surge and wind data
    # Fetch predictions for tide and surge
    try:
        station_id = app_data.get('station_id', '8415191')
        noaa_client = NOAAClient(station_id)
        start_dt = tide_df.index.min()
        end_dt = tide_df.index.max()
        
        all_predictions = []
        current_start = start_dt
        
        while current_start < end_dt:
            chunk_end = min(current_start + pd.DateOffset(months=1), end_dt)
            logger.info(f"Fetching predictions for {current_start.date()} to {chunk_end.date()}")
            chunk_predictions = noaa_client.fetch_predictions(current_start, chunk_end, use_hilo=True)
            
            if chunk_predictions is not None and not chunk_predictions.empty:
                all_predictions.append(chunk_predictions)
            
            current_start = chunk_end
        
        if all_predictions:
            predictions = pd.concat(all_predictions).sort_index()
            predictions = predictions[~predictions.index.duplicated(keep='first')]
            
            from presentation_app.data.processor import SurgeProcessor
            processor = SurgeProcessor()
            processed = processor.calculate_surge_from_predictions(tide_df, predictions, method='pchip')
            
            if 'predicted' in processed.columns:
                # Add predicted tide to row 1
                fig.add_trace(go.Scatter(
                    x=processed.index, 
                    y=processed['predicted'], 
                    mode='lines', 
                    name='Predicted Tide',
                    line=dict(color='#e67e22', width=2, dash='dash'),
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Predicted: %{y:.2f} ft<extra></extra>'
                ), row=1, col=1)
                logger.info(f"Added predicted tide line (interpolated from {len(predictions)} high/low points)")
            
            if 'surge' in processed.columns:
                # Add surge to row 2
                fig.add_trace(go.Scatter(
                    x=processed.index, 
                    y=processed['surge'], 
                    mode='lines', 
                    name='Storm Surge',
                    line=dict(color='#e74c3c', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Surge: %{y:+.2f} ft<extra></extra>'
                ), row=2, col=1)
                
                # Add zero line to surge plot
                fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'), opacity=0.5, row=2, col=1)
                logger.info(f"Added surge data")
    except Exception as e:
        logger.warning(f"Could not fetch NOAA predictions for full range: {e}", exc_info=True)
    
    # Row 3: Wind speed
    fig.add_trace(go.Scatter(
        x=weather_df.index, 
        y=weather_df['wind_speed'], 
        mode='lines', 
        name='Wind Speed',
        line=dict(color='#2ecc71', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)',
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Wind Speed: %{y:.1f} kts<extra></extra>'
    ), row=3, col=1)
    
    # Update axes
    y_min_tide = tide_df['water_level'].min() - 1
    y_max_tide = tide_df['water_level'].max() + 1
    y_max_wind = weather_df['wind_speed'].max() + 5
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Feet", range=[y_min_tide, y_max_tide], row=1, col=1)
    fig.update_yaxes(title_text="Feet", row=2, col=1)
    fig.update_yaxes(title_text="Knots", range=[0, y_max_wind], row=3, col=1)
    
    fig.update_layout(
        height=900,
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.05, xanchor='center', x=0.5)
    )
    
    return fig


def create_full_range_surge_chart(tide_df: pd.DataFrame, weather_df: pd.DataFrame) -> go.Figure:
    """Create a full-range surge chart showing actual minus predicted for the entire dataset."""
    fig = go.Figure()
    
    # Fetch and calculate surge for the full range
    try:
        station_id = app_data.get('station_id', '8415191')
        noaa_client = NOAAClient(station_id)
        start_dt = tide_df.index.min()
        end_dt = tide_df.index.max()
        
        # Fetch predictions in monthly chunks
        all_predictions = []
        current_start = start_dt
        
        while current_start < end_dt:
            chunk_end = min(current_start + pd.DateOffset(months=1), end_dt)
            chunk_predictions = noaa_client.fetch_predictions(current_start, chunk_end, use_hilo=True)
            
            if chunk_predictions is not None and not chunk_predictions.empty:
                all_predictions.append(chunk_predictions)
            
            current_start = chunk_end
        
        if all_predictions:
            predictions = pd.concat(all_predictions).sort_index()
            predictions = predictions[~predictions.index.duplicated(keep='first')]
            
            from presentation_app.data.processor import SurgeProcessor
            processor = SurgeProcessor()
            processed = processor.calculate_surge_from_predictions(tide_df, predictions, method='pchip')
            
            if 'surge' in processed.columns:
                # Add surge line
                fig.add_trace(go.Scatter(
                    x=processed.index, 
                    y=processed['surge'], 
                    mode='lines', 
                    name='Storm Surge',
                    line=dict(color='#e74c3c', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Surge: %{y:+.2f} ft<extra></extra>'
                ))
                
                # Add zero line
                fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
                
                logger.info(f"Added surge data to full-range chart")
    except Exception as e:
        logger.warning(f"Could not calculate surge for full range: {e}", exc_info=True)
    
    # Set y-axis range based on surge values if available
    if fig.data and hasattr(fig.data[0], 'y') and fig.data[0].y is not None:  # type: ignore
        y_values = fig.data[0].y  # type: ignore
        y_max = max(abs(min(y_values)), abs(max(y_values))) + 1
        y_range = [-y_max, y_max]
    else:
        y_range = [-5, 5]
    
    fig.update_layout(
        title='Storm Surge (Full Range)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Feet', range=y_range),
        height=300,
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def create_full_range_wind_chart(weather_df: pd.DataFrame) -> go.Figure:
    """Create a full-range wind speed chart for the entire dataset."""
    fig = go.Figure()
    
    # Add wind speed
    fig.add_trace(go.Scatter(
        x=weather_df.index, 
        y=weather_df['wind_speed'], 
        mode='lines', 
        name='Wind Speed',
        line=dict(color='#2ecc71', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)',
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Wind Speed: %{y:.1f} kts<extra></extra>'
    ))
    
    y_max = weather_df['wind_speed'].max() + 5
    
    fig.update_layout(
        title='Wind Speed (Full Range)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Knots', range=[0, y_max]),
        height=300,
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def build_frame_patches(frame):
    """Build Plotly patch updates and display strings for animation frame."""
    water_chart_patch = Patch()
    wind_chart_patch = Patch()
    water_chart_patch['data'][2]['x'] = [frame['timestamp'], frame['timestamp']]
    water_chart_patch['data'][3]['x'] = [frame['timestamp']]
    water_chart_patch['data'][3]['y'] = [frame['water_level']]
    water_chart_patch['data'][3]['text'] = [f"Water: {frame['water_level']:.1f} ft<br>Surge: {frame['surge']:+.1f} ft"]
    wind_chart_patch['data'][1]['x'] = [frame['timestamp'], frame['timestamp']]
    wind_chart_patch['data'][2]['x'] = [frame['timestamp']]
    wind_chart_patch['data'][2]['y'] = [frame['wind_speed']]
    wind_chart_patch['data'][2]['text'] = [f"{format_wind_speed(frame['wind_speed'])} @ {frame['wind_dir']:.0f}°"]
    time_str = frame['timestamp_str']
    surge_str = f"{frame['surge']:+.2f} ft"
    wind_str = f"{format_wind_speed(frame['wind_speed'])} @ {frame['wind_dir']:.0f}°"
    return water_chart_patch, wind_chart_patch, time_str, surge_str, wind_str


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Storm Surge Visualization - Presentation Mode"
server = app.server  # Expose Flask server for gunicorn
app_data = {'tide_df': None,'weather_df': None,'station_id': '8415191'}
# In-memory cache: LRU + TTL when enabled, plain dict otherwise
session_cache = LRUCacheTTL(max_size=CACHE_MAX_SIZE, ttl_seconds=CACHE_TTL_SECONDS) if CACHE_ENABLED else {}

try:
    loader = DataLoader()
    tide_file = BASE_PATH / 'tide_belfast.csv'
    weather_file = BASE_PATH / 'weather_belfast.csv'
    logger.info(f"Loading data from: tide={tide_file}, weather={weather_file}")
    tide_df = loader.load_tide_csv(tide_file)
    weather_df = loader.load_weather_csv(weather_file)
    app_data['tide_df'] = tide_df
    app_data['weather_df'] = weather_df
    logger.info(f"Loaded default data: {len(tide_df)} tide rows, {len(weather_df)} weather rows")
except Exception as e:
    logger.warning(f"Could not load default data: {e}")


def get_initial_dates():
    """Get initial date range from loaded tide and weather data."""
    if app_data['tide_df'] is not None and app_data['weather_df'] is not None:
        try:
            # Data is already loaded and processed with datetime index
            tide_df = app_data['tide_df']
            weather_df = app_data['weather_df']
            min_date_overlap = max(tide_df.index.min(), weather_df.index.min())
            max_date_overlap = min(tide_df.index.max(), weather_df.index.max())
            forced_center = pd.Timestamp('2024-01-10')
            center_date = forced_center
            min_date = min(forced_center, min_date_overlap)
            max_date = max_date_overlap
            logger.info(f"Initial dates: min={min_date}, max={max_date}, center={center_date}")
            return min_date, max_date, center_date
        except Exception as e:
            logger.error(f"Error getting initial dates: {e}", exc_info=True)
            pass
    return None, None, None

initial_min, initial_max, initial_center = get_initial_dates()

app.layout = html.Div([
    html.Div([dcc.Graph(id='surge-map', config={'displayModeBar': False}, animate=False, style={'width': '100%', 'height': '65vh'})], style={'width': '100%', 'position': 'relative', 'marginBottom': '20px'}),
    html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(id='water-level-chart', config={'displayModeBar': False}, animate=False, style={'height': '220px'})], width=6, style={'padding': '0 5px'}),
            dbc.Col([dcc.Graph(id='wind-speed-chart', config={'displayModeBar': False}, animate=False, style={'height': '220px'})], width=6, style={'padding': '0 5px'}),
        ], style={'margin': 0})
    ], style={'width': '100%', 'background': '#f8f9fa', 'padding': '8px 0', 'position': 'relative'}),
    # Full-range overview graphs at the bottom (combined with shared x-axis)
    html.Div([
        dcc.Graph(id='full-range-combined-chart', config={'displayModeBar': False}, animate=False)
    ], style={'width': '100%', 'background': '#ffffff', 'padding': '4px 0', 'marginTop': '12px'}),
    create_overlay_panel(min_date=initial_min, max_date=initial_max, center_date=initial_center),
], style={'margin': 0, 'padding': '12px', 'overflowY': 'auto', 'minHeight': '100vh', 'display': 'flex', 'flexDirection': 'column'})


@app.callback(Output('session-id', 'data'), Input('time-slider', 'id'), State('session-id', 'data'))
def ensure_session_id(_, existing):
    if existing:
        return existing
    return str(uuid.uuid4())


@app.callback(
    Output('full-range-combined-chart', 'figure'),
    Input('session-id', 'data'),
    prevent_initial_call=False
)
def populate_full_range_chart(session_id):
    """Populate the combined full-range chart with tide, surge, and wind subplots on page load."""
    if app_data['tide_df'] is None or app_data['weather_df'] is None:
        return create_empty_figure('No data available')
    
    try:
        combined_fig = create_full_range_combined_chart(app_data['tide_df'], app_data['weather_df'])
        return combined_fig
    except Exception as e:
        logger.error(f"Error creating full-range chart: {e}", exc_info=True)
        return create_empty_figure(f'Error: {str(e)}')


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
        # Data is already loaded and processed with datetime index
        tide_df = app_data['tide_df']
        weather_df = app_data['weather_df']
        min_date = max(tide_df.index.min(), weather_df.index.min())
        max_date = min(tide_df.index.max(), weather_df.index.max())
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
     Output('animation-data-store', 'data'),
     Output('current-time-display', 'children'),
     Output('current-surge-value', 'children'),
     Output('current-wind-value', 'children')],
    Input('center-date-picker', 'date'),
    State('session-id', 'data'),
    State('data-version', 'data')
)
def process_data(center_date, session_id, data_version):
    logger.info(f"process_data called with center_date={center_date}")
    if app_data['tide_df'] is None or app_data['weather_df'] is None:
        empty_fig = create_empty_figure('Upload data to begin')
        return empty_fig, empty_fig, empty_fig, "No data", 0, {}, None, "--:--", "--", "--"
    try:
        session_token = session_id or str(uuid.uuid4())
        cache_key = (session_token, str(center_date), data_version or 0)
        cached = cache_get(cache_key)
        if cached:
            return cached
        center_dt = pd.to_datetime(center_date)
        start_dt = center_dt - pd.Timedelta(hours=DATA_WINDOW_HOURS)
        end_dt = center_dt + pd.Timedelta(hours=DATA_WINDOW_HOURS)
        logger.info(f"Calculated window: center_dt={center_dt}, start_dt={start_dt}, end_dt={end_dt}")
        loader = DataLoader()
        # Data is already loaded and processed with datetime index
        tide_raw = app_data['tide_df']
        weather_raw = app_data['weather_df']
        available_min = max(tide_raw.index.min(), weather_raw.index.min())
        available_max = min(tide_raw.index.max(), weather_raw.index.max())
        logger.info(f"Available data range: {available_min} to {available_max}")
        tide_df, weather_df = loader.filter_by_window(tide_raw, weather_raw, start_dt, end_dt)
        logger.info(f"After filter_by_window: tide_df has {len(tide_df)} rows, weather_df has {len(weather_df)} rows")
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
        if predictions is None:
            predictions = pd.DataFrame()
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
            if hasattr(ts, 'isoformat') and hasattr(ts, 'strftime'):
                ts_iso = ts.isoformat()  # type: ignore
                ts_str = ts.strftime('%B %d, %Y %H:%M')  # type: ignore
            else:
                ts_iso = str(ts)
                ts_str = str(ts)
            animation_frames.append({'timestamp': ts_iso, 'timestamp_str': ts_str, 'surge': surge_val, 'wind_speed': wind_spd, 'wind_dir': wind_dir, 'water_level': row.get('water_level', 0)})
        current_time = anim_df.index[0]
        current_data = anim_df.iloc[0]
        wind_mode = 'arrows'
        map_fig = create_presentation_map(anim_df, center_lat, center_lon, current_time, station_name, wind_history_mode=wind_mode, wind_rose_overlay=True)
        water_chart = create_water_level_chart(anim_df, current_time, current_data)
        wind_chart = create_wind_speed_chart(anim_df, current_time, current_data)
        if isinstance(anim_df.index, pd.DatetimeIndex):
            marks = generate_slider_marks(anim_df.index)
        else:
            marks = generate_slider_marks(pd.DatetimeIndex(anim_df.index))
        anim_records = anim_df.reset_index().rename(columns={'index': 'dt'})
        if hasattr(anim_records['dt'], 'dt') and hasattr(anim_records['dt'].dt, 'strftime'):
            anim_records['dt'] = anim_records['dt'].dt.strftime('%Y-%m-%dT%H:%M:%S')  # type: ignore
        else:
            anim_records['dt'] = anim_records['dt'].astype(str)
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
        
        # Get initial frame values for display
        first_frame = animation_frames[0]
        time_str = first_frame['timestamp_str']
        surge_str = f"{first_frame['surge']:.2f} ft"
        wind_str = f"{format_wind_speed(first_frame['wind_speed'])} @ {first_frame['wind_dir']:.0f}°"
        
        result = (map_fig, water_chart, wind_chart, station_name, len(anim_df) - 1, marks, animation_store, time_str, surge_str, wind_str)
        cache_set(cache_key, result)
        cache_set(f"{cache_key}_df", anim_df)
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        empty_fig = create_empty_figure(f'Error: {str(e)}')
        return empty_fig, empty_fig, empty_fig, "Error", 0, {}, None, "--:--", "--", "--"


@app.callback(
    [Output('surge-map', 'figure', allow_duplicate=True),
     Output('water-level-chart', 'figure', allow_duplicate=True),
     Output('wind-speed-chart', 'figure', allow_duplicate=True),
     Output('current-time-display', 'children', allow_duplicate=True),
     Output('current-surge-value', 'children', allow_duplicate=True),
     Output('current-wind-value', 'children', allow_duplicate=True)],
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
    logger.info(f"update_time_position: time_idx={time_idx}, frame timestamp={frame.get('timestamp_str', 'N/A')}, total frames={len(frames)}")
    
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


def kill_port_processes(port: int):
    """Terminate any processes using the specified port."""
    import subprocess
    import signal
    
    try:
        # Use lsof to find processes using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            logger.info(f"Found {len(pids)} process(es) using port {port}, terminating...")
            
            for pid in pids:
                try:
                    pid_int = int(pid)
                    os.kill(pid_int, signal.SIGTERM)
                    logger.info(f"Terminated process {pid_int}")
                except (ValueError, ProcessLookupError) as e:
                    logger.debug(f"Could not terminate process {pid}: {e}")
            
            # Give processes time to shut down
            import time
            time.sleep(1)
        else:
            logger.debug(f"No processes found using port {port}")
    except FileNotFoundError:
        # lsof not available (unlikely on macOS, but handle gracefully)
        logger.warning("lsof command not found, cannot check for existing processes")
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout checking for processes on port {port}")
    except Exception as e:
        logger.warning(f"Error checking/killing processes on port {port}: {e}")


if __name__ == '__main__':
    logger.info("Starting Belfast Harbor Storm Surge & Wind Visualization - PRESENTATION MODE (Standalone)")
    port = int(os.getenv('PORT', '8052'))
    
    # Kill any existing processes using the port (only in parent process to avoid reloader conflicts)
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        kill_port_processes(port)
    
    logger.info(f"Server running on http://localhost:{port}")
    
    # Open browser automatically after a short delay (only in reloader child or non-debug mode)
    if os.environ.get('WERKZEUG_RUN_MAIN') or getattr(sys, 'frozen', False):
        def open_browser():
            time.sleep(2)  # Give server time to start
            url = f'http://localhost:{port}'
            logger.info(f"Opening browser to {url}")
            webbrowser.open(url)
        
        # Start browser opening in a separate thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    # Disable debug mode when running as PyInstaller bundle to avoid reloader issues
    debug_mode = not getattr(sys, 'frozen', False)
    app.run(debug=debug_mode, port=port)
