"""
Standalone Presentation App Configuration
"""
import os

# CSV Column Mappings (maps internal name -> CSV header name)
TIDE_CSV_COLUMNS = {
    'datetime': 'timestring [America/New_York]',
    'water_level': 'Water level [ft]'
}

WEATHER_CSV_COLUMNS = {
    'datetime': 'timestring [America/New_York]',
    'wind_speed': 'Wind speed [kn]',
    'wind_direction': 'Wind direction (coming from) [deg N]'
}

# Column Metadata (units and display labels)
TIDE_COLUMN_METADATA = {
    'water_level': {
        'units': 'ft',
        'label': 'Water Level'
    }
}

WEATHER_COLUMN_METADATA = {
    'wind_speed': {
        'units': 'kn',
        'label': 'Wind Speed'
    },
    'wind_direction': {
        'units': 'deg',
        'label': 'Wind Direction'
    }
}

# Timezone for local time
LOCAL_TIMEZONE = 'America/New_York'

# Station info
STATION_INFO = {
    '8415191': {
        'name': 'Belfast Harbor, ME',
        'lat': 44.428903,
        'lon': -69.004461
    },
    '8418150': {
        'name': 'Portland, ME',
        'lat': 43.6563,
        'lon': -70.2436
    }
}
DEFAULT_STATION = '8415191'

# Map
MAP_STYLE = 'open-street-map'
DEFAULT_ZOOM = 12
MAP_HEIGHT = 800

# Wind display
MIN_WIND_SPEED_DISPLAY = 1.0
WIND_ARROW_SCALE = 0.002

# Surge thresholds/colors
SURGE_THRESHOLDS = {
    'extreme_low': -1.5,
    'normal_low': -0.5,
    'normal_high': 0.5,
    'moderate': 1.5,
    'high': 2.5
}
SURGE_COLORS = {
    'extreme_low': '#1a472a',
    'normal': '#2ecc71',
    'moderate': '#f39c12',
    'high': '#e74c3c',
    'extreme': '#8b0000'
}

# Wind history
WIND_HISTORY_LENGTH = 6
WIND_FADE_OPACITY_MIN = 0.2
WIND_FADE_OPACITY_MAX = 0.8
WIND_HISTORY_MODE = 'arrows'

# NOAA API
NOAA_API_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
API_TIMEOUT = 30
PREDICTION_BUFFER_DAYS = 2

# Processing
TIDAL_WINDOW = '25h'
DATA_WINDOW_HOURS = 18  # Hours before/after center date for data window
SLIDER_MARK_STRIDE = 8  # Show every Nth midnight mark on slider

# Caching
CACHE_ENABLED = os.getenv('CACHE_ENABLED', '1') == '1'
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '32'))
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '900'))  # 15 minutes default
