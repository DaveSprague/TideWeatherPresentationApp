"""
CSV data loading utilities (standalone)
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple
from ..validators import validate_tide_data, validate_weather_data, sanitize_numeric_column

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial processing of CSV data"""
    
    @staticmethod
    def parse_raw_dataframes(tide_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse raw tide and weather DataFrames, adding datetime index."""
        tide_parsed = tide_df.copy()
        weather_parsed = weather_df.copy()
        tide_parsed['dt'] = pd.to_datetime(tide_parsed['timestring'], errors='coerce')
        weather_parsed['dt'] = pd.to_datetime(weather_parsed['timestring'], errors='coerce')
        return tide_parsed, weather_parsed
    
    @staticmethod
    def filter_by_window(tide_raw: pd.DataFrame, weather_raw: pd.DataFrame, 
                        start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter and prepare tide/weather data for a specific time window."""
        tide_df_local = tide_raw.set_index('dt')[['water_level']]
        tide_df_local['water_level'] = pd.to_numeric(tide_df_local['water_level'], errors='coerce')
        tide_df_local = tide_df_local[(tide_df_local.index >= start_dt) & (tide_df_local.index <= end_dt)].dropna()
        
        weather_df_local = weather_raw.set_index('dt')
        weather_df_local = weather_df_local[(weather_df_local.index >= start_dt) & (weather_df_local.index <= end_dt)]
        for col in ['wind_speed', 'wind_dir_from', 'pressure']:
            weather_df_local[col] = pd.to_numeric(weather_df_local[col], errors='coerce')
        weather_df_local = weather_df_local[['wind_speed', 'wind_dir_from', 'pressure']].dropna()
        
        return tide_df_local, weather_df_local
    
    @staticmethod
    def load_tide_csv(file_path: Path, start_date: Optional[pd.Timestamp] = None, 
                      end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        logger.info(f"Loading tide data from {file_path}")
        cols_tide = ['timestamp', 'timestring', 'water_level']
        df = pd.read_csv(file_path, comment='#', header=None, names=cols_tide, on_bad_lines='skip')
        if 'timestring' in df.columns and df['timestring'].notna().any():
            df['dt'] = pd.to_datetime(df['timestring'], errors='coerce')
        else:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df = df.set_index('dt')
        df['water_level'] = sanitize_numeric_column(df['water_level'], 'water_level', min_val=-10.0, max_val=30.0)
        df = df[['water_level']].dropna()
        if start_date and end_date:
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        is_valid, msg = validate_tide_data(df)
        if not is_valid:
            raise ValueError(f"Tide data validation failed: {msg}")
        return df
    
    @staticmethod
    def load_weather_csv(file_path: Path, start_date: Optional[pd.Timestamp] = None, 
                        end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        logger.info(f"Loading weather data from {file_path}")
        cols_wx = ['timestamp', 'timestring', 'solar', 'rain', 'wind_speed','wind_dir_from', 'gust', 'air_temp', 'vapor', 'pressure','humidity', 'temp_humid']
        df = pd.read_csv(file_path, comment='#', header=None, names=cols_wx, on_bad_lines='skip')
        if 'timestring' in df.columns and df['timestring'].notna().any():
            df['dt'] = pd.to_datetime(df['timestring'], errors='coerce')
        else:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df = df.set_index('dt')
        df['wind_speed'] = sanitize_numeric_column(df['wind_speed'], 'wind_speed', min_val=0.0, max_val=150.0)
        df['wind_dir_from'] = sanitize_numeric_column(df['wind_dir_from'], 'wind_dir_from', min_val=0.0, max_val=360.0)
        df['pressure'] = sanitize_numeric_column(df['pressure'], 'pressure', min_val=900.0, max_val=1100.0)
        df = df[['wind_speed', 'wind_dir_from', 'pressure']].dropna(subset=['wind_speed', 'wind_dir_from'])
        if start_date and end_date:
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        is_valid, msg = validate_weather_data(df)
        if not is_valid:
            raise ValueError(f"Weather data validation failed: {msg}")
        return df
    
    @staticmethod
    def merge_datasets(tide_df: pd.DataFrame, weather_df: pd.DataFrame, tolerance: str = '30min') -> pd.DataFrame:
        logger.info(f"Merging tide and weather data with {tolerance} tolerance")
        tide_df = tide_df.sort_index()
        weather_df = weather_df.sort_index()
        merged = pd.merge_asof(tide_df, weather_df, left_index=True, right_index=True, tolerance=pd.Timedelta(tolerance), direction='nearest')
        before_len = len(merged)
        merged = merged.dropna(subset=['water_level', 'wind_speed', 'wind_dir_from'])
        return merged
