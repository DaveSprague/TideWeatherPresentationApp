"""
CSV data loading utilities (standalone)
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple
from ..validators import validate_tide_data, validate_weather_data, sanitize_numeric_column
from ..config import TIDE_CSV_COLUMNS, WEATHER_CSV_COLUMNS, LOCAL_TIMEZONE

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial processing of CSV data"""
    
    @staticmethod
    def parse_raw_dataframes(tide_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse raw tide and weather DataFrames, adding datetime index."""
        tide_parsed = tide_df.copy()
        weather_parsed = weather_df.copy()
        tide_parsed['dt'] = pd.to_datetime(tide_parsed['datetime'], errors='coerce')
        weather_parsed['dt'] = pd.to_datetime(weather_parsed['datetime'], errors='coerce')
        return tide_parsed, weather_parsed
    
    @staticmethod
    def filter_by_window(tide_raw: pd.DataFrame, weather_raw: pd.DataFrame, 
                        start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter and prepare tide/weather data for a specific time window.
        Expects data that's already been loaded with datetime index."""
        # Data already has datetime index from load_tide_csv/load_weather_csv
        tide_df_local = tide_raw[['water_level']].copy()
        tide_df_local = tide_df_local[(tide_df_local.index >= start_dt) & (tide_df_local.index <= end_dt)].dropna()
        
        weather_df_local = weather_raw.copy()
        weather_df_local = weather_df_local[(weather_df_local.index >= start_dt) & (weather_df_local.index <= end_dt)]
        weather_df_local = weather_df_local[['wind_speed', 'wind_dir_from']].dropna()
        
        return tide_df_local, weather_df_local
    
    @staticmethod
    def load_tide_csv(file_path: Path, start_date: Optional[pd.Timestamp] = None, 
                      end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        logger.info(f"Loading tide data from {file_path}")
        # Read CSV with headers - first non-comment line is the header
        df = pd.read_csv(file_path, comment='#', on_bad_lines='skip')
        
        # Clean column names - remove trailing # and whitespace
        df.columns = df.columns.str.rstrip('#').str.rstrip()
        
        # Create reverse mapping: CSV column name -> internal name
        column_mapping = {csv_name: internal_name for internal_name, csv_name in TIDE_CSV_COLUMNS.items()}
        
        # Rename columns from CSV names to internal names
        df = df.rename(columns=column_mapping)
        
        # Parse datetime from the 'datetime' column (local time)
        df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.set_index('dt')
        
        # Sanitize and select water level column
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
        # Read CSV with headers - first non-comment line is the header
        df = pd.read_csv(file_path, comment='#', on_bad_lines='skip')
        
        # Clean column names - remove trailing # and whitespace
        df.columns = df.columns.str.rstrip('#').str.rstrip()
        
        # Create reverse mapping: CSV column name -> internal name
        column_mapping = {csv_name: internal_name for internal_name, csv_name in WEATHER_CSV_COLUMNS.items()}
        
        # Rename columns from CSV names to internal names
        df = df.rename(columns=column_mapping)
        
        # Parse datetime from the 'datetime' column (local time)
        df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.set_index('dt')
        
        # Sanitize numeric columns
        df['wind_speed'] = sanitize_numeric_column(df['wind_speed'], 'wind_speed', min_val=0.0, max_val=150.0)
        df['wind_direction'] = sanitize_numeric_column(df['wind_direction'], 'wind_direction', min_val=0.0, max_val=360.0)
        
        # Select and rename to standard internal names
        df = df[['wind_speed', 'wind_direction']].dropna(subset=['wind_speed', 'wind_direction'])
        df = df.rename(columns={'wind_direction': 'wind_dir_from'})
        
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
