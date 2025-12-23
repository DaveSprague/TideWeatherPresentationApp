"""
Validation utilities (standalone)
"""
import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def validate_tide_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    if df.empty:
        return False, "Tide data is empty"
    if 'water_level' not in df.columns:
        return False, "Missing required column: water_level"
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Index must be DatetimeIndex"
    wl = df['water_level'].dropna()
    if len(wl) == 0:
        return False, "No valid water level data found"
    return True, None


def validate_weather_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    if df.empty:
        return False, "Weather data is empty"
    for col in ['wind_speed', 'wind_dir_from']:
        if col not in df.columns:
            return False, f"Missing required column: {col}"
    if not isinstance(df.index, pd.DatetimeIndex):
        return False, "Index must be DatetimeIndex"
    return True, None


def sanitize_numeric_column(series: pd.Series, col_name: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce')
    if min_val is not None:
        series = series.where(series >= min_val)
    if max_val is not None:
        series = series.where(series <= max_val)
    return series
