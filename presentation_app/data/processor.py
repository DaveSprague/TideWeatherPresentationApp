"""
Storm surge calculation and data processing (standalone)
"""
import pandas as pd
import numpy as np
import logging
from scipy.interpolate import PchipInterpolator
from typing import Optional
from config import TIDAL_WINDOW, SURGE_THRESHOLDS, SURGE_COLORS

logger = logging.getLogger(__name__)


class SurgeProcessor:
    @staticmethod
    def calculate_surge_from_predictions(df: pd.DataFrame, predictions: pd.DataFrame, method: str = 'pchip') -> pd.DataFrame:
        logger.info(f"Calculating surge using {method} interpolation")
        if predictions is None or predictions.empty:
            logger.warning("No predictions available, using harmonic fallback")
            return SurgeProcessor._calculate_surge_harmonic(df)
        try:
            combined_idx = df.index.union(predictions.index).sort_values()
            pred_series = predictions['predicted'].reindex(combined_idx)
            if method == 'pchip' and len(predictions) > 3:
                pred_series = SurgeProcessor._interpolate_pchip(pred_series)
            elif method == 'cubic':
                pred_series = pred_series.interpolate(method='cubic')
            else:
                pred_series = pred_series.interpolate(method='linear')
            df['predicted'] = pred_series.reindex(df.index)
            df['predicted'] = df['predicted'].interpolate(method='linear').bfill().ffill()
            df['predicted'] = df['predicted'].clip(lower=-2.0, upper=15.0)
            df['surge'] = df['water_level'] - df['predicted']
            return df
        except Exception:
            return SurgeProcessor._calculate_surge_harmonic(df)

    @staticmethod
    def _interpolate_pchip(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < 4:
            return series.interpolate(method='cubic')
        start_time = series.index.min()
        valid_hours = [(t - start_time).total_seconds() / 3600 for t in valid.index]
        all_hours = [(t - start_time).total_seconds() / 3600 for t in series.index]
        pchip = PchipInterpolator(valid_hours, valid.values)
        values = pchip(all_hours)
        return pd.Series(values, index=series.index)

    @staticmethod
    def _calculate_surge_harmonic(df: pd.DataFrame) -> pd.DataFrame:
        df['predicted'] = df['water_level'].rolling(window=TIDAL_WINDOW, center=True, min_periods=12).mean()
        df['predicted'] = df['predicted'].bfill().ffill()
        df['surge'] = df['water_level'] - df['predicted']
        return df

    @staticmethod
    def resample_data(df: pd.DataFrame, interval: str = '15min') -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        resampled = df[numeric_cols].resample(interval).mean()
        resampled = resampled.interpolate(method='linear', limit=4)
        resampled = resampled.dropna()
        return resampled

    @staticmethod
    def get_surge_color(surge_value: float) -> str:
        if surge_value < SURGE_THRESHOLDS['extreme_low']:
            return SURGE_COLORS['extreme_low']
        elif surge_value < SURGE_THRESHOLDS['normal_high']:
            return SURGE_COLORS['normal']
        elif surge_value < SURGE_THRESHOLDS['moderate']:
            return SURGE_COLORS['moderate']
        elif surge_value < SURGE_THRESHOLDS['high']:
            return SURGE_COLORS['high']
        else:
            return SURGE_COLORS['extreme']
