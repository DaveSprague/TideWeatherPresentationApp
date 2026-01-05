"""
NOAA API integration (standalone)
"""
import requests
import pandas as pd
import logging
from datetime import timedelta
from typing import Optional
from config import NOAA_API_BASE, API_TIMEOUT, PREDICTION_BUFFER_DAYS, STATION_INFO, DEFAULT_STATION

logger = logging.getLogger(__name__)


class NOAAClient:
    def __init__(self, station_id: str = DEFAULT_STATION):
        self.station_id = station_id
        self.station_info = STATION_INFO.get(station_id, STATION_INFO[DEFAULT_STATION])
    
    def fetch_predictions(self, start_date: pd.Timestamp, end_date: pd.Timestamp, use_hilo: bool = True) -> Optional[pd.DataFrame]:
        buffer = timedelta(days=PREDICTION_BUFFER_DAYS)
        extended_start = start_date - buffer
        extended_end = end_date + buffer
        if use_hilo:
            return self._fetch_hilo_predictions(extended_start, extended_end)
        return self._fetch_interval_predictions(start_date, end_date)
    
    def _fetch_hilo_predictions(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        logger.info(f"Fetching high/low predictions for station {self.station_id}")
        params = {
            'station': self.station_id,
            'begin_date': start_date.strftime('%Y%m%d'),
            'end_date': end_date.strftime('%Y%m%d'),
            'product': 'predictions',
            'datum': 'MLLW',
            'time_zone': 'lst_ldt',
            'units': 'english',
            'format': 'json',
            'interval': 'hilo'
        }
        try:
            r = requests.get(NOAA_API_BASE, params=params, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if 'predictions' not in data or not data['predictions']:
                return None
            df = pd.DataFrame(data['predictions'])
            df['t'] = pd.to_datetime(df['t'])
            df['v'] = pd.to_numeric(df['v'], errors='coerce')
            df = df.set_index('t').sort_index()
            return df[['v']].rename(columns={'v': 'predicted'})
        except requests.exceptions.RequestException as e:
            logger.error(f"NOAA API request failed: {e}")
            return None
    
    def _fetch_interval_predictions(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        logger.info(f"Fetching 6-minute predictions for station {self.station_id}")
        params = {
            'station': self.station_id,
            'begin_date': start_date.strftime('%Y%m%d'),
            'end_date': end_date.strftime('%Y%m%d'),
            'product': 'predictions',
            'datum': 'MLLW',
            'time_zone': 'lst_ldt',
            'units': 'english',
            'format': 'json'
        }
        try:
            r = requests.get(NOAA_API_BASE, params=params, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if 'predictions' not in data or not data['predictions']:
                return None
            df = pd.DataFrame(data['predictions'])
            df['t'] = pd.to_datetime(df['t'])
            df['v'] = pd.to_numeric(df['v'], errors='coerce')
            df = df.set_index('t').sort_index()
            return df[['v']].rename(columns={'v': 'predicted'})
        except requests.exceptions.RequestException as e:
            logger.error(f"NOAA API request failed: {e}")
            return None
    
    def try_fallback_stations(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        fallback_stations = ['8413320', '8418150']
        for sid in fallback_stations:
            if sid not in STATION_INFO:
                continue
            original = self.station_id
            self.station_id = sid
            self.station_info = STATION_INFO[sid]
            df = self.fetch_predictions(start_date, end_date, use_hilo=True)
            self.station_id = original
            self.station_info = STATION_INFO[original]
            if df is not None:
                adjustment = STATION_INFO[sid].get('adjustment', 0.0)
                if adjustment:
                    df['predicted'] = df['predicted'] + adjustment
                return df
        return None
