"""
Data models for the TideWeather Presentation App.

Centralising the data model here supports the two main future goals:
  1. Multiple stations on the same map  – add more StationData instances
  2. Time/location-stamped flood images – FloodPhoto layer on the map
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class StationData:
    """All configuration and loaded data for a single tide/weather station.

    Replaces the previous ``app_data`` flat dict so the shape of per-station
    data is explicit and reusable.  When a second station is added, create
    another ``StationData`` and pass both to the map renderer.
    """
    station_id: str
    name: str
    lat: float
    lon: float
    tide_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    weather_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    # Resampled / processed DataFrame used for animation frames
    anim_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def has_data(self) -> bool:
        return self.tide_df is not None and self.weather_df is not None


@dataclass
class FloodPhoto:
    """A time and location-stamped photograph of actual flooding.

    Intended to be rendered as a camera-icon marker on the map.  When the
    animation time is within a threshold of ``timestamp`` the marker is
    highlighted and the photo can be displayed in a side panel or modal.

    ``image_path`` should be a path relative to the Dash ``assets/`` folder
    (e.g. ``photos/flooding_2024-01-10.jpg``) or a fully-qualified URL.
    Dash serves the ``assets/`` directory as static files automatically.
    """
    photo_id: str
    lat: float
    lon: float
    timestamp: pd.Timestamp
    image_path: str
    description: str = ""
    location_name: str = ""
