# Refactor Plan — claude/refactor-weather-code-1vTNo

Last updated: 2026-02-28

## Task List (in priority order)

| # | Task | Status |
|---|------|--------|
| 1 | Create `presentation_app/data/models.py` with `StationData` and `FloodPhoto` | ✅ Done |
| 2 | Refactor `app_data` from flat dict → `StationData`-based structure | ✅ Done |
| 3 | Make `create_presentation_map()` accept `list[StationData]` | ✅ Done |
| 4 | NOAA API deduplication | ✅ Done — `_fetch_full_range_predictions()` helper in `app.py` |
| 5 | CSV loader helper | ✅ Done — `DataLoader._read_csv_normalized()` in `loader.py` |
| 6 | Extract full-range prediction fetching | ✅ Done (merged with #4) |
| 7 | Fix hardcoded center date, magic numbers | ✅ Done — `DEFAULT_CENTER_DATE` in `config.py` |
| 8 | Wind rose NumPy / line formatting | ✅ Done — vectorised with `np.bincount`; warm colour ramp |
| 9 | Better wind rose colours (new) | ✅ Done — sky-blue → yellow → deep-red ramp via `_wind_rose_color()` |
| 10 | Surge reference rings on map (new) | ✅ Done — dashed circles at 1 ft / 2 ft / 3 ft via `build_surge_reference_circles()` |

## Key Design Decisions

### Surge visualisation (items 9–10)
- `build_surge_indicator_traces()` replaces the old fixed-size point marker with a filled
  polygon whose **radius scales with surge magnitude** (1 surge ft = `SURGE_RING_RADIUS_DEG_PER_FT`
  degrees). The circle grows to touch the corresponding reference ring.
- `build_surge_reference_circles()` draws dashed white rings at 1 ft / 2 ft / 3 ft inside
  the wind-rose centre hole (shown when `wind_rose_overlay=True`).
- Both constants (`SURGE_RING_RADIUS_DEG_PER_FT`, `SURGE_RING_LEVELS`) live in `config.py`.

### Multi-station map (#3)
- `create_presentation_map(stations: list[StationData], time_indices: list[int], ...)` — each
  `StationData` must have `anim_df` populated before the call.
- `_render_station_traces(fig, station, time_idx, ...)` is called once per station.
- Call sites use `dataclasses.replace(active_station, anim_df=anim_df)` to avoid mutating global state.

## Flood Photo Design Note

Most flexible approach: a small CSV or JSON metadata file listing each photo
(lat, lon, timestamp, filename) alongside a folder of images. User uploads the
metadata file the same way they currently upload tide/weather CSVs. Images live
in `assets/photos/` (Dash serves `assets/` as static files automatically), so
`image_path` is just a relative URL like:

    /assets/photos/flooding_2024-01-10_14-30.jpg

The `FloodPhoto` model in `models.py` is already defined and ready to wire up.

## Remaining Work

- **FloodPhoto map layer** — render camera-icon markers on the map, highlight near
  animation time, show image in a side panel or modal.

## Branch

All work goes on `claude/refactor-weather-code-1vTNo`. Never push to master.
