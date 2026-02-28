# Refactor Plan — claude/refactor-weather-code-1vTNo

Last updated: 2026-02-28

## Task List (in priority order)

| # | Task | Status |
|---|------|--------|
| 1 | Create `presentation_app/data/models.py` with `StationData` and `FloodPhoto` | ✅ Done |
| 2 | Refactor `app_data` from flat dict → `StationData`-based structure | ✅ Done |
| 3 | Make `create_presentation_map()` accept `list[StationData]` | ⬜ TODO |
| 4 | NOAA API deduplication | ⬜ TODO |
| 5 | CSV loader helper | ⬜ TODO |
| 6 | Extract full-range prediction fetching | ⬜ TODO |
| 7 | Fix hardcoded center date, magic numbers | ⬜ TODO |
| 8 | Wind rose NumPy / line formatting | ⬜ TODO |

## Flood Photo Design Note

Most flexible approach: a small CSV or JSON metadata file listing each photo
(lat, lon, timestamp, filename) alongside a folder of images. User uploads the
metadata file the same way they currently upload tide/weather CSVs. Images live
in `assets/photos/` (Dash serves `assets/` as static files automatically), so
`image_path` is just a relative URL like:

    /assets/photos/flooding_2024-01-10_14-30.jpg

## Branch

All work goes on `claude/refactor-weather-code-1vTNo`. Never push to master.
