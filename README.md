# Presentation App (Standalone)

A self-contained Dash app for presenting storm surge and wind data for Belfast, Maine. This package is split from the parent workspace and includes all its own dependencies and deployment files.

## Quick Start

- Python: 3.12 recommended

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://localhost:8052 (or whatever `PORT` you set)

## Features
- Map via Plotly `Scattermap` (Maplibre), no token required
- Wind overlays on-map: fading arrows + accumulating wind rose at the station
- Water level and wind speed charts with current value callouts
- Timeline scrubber updates map overlays live (wind rose and arrows)

## Structure
```
presentation_app_standalone/
  presentation_app/
    __init__.py
    app.py
    config.py
    utils.py
    validators.py
    data/
      __init__.py
      loader.py
      noaa_api.py
      processor.py
    components/
      __init__.py
      overlay_panel.py
    assets/
      presentation.css
  tide_belfast.csv
  weather_belfast.csv
  requirements.txt
  Procfile
  Dockerfile
  .gitignore
  README.md
```

## Deployment

### Render / Railway / Heroku (Procfile)
- `requirements.txt` includes `gunicorn`.
- `Procfile` already present:
```
web: gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:server
```

Deploy steps (Render-style):
1) Push to GitHub
2) Create a new Web Service from this repo
3) Set `Start command` to use the Procfile (Render auto-detects) or explicitly `gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:server`
4) Ensure `PORT` env var is provided by the platform (Render/Heroku/Railway do this automatically)

### Docker
The Dockerfile in this repo is configured for platforms that inject `$PORT` (e.g., DigitalOcean App Platform). To run locally:

```bash
docker build -t presentation-app .
# Pick a local port and pass it in
docker run -e PORT=8052 -p 8052:8052 presentation-app
```

Gunicorn will bind to `0.0.0.0:$PORT` and Dash will serve `app:server`.

## Configuration
- `CACHE_ENABLED`: enable LRU+TTL cache (`1` to enable, default `1`).
- `CACHE_MAX_SIZE`: maximum cache entries (default `32`).
- `CACHE_TTL_SECONDS`: time-to-live in seconds (default `900`).

Example:
```bash
export CACHE_ENABLED=1
export CACHE_MAX_SIZE=64
export CACHE_TTL_SECONDS=600
export PORT=8053
python app.py
```

## Notes
- Default center date set to 2024-01-10; window clamps to available data if needed.
- NOAA predictions fetched with small buffer for interpolation; falls back to nearby stations if needed.
- VS Code workspace settings included for `.venv` activation and Dash debugging (see .vscode/).

## TODO
- Add rainfall graph and optionally overlay rainfall intensity on the map.
- Add draggable sliders at the ends of the timeline to adjust the start/end datetimes for the visible window.
- Support multiple locations (wind, tide, rainfall) displayed simultaneously on the map, e.g., sites around Penobscot Bay.
