# Presentation App (Standalone)

A self-contained Dash app for presenting storm surge and wind data for Belfast, Maine. This package is split from the parent workspace and includes all its own dependencies and deployment files.

## Quick Start

- Python: 3.12 recommended

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m presentation_app.app
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
web: gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT presentation_app.app:app.server
```

Deploy steps (Render-style):
1) Push to GitHub
2) Create a new Web Service from this repo
3) Set `Start command` to use the Procfile (Render auto-detects) or explicitly `gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT presentation_app.app:app.server`
4) Ensure `PORT` env var is provided by the platform (Render/Heroku/Railway do this automatically)

### Docker
```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8052
EXPOSE 8052
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:${PORT}","presentation_app.app:app.server"]
```

Build and run:
```bash
docker build -t presentation-app .
docker run -p 8052:8052 -e PORT=8052 presentation-app
```

## Notes
- Default center date set to 2024-01-10; window clamps to available data if needed.
- NOAA predictions fetched with small buffer for interpolation; falls back to nearby stations if needed.
- VS Code workspace settings included for `.venv` activation and Dash debugging (see .vscode/).

## TODO
- Add rainfall graph and optionally overlay rainfall intensity on the map.
- Add draggable sliders at the ends of the timeline to adjust the start/end datetimes for the visible window.
- Support multiple locations (wind, tide, rainfall) displayed simultaneously on the map, e.g., sites around Penobscot Bay.
