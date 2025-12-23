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

Open http://localhost:8051

## Features
- Map via Plotly `Scattermap` (Maplibre), no token required
- Fading wind arrow history (no ribbon)
- Donut wind rose that accumulates up to current time
- Water level and wind speed charts with current value callouts

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
- `Procfile`:
```
web: gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT presentation_app.app:app.server
```

### Docker
```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8051
EXPOSE 8051
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:${PORT}","presentation_app.app:app.server"]
```

Build and run:
```bash
docker build -t presentation-app .
docker run -p 8051:8051 presentation-app
```

## Notes
- Default center date set to 2024-01-10; window clamps to available data if needed.
- NOAA predictions fetched with small buffer for interpolation; falls back to nearby stations if needed.
