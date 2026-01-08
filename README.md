# Presentation App (Standalone)

A self-contained Dash app for presenting storm surge and wind data for Belfast, Maine. This package is split from the parent workspace and includes all its own dependencies and deployment files.

## Getting Started

### Prerequisites
- **Python 3.12+** (recommended)
- **Git** (for cloning the repository)
- **VS Code** (optional, but recommended)

### Option 1: Clone with Git

1. **Clone the repository**
   ```bash
   git clone https://github.com/DaveSprague/TideWeatherPresentationApp.git
   cd TideWeatherPresentationApp
   ```

2. **Create and activate virtual environment**
   ```bash
   # macOS/Linux
   python3.12 -m venv .venv
   source .venv/bin/activate
   
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   
   Navigate to http://localhost:8052

### Option 2: Clone with VS Code

1. **Open VS Code** and press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)

2. **Type "Git: Clone"** and press Enter

3. **Paste the repository URL**
   ```
   https://github.com/DaveSprague/TideWeatherPresentationApp.git
   ```

4. **Choose a local folder** where you want to save the project

5. **Open the cloned repository** when prompted

6. **Create virtual environment**
   - Open the integrated terminal (`Ctrl+`\` or View → Terminal)
   - Run:
     ```bash
     python3.12 -m venv .venv
     ```

7. **Select Python interpreter**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `.venv` (should show `./venv/bin/python`)

8. **Install dependencies**
   - VS Code should prompt you to install dependencies
   - Or manually run in terminal:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

9. **Run the app**
   - Press `F5` to run with debugger, or
   - Use the "Run Dash App" task from the task menu, or
   - Run in terminal:
     ```bash
     python app.py
     ```

10. **Open in browser**
    
    VS Code may open it automatically, or navigate to http://localhost:8052

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

## Future Enhancements

### Advanced Surge Analysis
- **Multivariate regression analysis** to separate influences of wind, atmospheric pressure, and river discharge on water levels
- **Integration with river discharge data** from USGS stream gauges (Penobscot River at Eddington #01036390)
- **CTD sensor data integration** from UMaine Civil Engineering Dept. (Passagassawakeag River and Penobscot River sensors)
  - Salinity measurements to detect freshwater influence
  - Temperature/density stratification effects
  - Direct water level measurements from pressure sensors
- **Correlation analysis tools** to quantify wind-surge relationships and identify river-driven vs. wind-driven surge events
- **Seasonal pattern analysis** to distinguish spring runoff effects from meteorological forcing
- **Atmospheric pressure effects** - add inverse barometer correction for storm surge predictions (1 mb drop ≈ 1 cm rise)

### Data Sources
- UMaine Physical Oceanography Group (Dr. Lauren Ross) - CTD data
- USGS Water Data for the Nation - stream discharge
- NERACOOS - regional ocean observing data
