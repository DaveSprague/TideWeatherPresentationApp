# Storm Surge Visualization App - Development Context

**Last Updated:** January 5, 2026

## Project Overview

A Dash/Plotly web application for visualizing Belfast Harbor storm surge and wind patterns. Supports multi-window/multi-tab sessions with independent data and animation states. Deployed on Render.

**Key Features:**
- Interactive animated timeline showing surge, water level, and wind patterns
- Session-scoped state (each browser tab maintains independent session)
- Play/Pause/Reset animation controls with adjustable speed
- Data upload (CSV tide/weather data)
- Sample data with NOAA API integration (optional)

---

## Architecture

### Core Components

**`presentation_app/app.py`**
- Main Dash application entry point
- Session management with per-tab UUID and session cache
- Callback definitions for map/chart updates, animation control
- Multi-callback architecture for smooth animation without UI blocking

**`presentation_app/components/overlay_panel.py`**
- Control panel overlay (date picker, play/pause/reset buttons, sliders)
- Session-scoped dcc.Store components (data-store, animation-data-store, session-id, data-version)
- Animation interval (initially disabled, enabled only on play)

**`presentation_app/utils.py`**
- `create_presentation_map()`: Builds Plotly Scattermap with wind rose overlay, surge marker, current wind arrow
- Wind geometry functions for visualizing wind direction/speed

**`presentation_app/data/`**
- `loader.py`: CSV data loading and validation
- `processor.py`: Data merging, surge calculations, wind processing
- `noaa_api.py`: Optional NOAA predictions (may return None if unavailable)

### Data Flow

1. User uploads or loads sample data → parsed and cached by session
2. Animation frames generated on-demand and stored in session cache
3. Animation interval triggers slider updates
4. Slider updates (via `update_time_position` callback) rebuild map/charts from cached data
5. Each tab has independent session → independent data/animation state

---

## Current Implementation Status

### ✅ Completed
- **Session isolation**: Per-tab dcc.Store with session_type='session' + session UUID
- **Performance optimization**: 
  - `animate=False` on all Graph components
  - `transition={'duration': 0}` for instant updates
  - `uirevision='map-constant'` on map
  - In-memory session cache (session_id, center_date, data_version) keyed tuples
- **Deployment**: Gunicorn + Flask server, Render with Python 3.12, health check endpoint
- **Import fixes**: All absolute imports converted to relative imports throughout package
- **Animation stability**: No flashing during timeline playback
- **UI/UX**: 
  - Animation starts disabled (paused until play button clicked)
  - Speed slider (50-1000ms) for animation interval control
  - Step size selector (1x, 2x, 4x, 8x) for frame jumping

### 🟡 Known Limitations

**Map Panning During Animation:**
- Users must pause animation to pan/zoom the map
- Auto-pause/resume callbacks were attempted but broke panning entirely (reverted)
- Trade-off: Simpler code, cleaner event handling
- Users can still pause → pan → resume without issue

**NOAA Predictions:**
- API may return None/unavailable data
- Code handles gracefully but surge calculation skipped if no predictions
- Not a blocker for visualization with user data

---

## Recent Changes (January 5, 2026)

**Commit:** "Remove map-interaction-timeout and cleanup map interaction callbacks"

### What Changed
- ❌ Removed `pause_on_map_interaction` callback (was hijacking map relayoutData)
- ❌ Removed `resume_after_map_interaction` callback
- ❌ Removed `map-interaction-timeout` Interval component from layout

### Why
- Auto-pause/resume was breaking map panning completely
- Plotly relayoutData events were being intercepted and interfering with normal map interaction
- Simpler approach: let users manually pause if they want to pan

### Result
- ✅ Map panning restored to normal smooth behavior
- ✅ Animation starts paused (disabled=True) as intended
- ✅ Cleaner callback architecture (only 2 callbacks touch animation-interval)

---

## Technical Details

### Session Management
```python
# Each tab gets unique session via dcc.Store(storage_type='session')
session_id = str(uuid.uuid4())  # Per-tab unique ID
session_cache = {}  # Global dict keyed by (session_id, center_date, data_version)
```

### Animation Control
```python
# Animation interval starts disabled, only enabled when play button clicked
dcc.Interval(id='animation-interval', interval=250, disabled=True)

# control_animation callback:
# - Play button → disabled=False
# - Pause button → disabled=True
# - Reset button → value=0, disabled=True
# - Each interval tick increments value by step_size
# - At max value → auto-pauses (disabled=True)
```

### Performance Optimization
```python
# Graph components configured for instant updates
dcc.Graph(id='surge-map', animate=False, style={...})
# With transition duration 0 and uirevision constant
# No Plotly animations, instant redraws each interval tick
```

### Deployment (Render)
```
# Procfile
web: gunicorn presentation_app.app:server

# runtime.txt
python-3.12.1

# Health check endpoint
@app.server.route('/health')
def health_check():
    return {'status': 'ok'}, 200
```

---

## Next Steps / Considerations

### High Priority
- [ ] **Test deployment**: Verify app runs on Render with these changes
- [ ] **Test multi-tab**: Confirm session isolation works (open app in 2 tabs, verify independent state)
- [ ] **Test animation**: Verify starts paused, plays on click, respects speed slider

### Medium Priority
- [ ] **Error handling**: NOAA API failures should gracefully fall back to water level only
- [ ] **Data validation**: More robust CSV parsing with better error messages
- [ ] **UI refinements**: Loading states, error notifications

### Low Priority (Nice-to-have)
- [ ] **Panning during animation**: Consider simpler solution (e.g., pause on first pan event, resume after timeout)
- [ ] **Map animations**: Custom marker animations for surge/wind indicators
- [ ] **Data caching**: Persistent cache of NOAA predictions to avoid repeated API calls
- [ ] **Mobile responsiveness**: Better layout for smaller screens

---

## Commands & Debugging

### Run Local Dev Server
```bash
cd presentation_app_standalone
python -m presentation_app.app
# Or: gunicorn presentation_app.app:server --reload
```

### Test Imports
```bash
python -c "from presentation_app import app; print('✓ Imports OK')"
```

### View Logs
```bash
# Docker logs (if deployed)
heroku logs --app app-name --tail
```

### Check Git Status
```bash
git log --oneline -5  # Recent commits
git status           # Current changes
```

---

## File Structure
```
presentation_app_standalone/
├── Dockerfile
├── Procfile
├── runtime.txt
├── requirements.txt
├── README.md
├── tide_belfast.csv
├── weather_belfast.csv
├── DEVELOPMENT_CONTEXT.md  ← You are here
└── presentation_app/
    ├── __init__.py
    ├── app.py                    [Main app, callbacks, session mgmt]
    ├── config.py                 [Constants, paths, config]
    ├── utils.py                  [create_presentation_map]
    ├── validators.py             [CSV validation]
    ├── assets/
    │   └── presentation.css      [Styling]
    ├── components/
    │   ├── __init__.py
    │   └── overlay_panel.py       [Control panel UI, stores]
    └── data/
        ├── __init__.py
        ├── loader.py             [CSV loading]
        ├── processor.py          [Data merging, surge calc]
        └── noaa_api.py           [NOAA integration]
```

---

## Questions to Ask Copilot

When reopening this project, you can ask:
- "What's the current status of [feature]?"
- "Should we implement [feature]?"
- "Can you explain how [component] works?"
- "Fix this error: [error message]"
- "I'm getting this behavior: [description], can you fix it?"

Copilot will have full context of the architecture, recent changes, and known tradeoffs.
