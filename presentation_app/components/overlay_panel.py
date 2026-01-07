"""
Compact overlay control panel for presentation mode (standalone)
"""
import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd
from typing import Optional


def create_overlay_panel(min_date: Optional[pd.Timestamp] = None,
                        max_date: Optional[pd.Timestamp] = None,
                        center_date: Optional[pd.Timestamp] = None) -> html.Div:
    if center_date is None:
        if min_date and max_date:
            center_date = min_date + (max_date - min_date) / 2
        else:
            center_date = min_date

    return html.Div([
        html.Div([
            html.H5("Storm Surge Visualization", className="mb-2"),
            html.Div(id='station-name-display', className="small text-muted mb-3")
        ]),
        html.Div([
            html.Div(id='current-time-display', className="time-display mb-2", children="--:--")
        ]),
        dbc.Row([
            dbc.Col([
                html.Div("Surge", className="stat-label"),
                html.Div(id='current-surge-value', className="stat-value", children="--")
            ], width=5),
            dbc.Col([
                html.Div("Wind", className="stat-label"),
                html.Div(id='current-wind-value', className="stat-value wind-value", children="--")
            ], width=7),
        ], className="mb-3"),
        html.Div([
            dbc.ButtonGroup([
                dbc.Button(html.I(className="bi bi-play-fill"), id='play-button', color="success", size="sm"),
                dbc.Button(html.I(className="bi bi-pause-fill"), id='pause-button', color="warning", size="sm"),
                dbc.Button(html.I(className="bi bi-skip-backward-fill"), id='reset-button', color="secondary", size="sm"),
            ], className="mb-2 w-100")
        ]),
        html.Div([
            dcc.Slider(id='time-slider', min=0, max=100, value=0, step=1, marks={}, tooltip={"placement": "bottom", "always_visible": False}, updatemode='drag', className="mb-2")
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Speed", className="small mb-1"),
                dcc.Slider(id='speed-slider', min=50, max=1000, value=500, marks={50: '20/s', 500: '2/s', 1000: '1/s'}, tooltip={"placement": "bottom"})
            ], width=7),
            dbc.Col([
                html.Label("Step", className="small mb-1"),
                dcc.Dropdown(id='step-size-selector', options=[{'label':'1x','value':1},{'label':'2x','value':2},{'label':'4x','value':4},{'label':'8x','value':8}], value=1, clearable=False, className="step-dropdown")
            ], width=5),
        ], className="mb-3"),
        # Wind history mode control removed (redundant) and compact water chart removed
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        dcc.Upload(id='upload-tide-data', children=html.Div(['Tide CSV']), style={'width':'100%','height':'40px','lineHeight':'40px','borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px','textAlign':'center','fontSize':'12px'}),
                    ], width=6),
                    dbc.Col([
                        dcc.Upload(id='upload-weather-data', children=html.Div(['Weather CSV']), style={'width':'100%','height':'40px','lineHeight':'40px','borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px','textAlign':'center','fontSize':'12px'}),
                    ], width=6),
                ], className="mb-2"),
                dbc.Button("Use Sample Data", id='load-sample-data', color="secondary", size="sm", className="w-100 mb-2"),
                html.Div([
                    html.Label("Center Date (±18 hours)", className="small mb-1"),
                    dcc.DatePickerSingle(id='center-date-picker', min_date_allowed=min_date, max_date_allowed=max_date, date=center_date, className="w-100")
                ])
            ], title="Data & Settings", className="small")
        ], start_collapsed=True, flush=True),
        dcc.Interval(id='animation-interval', interval=500, disabled=True),
        # Session-scoped stores so each browser tab keeps its own data
        dcc.Store(id='data-store', storage_type='session'),
        dcc.Store(id='animation-data-store', storage_type='session'),
        dcc.Store(id='session-id', storage_type='session'),
        dcc.Store(id='data-version', storage_type='session', data=0),
    ], id='overlay-panel', className='overlay-panel')
