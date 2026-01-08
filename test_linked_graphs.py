"""
Test program to figure out how to properly link/synchronize Dash/Plotly graphs
so that zoom and pan actions on one graph affect all linked graphs.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=100, freq='H')
data1 = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
data2 = np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
data3 = np.random.normal(10, 2, 100)

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Linked Graph Test"),
    html.P("Test different methods to synchronize zoom/pan across multiple graphs"),
    
    html.H3("Method 1: Using relayoutData callback"),
    dcc.Graph(id='graph1-m1'),
    dcc.Graph(id='graph2-m1'),
    dcc.Graph(id='graph3-m1'),
    
    html.Hr(),
    
    html.H3("Method 2: Using subplot with shared x-axis"),
    dcc.Graph(id='graph-subplot'),
    
    html.Div(id='debug-output', style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0'})
])

# Method 1: Create initial figures
@app.callback(
    [Output('graph1-m1', 'figure'),
     Output('graph2-m1', 'figure'),
     Output('graph3-m1', 'figure')],
    Input('graph1-m1', 'id')
)
def create_initial_figures(_):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dates, y=data1, mode='lines', name='Data 1'))
    fig1.update_layout(
        title='Graph 1 (Sine wave)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        height=300
    )
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates, y=data2, mode='lines', name='Data 2', line=dict(color='orange')))
    fig2.update_layout(
        title='Graph 2 (Cosine wave)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        height=300
    )
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=dates, y=data3, mode='lines', name='Data 3', line=dict(color='green')))
    fig3.update_layout(
        title='Graph 3 (Random)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
        height=300
    )
    
    return fig1, fig2, fig3

# Method 1: Sync zoom/pan using relayoutData
@app.callback(
    [Output('graph2-m1', 'figure', allow_duplicate=True),
     Output('graph3-m1', 'figure', allow_duplicate=True)],
    [Input('graph1-m1', 'relayoutData')],
    [State('graph2-m1', 'figure'),
     State('graph3-m1', 'figure')],
    prevent_initial_call=True
)
def sync_from_graph1(relayout_data, fig2, fig3):
    if relayout_data and any(k.startswith('xaxis') for k in relayout_data.keys()):
        # Extract x-axis range
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            x_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
            
            # Update both figures
            fig2['layout']['xaxis']['range'] = x_range
            fig3['layout']['xaxis']['range'] = x_range
            
            return fig2, fig3
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('graph1-m1', 'figure', allow_duplicate=True),
     Output('graph3-m1', 'figure', allow_duplicate=True)],
    [Input('graph2-m1', 'relayoutData')],
    [State('graph1-m1', 'figure'),
     State('graph3-m1', 'figure')],
    prevent_initial_call=True
)
def sync_from_graph2(relayout_data, fig1, fig3):
    if relayout_data and any(k.startswith('xaxis') for k in relayout_data.keys()):
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            x_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
            
            fig1['layout']['xaxis']['range'] = x_range
            fig3['layout']['xaxis']['range'] = x_range
            
            return fig1, fig3
    
    return dash.no_update, dash.no_update

@app.callback(
    [Output('graph1-m1', 'figure', allow_duplicate=True),
     Output('graph2-m1', 'figure', allow_duplicate=True)],
    [Input('graph3-m1', 'relayoutData')],
    [State('graph1-m1', 'figure'),
     State('graph2-m1', 'figure')],
    prevent_initial_call=True
)
def sync_from_graph3(relayout_data, fig1, fig2):
    if relayout_data and any(k.startswith('xaxis') for k in relayout_data.keys()):
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            x_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
            
            fig1['layout']['xaxis']['range'] = x_range
            fig2['layout']['xaxis']['range'] = x_range
            
            return fig1, fig2
    
    return dash.no_update, dash.no_update

# Method 2: Using subplots
@app.callback(
    Output('graph-subplot', 'figure'),
    Input('graph-subplot', 'id')
)
def create_subplot_figure(_):
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Sine Wave', 'Cosine Wave', 'Random Data')
    )
    
    fig.add_trace(go.Scatter(x=dates, y=data1, mode='lines', name='Data 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=data2, mode='lines', name='Data 2', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=data3, mode='lines', name='Data 3', line=dict(color='green')), row=3, col=1)
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Subplot with Shared X-Axis (Automatically Linked)"
    )
    
    return fig

# Debug output
@app.callback(
    Output('debug-output', 'children'),
    [Input('graph1-m1', 'relayoutData'),
     Input('graph2-m1', 'relayoutData'),
     Input('graph3-m1', 'relayoutData')]
)
def show_debug(r1, r2, r3):
    return html.Pre(f"Graph 1 relayout: {r1}\nGraph 2 relayout: {r2}\nGraph 3 relayout: {r3}")

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
