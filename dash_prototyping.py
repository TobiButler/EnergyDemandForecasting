import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import base64

# Create Dash app
app = dash.Dash(__name__)

### Define the four pages ### only one page will ever be active at a time

# Define landing page layout
landing_page_layout = html.Div([
    html.H1('Energy Demand Forecasting Application'),
    html.Div([
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Page 3', href='/page-3'),
    ])
])


encoded_image = base64.b64encode(open("Static Visuals/ed_outliers.png", 'rb').read())
# Define page 1
page1_layout = html.Div([
    html.H1('Part 1: Exploring the Data'),
    html.Div([
        html.H3("Describing the Dataset"),
        html.Div("Data was collected from EIA, NOAA, BLS"),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), height=300),
            html.Img(src='/Static Visuals/Untitled.jpeg'),
            html.Img(src='/home/tobi/Desktop/Capstone/EnergyDemandForecasting/Static Visuals/ed_outliers.png'),
        ])
        
    ]),
    html.Div([
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Page 3', href='/page-3'),
    ])
])

# Define page 2
page2_layout = html.Div([
    html.H1('Page 2'),
    html.Div([
        dcc.Graph(id='page2-graph1'),
        dcc.Graph(id='page2-graph2'),
        dcc.Graph(id='page2-graph3')
    ]),
    html.Div([
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Page 3', href='/page-3'),
    ])
])

# Define Page 3
page3_layout = html.Div([
    html.H1('Page 3'),
    dcc.Graph(
        id='time-series-plot',
        figure=go.Figure(data=[go.Scatter(x=pd.date_range('2022-01-01', periods=100), y=np.random.randn(100), mode='lines')]),
    ),
    dcc.Slider(
        id='future-slider',
        min=0,
        max=365,
        step=1,
        value=0,
        marks={i: str(i) for i in range(0, 366, 10)}
    ),
    html.Div(id='slider-output-container'),
    html.Div([
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Page 3', href='/page-3'),
    ])
])

# Define initial starting layout for the app
app.layout = landing_page_layout
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])

# Define callbacks to update content dynamically
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('future-slider', 'value')]
)
def update_time_series_plot(value):
    future_date = pd.Timestamp('2022-01-01') + pd.Timedelta(days=value)
    x_values = pd.date_range(start='2022-01-01', periods=100) + pd.Timedelta(days=value)
    y_values = np.random.randn(100)
    return {'data': [go.Scatter(x=x_values, y=y_values, mode='lines')]}

# Define callback for slider output
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('future-slider', 'value')]
)
def update_slider_output(value):
    return f'Future days: {value}'

# Define callback to display page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/page-1':
        return page1_layout
    elif pathname == '/page-2':
        return page2_layout
    elif pathname == '/page-3':
        return page3_layout
    else:
        return landing_page_layout
    


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')



