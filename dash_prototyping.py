# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layouts for each page
# page1_layout = html.Div([
#     html.H1("Page 1"),
#     html.P("This is Page 1 content."),
# ])

# page2_layout = html.Div([
#     html.H1("Page 2"),
#     html.P("This is Page 2 content."),
# ])

# page3_layout = html.Div([
#     html.H1("Page 3"),
#     html.P("This is Page 3 content."),
# ])

# # Define app layout with navigation
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False, pathname='127.0.0.1'),
#     html.Div(id='page-content')
# ])

# # Callback to update page content based on URL path
# @app.callback(
#     Output('page-content', 'children'),
#     [Input('url', 'pathname')]
# )
# def display_page(pathname):
#     if pathname == '/page-1':
#         return page1_layout
#     elif pathname == '/page-2':
#         return page2_layout
#     elif pathname == '/page-3':
#         return page3_layout
#     else:
#         return '404: Page not found'

# if __name__ == '__main__':
#     app.run_server(debug=True)


# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output

# app = dash.Dash(__name__)

# # Define the layout for the landing page
# landing_page_layout = html.Div([
#     html.H1('Welcome to the Dash Multi-Page App'),
#     html.Hr(),  # Horizontal line for separation
#     html.H3('Choose a page to navigate:'),
#     html.Ul([
#         html.Li(html.A('Page 1', href='/page-1')),  # Link to Page 1
#         html.Li(html.A('Page 2', href='/page-2')),  # Link to Page 2
#         html.Li(html.A('Page 3', href='/page-3')),  # Link to Page 3
#     ])
# ])

# # Define the layout for Page 1
# page1_layout = html.Div([
#     html.H1('Page 1'),
#     html.P('This is Page 1 content.'),
#     html.A('Go to Page 2', href='/page-2'),  # Hyperlink to Page 2
#     html.Br(),  # Add line break for spacing
#     html.A('Go to Page 3', href='/page-3'),  # Hyperlink to Page 3
# ])

# # Define the layout for Page 2
# page2_layout = html.Div([
#     html.H1('Page 2'),
#     html.P('This is Page 2 content.'),
#     html.A('Go to Page 1', href='/page-1'),  # Hyperlink to Page 1
#     html.Br(),  # Add line break for spacing
#     html.A('Go to Page 3', href='/page-3'),  # Hyperlink to Page 3
# ])

# # Define the layout for Page 3
# page3_layout = html.Div([
#     html.H1('Page 3'),
#     html.P('This is Page 3 content.'),
#     html.A('Go to Page 1', href='/page-1'),  # Hyperlink to Page 1
#     html.Br(),  # Add line break for spacing
#     html.A('Go to Page 2', href='/page-2'),  # Hyperlink to Page 2
# ])

# # Define the overall layout of the app with a Location component
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])

# # Callback to update the page content based on the URL path
# @app.callback(
#     Output('page-content', 'children'),
#     [Input('url', 'pathname')]
# )
# def display_page(pathname):
#     if pathname == '/':
#         return landing_page_layout
#     elif pathname == '/page-1':
#         return page1_layout
#     elif pathname == '/page-2':
#         return page2_layout
#     elif pathname == '/page-3':
#         return page3_layout
#     else:
#         return html.H1('Page Not Found')

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0')

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd

# Create Dash app
app = dash.Dash(__name__)

# Define landing page layout
landing_page_layout = html.Div([
    html.H1('Welcome to My Dash App'),
    html.Div([
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Page 3', href='/page-3'),
    ])
])

# Define layout and figures for each page
page1_layout = html.Div([
    html.H1('Page 1'),
    html.Div([
        dcc.Graph(id='page1-graph1'),
        dcc.Graph(id='page1-graph2'),
        dcc.Graph(id='page1-graph3')
    ]),
    html.Div([
        dcc.Link('Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Page 2', href='/page-2'),
        html.Br(),
        dcc.Link('Page 3', href='/page-3'),
    ])
])

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

# Define app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

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



