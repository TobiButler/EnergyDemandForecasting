import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import base64
from PIL import Image
import os
import pickle as pkl
import io

# Define global variables
IMAGE_WIDTH = 750
saved_directory = r"Saved"
dependent_variable = r"Energy Demand (MWH)"

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

### Define the four pages ### only one page will ever be active at a time

# Define landing page default layout
navigation_button_ids = dict(zip(["homepage", "page1", "page2", "page3"], ["lp", "p1", "p2", "3"]))
landing_page_layout = html.Div([
    html.H1('Energy Demand Forecasting Application'),
    html.Div(
        """This small web application is designed to provide an interactive way of understanding how data collected from the U.S. Energy Information Agency (EIA), 
        the U.S. National Oceanic Atmospheric Agency (NOAA), and the U.S. Bureau of Labor Statistics (BLS), was analyzed and used to develop and Energy Demand 
        Forecasting Pipeline. """
    ),
    html.Br(),
    html.Div("""This page provides a user guide for navigating the application and using the other pages. Feel free to leave this page and head to one of the 
             others using the buttons at the bottom at any time."""),
    html.Br(),
    html.Div( # include user guide here
        [
        html.H2("User Guide"),
        html.Br(),
        html.H4("Page 1: Exploratory Data Analysis"),
        html.Div("""Describe it"""),
        html.Br(),
        html.H4("Page 2: Fitting the Forecasting Model"),
        html.Div(),
        html.Br(),
        html.H4("Page 3: The Final Product"),
        html.Div(),
        ]
    ),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('Reset Home Page', id=navigation_button_ids['homepage'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 1', id=navigation_button_ids['page1'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 2', id=navigation_button_ids['page2'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 3', id=navigation_button_ids['page3'], n_clicks=0, style={'width': '25%'}),
    ])
])

# HERE want to add dropdown for each type of plot. Only offer the relevant variables for each type of plot 
# (ex: only show outliers plotted for variables that have outliers.)


# Specify the directory path for raw-time-series plots
rts_directory = r"{}/Plotly Figures/Raw Time Series".format(saved_directory)
variables = [x[:-4] for x in os.listdir(rts_directory)] # Get a list of associated variables
rts_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for outlier identification plots
outlier_directory = r"{}/Plotly Figures/Outlier Detection".format(saved_directory)
variables = [x[:-4] for x in os.listdir(outlier_directory)] # Get a list of associated variables
outlier_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for distribution plots
# distribution_directory = r"Static Visuals/Distributions"
# variables = [x[:-4] for x in os.listdir(distribution_directory)] # Get a list of associated variables
# distribution_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for scatterplots
scatterplot_directory = r"{}/Static Visuals/Scatterplots".format(saved_directory)
variables = [x[:-4] for x in os.listdir(scatterplot_directory)] # Get a list of associated variables
scatterplot_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for scatterplots
decomposition_directory = r"{}/Static Visuals/Decompositions".format(saved_directory)
variables = [x[:-4] for x in os.listdir(decomposition_directory)] # Get a list of associated variables
decomposition_dropdown_options = [{"label":variable, "value":variable} for variable in variables]


# Define page 1 default layout
page1_layout = html.Div([
    html.H1('Part 1: Exploring the Data'),
    html.Div([
        html.H3("Describing the Dataset"),
        html.Div("Data was collected from EIA, NOAA, BLS"),
        html.Div([
            dcc.Dropdown(
                id='rts-dropdown',
                options=rts_dropdown_options,
                value=dependent_variable  # Default value
            ),
            html.Div("Raw Time Series Plot"),
            html.Div([], id='raw-time-series'),
            html.Br(),
            dcc.Dropdown(
                id='outliers-dropdown',
                options=outlier_dropdown_options,
                value=dependent_variable  # Default value
            ),
            html.Div("Time Series Plot with Outliers Identified"),
            html.Div([], id='outliers'),
            html.Br(),
            dcc.Dropdown(
                id='scatterplots-dropdown',
                options=scatterplot_dropdown_options,
                value=dependent_variable  # Default value
            ),
            html.Img(id='distribution'),
            html.Br(),
            html.Img(id='scatterplot'),

            html.Br(),
            html.Div("Below are the time series decomposition plots for available time series variables."),
            dcc.Dropdown(
                id='decompositions-dropdown',
                options=decomposition_dropdown_options,
                value=dependent_variable  # Default value
            ),
            html.Img(id='decomposition'),
        ])
        
    ]),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('Home Page', id=navigation_button_ids['homepage'], n_clicks=0, style={'width': '25%'}),
        html.Button('Reset Page 1', id=navigation_button_ids['page1'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 2', id=navigation_button_ids['page2'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 3', id=navigation_button_ids['page3'], n_clicks=0, style={'width': '25%'}),
    ])
])

# Define page 2 default layout
page2_layout = html.Div([
    html.H1('Page 2'),
    html.Div([
        dcc.Graph(id='page2-graph1'),
        dcc.Graph(id='page2-graph2'),
        dcc.Graph(id='page2-graph3')
    ]),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('Home Page', id=navigation_button_ids['homepage'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 1', id=navigation_button_ids['page1'], n_clicks=0, style={'width': '25%'}),
        html.Button('Reset Page 2', id=navigation_button_ids['page2'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 3', id=navigation_button_ids['page3'], n_clicks=0, style={'width': '25%'}),
    ])
])


# Define Page 3 default layout
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
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('Home Page', id=navigation_button_ids['homepage'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 1', id=navigation_button_ids['page1'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 2', id=navigation_button_ids['page2'], n_clicks=0, style={'width': '25%'}),
        html.Button('Reset Page 3', id=navigation_button_ids['page3'], n_clicks=0, style={'width': '25%'}),
    ])
])

### End Default Page Layout Definitions ###

### Define Callbacks for Navigation ###

# Callback for all page navigation
@app.callback(Output('url', 'pathname'), [Input(navigation_button_ids["homepage"], 'n_clicks'), 
    Input(navigation_button_ids["page1"], 'n_clicks'), Input(navigation_button_ids["page2"], 'n_clicks'),
    Input(navigation_button_ids["page3"], 'n_clicks')
    ]
)
def page_navigation(*buttons):
    ctx = dash.callback_context
    button_id = ctx.triggered_id
    if button_id == navigation_button_ids["page1"]:
        return '/page-1'
    elif button_id == navigation_button_ids["page2"]:
        return '/page-2'
    elif button_id == navigation_button_ids["page3"]:
        return '/page-3'
    else:
        return ""

### Define Callbacks for Page 1 ###
@app.callback(Output('raw-time-series', 'children'), [Input('rts-dropdown', 'value')])
def update_rts(variable):
    # update raw time series plot
    path = r"{}/Plotly Figures/Raw Time Series/{}.pkl".format(saved_directory, variable)
    raw_time_series_figure = load_plotly_figure(path)
    return raw_time_series_figure

@app.callback(Output('outliers', 'children'), [Input('outliers-dropdown', 'value')])
def update_outliers_fig(variable):
    # update outliers plot
    path = r"{}/Plotly Figures/Outlier Detection/{}.pkl".format(saved_directory, variable)
    outliers_figure = load_plotly_figure(path)
    return outliers_figure

@app.callback(
    [Output('distribution', 'src'), Output('scatterplot', 'src')], [Input('scatterplots-dropdown', 'value')]
)
def update_distribution_scatterplot(variable):
    # update scatterplot
    path = r"{}/Static Visuals/Scatterplots/{}.png".format(saved_directory, variable)
    scatterplot_image = load_static_image(path=path)
    
    # update distribution plot
    path = r"{}/Static Visuals/Distributions/{}.png".format(saved_directory, variable)
    distribution_image = load_static_image(path=path)

    return distribution_image, scatterplot_image

@app.callback(Output('decomposition', 'src'), [Input('decompositions-dropdown', 'value')])
def update_decomposition_plot(variable):
    # update time-series decomposition
    path = r"{}/Static Visuals/Decompositions/{}.png".format(saved_directory, variable)
    decomposition_image = load_static_image(path=path)
    return decomposition_image

# @app.callback(
#     [Output('raw-time-series', 'children'), Output('outliers', 'children'), 
#         Output('distribution', 'src'), Output('scatterplot', 'src'), 
#         Output('decomposition', 'src')],
#     [Input('page1-dropdown', 'value')]
# )
# def update_page1_graphs(variable):
#     # update raw time series plot
#     path = r"Plotly Figures/Raw Time Series/{}.pkl".format(variable)
#     raw_time_series_figure = load_plotly_figure(path)

#     # update outliers plot
#     path = r"Plotly Figures/Outlier Detection/{}.pkl".format(variable)
#     outliers_figure = load_plotly_figure(path)
        
#     # update distribution plot
#     path = r"Static Visuals/Distributions/{}.png".format(variable)
#     distribution_image = load_static_image(path=path)
        
#     # update scatterplot
#     path = r"Static Visuals/Scatterplots/{}.png".format(variable)
#     scatterplot_image = load_static_image(path=path)
        
#     # update time-series decomposition
#     path = r"Static Visuals/Decompositions/{}.png".format(variable)
#     decomposition_image = load_static_image(path=path)

#     return raw_time_series_figure, outliers_figure, distribution_image, scatterplot_image, decomposition_image

### Define Callbacks for Page 2 ###

# callback (model selection) -> page 2 layout

# when VARMA-GARCH model is selected:

# callback (variable selection) -> time series plot of training data

# callback (hyperparamter selection) -> time series plot with evaluation results from this model

# callback (variable selection) -> time series plot showing evaluation impact of randomizing variable

# when BAYESIAN ES-LSTM model is selected:

# callback (variable selection) -> time series plot of training data

# callback (hyperparamter selection) -> time series plot with evaluation results from this model

# callback (variable selection) -> time series plot showing evaluation impact of randomizing variable

### Define Callbacks for Page 3 Visuals ###

# callback (slider) -> time series plot showing forecasts, slider values




# Define initial starting layout for the app
# app.layout = landing_page_layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
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
    

### Helper Functions ###
########################
    

def load_plotly_figure(path):
    with open(path, 'rb') as file:
        figure = pkl.load(file)

    # Convert the Plotly figure to a Dash graph object
    graph = dcc.Graph(
        figure=figure
    )

    return graph


def load_static_image(path):
    # Read the image file
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize the image while preserving aspect ratio
        img.thumbnail((IMAGE_WIDTH, IMAGE_WIDTH))
        # Convert the image to RGBA if it's not already in that mode
        img = img.convert('RGBA')
        # Create a white background image to place the resized image on
        bg = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_WIDTH), (255, 255, 255, 255))
        bg.paste(img, (int((IMAGE_WIDTH - img.width) / 2), int((IMAGE_WIDTH - img.height) / 2)), img)
        # Save the resized and centered image to a temporary file
        with io.BytesIO() as temp_image_buffer:
            bg.save(temp_image_buffer, format='PNG')
            temp_image_buffer.seek(0)
            # Encode the resized image to base64 format
            encoded_image = base64.b64encode(temp_image_buffer.read()).decode('utf-8')

    # Create an HTML img element with the encoded image
    # image_element = html.Img(src='data:image/png;base64,{}'.format(encoded_image))

    return 'data:image/png;base64,{}'.format(encoded_image)
    

# Run the application when script is run
if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8050)
    app.run_server(debug=True, host='localhost', port=8050)
