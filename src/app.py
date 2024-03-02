import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import os
import pickle as pkl

# import custom functions
import dash_app_functions as daf

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


# load evaluation plot
with open(r"{}/Plotly Figures/Forecasting/best_model_cross_validation.pkl".format(saved_directory), 'rb') as file:
    figure = pkl.load(file)
evaluation_plot = dcc.Graph(figure=figure)

# load model performance report details

# Define page 2 default layout
page2_layout = html.Div([
    html.H1('Page 2: Cross Validation and Final Evaluation'),
    html.H3("Hyperparameter Tuning (under development)"),
    html.Div(["""The forecasting model for this pipeline was tuned using grid-search hyperparameter tuning and 5-Folds 
              rolling cross validation. Details about this process will be provided in the next update."""]),
    html.Div([], id="cross-validation-plot"),
    html.Br(),
    html.H3("Final Model Evaluation"),
    html.Div([r"""Once an optimal set of hyperparameters were obtained for the underlying forecasting model, its predictive 
              performance was evaluated on a holdout dataset containing the most recent 10% of Energy Demand Observations. 
              The time series plot below presents the forecasting model's predictions over this holdout time period along with 
              the actual observed values for comparison."""]),
    html.Div([evaluation_plot], id="final-evaluation-plot"),
    html.Br(),
    html.Div([r"""The following report describes the performance of the model compared to a baseline moving average using multiple metrics. 
              In future updates, forecasts from the EIA will be included for additional comparison."""]),
    html.Div([], id="final-evaluation-results"),
    html.Br(),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('Home Page', id=navigation_button_ids['homepage'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 1', id=navigation_button_ids['page1'], n_clicks=0, style={'width': '25%'}),
        html.Button('Reset Page 2', id=navigation_button_ids['page2'], n_clicks=0, style={'width': '25%'}),
        html.Button('Page 3', id=navigation_button_ids['page3'], n_clicks=0, style={'width': '25%'}),
    ])
])


# load future forecasts plot
with open(r"{}/Plotly Figures/Forecasting/future_forecasts.pkl".format(saved_directory), 'rb') as file:
    figure = pkl.load(file)
x_range = figure.layout.xaxis.range
# Generate hourly datetime range using pandas
two_year_hourly_range = pd.date_range(start=x_range[0], end=x_range[1], freq='H')
future_forecasts_plot = dcc.Graph(figure=figure)

# Define Page 3 default layout
page3_layout = html.Div([
    html.H1('Page 3: Forecasting into the Future'),
    html.Div("Future Forecasts"),
    # html.Div([future_forecasts_plot], id='future-forecasts-plot'),
    html.Div([], id='future-forecasts-plot'),
    html.Br(),
    dcc.Slider(
        id='future-slider',
        min=1,
        max=365*2,
        step=1,
        value=1,
        # marks = {np.datetime_as_string(i, unit='h'): i for i in hourly_range}
        marks={i: str(i) for i in range(0, 365*2*24, 100)}
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
    raw_time_series_figure = daf.load_plotly_figure(path)
    return raw_time_series_figure

@app.callback(Output('outliers', 'children'), [Input('outliers-dropdown', 'value')])
def update_outliers_fig(variable):
    # update outliers plot
    path = r"{}/Plotly Figures/Outlier Detection/{}.pkl".format(saved_directory, variable)
    outliers_figure = daf.load_plotly_figure(path)
    return outliers_figure

@app.callback(
    [Output('distribution', 'src'), Output('scatterplot', 'src')], [Input('scatterplots-dropdown', 'value')]
)
def update_distribution_scatterplot(variable):
    # update scatterplot
    path = r"{}/Static Visuals/Scatterplots/{}.png".format(saved_directory, variable)
    scatterplot_image = daf.load_static_image(path, IMAGE_WIDTH)
    
    # update distribution plot
    path = r"{}/Static Visuals/Distributions/{}.png".format(saved_directory, variable)
    distribution_image = daf.load_static_image(path, IMAGE_WIDTH)

    return distribution_image, scatterplot_image

@app.callback(Output('decomposition', 'src'), [Input('decompositions-dropdown', 'value')])
def update_decomposition_plot(variable):
    # update time-series decomposition
    path = r"{}/Static Visuals/Decompositions/{}.png".format(saved_directory, variable)
    decomposition_image = daf.load_static_image(path, IMAGE_WIDTH)
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
    Output('future-forecasts-plot', 'children'),
    [Input('future-slider', 'value')]
)
def update_time_series_plot(value):
    with open(r"Saved/Plotly Figures/Forecasting/future_forecasts.pkl", 'rb') as file:
        figure = pkl.load(file)
    x_min = two_year_hourly_range[0]
    x_max = two_year_hourly_range[value*24]
    figure.update_xaxes(range = [x_min, x_max])

    # Convert the Plotly figure to a Dash graph object
    graph = dcc.Graph(
        figure=figure
    )

    return graph

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
    

# Run the application when script is run
if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8050)
    # app.run_server(debug=True, host='localhost', port=8050)
    # app.run_server(debug=True, host='localhost')
    # app.run_server(debug=False, host='127.0.0.1', port="8000")
    app.run_server(debug=True)
