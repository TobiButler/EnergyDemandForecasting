"""
Author: Tobias Butler
Last Modified: 03/04/2024
Description: This Python module contains a Dash application
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import os
import pickle as pkl
import gc
from datetime import datetime

# import custom functions
import dash_app_functions as daf

# Define global variables
IMAGE_WIDTH = 750
saved_directory = r"Saved"
dependent_variable = r"Energy Demand (MWH)"
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
explanation_text_style = {"width":"1000px"}
dropdown_menu_style = {"width":"500px"}

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

### Define the four pages ### only one page will ever be active at a time

# Define landing page default layout
navigation_button_ids = dict(zip(["homepage", "page1", "page2", "page3"], ["lp", "p1", "p2", "3"]))
landing_page_layout = html.Div([
    html.H1('Residential Energy Demand Forecasting Application'),
    html.H5("Created by Tobias Butler. Last updated {}".format(now)),
    html.Br(),
    html.H3("About This Application"),
    html.Div(
        """This small web application is designed to provide an interactive way of understanding how data collected from the U.S. Energy Information Administration (EIA), 
        the U.S. National Oceanic Atmospheric Agency (NOAA), and the U.S. Bureau of Labor Statistics (BLS), was analyzed and used to develop a Residential Energy Demand 
        Forecasting Pipeline. All data collected is localized around New York City. More specific details will be provided in future updates.""", style=explanation_text_style
    ),
    html.Br(),
    html.Div("""This page provides a user guide for navigating the application and using the other pages. To navigate to the other pages, use the buttons at the bottom of 
        the page. They look like this:""", style=explanation_text_style),
    html.Br(),
    html.Img(id='example-navigation-buttons', className="image_with_border", style={"width":"1500px"}, src=r'assets/example_navigation_buttons.png', 
            alt="Image showing example of dropdown menu"),
    html.Br(),html.Br(),
    html.Div( # user guide
        [
        html.H2("User Guide"),
        html.Div("""This web application has been broken into three pages, each of which present information about a different component of the pipeline. The first 
            page presents some of the visualizations used during the exploratory data analysis to help guide data cleansing and processing. The second page presents 
            how the underlying forecast model was fit, tuned, and evaluated using holdout data. The third page applies the model and shows its estimated energy 
            demand forecasts into the future.""", style=explanation_text_style),
        html.Br(),
        html.Div("""This application contains both static images and interactive Plotly figures. When hovering over part of a visual, if the cursor changes its shape 
            to a crosshair or clickable icon, then the visual is interactive. Plotly figures enable users to zoom and to hide/unhide certain elements of the visual. 
            See an example of this interactivity below.""", style=explanation_text_style),
        html.H5("Example of Interactive Visuals"),
        html.Img(id='example_plotly_interactivity', className="image_with_border", style={"width":"1000px"}, src=r'assets/example_plotly_interactivity.gif', 
            alt="Image showing example of dropdown menu"),
        html.Br(),html.Br(),
        html.Div("""Feel free to click around and see what you are able to do to each visual. Plotly figures always have a "reset axes" button in the upper right corner 
            that will revert the visual back to its original form. That button looks like this:""", style=explanation_text_style),
        html.Img(id='example_plotly_reset', className="image_with_border", style={"width":"150px"}, src=r'assets/example_plotly_reset.png', 
            alt="Image showing example of dropdown menu"),
        html.Br(),html.Br(),html.Br(),
        html.H3("Page 1: Exploring the Data"),
        html.Div("""The first page of this application presents several exploratory visuals to help understand the behavior of each variable in the pipeline's dataset. 
            To avoid overcrowding the page, each type of visual is only presented for one variable at a time. In order to change which variable is being displayed for 
            each type of visual, there are dropdown menus that can be used to select from any of the relevant variables. Use these to explore the behavior of different 
            variables within the dataset. Below is an example of what the dropdown menus look like:""", style=explanation_text_style),
        html.H5("Example of Dropdown Menu:"),
        html.Img(id='example1', className="image_with_border", src=r'assets/example1.png', alt="Image showing example of Plotly figure interactivity"),
        html.Br(),html.Br(),
        html.Div("""The available exploratory visuals included on Page 1 include:"""),
        html.Ul([html.Li("Raw time series plots (with outliers highlighted)"), html.Li("Variable Distributions"), 
            html.Li("Scatterplots between Energy Demand and other variables"), html.Li("Time series decompositions (into trend and seasonal components)")]),
        html.Br(),html.Br(),
        html.H3("Page 2: Explaining the Model"),
        html.Div("""The second page of the application presents an overview of the model fitting and tuning methods, along with a final performance evaluation and report 
            (that is still under development). The report contains accuracy metrics for the forecasting model and compare's its predictive performance to alternative baseline 
            models. This information is intended to help users understand the usefulness and/or limitations of the forecasting pipeline before 
            proceeding to the final page to see it in action.""", style=explanation_text_style),
        html.Br(),
        html.Div("""On this page, users have the ability to download (as an html file) some of the visuals and tabular performance metrics displayed. This way, one can easily 
        share the predictive capabilities of the model with others without needing to delve into the entire application. Look for a clickable link similar to that shown below:""",
            style=explanation_text_style),
        html.H5("Example of Download Feature:"),
        html.Img(id='example2', className="image_with_border", src=r'assets/example2.png', alt="example of download button on page 2"),
        html.Br(),html.Br(),html.Br(),
        html.H3("Page 3: The Final Product"),
        html.Div("""The third page of the application presents forecasts from the model up to two years into the future. Users can adjust the forecasting horizon of the 
            model by moving the slider underneath the forecasting visualization (shown below).""", style=explanation_text_style),
        html.Img(id='example3', className="image_with_border", style={"width":"1500px"}, src=r'assets/example3.png', alt="example of slider on page 3 (The Final Product)"),
        html.Br(),html.Br(),html.Br(),
        html.Div("""Users can also download these future forecasts as a csv file by clicking on the "Download Future Forecasts (.csv)" button.""", style=explanation_text_style),
        html.Img(id='example_download_ff', className="image_with_border", style={"width":"200px"}, src=r'assets/example_download_ff.png', 
            alt="example of download button on page 3 (The Final Product)")
        ]
    ),
    html.Br(),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('Reset This Page', id=navigation_button_ids['homepage'], n_clicks=0, className="navigation_button"),
        html.Button('Exploring the Data', id=navigation_button_ids['page1'], n_clicks=0, className="navigation_button"),
        html.Button('Evaluation Results', id=navigation_button_ids['page2'], n_clicks=0, className="navigation_button"),
        html.Button('The Final Product', id=navigation_button_ids['page3'], n_clicks=0, className="navigation_button"),
    ]),
    html.Br()
])


# Specify the directory path for raw-time-series plots
rts_directory = r"{}/Plotly Figures/Raw Time Series".format(saved_directory)
variables = [x[:-4] for x in os.listdir(rts_directory)] # Get a list of associated variables
rts_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for outlier identification plots
outlier_directory = r"{}/Plotly Figures/Outlier Detection".format(saved_directory)
variables = [x[:-4] for x in os.listdir(outlier_directory)] # Get a list of associated variables
outlier_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for scatterplots
scatterplot_directory = r"{}/Static Visuals/Scatterplots".format(saved_directory)
variables = [x[:-4] for x in os.listdir(scatterplot_directory)] # Get a list of associated variables
scatterplot_dropdown_options = [{"label":variable, "value":variable} for variable in variables]

# specify the directory path for scatterplots
decomposition_directory = r"{}/Static Visuals/Decompositions".format(saved_directory)
variables = [x[:-4] for x in os.listdir(decomposition_directory) if "_bar" not in str(x)] # Get a list of associated variables
decomposition_dropdown_options = [{"label":variable, "value":variable} for variable in variables]


# Define page 1 default layout
page1_layout = html.Div([
    html.H1('Part 1: Exploring the Data'),
    html.Div([
        html.H3("Describing the Dataset"),
        html.Div("""Historical hourly residential energy demand data in New York City, recorded by the New York 
            Independent System Operator (NYISO), was obtained from the EIA and combined with several weather and 
            economic related variables. Hourly weather data was recorded by the New York Central Park weather 
            station and was gathered from the NOAA. Monthly economic inidcators for New York City were obtained 
            from the Bureau of Labor Statistics.""", style=explanation_text_style),
        html.Br(), html.Br(),
        html.H3("Raw Time Series Plots"),
        html.Div("""The following figure shows a raw plot of the selected time series variable. The start date 
            is the earliest point at which hourly residential energy demand was available through the NYISO.
            Other variables can be selected from the dropdown menu to see their raw time series distribution.""", 
            style=explanation_text_style),
            html.Br(),
        html.Div([
            dcc.Dropdown(
                id='rts-dropdown',
                options=rts_dropdown_options,
                value=dependent_variable,
                style=dropdown_menu_style
            ),
            html.Div([], id='raw-time-series'),
            html.Br(),html.Br(),html.Br(),
            html.H3("Outlier Identification"),
            html.Div("""For each of the time series variables, outliers were detected using moving average distribution estimation, similar to that described by Blázquez-García et al. (2022).
                A conservative probability threshold of 0.001 was used alongside an N of 1000 in order to detect observations with extremely low probabilities of 
                being generated from the same distribution as the nearest 1000 observations. Once identified, outliers were temporarily replaced with missing values 
                until filled during imputation. In the following figure, these points have been highlighted in red.""", style=explanation_text_style),
                html.Br(),
            dcc.Dropdown(
                id='outliers-dropdown',
                options=outlier_dropdown_options,
                value=dependent_variable,
                style=dropdown_menu_style
            ),
            html.Div([], id='outliers'),
            html.Br(),html.Br(),html.Br(),html.Br(),
            html.H3("Distributions and Bivariate Relationships"),
            html.Div(r"""For all variables in the dataset, distributions were produced to help recognize any ill-conditioned distributions that required transformations, 
                and scatterplots between them and the dependent variable, Energy Demand (MWH), were produced to help understand the types of relationships 
                (linear or nonlinear) present in the dataset. These two types of visuals are shown below for the variable selected below. By default, when 
                "Energy Demand (MWH)" is selected, a heatmap of correlation coefficients between all variables is shown.""", style=explanation_text_style),
            html.Br(),
            dcc.Dropdown(
                id='scatterplots-dropdown',
                options=scatterplot_dropdown_options,
                value=dependent_variable,
                style=dropdown_menu_style
            ),
            html.Div(style={'display': 'flex', "width":"90%", 'align-items': 'flex-start'}, children=[
                html.Img(id='distribution', style={'width': '50%', 'margin-right': '100px', 'object-fit': 'contain'}),
                html.Img(id='scatterplot', style={'width': '50%', 'object-fit': 'contain'})
            ]),
            html.Br(),html.Br(),html.Br(),html.Br(),
            html.H3("Time Series Decomposition Plots"),
            html.Div("""Lastly, it is important to understand how each time series variable, especially the dependent variable, is impacted by trend and seasonality components. 
                Below, time series decomposition plots estimating trend, yearly seasonality, weekly seasonality, and daily seasonality are provided for each time series variable 
                in the dataset.""", style=explanation_text_style),
            html.Br(),
            dcc.Dropdown(
                id='decompositions-dropdown',
                options=decomposition_dropdown_options,
                value=dependent_variable,
                style=dropdown_menu_style
            ),
            # html.Img(id='decomposition', style={'width':'750px'}, alt="Time series decomposition of variable selected above"),
            html.Div(style={'display': 'flex', "width":"95%", 'align-items': 'flex-start'}, children=[
                html.Img(id='decomposition', style={'width': '25%', 'margin-right': '200px', 'object-fit': 'contain'}),
                html.Img(id='decomposition-bar', style={'width': '40%', 'object-fit': 'contain'})
            ]),
        ]),
        html.Br(),html.Br(),
    ]),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('User Guide', id=navigation_button_ids['homepage'], n_clicks=0, className="navigation_button"),
        html.Button('Reset This Page', id=navigation_button_ids['page1'], n_clicks=0, className="navigation_button"),
        html.Button('Evaluation Results', id=navigation_button_ids['page2'], n_clicks=0, className="navigation_button"),
        html.Button('The Final Product', id=navigation_button_ids['page3'], n_clicks=0, className="navigation_button"),
    ])
])



# Define page 2 default layout
page2_layout = html.Div([
    html.H1('Part 2: Forecasting Model Evaluation'),
    # html.H3("Hyperparameter Tuning (under development)"),
    # html.Div(["""The forecasting model for this pipeline was tuned using grid-search hyperparameter tuning and 5-Folds 
    #           rolling cross validation. Details about this process will be provided in a future update."""], style=explanation_text_style),
    html.H3("Forecasting Model Architecture"),
    html.Div([r"""The forecasting ensemble developed for this project is a combination of Prophet, vector autoregressive (VAR) and long short-term memory (LSTM) architectures. """], style=explanation_text_style),
    html.Br(),
    html.H4("Prophet"),
    html.Div([r"""Prophet is a forecasting tool developed by the data science team at Facebook. It is designed to handle time-series data with multiple strong 
        seasonal patterns through the use of fourier series. The model looks specifically at daily, weekly, and yearly patterns, and fits a sinusoidal fourier series 
        to each detected seasonal component. Then, each series can be summed to produce a new series that represents the whole time series. As a result, Prophet models 
        are particularly useful for analyzing the various seasonal components of a time series and for forecasting in settings where datasets may have irregularities, 
        missing data, or outliers (Taylor & Letham, 2017)."""], style=explanation_text_style),
    html.Br(),html.Br(),
    html.H4("VAR"),
    html.Div([r"""The VAR statistical forecasting model is a generalization of the autoregressive moving average (ARMA) model, designed for analyzing multivariate 
        time series data. In a VAR model, each variable is modeled as a linear function of its past values and the past values of other variables in the system, but 
        there are no moving average terms. The order of a VAR model, often denoted as the hyperparameter p, indicates the number of lagged values included in the model 
        for each variable. For example, a VAR(2) model uses the two most recent observations for each variable to predict the subsequent vector of observations (Guefano et al., 2021). 
        In this project, a VAR model has been defined to produce point forecasts from the residual components of each time series variable (as estimated by the Prophet models) 
        and another VAR model has been defined to produce variance or error forecasts (prediction intervals). """], style=explanation_text_style),
    html.Br(),html.Br(),
    html.H4("LSTM"),
    html.Div([r"""The LSTM is a RNN variant designed with an additional type of cell, called a forget gate, that helps to mitigate the effects of the vanishing gradient problem. 
        It specializes in handling sequential data and is widely used for natural language processing and time series forecasting. LSTM models have been shown to be capable of 
        outperforming statistical autoregressive models on both short-term and long-term forecasts, but generally require larger amounts of data and specially tuned hyperparameters 
        in order to do so (Kaur et al., 2022). In addition to the regular hidden state and memory state cells that enable RNNs to learn temporal patterns, adding dropout to the 
        network can be used to mimic the behavior of a Bayesian network, in which predictions are probabilistic rather than deterministic (Gal & Ghahramani, 2015). In this project, 
        a LSTM model will be trained and tuned using the clean dataset and an Adam optimization algorithm. The LSTM will be defined to predict day-ahead predictions, and the learning 
        rate, hidden state size, input size, and dropout hyperparameters will all be tuned using five folds rolling window cross validation. Probabilistic forecasts will be obtained by 
        repeatedly applying the LSTM 100 times to obtain an estimated prediction interval over its point forecasts."""], style=explanation_text_style),
    html.Br(),html.Br(),html.Br(),
    html.H3("Evaluation"),
    html.Div([r"""An optimal set of hyperparameters were obtained for each forecasting architecture described above using five-fold rolling cross validation. Below, the long-term and 
            short-term forecasting capabilities of the resulting ensemble is evaluated on a holdout dataset containing the most recent 10% of Energy Demand Observations. 
            Performance was measured using multple accuracy metrics and compared to baseline models using a Wilcoxon Signed Rank test."""], style=explanation_text_style),
    html.Br(),html.Br(),
    html.H4("Long-term Forecasting Evaluation"),
    html.Div([r"""The time series plot below presents the forecasting ensemble's one-shot (long-term) predictions over the holdout time period along with 
        the actual observed values for comparison."""], style=explanation_text_style),
    html.Div([], id="long-term-cv-plot"),
    html.Br(),
    html.H4("Long-term Evaluation Results"),
    html.Div([r"""The following report describes the performance of the ensemble's long-term forecasts compared to a baseline yearly moving average using multiple metrics."""], style=explanation_text_style),
    html.Br(),
    html.Div(style={'width': '100%'}, children=[], id="long-term-cv-results"),
    html.Br(), html.Br(),
    html.H4("Day-ahead Forecasting Evaluation"),
    html.Div([r"""The time series plot below presents the forecasting ensemble's daily updated (short-term) predictions over the holdout time period along with 
        the actual observed values for comparison."""], style=explanation_text_style),
    html.Div([], id="short-term-cv-plot"),
    html.Br(),
    html.H4("Day-ahead Evaluation Results"),
    html.Div([r"""The following report describes the performance of the ensemble's day-ahead forecasts compared to a baseline daily moving average and comppared to EIA forecasts using 
        multiple metrics. The EIA forecasts are day-ahead forecasts provided by the EIA through their open-access database."""], style=explanation_text_style),
    html.Br(),
    html.Div(style={'width': '100%'}, children=[], id="short-term-cv-results"),
    html.Br(),html.Br(),html.Br(),
    html.H3("Sensitivity Analysis"),
    html.Div([r"""A permutation analysis was conducted to estimate the relative contributions of each variable to the effectiveness of the forecasting ensemble. In this analysis, 
        variables within the holdout evaluation dataset were permuted one at a time and the resulting decreases in the MSE loss function were normalized to sum to 1. The results of 
        this permutation feature importance analysis is shown below. Please note that this is a model-dependent approach to estimating feature importances and that the impacts from 
        strongly related predictor variables are not being taken into account."""], style=explanation_text_style),
    html.Br(),
    html.Img(id='sensitivity-analysis', style={'width':'750px'}, alt="Sensitivity Analysis for Forecasting Model"),
    html.Br(),html.Br(),
    html.Div([
    html.A('Download Model Summary Performance Report (.html)', href='assets/model_performance_report.html', download='Forecasting Model Performance Report.html')
    ]),
    html.Br(),html.Br(),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('User Guide', id=navigation_button_ids['homepage'], n_clicks=0, className="navigation_button"),
        html.Button('Exploring the Data', id=navigation_button_ids['page1'], n_clicks=0, className="navigation_button"),
        html.Button('Reset This Page', id=navigation_button_ids['page2'], n_clicks=0, className="navigation_button"),
        html.Button('The Final Product', id=navigation_button_ids['page3'], n_clicks=0, className="navigation_button"),
    ])
])


# Define Page 3 default layout
page3_layout = html.Div([
    html.H1('Part 3: Forecasting into the Future'),
    html.Div("Future Forecasts"),
    html.Div([], id='future-forecasts-plot'),
    html.Br(),
    dcc.Slider(
        id='future-slider',
        min=1,
        max=365*2,
        step=1,
        value=1,
        marks={i: str(i) for i in range(0, 365*2*24, 100)}
    ),
    html.Div(id='slider-output-container'),
    html.Br(),
    html.A('Download Future Forecasts (.csv)', href='assets/future_forecasts.csv', download='NYHourlyResidentialEnergyDemandForecasts.csv'),
    html.Br(),html.Br(),html.Br(),
    html.Div(style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}, children=[
        html.Button('User Guide', id=navigation_button_ids['homepage'], n_clicks=0, className="navigation_button"),
        html.Button('Exploring the Data', id=navigation_button_ids['page1'], n_clicks=0, className="navigation_button"),
        html.Button('Evaluation Results', id=navigation_button_ids['page2'], n_clicks=0, className="navigation_button"),
        html.Button('Reset This Page', id=navigation_button_ids['page3'], n_clicks=0, className="navigation_button"),
    ])
])

### End Default Page Layout Definitions ###


### Define Callbacks for Navigation ###

# update url based on button clicked
@app.callback(Output('url', 'pathname'), [Input(navigation_button_ids["homepage"], 'n_clicks'), 
    Input(navigation_button_ids["page1"], 'n_clicks'), Input(navigation_button_ids["page2"], 'n_clicks'),
    Input(navigation_button_ids["page3"], 'n_clicks')
    ]
)
def page_navigation(*buttons):
    gc.collect()
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

# update page layout/content based on url
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


### Define Callbacks for Page 1 ###
# update raw time series plot
@app.callback(Output('raw-time-series', 'children'), [Input('rts-dropdown', 'value')])
def update_rts(variable):
    path = r"{}/Plotly Figures/Raw Time Series/{}.pkl".format(saved_directory, variable)
    raw_time_series_figure = daf.load_plotly_figure(path)
    return raw_time_series_figure

# update outliers plot
@app.callback(Output('outliers', 'children'), [Input('outliers-dropdown', 'value')])
def update_outliers_fig(variable):
    path = r"{}/Plotly Figures/Outlier Detection/{}.pkl".format(saved_directory, variable)
    outliers_figure = daf.load_plotly_figure(path)
    return outliers_figure

# update distribution plot and scatterplot
@app.callback(
    [Output('distribution', 'src'), Output('scatterplot', 'src')], [Input('scatterplots-dropdown', 'value')]
)
def update_distribution_scatterplot(variable):
    path = r"{}/Static Visuals/Scatterplots/{}.png".format(saved_directory, variable)
    scatterplot_image = daf.load_static_image(path, IMAGE_WIDTH)
    path = r"{}/Static Visuals/Distributions/{}.png".format(saved_directory, variable)
    distribution_image = daf.load_static_image(path, IMAGE_WIDTH)
    return distribution_image, scatterplot_image

# update time-series decomposition
@app.callback([Output('decomposition', 'src'), Output('decomposition-bar', 'src')], [Input('decompositions-dropdown', 'value')])
def update_decomposition_plot(variable):
    path = r"{}/Static Visuals/Decompositions/{}.png".format(saved_directory, variable)
    decomposition_image = daf.load_static_image(path, IMAGE_WIDTH)
    path = r"{}/Static Visuals/Decompositions/{}_bar.png".format(saved_directory, variable)
    decomposition_bar_image = daf.load_static_image(path, IMAGE_WIDTH)
    return decomposition_image, decomposition_bar_image


### Define Callbacks for Page 2 ### can combine most of these into one

# update final model evaluation (on holdout data) plot
@app.callback(
    Output('long-term-cv-plot', 'children'),
    [Input('url', 'pathname')]
)
def update_long_term_cv_plot(pathname):
    if pathname == '/page-2':
        with open(r"assets/long_term_evaluation_plot.pkl", 'rb') as file:
            figure = pkl.load(file)
        evaluation_plot = dcc.Graph(figure=figure)
        return evaluation_plot
    
@app.callback(
    Output('short-term-cv-plot', 'children'),
    [Input('url', 'pathname')]
)
def update_short_term_cv_plot(pathname):
    if pathname == '/page-2':
        with open(r"assets/short_term_evaluation_plot.pkl", 'rb') as file:
            figure = pkl.load(file)
        evaluation_plot = dcc.Graph(figure=figure)
        return evaluation_plot
    
@app.callback(
    Output('sensitivity-analysis', 'src'),
    [Input('url', 'pathname')]
)
def update_sensitivity_analysis(pathname):
    if pathname == '/page-2':
        path = r"assets/sensitivity_analysis.png"
        sensitivity_analysis = daf.load_static_image(path, IMAGE_WIDTH)
        return sensitivity_analysis
    
# update table when user navigates to page 2
@app.callback(Output('long-term-cv-results', 'children'), [Input('url', 'pathname')])
def update_long_term_table(pathname):
    if pathname == '/page-2':
        # Read new CSV file or generate new DataFrame
        eval_df = pd.read_csv('assets/long_term_performance_report.csv')
        columns = [{'name': col, 'id': col} for col in eval_df.columns]
        table = dash_table.DataTable(id="evaluation-table", data=eval_df.to_dict('records'), columns=columns,
            style_table={"width": "50%"}, style_header={"fontWeight": "bold", "textAlign": "center"}, style_cell={
            "whiteSpace": "normal", "textAlign": "left"})
        return [html.H5("Model Evaluation Results"), table]
    
# update table when user navigates to page 2
@app.callback(Output('short-term-cv-results', 'children'), [Input('url', 'pathname')])
def update_short_term_table(pathname):
    if pathname == '/page-2':
        # Read new CSV file or generate new DataFrame
        eval_df = pd.read_csv('assets/short_term_performance_report.csv')
        columns = [{'name': col, 'id': col} for col in eval_df.columns]
        table = dash_table.DataTable(id="evaluation-table", data=eval_df.to_dict('records'), columns=columns,
            style_table={"width": "50%"}, style_header={"fontWeight": "bold", "textAlign": "center"}, style_cell={
            "whiteSpace": "normal", "textAlign": "left"})
        return [html.H5("Model Evaluation Results"), table]
        


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

# update future forecasts plot based on slider value
@app.callback(
    Output('future-forecasts-plot', 'children'),
    [Input('future-slider', 'value')]
)
def update_future_forecasts_plot(value):
    with open(r"Saved/Plotly Figures/Forecasting/future_forecasts.pkl", 'rb') as file:
        figure = pkl.load(file)
    x_range = figure.layout.xaxis.range

    # Generate hourly datetime range using pandas
    two_year_hourly_range = pd.date_range(start=x_range[0], end=x_range[1], freq='H')
    x_min = two_year_hourly_range[0]
    x_max = two_year_hourly_range[value*24]
    figure.update_xaxes(range = [x_min, x_max])

    # Convert the Plotly figure to a Dash graph object
    graph = dcc.Graph(
        figure=figure
    )

    return graph

# update slider output
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('future-slider', 'value')]
)
def update_slider_output(value):
    return f'Future days: {value}'



# Define initial starting layout for the app
# app.layout = landing_page_layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])



# Run the application when script is run
if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8050) # for testing
    # app.run_server(debug=True, host='localhost', port=8050)
    app.run(host='0.0.0.0', port=8050) # for render deployment
