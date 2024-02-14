import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.stats import norm
import itertools
import plotly.graph_objects as go
import pickle as pkl


# run all data cleaning functions
def main():
    pass


# function to split dataset into training and holdout test datasets
def split_preliminary_dataset(path_to_preliminary_dataset:str=None, path_to_holdout_dataset:str=None):
    if path_to_preliminary_dataset is None: path_to_preliminary_dataset = r"Datasets/preliminary.csv"
    prelim_dataset = pd.read_csv(path_to_preliminary_dataset, index_col=0)

    # use last 10% of data for final evaluation
    evaluation_length = int(np.ceil(prelim_dataset.shape[0] * 0.1))
    holdout_test_data = prelim_dataset.iloc[-evaluation_length:, :]["Energy Demand (MWH)"]

    # remove test data from the rest of the dataset
    training_data = prelim_dataset[~prelim_dataset.index.isin(holdout_test_data.index)]

    # save energy demand from holdout test dataset for later use 
    if path_to_holdout_dataset is None: path_to_holdout_dataset = r"Datasets/holdout.csv"
    holdout_test_data.to_csv(path_to_holdout_dataset)

    return training_data


# function to create time series plot for all variables in dataset
def raw_time_series_plots(prelim_training_data:pd.DataFrame=None, path_to_plots:str=None):
    # load dataset if not provided
    if prelim_training_data is None: prelim_training_data = split_preliminary_dataset()
    
    # loop through variables and produce time series plot for each
    if path_to_plots is None: path_to_plots = r"Plotly Figures/Raw Time Series"
    for variable in prelim_training_data:
        fig = go.Figure()

        # Add a time series line plot
        fig.add_trace(go.Scatter(x=prelim_training_data.index, y=prelim_training_data[variable], mode='lines', name='Time Series'))

        # Customize the layout
        fig.update_layout(
            title='Time Series Plot of {}'.format(variable),
            xaxis_title='Time',
            yaxis_title='{}'.format(variable),
            template='plotly_dark'  # Use a dark theme
        )

        variable = variable.replace(r"/", "-")
        with open(r"{}/{}.pkl".format(path_to_plots, variable), 'wb') as file:
            pkl.dump(fig, file=file)


# function to identify outliers for all time series variables in dataset
def identify_outliers(prelim_training_data:pd.DataFrame=None, ):
    pass


# function to fill missing values using univariate Prophet forecasting models
def impute_missing_values(prelim_training_data:pd.DataFrame, path_to_clean_dataset:str=None, path_to_prophet_models:str=None):
    
    
    
    if path_to_clean_dataset is None: path_to_clean_dataset = r"Datasets/clean_training.csv"
    if path_to_prophet_models is None: path_to_prophet_models = r"Models/Prophet"



if __name__ == "__main__":
    main()