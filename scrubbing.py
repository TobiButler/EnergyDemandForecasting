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
def identify_outliers(transformed_training_data:pd.DataFrame=None, path_to_plots:str=None):
    # load dataset if not provided
    if transformed_training_data is None: transformed_training_data = transform_variables()
    
    # Plot the time series with outliers marked
    outliers_removed_data = transformed_training_data.copy()
    for variable in transformed_training_data.select_dtypes("number").columns:
        ts = transformed_training_data[variable].copy()
        outliers = detect_outliers(ts, p=0.001, n=1000)
        outliers_removed_data.loc[outliers, variable] = np.nan # replace outliers with NAN to remove them
        outliers = ts[outliers]

        # create plotly time series plot with outliers marked in red
        fig = go.Figure()

        # Add a time series line plot
        fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Time Series'))
        fig.add_trace(go.Scatter(x=outliers.index, y=outliers, mode='markers', name='Outliers'))

        # Customize the layout
        fig.update_layout(
            title='Outlier Identification for {}'.format(variable),
            xaxis_title='Time',
            yaxis_title='{}'.format(variable),
            template='plotly_dark'  # Use a dark theme
        )

        # define directory to store plots if not provided
        if path_to_plots is None: path_to_plots = r"Plotly Figures/Outlier Detection"

        # save plotly figure
        variable = variable.replace(r"/", "-")
        with open(r"{}/{}.pkl".format(variable), 'wb') as file:
            pkl.dump(fig, file=file)

    return outliers_removed_data # preliminary dataset that has outliers replaced with NAN


# transform variables where appropriate
def transform_variables(prelim_training_data:pd.DataFrame=None, **kwargs):
    if prelim_training_data is None: prelim_training_data = split_preliminary_dataset(**kwargs)
    # don't want to apply outlier detection to HourlyPrecipitation since it behaves more like a binary variable than a normally distributed one.

    # Define bin edges and labels
    bins = [-float('inf'), 0.0001, 0.05, 0.33, float('inf')]
    labels = ['None', 'Light Rain', 'Medium Rain', 'Heavy Rain']

    # Bin the 'values' column
    transformed_training_data = prelim_training_data.copy() # don't overwrite dataset
    transformed_training_data.loc[:, "HourlyPrecipitation"] = pd.cut(transformed_training_data['HourlyPrecipitation'], bins=bins, labels=labels)

    return transformed_training_data


# function to fill missing values using univariate Prophet forecasting models
def impute_missing_values(outliers_removed_data:pd.DataFrame=None, path_to_clean_dataset:str=None, path_to_prophet_models:str=None, **kwargs):
    if outliers_removed_data is None: outliers_removed_data = identify_outliers(**kwargs)
    prophet_models = {}
    interpolations = {}
    clean_data = outliers_removed_data.copy()

    # define path to clean dataset
    if path_to_clean_dataset is None: path_to_clean_dataset = r"Datasets/clean_training.csv"

    # define path to directory containing pickled Prophet models
    if path_to_prophet_models is None: path_to_prophet_models = r"Models/Prophet"

    # fit a Prophet model for each time-series variable
    for variable in outliers_removed_data.select_dtypes("number").columns:
        print(f"Interpolating for variable {variable}")
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, 
            changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
        df = outliers_removed_data[[variable]].reset_index().rename(columns={"index":"ds", variable:"y"})
        model.fit(df)
        prophet_models[variable] = model
        interpolated_values = model.predict(df)
        interpolations[variable] = interpolated_values
        to_impute = outliers_removed_data[variable].isna().values
        clean_data.loc[to_impute, variable] = interpolated_values["yhat"][to_impute].values

        # save model
        variable = variable.replace(r"/", "-")
        with open("{}/{}".format(path_to_prophet_models, variable), "wb") as file:
            pkl.dump(model, file=file)

    # save clean dataset
    clean_data.to_csv(path_to_clean_dataset)


def distribution_plots(clean_training_data:pd.DataFrame=None, path_to_plots:str=None, **kwargs):
    if clean_training_data is None: 
        try: clean_training_data = pd.read_csv(r"Datasets/clean_training.csv", index_col=0)
        except: clean_training_data = impute_missing_values(**kwargs)

    # define where to save the distribution plots
    if path_to_plots is None: path_to_plots = r"Static Visuals/Distributions"

    # create distribution plots
    for variable in clean_training_data.select_dtypes("number").columns:
        # calculate mean and standard deviation
        mean = clean_training_data[variable].mean()
        std = clean_training_data[variable].std()
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
        sns.histplot(clean_training_data[variable])
        ax.set_title(f"{variable}: Distribution")
        ax.set_xlabel(variable)
        ax.set_ylabel("Number of Occurrences")
        ylim = ax.get_ylim()
        ax.vlines(x=mean, ymin=ylim[0], ymax=ylim[1], color="black", linewidth=3, label="Mean: {:.2f}".format(mean))
        ax.fill_between(x=[mean-std, mean+std], y1=ylim[0], y2=ylim[1], alpha=0.5, label="STD: {:.2f}".format(std))
        ax.fill_between(x=[mean-3*std, mean+3*std], y1=ylim[0], y2=ylim[1], alpha=0.2, label="Within 3 STDs".format(std))
        ax.legend()

    for variable in [x for x in clean_training_data.columns if x not in clean_training_data.select_dtypes("number").columns]:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
        sns.countplot(variable)
        ax.set_title(f"{variable}: Distribution")
        ax.set_xlabel(variable)
        ax.set_ylabel("Number of Observations")
        ax.legend()
        # Save the plot as a PNG image with 300 pixels per inch (ppi)
        fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)

    

# function to produce static scatterplots between Energy Demand and each other variable
def scatterplots(clean_training_data:pd.DataFrame=None, path_to_plots:str=None, **kwargs):
    if clean_training_data is None: 
        try: clean_training_data = pd.read_csv(r"Datasets/clean_training.csv", index_col=0)
        except: clean_training_data = impute_missing_values(**kwargs)

    # define where to save the distribution plots
    if path_to_plots is None: path_to_plots = r"Static Visuals/Scatterplots"

    # create distribution plots
    dependent_variable = "Energy Demand (MWH)"
    for variable in clean_training_data.columns:
        if variable == dependent_variable: # plot heatmap of correlation coefficients
            correlations = clean_training_data.corr()
            fig = plt.figure(figsize=(30,30))
            ax = fig.add_subplot()
            sns.heatmap(correlations, annot=True, fmt=".2f")
            plt.fontsize=20
            plt.title("Correlations Between Variables", size=20)

            # Save the plot as a PNG image with 300 pixels per inch (ppi)
            plt.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
        else:
            # Calculate the correlation coefficient
            corr = np.corrcoef(clean_training_data[variable], clean_training_data[dependent_variable])[0, 1]

            # plot scatter plot
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot()
            sns.regplot(x=clean_training_data[variable], y=clean_training_data["Energy Demand (MWH)"], label="Correlation: {:.2f}".format(corr))
            ax.set_title("Relationship between Energy Demand and {}".format(variable))
            ax.set_xlabel(variable)
            ax.set_ylabel("Energy Demand (MWH)")
            ax.legend()

            # Save the plot as a PNG image with 300 pixels per inch (ppi)
            fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)


# function to produce time series decompositions for each numerical variable (and save the residuals for future modeling)
def time_series_decompositions(path_to_prophet_models:str=None):
    if path_to_prophet_models is None: path_to_prophet_models = r"Models/Prophet"
    # for variable in 
    fig = prophet_models[variable].plot_components(interpolations[variable])
    plt.suptitle(f"{variable}: Time Series Decomposition")
    plt.tight_layout()
    plt.show()



# a helper function for identifying outliers
def detect_outliers(data:pd.Series, n:int=1000, p:float=0.001):
    """
    Detect outliers in a time series using a moving average estimation model.

    Parameters:
    - data: Time series data.
    - n: Window size for the moving average (default: 1000).
    - p: Initial threshold probability (default: 0.001).

    Returns:
    - outliers: Boolean array indicating whether each data point is an outlier.
    """

    # define array to hold boolean identifications
    outliers = np.zeros_like(data, dtype=bool)
    half_n = n // 2 # define upper and lower window sizes

    for i in range(half_n, len(data) - half_n):
        window = data[i - half_n : i + half_n + 1] # define window
        mean = np.mean(window)
        std_dev = np.std(window)
            
        z_score = (data[i] - mean) / std_dev
        probability = norm.cdf(z_score) # calculate probability of generating sample from points around it

        # check if sample is highly unlikely to be generated from the same distribution as the points around it
        if (probability >= (1 - p/2)) or (probability <= p/2):
            outliers[i] = True

    return outliers



if __name__ == "__main__":
    main()