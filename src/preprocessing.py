"""
Created by: Tobias Butler
Last Modified: 02/22/2024
Description: This Python module contains functionality to clean and process a preliminary dataset containing energy 
    demand, weather, and economic related data obtained from the EIA, NOAA, and BLS. Running it as a script requires 
    an appropriately named csv file in the local working directory and will produce a new dataset and optional 
    visuals saved to the local working directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from scipy.stats import norm
import plotly.graph_objects as go
import pickle as pkl
import sys
import os
import io
import base64

# prevent logging when fitting Prophet models
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

png_width = 750


class PreprocessingPipeline():
    def __init__(self, saved_directory_name:str="Saved", save_datasets:bool=True, produce_eda_plots:bool=True) -> None:
        """
        Parameters:
        ----------
        saved_directory_name (str): the name of a directory that will be created to hold the clean dataset produced, forecasting models, 
            and optional exploratory data analysis visualizations.

        produce_plots (bool): Determines whether EDA plots are produced and saved in the local directory.
        """
        self.saved_directory = saved_directory_name
        self.produce_eda_plots = produce_eda_plots
        self.save_datasets = save_datasets
        self.eda_error = "This ProcessingPipeline has not been instantiated to produce eda plots. You must set \"produce_eda_plots\" to True."

        # Create file system for saving datasets and figures

        # Define subdirectories
        self.subdirectories = {
            'Datasets': [],
            'Models': ['Prophet'],
            'Plotly Figures': ['Raw Time Series', 'Outlier Detection'],
            'Static Visuals':['Distributions', 'Scatterplots', 'Decompositions']
        }

        # Create the main directory if it doesn't exist
        if not os.path.exists(self.saved_directory):
            os.makedirs(self.saved_directory)

        # Create subdirectories
        for subdir, subsubdirs in self.subdirectories.items():
            subdir_path = os.path.join(self.saved_directory, subdir)
            if not os.path.exists(subdir_path): os.makedirs(subdir_path)
            for subsubdir in subsubdirs:
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                if not os.path.exists(subsubdir_path): os.makedirs(subsubdir_path)



    """
    A method to run all data cleaning and processing functions of the class. It also has an optional 
        argument to produce exploratory data analysis visualizations.
    """
    def process_dataset(self, preliminary_dataset:pd.DataFrame=None, path_to_prelim_dataset:str=None, split_into_train_holdout:bool=False, **kwargs) -> pd.DataFrame:
        """
        Parameters:
        ----------
        preliminary_dataset (pandas.DataFrame): a raw dataset containing energy demand, weather, and economic 
            data gathered from the EIA, NOAA, and BLS.

        path_to_prelim_dataset (str): a path to a csv file containing the dataset described above.

        Returns:
        ----------
        pandas.DataFrame: a clean dataset with reduced outliers and no missing values. This dataset is well-suited for exploratory 
            data analysis and predictive modeling with a model that is able to capture trend and seasonal patterns (TBATS, Holtz-Winters, 
            Prophet, LSTM, N-BEATS, etc.)

        pandas.DataFrame: Also a clean dataset but with all trend and seasonal components removed from each variable. This dataset is suited 
            for predictive modeling by a autoregressive model that does not capture trend or seasonal patterns (ARIMA, VAR, etc.)
        """
        # if producing eda plots, delete current contents of directories
        if self.produce_eda_plots:
            for subdir, subsubdirs in self.subdirectories.items():
                subdir_path = os.path.join(self.saved_directory, subdir)
                for subsubdir in subsubdirs:
                    subsubdir_path = os.path.join(subdir_path, subsubdir)
                    files = os.listdir(subsubdir_path)
                    for file in files:
                        file_path = os.path.join(subsubdir_path, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")

        # Create the main directory if it doesn't exist
        if not os.path.exists(self.saved_directory):
            os.makedirs(self.saved_directory)

        # Create subdirectories
        
        
        # get path to preliminary dataset if not provided
        if preliminary_dataset is None:
            if path_to_prelim_dataset is None:
                preliminary_dataset = pd.read_csv(r"{}/Datasets/preliminary.csv".format(self.saved_directory))
            else:
                preliminary_dataset = pd.read_csv(path_to_prelim_dataset)
        
        if split_into_train_holdout:
            # split dataset into training and holdout evaluation sets
            training_data, holdout_test_data = self.split_preliminary_dataset(prelim_dataset=preliminary_dataset)
        else: # process all data
            training_data = preliminary_dataset.copy()

        if self.produce_eda_plots:
            # produce raw time series plots
            self.raw_time_series_plots(prelim_training_data=training_data)

        # convert numerical variables to ordinal categorical where appropriate (ex: Hourly Precipitation)
        transformed_training_data = self.transform_variables(prelim_training_data=training_data)

        # identify outliers and produce plots
        outliers_removed = self.identify_outliers(transformed_training_data=transformed_training_data)

        # impute missing data and outlier values
        clean_training_data = self.impute_missing_values(outliers_removed_data=outliers_removed)

        if self.produce_eda_plots:
            # produce distribution plots
            self.distribution_plots(clean_training_data=clean_training_data)

            # produce scatterplots with dependent variable
            self.scatterplots(clean_training_data=clean_training_data)

        # produce time series decomposition plots
        residual_components = self.time_series_decompositions(clean_training_data=clean_training_data, save_residuals=True)

        return clean_training_data, residual_components

    """
    This function takes a preliminary dataset and applies a 90-10 split to separate it into training and holdout 
        evaluation data. The training dataset is returned while the holdout dataset is saved to a local directory 
        for future use.
    """
    def split_preliminary_dataset(self, prelim_dataset:pd.DataFrame=None, path_to_preliminary_dataset:str=None, path_to_holdout_dataset:str=None):
        """
        Parameters:
        ----------
        preliminary_dataset (pandas.DataFrame): a raw dataset containing energy demand, weather, and economic 
            data gathered from the EIA, NOAA, and BLS.

        path_to_prelim_dataset (str): a path to a csv file containing the dataset described above.

        path_to_holdout_dataset (str): the path where the 10% holdout data will be saved as a csv file.

        Returns:
        ----------
        pandas.DataFrame: the 90% training data ready for exploratory data analysis or predictive modeling.
        """

        # if dataset not provided, load from local directory
        if prelim_dataset is None:
            if path_to_preliminary_dataset is None: path_to_preliminary_dataset = r"{}/Datasets/preliminary.csv".format(self.saved_directory)
            prelim_dataset = pd.read_csv(path_to_preliminary_dataset, index_col=0)

        # use last 10% of data for final evaluation
        evaluation_length = int(np.ceil(prelim_dataset.shape[0] * 0.1))
        holdout_test_data = prelim_dataset.iloc[-evaluation_length:, :]["Energy Demand (MWH)"]

        # remove test data from the rest of the dataset
        training_data = prelim_dataset[~prelim_dataset.index.isin(holdout_test_data.index)]

        # save energy demand from holdout test dataset for later use 
        if self.save_datasets:
            if path_to_holdout_dataset is None: path_to_holdout_dataset = r"{}/Datasets/holdout.csv".format(self.saved_directory)
            holdout_test_data.to_csv(path_to_holdout_dataset)

        return training_data, holdout_test_data


    """
    This function takes a training dataset and creates a series of time series plots which it saves 
        for future use. By default, these plots are saved in the local directory.
    """
    def raw_time_series_plots(self, prelim_training_data:pd.DataFrame, path_to_plots:str=None):
        """
        Parameters:
        ----------
        prelim_training_data (pandas.DataFrame): a training dataset returned by split_preliminary_dataset()

        path_to_plots (str): Determines where to save the created time series plots. By default, 
            they are saved in the local directory.
        """
        # make sure plots are supposed to be produced
        if not self.produce_eda_plots: raise AttributeError(self.eda_error)
        
        # loop through variables and produce time series plot for each
        if path_to_plots is None: path_to_plots = r"{}/Plotly Figures/Raw Time Series".format(self.saved_directory)
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

            # save plotly figure as pickled file
            variable = variable.replace(r"/", "-") # change variable name for file path
            with open(r"{}/{}.pkl".format(path_to_plots, variable), 'wb') as file:
                pkl.dump(fig, file=file)


    # function to identify outliers for all time series variables in dataset
    """
    This function takes a dataset with any potential data transformations already made. It applies 
        univariate moving average estimation to each time series variable in the dataset provided. 
        The method is described by Blázquez-García, et al. (2022) in A Review on Outlier/Anomaly 
        Detection in Time Series Data. ACM Computing Surveys, 54(3), 1–33. 
    """
    def identify_outliers(self, transformed_training_data:pd.DataFrame=None, path_to_plots:str=None):
        """
        Parameters:
        ----------
        tranformed_training_data (pandas.DataFrame): a dataset with transformations (binning, 
            probability transforms, etc.) already applied

        path_to_plots (str): a path to a directory to save plotly figures showing outliers as pickled 
            files for future use.
        """
        # load dataset if not provided, call function to obtain
        if transformed_training_data is None: transformed_training_data = self.transform_variables()
        
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
            if path_to_plots is None: path_to_plots = r"{}/Plotly Figures/Outlier Detection".format(self.saved_directory)

            if self.produce_eda_plots:
                # save plotly figure
                variable = variable.replace(r"/", "-")
                with open(r"{}/{}.pkl".format(path_to_plots, variable), 'wb') as file:
                    pkl.dump(fig, file=file)

        return outliers_removed_data # preliminary dataset that has outliers replaced with NAN


    """
    This function takes a raw training dataset and applies the following transformations:
        - HourlyPrecipitation is binned to produce a categorical variable
    """
    def transform_variables(self, prelim_training_data:pd.DataFrame, **kwargs):
        """
        Parameters:
        ----------
        prelim_training_data (pandas.DataFrame): a raw training dataset that has been obtained after 
            splitting the original dataset into 90-10 training-holdout datasets.

        Returns:
        ----------
        pandas.DataFrame: a dataset with all variable transformations applied
        """
        # try to transform variables, ignoring an error if those variables are not present in the dataset
        try:
            # transform HourlyPrecipitation to categorical by binning
            # Define bin edges and labels
            bins = [-float('inf'), 0.0001, 0.05, 0.33, float('inf')]
            labels = ['None', 'Light Rain', 'Medium Rain', 'Heavy Rain']

            # Apply binning
            transformed_training_data = prelim_training_data.copy() # don't overwrite dataset
            transformed_training_data.loc[:, "HourlyPrecipitation"] = pd.cut(transformed_training_data['HourlyPrecipitation'], 
                bins=bins, labels=labels)
            
        except KeyError: pass

        return transformed_training_data


    """
    This function takes a dataset that has already had outliers replaced with missing values. It fits Prophet 
        forecasting models to each time series variable and uses interpolation to fill any missing values. 
        Missing values in categorical variables are replaced with their most common value for simplicity.
    """
    # function to fill missing values using univariate Prophet forecasting models
    def impute_missing_values(self, outliers_removed_data:pd.DataFrame=None, path_to_clean_dataset:str=None, 
        path_to_prophet_models:str=None, **kwargs):
        """
        Parameters:
        ----------
        outliers_removed_data (pandas.DataFrame): a dataset with outliers removed (replaced with NAN)

        path_to_clean_dataset (str): a path where the clean dataset (with missing values imputed) will be saved

        path_to_prophet_models (str): a path to a folder where the fit Prophet models will be saved for later use

        Returns:
        ----------
        pandas.DataFrame: a clean dataset with no missing values.
        """

        # if dataset not provided, call function to obtain
        if outliers_removed_data is None: outliers_removed_data = self.identify_outliers(**kwargs)

        # copy the structure of the input dataset for the output dataset
        clean_data = outliers_removed_data.copy()

        # define path to save clean dataset
        if path_to_clean_dataset is None: path_to_clean_dataset = r"{}/Datasets/clean_training.csv".format(self.saved_directory)

        # define path to directory containing pickled Prophet models
        if path_to_prophet_models is None: path_to_prophet_models = r"{}/Models/Prophet".format(self.saved_directory)

        # fit a Prophet model for each time-series variable
        time_series_variables = list(outliers_removed_data.select_dtypes("number").columns)
        for variable in time_series_variables:
            print(f"Interpolating for variable {variable}")
            df = outliers_removed_data[[variable]].reset_index().rename(columns={"index":"ds", variable:"y"})

            # define prophet model with all seasonality components and high regularization
            model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, 
                changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
            
            # fit model
            model.fit(df)

            # calculate interpolations
            interpolated_values = model.predict(df)
            to_impute = outliers_removed_data[variable].isna().values
            clean_data.loc[to_impute, variable] = interpolated_values["yhat"][to_impute].values

            # save fit model
            variable = variable.replace(r"/", "-")
            with open("{}/{}.pkl".format(path_to_prophet_models, variable), "wb") as file:
                pkl.dump(model, file=file)
        
        for variable in [x for x in outliers_removed_data.columns if x not in time_series_variables]:
            print(f"Interpolating for variable {variable}")
            to_impute = outliers_removed_data[variable].isna().values
            clean_data.loc[to_impute, variable] = clean_data[variable].mode()

        # save clean dataset
        if self.save_datasets:
            clean_data.to_csv(path_to_clean_dataset)
        return clean_data


    """
    This function takes a clean dataset (with outliers and missing values imputed) and produces a series of 
        visuals displaying the distributions of every variable in the dataset. The distributions of continuous 
        variables are visualized using histograms while the distributions of categorical variables are 
        visualized using bart charts.
    """
    def distribution_plots(self, clean_training_data:pd.DataFrame=None, path_to_clean_dataset:str=None,
        path_to_plots:str=None, **kwargs):
        """
        Parameters:
        ----------
        clean_training_data (pandas.DataFrame): a clean dataset with outliers and missing values imputed

        path_to_clean_dataset (str): a path to a csv file containing the clean dataset

        path_to_plots (str): a path to the directory where distribution plots will be saved as png files for later use
        """
        # make sure plots are supposed to be produced
        if not self.produce_eda_plots: raise AttributeError(self.eda_error)

        # if dataset not provided, try loading from local directory
        if clean_training_data is None: 
            if path_to_clean_dataset is None: path_to_clean_dataset = r"{}/Datasets/clean_training.csv".format(self.saved_directory)
            clean_training_data = pd.read_csv(clean_training_data, index_col=0)
        else: clean_training_data = clean_training_data.copy() # avoid overwriting

        # define where to save the distribution plots
        if path_to_plots is None: path_to_plots = r"{}/Static Visuals/Distributions".format(self.saved_directory)

        # create distribution plots for continuous variables
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
            # Save the plot as a PNG image with 300 pixels per inch (ppi)
            variable = variable.replace(r"/", "-")
            # fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
            save_png_encoded(r'{}/{}.png'.format(path_to_plots, variable), fig)
            plt.close()

        # plot distributions for categorical variables
        for variable in [x for x in clean_training_data.columns if x not in clean_training_data.select_dtypes("number").columns]:
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot()
            sns.countplot(data=clean_training_data, x=variable)
            ax.set_title(f"{variable}: Distribution")
            ax.set_xlabel(variable)
            ax.set_ylabel("Number of Observations")
            ax.bar_label(ax.containers[0])
            # Save the plot as a PNG image with 300 pixels per inch (ppi)
            variable = variable.replace(r"/", "-")
            # fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
            save_png_encoded(r'{}/{}.png'.format(path_to_plots, variable), fig)
            plt.close()

        

    # function to produce static scatterplots between Energy Demand and each other variable
    """
    This function takes a clean dataset and produces a series of scatterplots between each predictor 
        variable and the dependent variable. It saves them to a directory as png files for later use.
    """
    def scatterplots(self, clean_training_data:pd.DataFrame=None, path_to_clean_dataset:str=None, path_to_plots:str=None, 
        **kwargs):
        """
        Parameters:
        ----------
        clean_training_data (pandas.DataFrame): a clean dataset with outliers and missing values imputed

        path_to_clean_dataset (str): a path to a csv file containing the clean dataset

        path_to_plots (str): a path to the directory where the scatterplots will be saved as png files for later use
        """
        # make sure plots are supposed to be produced
        if not self.produce_eda_plots: raise AttributeError(self.eda_error)

        # if dataset not provided, load from local directory
        if clean_training_data is None: 
            if path_to_clean_dataset is None: path_to_clean_dataset = r"{}/Datasets/clean_training.csv".format(self.saved_directory)
            clean_training_data = pd.read_csv(path_to_clean_dataset, index_col=0)
        else: clean_training_data = clean_training_data.copy() # avoid overwriting

        # define where to save the scatterplots
        if path_to_plots is None: path_to_plots = r"{}/Static Visuals/Scatterplots".format(self.saved_directory)

        # create scatterplots for continuous predictors
        dependent_variable = "Energy Demand (MWH)"
        continuous_variables = list(clean_training_data.select_dtypes("number").columns)
        for variable in continuous_variables:
            if variable == dependent_variable: # plot heatmap of correlation coefficients
                correlations = clean_training_data.select_dtypes("number").corr()
                fig = plt.figure(figsize=(30,30))
                ax = fig.add_subplot()
                sns.heatmap(correlations, annot=True, fmt=".2f", annot_kws={"fontsize":24})
                ax.set_title("Correlations Between {} and Predictors".format("Energy Demand (MWH)"), size=28)
                ax.set_yticklabels(ax.get_yticklabels(), size = 20)
                ax.set_xticklabels(ax.get_xticklabels(), size = 20)
                plt.yticks(rotation=0) 
                plt.xticks(rotation=90) 

                # Save the plot as a PNG image with 300 pixels per inch (ppi)
                variable = variable.replace(r"/", "-")
                # plt.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
                save_png_encoded(r'{}/{}.png'.format(path_to_plots, variable), fig)
                plt.close()
            else:
                # Calculate the correlation coefficient
                corr = np.corrcoef(clean_training_data[variable], clean_training_data[dependent_variable])[0, 1]

                # plot scatter plot
                fig = plt.figure(figsize=(8,5))
                ax = fig.add_subplot()
                sns.regplot(x=clean_training_data[variable], y=clean_training_data[dependent_variable], 
                    label="Correlation: {:.2f}".format(corr), line_kws=dict(color="r"), order=3)
                ax.set_title("Relationship between Energy Demand and {}".format(variable))
                ax.set_xlabel(variable)
                ax.set_ylabel(dependent_variable)
                ax.legend()

                # Save the plot as a PNG image with 300 pixels per inch (ppi)
                variable = variable.replace(r"/", "-")
                # fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
                save_png_encoded(r'{}/{}.png'.format(path_to_plots, variable), fig)
                plt.close()

        # scatter plots for categorical predictors
        for variable in [x for x in clean_training_data.columns if x not in continuous_variables]:
            # calculate average value of dependent variable for each category of predictor
            means = clean_training_data.groupby(by=[variable])[dependent_variable].mean()

            # scatter plot
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot()
            sns.scatterplot(data=clean_training_data, x=variable, y=dependent_variable)
            sns.lineplot(means, label="OLS Regression Line", color="red")
            ax.set_title("Relationship between Energy Demand and {}".format(variable))
            ax.set_xlabel(variable)
            ax.set_ylabel(dependent_variable)
            ax.legend()

            # Save the plot as a PNG image with 300 pixels per inch (ppi)
            variable = variable.replace(r"/", "-")
            # fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
            save_png_encoded(r'{}/{}.png'.format(path_to_plots, variable), fig)
            plt.close()


    # function to produce time series decompositions for each numerical variable (and save the residuals for future modeling)
    """
    This function takes a clean dataset and produces a series of time-series decomposition plots for each continuous variable 
        using the Prophet models fit previously. It saves them in a local directory for later use.
    """
    def time_series_decompositions(self, clean_training_data:pd.DataFrame=None, path_to_prophet_models:str=None, 
        path_to_plots:str=None, save_residuals:bool=True, path_to_residual_dataset:str=None, produce_plots:bool=True, 
        **kwargs):
        """
        Parameters:
        ----------
        clean_training_data (pandas.DataFrame): a clean dataset with outliers and missing values imputed

        path_to_clean_dataset (str): a path to a csv file containing the clean dataset

        path_to_plots (str): a path to the directory where the scatterplots will be saved as png files for later use

        Returns:
        ----------
        """
        # get clean training dataset if not provided
        if clean_training_data is None: 
            try: clean_training_data = pd.read_csv(r"{}/Datasets/clean_training.csv".format(self.saved_directory), index_col=0)
            except: clean_training_data = self.impute_missing_values(**kwargs)
        else: clean_training_data = clean_training_data.copy() # avoid overwriting

        # get path to plots if not provided
        if path_to_plots is None: path_to_plots = r"{}/Static Visuals/Decompositions".format(self.saved_directory)

        # get path to prophet models if not provided
        if path_to_prophet_models is None: path_to_prophet_models = r"{}/Models/Prophet".format(self.saved_directory)

        # loop through each variable, producing time series decomposition and optionally saving residuals
        for variable in clean_training_data.select_dtypes("number").columns:
            file_variable = variable.replace(r"/", "-")
            with open(r"{}/{}.pkl".format(path_to_prophet_models, file_variable), "rb") as file:
                model = pkl.load(file=file)
            df = clean_training_data[[variable]].reset_index().rename(columns={"index":"ds", variable:"y"})
            forecasts = model.predict(df)
            forecasts.index = pd.to_datetime(clean_training_data.index)
            forecasts['y'] = df['y'].values
            forecasts["residual"] = df["y"].values - forecasts["yhat"].values

            if save_residuals: clean_training_data.loc[:,variable] = forecasts["residual"].values

            if produce_plots and self.produce_eda_plots:
                # define figure
                fig = plt.figure(figsize=(8,15))

                # plot original time series
                ax = fig.add_subplot(6,1,1)
                sns.lineplot(x=forecasts["ds"].index, y=forecasts['y'])
                ax.set_title("Original Time Series for {}".format(variable))
                ax.set_xlabel("Time")
                ax.set_ylabel(variable)

                # plot trend component
                ax = fig.add_subplot(6,1,2)
                sns.lineplot(x = forecasts["ds"], y=forecasts["trend"])
                ax.set_title("Trend Component for {}".format(variable))
                ax.set_xlabel("Time")
                ax.set_ylabel(variable)

                # plot yearly seasonality
                ax = fig.add_subplot(6,1,3)
                sns.lineplot(x = forecasts["ds"], y=forecasts["yearly"])
                ax.set_title("Yearly Seasonality for {}".format(variable))
                ax.set_xlabel("Time")
                ax.set_ylabel(variable)

                # plot weekly seasonality
                ax = fig.add_subplot(6,1,4)
                sns.lineplot(x = forecasts["ds"], y=forecasts["weekly"])
                ax.set_title("Weekly Seasonality for {}".format(variable))
                ax.set_xlabel("Time")
                ax.set_ylabel(variable)
                # change axis limits
                ax.set_xlim(forecasts["ds"].iloc[0], forecasts["ds"].iloc[100*7])
                labels = [str(x)[-8:-2] for x in ax.get_xticklabels()]
                ax.set_xticklabels(labels)

                # plot daily seasonlity
                ax = fig.add_subplot(6,1,5)
                sns.lineplot(x = forecasts["ds"], y=forecasts["daily"])
                ax.set_title("Daily Seasonality for {}".format(variable))
                ax.set_xlabel("Time")
                ax.set_ylabel(variable)
                # change axis limits
                ax.set_xlim(forecasts["ds"].iloc[0], forecasts["ds"].iloc[100])
                labels = [str(x)[-10:-5] for x in ax.get_xticklabels()]
                ax.set_xticklabels(labels)

                # plot residual component
                ax = fig.add_subplot(6,1,6)
                sns.lineplot(x = forecasts["ds"], y=forecasts["residual"])
                ax.set_title("Residual Component for {}".format(variable))
                ax.set_xlabel("Time")
                ax.set_ylabel(variable)

                # save figure
                plt.tight_layout()
                # Save the plot as a PNG image with 300 pixels per inch (ppi)
                variable = variable.replace(r"/", "-")
                # fig.savefig(r'{}/{}.png'.format(path_to_plots, variable), dpi=300)
                save_png_encoded(r'{}/{}.png'.format(path_to_plots, variable), fig)
                plt.close()
        
        if save_residuals and self.save_datasets:
            if path_to_residual_dataset is None: path_to_residual_dataset = r"{}/Datasets/residuals.csv".format(self.saved_directory)
            clean_training_data.to_csv(path_to_residual_dataset)

        return clean_training_data # really the residuals


    # obtain time series residuals from decompositions
    def calculate_residuals(self, clean_training_data:pd.DataFrame=None, path_to_clean_dataset:str=None, 
        path_to_prophet_models:str=None, path_to_residual_dataset:str=None, save_residuals:bool=True, **kwargs):
        """
        Parameters:
        ----------

        Returns:
        ----------
        """

        # get clean training dataset if not provided
        if clean_training_data is None:
            if path_to_clean_dataset is None: clean_training_data = pd.read_csv(r"{}/Datasets/clean_training.csv".format(self.saved_directory))
            else: clean_training_data = pd.read_csv(path_to_clean_dataset)

        # get residuals from time series decompositions without making plots
        return self.time_series_decompositions(clean_training_data=clean_training_data, path_to_prophet_models=path_to_prophet_models, 
            save_residuals=save_residuals, path_to_residual_dataset=path_to_residual_dataset, no_plots=True, **kwargs)




"""
The main function to run all data cleaning and processing functions within this module. It also has an optional 
    argument to produce exploratory data analysis visualizations.
"""
def main(preliminary_dataset, saved_directory:str, produce_plots:bool=True, **kwargs) -> pd.DataFrame:
    """
    Parameters:
    ----------
    preliminary_dataset (str or pandas.DataFrame): either a string path to a csv file containing a raw dataset or the dataset itself 
        as a pandas DataFrame. The dataset is intended to contain energy demand, weather, and economic data collected from the 
        EIA, NOAA, and BLS.

    saved_directory (str): the path to the directory that is going to hold processing outputs (clean dataset, Prophet forecasting model, 
        optional eda visualizations)

    produce_plots (bool): Determines whether EDA plots are produced and saved in the local directory.

    reduced_variables (bool): Determines whethere to use include all variables while processing the data or a subset 
        of the variables that are correlated with the dependent variable and less correlated with other predictor variables.

    Returns:
    ----------
    pandas.DataFrame: a clean dataset with reduced outliers and no missing values. This dataset is well-suited for exploratory 
        data analysis and predictive modeling with a model that is able to capture trend and seasonal patterns (TBATS, Holtz-Winters, 
        Prophet, LSTM, N-BEATS, etc.)

    pandas.DataFrame: Also a clean dataset but with all trend and seasonal components removed from each variable. This dataset is suited 
        for predictive modeling by a autoregressive model that does not capture trend or seasonal patterns (ARIMA, VAR, etc.)
    """    
    # instantiate ProcessingPipeline
    pipeline = PreprocessingPipeline(saved_directory_name=saved_directory, produce_eda_plots=produce_plots)

    # run pipeline from start to finish
    if type(preliminary_dataset) is pd.DataFrame:
        return pipeline.process_dataset(preliminary_dataset=preliminary_dataset)
    else: 
        return pipeline.process_dataset(path_to_prelim_dataset=preliminary_dataset)



### Helper Functions ###
########################


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


def save_png_encoded(filepath:str, fig:plt.Figure):
    # Get the original dimensions of the figure
    original_width, original_height = fig.get_size_inches()

    # Calculate the aspect ratio
    aspect_ratio = original_height / original_width

    # Set the new width to 750 pixels
    new_width_inches = png_width / 100  # Convert pixels to inches

    # Calculate the corresponding height
    new_height_inches = new_width_inches * aspect_ratio

    # Set the new dimensions for the figure
    fig.set_size_inches(new_width_inches, new_height_inches)

    # Save the figure to a buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=200)
    buffer.seek(0)

    # Encode the buffer contents as base64
    base64_encoded = base64.b64encode(buffer.read()).decode()

    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_encoded))

    # Embed the base64-encoded image in an HTML img tag
    # html_img = f'<img src="data:image/png;base64,{base64_encoded}">'


# # want to be able to call this function to fit all models, save them, and return them. Then use them in whatever this function was called from.
# def fit_prophet_models(outliers_removed_dataset:pd.DataFrame, tune_hyperparameters:bool=False, 
#     path_to_prophet_models:str=None):
#     """
#     Parameters:
#     ----------
#     tune_hyperparameters (bool): Determines whether to perform cross-validation to determine optimal 
#         hyperparameter values for each Prophet model. This adds a significant amount of computation time. 
#         By default, it is False and high regularization hyperparameters are used for every model.
#     """
#     if tune_hyperparameters: pass
#     # fit Prophet model for each numeric variable in the dataset
#     prophet_models = {}
#     time_series_variables = list(outliers_removed_dataset.select_dtypes("number").columns)
#     for variable in time_series_variables:
#         print(f"Interpolating for variable {variable}")
#         df = outliers_removed_dataset[[variable]].reset_index().rename(columns={"index":"ds", variable:"y"})

#         # define prophet model with all seasonality components and high regularization
#         model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, 
#             changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
        
#         # fit model
#         model.fit(df)

#         # save fit model
#         prophet_models[variable] = model
#         variable = variable.replace(r"/", "-")
#         with open("{}/{}.pkl".format(path_to_prophet_models, variable), "wb") as file:
#             pkl.dump(model, file=file)

#         return prophet_models

    # grid search hyperparameter tune for prophet models

    # save results to "Tuning Results/prophet.pkl"




if __name__ == "__main__":
    main()