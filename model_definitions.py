"""
Author: Tobias Butler
Last Modified: 02/24/2024
Description: 

Still to do: 
- add hyperparameter tuning options
- add functions to preprocess new data
- estimate trend in predicted squared error (to simulate increased uncertainty over 
    longer forecasting periods)
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle as pkl

# prevent logging when fitting Prophet models
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


"""
A class representing a probabilistic forecasting model. It fits basic statistical models for 
    both point forecasts and their squared errors.
"""
class Forecaster():
    """
    Parameters:
    ------------
    path_to_saved_files (str): a path to a directory that contains datasets, figures, and models 
        for this project.
    """
    def __init__(self, path_to_saved_files:str="Saved") -> None:
        self.point_prophet_model = None
        self.point_var_model = None
        self.error_prophet_model = None
        self.error_var_model = None
        self.path_to_saved_files = path_to_saved_files



    """
    
    """
    def fit_from_clean(clean_dataset:pd.DataFrame=None, path_to_clean_dataset:pd.DataFrame=None, 
        hyperparameters:dict=dict(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01, 
            point_var_lags=10, minimum_error_prediction=None, error_trend=1e-4
        ), **kwargs):
        """
        Parameters:
        ----------

        """
        # fit Prophet models then call fit_from_residuals


    """
    
    """
    def fit_from_residuals(self, dependent_variable:str, point_prophet_model:Prophet=None, residuals:pd.DataFrame=None, 
        path_to_residuals:str=None, path_to_prophet_models:str=None, strong_predictors:list=None, **kwargs):
        """
        
        dependent_variable (str):  ex: "Energy Demand (MWH)"

        strong_predictors (list[str]): ["HourlyDryBulbTemperature"]
        """
        # if prophet model not provided, try loading from saved files directory
        if point_prophet_model is None:
            if path_to_prophet_models is None: path_to_prophet_models = r"Saved/Models/Prophet"
            with open(r"{}/{}.pkl".format(path_to_prophet_models, dependent_variable), "rb") as file:
                point_prophet_model = pkl.load(file)

        # save provided point forecasting prophet model for making predictions
        self.point_prophet_model = point_prophet_model

        # if no residual dataset provided, try loading from saved files directory
        if residuals is None: 
            if path_to_residuals is None: path_to_residuals = r"Saved/Datasets/residuals.csv"
            residuals = pd.read_csv(path_to_residuals, index_col=0)

        # fit point forecasting VAR model
        self.fit_point_var(residuals=residuals, dependent_variable=dependent_variable, strong_predictors=strong_predictors)


    """
    
    """
    def fit_point_var(self, dependent_variable:str, residuals:pd.DataFrame, strong_predictors:list[str]=[]):
        """
        Parameters:
        ----------

        Returns:
        ----------

        """
        # convert categorical variables to encoded numerical using one-hot encoding
        dummified_data = pd.get_dummies(residuals, drop_first=True)

        # define the most important variables. These will not be reduced using PCA
        not_pca_vars = [dependent_variable] + strong_predictors

        # the rest of the variables will be transformed using PCA
        not_pca_data = dummified_data[not_pca_vars]
        pca_data = dummified_data.drop(columns=not_pca_vars)

        # Standardize the data being used in PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)

        # Perform PCA to reduce number of features in dataset
        pca = PCA(n_components=scaled_data.shape[1]) 
        pca.fit(scaled_data)

        # keep principal components that contribute more than 0.1 of the total variance
        num_pcs_to_keep = sum(pca.explained_variance_ >= 1e-1)

        # refit PCA with the desired number of principal components
        pca = PCA(n_components=num_pcs_to_keep)  
        pca.fit(scaled_data)

        # Transform the data using the fitted PCA
        orthogonal_data = pca.transform(scaled_data)
        orthogonal_data = pd.DataFrame(data=orthogonal_data, columns=["PC{}".format(x) for x in range(orthogonal_data.shape[1])], index=not_pca_data.index)

        # combine primary variables with transformed dataset
        endogenous_data = pd.concat([not_pca_data, orthogonal_data], axis=1)

        # Fit VAR model
        point_var_model = VAR(endogenous_data)
        point_var_result = point_var_model.fit()
        self.point_var_model = point_var_result
        self.point_var_context = endogenous_data.values[-point_var_result.k_ar:]

        # define the errors of the model
        errors = endogenous_data.values[1:] - point_var_result.resid.values

        # convert to squared errors for modeling variance
        squared_errors = pd.DataFrame(data=(errors**2), columns=endogenous_data.columns, index=endogenous_data.index[1:])

        return squared_errors
    


    def fit_error_prophet_models(self, dependent_variable:str, squared_errors:pd.DataFrame, use_var_model:bool=True):
        # fit Prophet model on the residuals of each variable included in the VAR model
        if use_var_model: 
            error_residuals = squared_errors.copy()
            for variable in squared_errors.columns:
                df = squared_errors[[variable]].reset_index().rename(columns={"index":"ds", variable:"y"})

                # define prophet model with all seasonality components and high regularization
                error_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, 
                    changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
                
                # fit model
                error_model.fit(df)

                forecasts = error_model.predict(df)
                forecasts["yhat"] = np.maximum(0, forecasts["yhat"].values) # adjust forecasts to be positive
                forecasts["residual"] = df["y"].values - forecasts["yhat"].values
                error_residuals.loc[:,variable] = forecasts["residual"].values

                # save squared error prophet model as class attribute for making predictions
                if variable == dependent_variable: 
                    self.error_prophet_model = error_model
                    self.min_squared_error = squared_errors[dependent_variable].median()

            # fit VAR model on the residuals of the squared error prophet models
            error_var_model = VAR(error_residuals)
            error_var_result = error_var_model.fit()
            self.error_var_model = error_var_result
            self.error_var_context = error_residuals.values[-error_var_model.k_ar:]

        # fit just a basic Prophet model on the squared errors of the dependent variable
        else:
            df = squared_errors[[dependent_variable]].reset_index().rename(columns={"index":"ds", variable:"y"})
            # define prophet model with all seasonality components and high regularization
            error_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, 
                changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
            
            # fit model
            error_model.fit(df)

            # save as class attribute
            self.error_prophet_model = error_model

            # save minimum (safety expected squared error)
            self.min_squared_error = squared_errors[dependent_variable].median()
        
    """
    
    """
    def predict(self, hours_ahead:int):
        # predict with point forecasting prophet model
        if self.point_prophet_model is not None:
            df = self.point_prophet_model.make_future_dataframe(periods=hours_ahead+1, freq="H")
            point_prophet_forecasts = self.point_prophet_model.predict(df)[["ds", "yhat"]]
            # point_prophet_forecasts = point_prophet_forecasts[point_prophet_forecasts["ds"].isin(evaluation_data.index)]["yhat"].values
            point_prophet_forecasts = point_prophet_forecasts["yhat"].values[-hours_ahead:]
        else:
            raise AttributeError("The model must be fit before predictions can be made.")

        # predict with point forecasting VAR model
        if self.point_var_model is not None:
            point_var_forecasts = self.point_var_model.forecast(self.point_var_context, steps=hours_ahead)[:,0]

        # predict with squared error forecasting Prophet model
        if self.error_prophet_model is not None:
            df = self.error_prophet_model.make_future_dataframe(periods=hours_ahead+1, freq="H")
            error_prophet_forecasts = self.error_prophet_model.predict(df)[["ds", "yhat"]]
            error_prophet_forecasts = error_prophet_forecasts["yhat"].values[-hours_ahead:]
            error_forecasts = error_prophet_forecasts

        # predict with squared error forecasting VAR model
        if self.error_var_model is not None:
            error_var_forecasts = self.error_var_model.forecast(self.error_var_context, steps=hours_ahead)[:,0]
            error_forecasts = error_forecasts + error_var_forecasts

        error_forecasts = np.maximum(self.min_squared_error, error_forecasts) # set minimum squared error

        return point_prophet_forecasts + point_var_forecasts, error_forecasts
        


    # provide new dataset for observations that have occurred since the model was fit. Does not refit the model.
    def update_model():
        pass


    def cross_validate():
        pass


    # want to split this into sections to avoid repeating all parts everytime if not necessary...
    def tune_hyperparameters(tune_point_forecaster:bool, tune_error_forecaster:bool):
        pass
