"""
Author: Tobias Butler
Last Modified: 02/27/2024
Description: This module contains a Forecaster class and the associated methods to fit a probabilistic 
    forecasting model and use it to make predictions. It also contains some hyperparameter tuning functionality.

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
from sklearn.model_selection import TimeSeriesSplit

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
    def __init__(self) -> None:
        """
        Parameters:
        ------------
        """
        self.point_prophet_model = None
        self.point_var_model = None
        self.error_prophet_model = None
        self.error_var_model = None
        self.error_trend = None



    """
    This method fits the forecasting model. It requires a clean dataset, a string dependent variable, and takes 
        a dictionary of optional hyperparameters.
    """
    def fit(self, clean_training_data:pd.DataFrame, dependent_variable:str, strong_predictors:list=[], 
            hyperparameters:dict=dict(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01, 
            point_var_lags=10, minimum_error_prediction=None, error_trend=1e-4
        ), **kwargs):
        """
        Parameters:
        ----------
        clean_training_data (pandas.DataFrame): a clean dataset (outliers removed, no missing values) to be used 
            for fitting the model parameters.

        dependent_variable (str): The name of the dependent variable that will be predicted.

        strong_predictors (list[str]): A list of variables to keep separate from the PCA dimensionality reduction 
            applied prior to fitting the model.

        hyperparameters (dict): all optional hyperparamters for the model (see method definition for default values)
        """
        # fit point forecasting Prophet models 
        residuals = self.fit_prophet_models(clean_training_data=clean_training_data, dependent_variable=dependent_variable, **hyperparameters)

        # fit point forecasting VAR model
        squared_errors = self.fit_point_var(residuals=residuals, dependent_variable=dependent_variable, strong_predictors=strong_predictors)
        self.fit_error_forecaster(dependent_variable=dependent_variable, squared_errors=squared_errors, **hyperparameters)


    """
    This is a private method that fits univariate Prophet models to all variables in the dataset provided and returns the residual components
    """
    def _fit_prophet_models(self, clean_training_data:pd.DataFrame, dependent_variable:str, changepoint_prior_scale:float=0.001, 
        seasonality_prior_scale:float=0.01, **kwargs):
        """
        Parameters:
        ----------
        clean_training_data (pandas.DataFrame): a clean dataset (outliers removed, no missing values) to be used 
            for fitting the model parameters.

        dependent_variable (str): The name of the dependent variable that will be predicted.

        changepoint_prior_scale (float): regularization parameter for the Trend component of the Prophet models (smaller values 
            cause more regularization). Default value is 0.001.

        seasonality_prior_scale (0.01): regularization parameter for the Seasonal components of the Prophet models (smaller values 
            cause more regularization). Default value is 0.01.

        Returns:
        ----------
        pandas.DataFrame: a dataset containing residual components from the fit Prophet models
        """
        # fit Prophet model for each numeric variable in the dataset
        residuals = clean_training_data.copy()
        time_series_variables = list(clean_training_data.select_dtypes("number").columns)
        for variable in time_series_variables:
            df = clean_training_data[[variable]].reset_index().rename(columns={"index":"ds", variable:"y"})

            # define prophet model with all seasonality components and high regularization
            model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=True, 
                changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
            
            # fit model
            model.fit(df)
            
            # save model of dependent variable
            if variable == dependent_variable:
                self.point_prophet_model = model

            # get residuals for all other models
            forecasts = model.predict(df)
            forecasts.index = pd.to_datetime(clean_training_data.index)
            forecasts['y'] = df['y'].values
            forecasts["residual"] = df["y"].values - forecasts["yhat"].values

            residuals.loc[:,variable] = forecasts["residual"].values

            return residuals

    """
    This is a private method that fits a vector autoregressive (VAR) model to the residuals of the point forecasting Prophet 
        models.
    """
    def _fit_point_var(self, dependent_variable:str, residuals:pd.DataFrame, strong_predictors:list[str]=[]):
        """
        Parameters:
        ----------
        dependent_variable (str): The name of the dependent variable that will be predicted.

        residuals (pandas.DataFrame): a dataset containing residual components from the fit Prophet models. 
            The VAR model will be fit to these residuals.

        strong_predictors (list): A list of variables to keep separate from the PCA dimensionality reduction 
            applied prior to fitting the model.

        Returns:
        ----------
        pandas.DataFrame: a dataset containing squared errors (residuals) from the fit VAR model.
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
        orthogonal_data = pd.DataFrame(data=orthogonal_data, columns=["PC{}".format(x) for x in 
            range(orthogonal_data.shape[1])], index=not_pca_data.index)

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
        squared_errors = pd.DataFrame(data=(errors**2), columns=endogenous_data.columns, 
            index=endogenous_data.index[1:])

        return squared_errors
    

    """
    This is a private method that fits the error (or variance) forecasting component of the model
    """
    def _fit_error_forecaster(self, dependent_variable:str, squared_errors:pd.DataFrame, 
        minimum_error_prediction:float=None, error_trend:float=1e-3, use_var_model:bool=True, **kwargs):
        """
        Parameters:
        ----------
        dependent_variable (str): The name of the dependent variable that will be predicted.

        squared_errors (pandas.DataFrame): a dataset containing squared errors (residuals) from the 
            fit point forecasting VAR model.

        minimum_error_prediction (float): A hyperparameter used to constrain the forecasted errors. This 
            is the smallest error that the model is able to predict.

        error_trend (float): A hyperparameter trend component added to error forecasts. When positive, 
            this value causes the forecasted error values to increase as the forecasting range increases. 
            This is designed to represent the uncertainty of forecasts made on new data.

        use_var_model (bool): determines whether a VAR model is fit on the residuals of the error forecasting 
            Prophet models. In practice, fitting this additional model has not seemed to be very effective.
        """
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

            # fit VAR model on the residuals of the squared error prophet models
            error_var_model = VAR(error_residuals)
            error_var_result = error_var_model.fit()
            self.error_var_model = error_var_result
            self.error_var_context = error_residuals.values[-error_var_result.k_ar:]

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
        if minimum_error_prediction is None: # use median error as default
            minimum_error_prediction = squared_errors[dependent_variable].median()
        self.min_squared_error = minimum_error_prediction

        # save error trend
        self.error_trend = error_trend
        

    """
    This method makes probabilistic forecasts into the future. The model is required to be fit first. It returns 
        both a series of point forecasts and a series of variance forecasts.
    """
    def predict(self, hours_ahead:int):
        """
        Parameters:
        ----------
        hours_ahead (int): The number of probabilistic forecasts to make

        Returns:
        ----------
        pandas.Series: Predicted point forecasts (float) with time as index (datetime64[ns])

        pandas.Series: Predicted variance (float) for each point forecast with time as index (datetime64[ns])
        """
        # predict with point forecasting prophet model
        if self.point_prophet_model is not None:
            df = self.point_prophet_model.make_future_dataframe(periods=hours_ahead, freq="H")
            point_prophet_forecasts = self.point_prophet_model.predict(df)[["ds", "yhat"]].set_index("ds")
            point_prophet_forecasts = point_prophet_forecasts["yhat"].iloc[-hours_ahead:]
        else:
            raise AttributeError("The model must be fit before predictions can be made.")

        # predict with point forecasting VAR model
        if self.point_var_model is not None:
            point_var_forecasts = self.point_var_model.forecast(self.point_var_context, steps=hours_ahead)[:,0]

        # predict with squared error forecasting Prophet model
        if self.error_prophet_model is not None:
            df = self.error_prophet_model.make_future_dataframe(periods=hours_ahead, freq="H")
            error_prophet_forecasts = self.error_prophet_model.predict(df)[["ds", "yhat"]].set_index("ds")
            error_prophet_forecasts = error_prophet_forecasts["yhat"].iloc[-hours_ahead:]
            error_forecasts = error_prophet_forecasts

        # predict with squared error forecasting VAR model
        if self.error_var_model is not None:
            error_var_forecasts = self.error_var_model.forecast(self.error_var_context, steps=hours_ahead)[:,0]
            error_forecasts = error_forecasts + error_var_forecasts

        error_forecasts = np.maximum(self.min_squared_error, error_forecasts) # set minimum squared error

        # add trend to error forecasts
        error_trend = np.cumsum(np.ones(error_forecasts.shape[0]) * self.error_trend)
        error_forecasts = error_forecasts + error_trend

        return point_prophet_forecasts + point_var_forecasts, error_forecasts


    """
    This method takes data and a set of hyperparameters and conducts rolling cross-validation using MSE and weighted MSE metrics.
    """
    def cross_validate(self, num_folds:int, clean_training_data:pd.DataFrame, dependent_variable:str, hyperparameters:dict,
        strong_predictors:list[str]=[], return_predictions:bool=False):
        """
        Parameters:
        ----------
        num_folds (int): The number of rolling windows to use for cross-validation. More folds reduces variance of 
            the result but increases computational complexity.

        clean_training_data (pandas.DataFrame): a clean dataset (outliers removed, no missing values) to be used 
            for fitting the model parameters.
        
        dependent_variable (str): The name of the dependent variable that will be predicted.

        hyperparameters (dict): a set of hyperparameters used to fit the model

        strong_predictors (list): A list of variables to keep separate from the PCA dimensionality reduction 
            applied prior to fitting the model.

        return_predictions (bool): Determines whether raw MSE and weighted MSE metric values are returned or the predictions 
            and ground truth values so that one can apply their own evalutation metric.
        
        Returns:
        ----------
        if return_predictions:
            float: the average MSE result from the cross-validation
            float: the average weighted MSE result from the cross-validation
        else:
            list[list]: a list of predicted point forecasts and variance forecasts
            list[np.ndarray]: a list of ground truth values that can be used to evaluate the cross-validation predictions.
        """
        # initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=num_folds)

        if return_predictions:
            model_predictions=[]
            truth_values = []
        else:
            model_mse_values = []
            model_wmse_values = []

        for train_index, test_index in tscv.split(clean_training_data):
            train_data, test_data = clean_training_data.iloc[train_index], clean_training_data.iloc[test_index]

            # fit model on the train_data
            self.fit(clean_training_data=train_data, dependent_variable=dependent_variable, 
                strong_predictors=strong_predictors, hyperparameters=hyperparameters)
            
            point_forecasts, error_forecasts = self.predict(test_data.shape[0])

            if return_predictions:
                model_predictions.append([point_forecasts, error_forecasts])
                truth_values.append(test_data[dependent_variable].values)
            
            else:
                # calculate MSE
                mse = np.sum((point_forecasts - (test_data[dependent_variable].values))**2)/test_data.shape[0]

                # calculate weighted MSE
                relative_confidence = 1/np.sqrt(error_forecasts) # weight by 1 over standard deviation
                relative_confidence = relative_confidence/np.sum(relative_confidence) * error_forecasts.shape[0] # normalize
                wmse = np.sum(relative_confidence * (point_forecasts - (test_data[dependent_variable].values))**2)/test_data.shape[0]

                model_mse_values.append(mse)
                model_wmse_values.append(wmse)

        if return_predictions:
            return model_predictions, truth_values
        else:
            avg_mse = sum(model_mse_values)/len(model_mse_values)
            avg_wmse = sum(model_wmse_values)/len(model_wmse_values)
            return avg_mse, avg_wmse


    """
    This method takes a list of hyperparameter sets and evaluates each using cross-validation.
    """
    def tune_hyperparameters(self, clean_training_data:pd.DataFrame, dependent_variable:str, 
        hyperparameter_sets:list[dict], num_cv_folds:int=5, strong_predictors:list[str]=[]):
        """
        Parameters:
        ----------
        clean_training_data (pandas.DataFrame): a clean dataset (outliers removed, no missing values) to be used 
            for fitting the model parameters.
        
        dependent_variable (str): The name of the dependent variable that will be predicted.

        hyperparameter_sets (list): a list of hyperparameter sets to use for fitting and evaluating the model. Each 
            set will be evaluated using rolling cross-validation.

        num_folds (int): The number of rolling windows to use for cross-validation. More folds reduces variance of 
            the result but increases computational complexity.

        strong_predictors (list): A list of variables to keep separate from the PCA dimensionality reduction 
            applied prior to fitting the model.

        Returns:
        ----------
        list[tuple]: a list of tuples. Each tuple will contain a hyperparameter set and its evaluation results.
        """

        # keep track of performances of each set
        hyperparameter_results = []
        for set in hyperparameter_sets: # for each set of hyperparameters, perform cross validation
            mse, wmse = self.cross_validate(num_cv_folds, clean_training_data, dependent_variable, set, 
                strong_predictors=strong_predictors)
            hyperparameter_results.append([set, mse, wmse])

        # return results as a list of tuples where each tuple contains the hyperparameter dict, mse, and wmse
        return hyperparameter_results


    # TODO:
    # provide new dataset for observations that have occurred since the model was fit. Does not refit the model.
    def update_model(self,):
        pass


    # create plotly figure that displays forecasts over the next n hours ahead
    def plot_future(self, n_hours_ahead:int):
        pass