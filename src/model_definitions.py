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
import datetime
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from sklearn.model_selection import TimeSeriesSplit
import torch as t
from copy import deepcopy
import warnings

# prevent logging when fitting Prophet models
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

class LSTM(t.nn.Module):
    """
    Constructor for the LSTM class. Currently, the general architecture of this class cannot be adjusted from outside of it. 
        The only adjustable attributes are the network's input size, hidden state size, number of layers, output size, and dropout.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, training_sequence_length:int, dropout:float = 0, **kwargs):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = t.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = t.nn.Dropout(dropout)
        self.fc = t.nn.Sequential(
            t.nn.Dropout(dropout),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.Dropout(dropout),
            t.nn.Linear(hidden_size, output_size),
            t.nn.Softplus() 
        ) 
        self.c0 = None
        self.h0 = None
        self.training_sequence_length = training_sequence_length
        self.input_scaling = None
        self.input_time_scaling = None

    """
    This method runs the network. It evaluates the network as a function to a batch of input samples with size 
        (sequence length x number of features). This means that all tensor arguments for this method should have 3 
        dimensions (batch size, sequence length, number of features)
    """
    def forward(self, x_observed, x_time, bayesian_predict:bool=True):
        if bayesian_predict: self.train() 
        else: self.eval()
        x = t.cat([x_observed, x_time], dim=-1)
        device = next(self.parameters()).device
        h0 = t.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = t.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        # Forward propagate LSTM
        out, (h0, c0) = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
    """
    
    """
    def predict_variance(self, x_observed, x_time, n:int=100):
        dims = [1]*x_observed.dim()
        x_observed = x_observed.repeat(n, *dims)
        x_time = x_time.repeat(n, *dims)
        predictions = self.forward(x_observed, x_time, bayesian_predict=True)
        return t.var(predictions).item()
    


"""
A class representing a probabilistic forecasting model. It fits basic statistical models for 
    both point forecasts and their squared errors.
"""
class Forecaster():
    def __init__(self, short_term_horizon:int=24) -> None:
        """
        Parameters:
        ------------
        """
        # Prophet-VAR model attributes
        self.point_prophet_model = None
        self.point_var_model = None
        self.error_prophet_model = None
        self.error_var_model = None
        self.error_trend = None

        # lstm related attributes
        self.short_term_horizon = short_term_horizon
        self.lstm = None
        self.dependent_variable = None
        self.predictor_variables = None



    """
    This method fits the forecasting model. It requires a clean dataset, a string dependent variable, and takes 
        a dictionary of optional hyperparameters.
    """
    def fit(self, clean_training_data:pd.DataFrame, dependent_variable:str, strong_predictors:list=[], 
        fit_pv:bool=True, fit_lstm:bool=True, hyperparameters:dict=dict(changepoint_prior_scale=0.001, 
        seasonality_prior_scale=0.01, point_var_lags=10, minimum_error_prediction=None, error_trend=1e-4, 
        lr=0.0005, dropout=0.1, batch_size=100, sequence_length=3*7*24, patience=4, loss_scalar=1e4), 
        lstm_device:str="cpu", verbose:bool=True, **kwargs):
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
        if fit_pv:
            # fit point forecasting Prophet models 
            if verbose: print("Fitting Prophet-VAR Model")
            residuals = self._fit_prophet_models(clean_training_data=clean_training_data, dependent_variable=dependent_variable, **hyperparameters)

            # fit point forecasting VAR model
            squared_errors = self._fit_point_var(residuals=residuals, dependent_variable=dependent_variable, strong_predictors=strong_predictors)

            # fit variance forecasting Prophet model
            self._fit_error_forecaster(dependent_variable=dependent_variable, squared_errors=squared_errors, **hyperparameters)
        
        if fit_lstm:
            # fit lstm model
            if verbose: print("Fitting LSTM Model")
            lstm = self.format_fit_lstm(clean_training_data, proportion_validation=0.1, lstm_device=lstm_device, verbose=verbose, **hyperparameters)


    """
    This is a private method that fits a short-term LSTM model using all variables in the dataset provided. The forecasting horizon that the LSTM 
        is trained to predict is a class attribute.
    """
    def _fit_lstm(self, clean_training_data:pd.DataFrame, dependent_variable:str, lr:float=0.0005, dropout=0.5,
        batch_size=100, sequence_length=3*7*24, patience=4, loss_scalar=1e4, **kwargs):
        """
        
        """
        pass
        self.lstm = None


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

        # Fit VAR model (ignoring warnings because I cannot figure out how to prevent a ValueWarning.)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # point_var_model = VAR(endogenous_data, freq=datetime.timedelta(hours=1))
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
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
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
    def long_term_predict(self, hours_ahead:int): # also want to add option to provide start time and previous observations so that the model can predict at any time.
        """
        Parameters:
        ----------
        hours_ahead (int): The number of probabilistic forecasts to make

        Returns:
        ----------
        pandas.Series: Predicted point forecasts (float) with time as index (datetime64[ns])

        pandas.Series: Predicted variance (float) for each point forecast with time as index (datetime64[ns])
        """
        # hours_ahead must be positive
        if (hours_ahead <= 0) or (hours_ahead % 1 > 0): raise SystemExit("The argument \"hours_ahead\" must be a positive integer.")
        
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

        return pd.DataFrame(index=point_prophet_forecasts.index, data={"Point Forecasts":point_prophet_forecasts + point_var_forecasts, 
            "Variance Forecasts": error_forecasts})
    

    """
    This method is used for making predictions on evaluation data while updating the model after a given number of time-steps. 
        In this way, we can specifically evaluate the model's ability to make forecasts N hours ahead, where N is a class attribute 
        defined during class instantiation. It is required that a LSTM model has been fit since the short-term forecasts rely heavily 
        on the LSTM. This method can be used to compare the performance of a fit Forecaster object with day-ahead forecasts from the EIA.
    """
    def short_term_predict(self, input_data:pd.DataFrame, expected_output_size:int, lstm_sequence_length:int=None, ensemble_weights:dict={"Prophet-VAR":0, "LSTM":1}):
        """
        Parameters:
        ----------
        input_data (pandas.DataFrame): a dataframe containing the same variables and datetime index as that used to fit the Prophet-VAR model and the LSTM. 
            Depending on the value of "pv_lstm_weights" passed, it may be necessary for this model to contain at least as many observations as the sequence 
            length used as input to this object's VAR model and LSTM. The order of columns in this dataframe must match the order of columns used to train the models.
        
        expected_output_size (int): The number of short-term predictions the user is expecting to obtain from this method. This argument is included as a 
            check that the user understands how short-term predictions are being made and how many initial observations are required before any forecasts can 
            start to be made.

        pv_lstm_weights (dict): determines how to weight the Prophet-VAR and LSTM models when using them as an ensemble to produce short-term predictions.
            If any model's weight is set to 0, it will not be used to make short-term predictions.
            
        Returns:
        ----------

        """
        # check that columns match what was used to train the models
        # print(list(pd.get_dummies(input_data, drop_first=True).columns))
        # print(([self.dependent_variable] + list(self.predictor_variables)))
        if list(pd.get_dummies(input_data, drop_first=True).columns) != ([self.dependent_variable] + list(self.predictor_variables)):
            raise SystemExit("The names and order of the columns provided in the input dataset do not match those used to train the LSTM model. " \
                "Please ensure that the input dataset matches that used to train the LSTM model.")

        # define datetime index for the predictions being made
        prediction_index = input_data.index[lstm_sequence_length+self.short_term_horizon:]

        # prediction placeholders
        ensemble_point_forecasts = 0
        ensemble_error_forecasts = 0

        # make predictions using the Prophet-VAR model 
        # HERE: TODO: Need to save all prophet models and principal components during PV fitting and create a pipeline that 
        # goes from clean input data to transformed residuals that can be used by the VAR model.
        if ensemble_weights["Prophet-VAR"] > 0:
            if self.point_var_model is None: raise AttributeError("Must fit a Prophet-VAR model before it can be used to make short-term predictions.")
            # predict using Prophet point forecaster
            prophet_df = pd.DataFrame({'ds': input_data.index})
            point_prophet_forecasts = self.point_prophet_model.predict(prophet_df)[["ds", "yhat"]].set_index("ds")
            point_prophet_forecasts = point_prophet_forecasts["yhat"].iloc[lstm_sequence_length:] # should match expected_output_size

            # predict with squared error forecasting Prophet model
            if self.error_prophet_model is not None:
                error_prophet_forecasts = self.error_prophet_model.predict(prophet_df)[["ds", "yhat"]].set_index("ds")
                error_prophet_forecasts = error_prophet_forecasts["yhat"].iloc[lstm_sequence_length:]
                error_forecasts = error_prophet_forecasts

            ### predict using VAR model ###

            # apply function to get transformed residuals for point forecasting VAR
            # point_var_df = self....

            # apply function to get transformed residuals for error forecasting VAR
            # error_var_df = self....

            # predict with point forecasting VAR model
            # if self.point_var_model is not None:
            #     point_var_forecasts = self.point_var_model.forecast(point_var_context, steps=self.short_term_horizon)[:,0]
            

            # predict with squared error forecasting VAR model
            # if self.error_var_model is not None:
            #     error_var_forecasts = self.error_var_model.forecast(error_var_context, steps=self.short_term_horizon)[:,0]
            #     error_forecasts = error_forecasts + error_var_forecasts

            error_forecasts = np.maximum(self.min_squared_error, error_forecasts) # set minimum squared error

            # add trend to error forecasts
            error_trend = np.cumsum(np.ones(error_forecasts.shape[0]) * self.error_trend)
            error_forecasts = error_forecasts + error_trend

            # return point_prophet_forecasts + point_var_forecasts, error_forecasts

            

        # make predictions using the LSTM model
        if ensemble_weights["LSTM"] > 0:
            if self.lstm is None: raise AttributeError("Must fit an LSTM model before it can be used to make short-term predictions.")

            # format input data for LSTM
            if lstm_sequence_length is None: lstm_sequence_length = self.lstm.training_sequence_length
            loader, _, _, _ = self.format_lstm_data(input_data, sequence_length=lstm_sequence_length, batch_size=input_data.shape[0]-lstm_sequence_length, 
                forecasting_steps_ahead=self.short_term_horizon, proportion_validation=0)
            with t.no_grad():
                inputs, time_inputs = [],[]
                for input, time_input, _ in loader:
                    inputs.append(input)
                    time_inputs.append(time_input)
                inputs = t.cat(inputs)
                time_inputs = t.cat(time_inputs)

                # put input onto same device as the model
                device = next(self.lstm.parameters()).device
                inputs = inputs.to(device=device)
                time_inputs = time_inputs.to(device=device)
                
                # make short-term predictions
                point_forecasts = (self.lstm(inputs, time_inputs, bayesian_predict=False)[:,0].cpu().numpy() * 
                        (self.lstm.input_scaling[1][self.dependent_variable]-self.lstm.input_scaling[0][self.dependent_variable]) + self.lstm.input_scaling[0][self.dependent_variable])

                # predict error variances
                error_forecasts = []
                for input, time_input in zip(inputs, time_inputs):
                    error_forecast = (self.lstm.predict_variance(input, time_input) * 
                    (self.lstm.input_scaling[1][self.dependent_variable]-self.lstm.input_scaling[0][self.dependent_variable]) + self.lstm.input_scaling[0][self.dependent_variable])
                    error_forecasts.append(error_forecast)
                error_forecasts = np.array(error_forecasts)
                # return error_forecasts

            # weight by ensemble weight
            ensemble_point_forecasts = ensemble_point_forecasts + ensemble_weights["LSTM"]*point_forecasts
            ensemble_error_forecasts = ensemble_error_forecasts + ensemble_weights["LSTM"]*error_forecasts

        
        # Divide forecasts by the sum of all ensemble weights (in case they do not sum to 1)
        weights_sum = sum(list(ensemble_weights.values()))
        ensemble_point_forecasts = ensemble_point_forecasts / weights_sum
        ensemble_error_forecasts = ensemble_error_forecasts / weights_sum

        return pd.DataFrame(index=prediction_index, data={"Point Forecast":ensemble_point_forecasts, "Variance Forecast":ensemble_error_forecasts})




    """
    This method takes data and a set of hyperparameters and conducts rolling cross-validation  on a Prophet-VAR ensemble using MSE and weighted MSE metrics. 
        This method does not cross validate an lstm model. See cross_validate_lstm() for that.
    """
    def cross_validate_pv(self, num_folds:int, clean_training_data:pd.DataFrame, dependent_variable:str, pv_hyperparameters:dict=None,
        strong_predictors:list[str]=[], return_predictions:bool=False):
        """
        Parameters:
        ----------
        num_folds (int): The number of rolling windows to use for cross-validation. More folds reduces variance of 
            the result but increases computational complexity.

        clean_training_data (pandas.DataFrame): a clean dataset (outliers removed, no missing values) to be used 
            for fitting the model parameters.
        
        dependent_variable (str): The name of the dependent variable that will be predicted.

        cv_hyperparameters (dict): a set of hyperparameters used to fit a Prophet-VAR combined model

        lstm_hyperparameters (dict): a set of hyperparamters used to fit a LSTM model

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
                strong_predictors=strong_predictors, hyperparameters=pv_hyperparameters, fit_lstm=False, verbose=False)
            
            forecasts = self.long_term_predict(test_data.shape[0])
            point_forecasts = forecasts["Point Forecasts"].values
            error_forecasts = forecasts["Variance Forecasts"].values

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
    This method takes a clean dataset and a set of hyperparameters and conducts rolling cross-validation of a LSTM network using the hyperparamters 
        provided and using MSE and weighted MSE metrics. This method does not cross validate an Prophet-VAR model. See cross_validate_pf() for that.
    """
    def cross_validate_lstm(self, num_folds:int, clean_training_data:pd.DataFrame, dependent_variable:str, 
        lstm_hyperparameters:dict=None, early_stopping_prop:float=0.1, device:str="cpu", strong_predictors:list[str]=[], 
        return_predictions:bool=False, verbose_validation:bool=False, verbose_training:bool=False):
        """
        
        """
        # initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=num_folds)

        if return_predictions:
            model_predictions=[]
            truth_values = []
        else:
            model_mse_values = []
            model_wmse_values = []

        counter=1
        for train_index, test_index in tscv.split(clean_training_data):
            if verbose_validation:
                print("Training Model for CV Fold {} out of {}.".format(counter, num_folds))
            train_data, test_data = clean_training_data.iloc[train_index], clean_training_data.iloc[test_index]

            # format training dataset
            train_loader, val_loader, input_scaling, input_time_scaling = self.format_lstm_data(train_data, sequence_length=lstm_hyperparameters["sequence_length"], 
                batch_size=lstm_hyperparameters["batch_size"], forecasting_steps_ahead=self.short_term_horizon, proportion_validation=early_stopping_prop)

            # format validation dataset
            validation_last_train_index = -(lstm_hyperparameters["sequence_length"]+self.short_term_horizon)
            temp_test_data = pd.concat([train_data[validation_last_train_index:], test_data], axis=0) # add last sequence length of the training data to start of validation data
            # return test_data
            _, validation_loader, _, _  = self.format_lstm_data(temp_test_data, sequence_length=lstm_hyperparameters["sequence_length"], batch_size=lstm_hyperparameters["batch_size"], 
                forecasting_steps_ahead=self.short_term_horizon, proportion_validation=1, input_scaling=input_scaling, input_time_scaling=input_time_scaling)
            
            # define input size
            input_size = train_loader.dataset[0][0].shape[-1] + train_loader.dataset[0][1].shape[-1]

            # define lstm model to be fit/trained
            model = LSTM(input_size, lstm_hyperparameters["hidden_size"], lstm_hyperparameters["num_layers"], 1, lstm_hyperparameters["sequence_length"], dropout=lstm_hyperparameters["dropout"])
            # return train_loader, val_loader
            
            # fit the lstm using the data and hyperparameters provided
            model = self.fit_lstm(model=model, train_loader=train_loader, val_loader=val_loader, device=device, **lstm_hyperparameters, verbose=verbose_training, overwrite_class_model=False, 
                input_scaling=input_scaling, input_time_scaling=input_time_scaling)
            
            with t.no_grad():
                # obtain validation predictions
                validation_inputs, validation_time_inputs, validation_targets = [],[],[]
                for inputs, time_inputs, targets in validation_loader:
                    validation_inputs.append(inputs)
                    validation_time_inputs.append(time_inputs)
                    validation_targets.append(targets)
                validation_inputs = t.cat(validation_inputs)
                validation_time_inputs = t.cat(validation_time_inputs)
                validation_targets = t.cat(validation_targets)

                # return model, validation_inputs, validation_time_inputs, validation_targets
                
                # predict point forecast values
                point_forecasts = (model(validation_inputs, validation_time_inputs)[:,0].cpu().numpy() * 
                    (input_scaling[1][dependent_variable]-input_scaling[0][dependent_variable]) + input_scaling[0][dependent_variable])

                # predict error variances
                error_forecasts = []
                for input, time_input in zip(validation_inputs, validation_time_inputs):
                    error_forecast = (model.predict_variance(input, time_input) * 
                        (input_scaling[1][dependent_variable]-input_scaling[0][dependent_variable]) + input_scaling[0][dependent_variable])
                    error_forecasts.append(error_forecast)
                error_forecasts = np.array(error_forecasts)

                if return_predictions:
                    model_predictions.append([point_forecasts, error_forecasts])
                    truth_values.append(validation_targets.cpu().numpy())
                
                else:
                    # calculate MSE
                    mse = np.sum((point_forecasts - (test_data[dependent_variable].values))**2)/test_data.shape[0]

                    # calculate weighted MSE
                    relative_confidence = 1/np.sqrt(error_forecasts) # weight by 1 over standard deviation
                    relative_confidence = relative_confidence/np.sum(relative_confidence) * error_forecasts.shape[0] # normalize
                    # return model, relative_confidence, point_forecasts, test_data
                    wmse = np.sum(relative_confidence * (point_forecasts - (test_data[dependent_variable].values))**2)/test_data.shape[0]

                    model_mse_values.append(mse)
                    model_wmse_values.append(wmse)
                if verbose_validation:
                    print("Validation Fold {} out of {}. MSE: {}, WMSE: {}.".format(counter, num_folds, mse, wmse))

                counter = counter + 1

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
        pv_hyperparameter_sets:list[dict]=None, lstm_hyperparameter_sets:list[dict]=None, num_cv_folds:int=5, 
        previous_pv_results:pd.DataFrame=None, previous_lstm_results:pd.DataFrame=None, strong_predictors:list[str]=[], 
        verbose:bool=True, **kwargs):
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
        # tune hyperparameters for Prophet-VAR model
        if pv_hyperparameter_sets is None: pv_hyperparameter_results = None
        else: 
            # HERE create new dataframe to hold results unless dataframe is provided, then use that...
            if verbose: print("Tuning Prophet-VAR Model using hyperparameter sets provided.")
            pv_hyperparameter_results = []
            for set in pv_hyperparameter_sets: # for each set of hyperparameters, perform cross validation
                mse, wmse = self.cross_validate_pv(num_cv_folds, clean_training_data, dependent_variable, pv_hyperparameters=set, 
                    strong_predictors=strong_predictors, **kwargs)
                if verbose: print("Hyperparameters: {}; MSE: {:.3f}; WMSE: {:.3f}".format(set, mse, wmse))
                pv_hyperparameter_results.append([set, mse, wmse])

        # tune hyperparameters for Prophet-VAR model
        if lstm_hyperparameter_sets is None: lstm_hyperparameter_results = None
        else: 
            if verbose: print("Tuning LSTM Model using hyperparameter sets provided.")
            lstm_hyperparameter_results = []
            for set in lstm_hyperparameter_sets: # for each set of hyperparameters, perform cross validation
                mse, wmse = self.cross_validate_lstm(num_cv_folds, clean_training_data, dependent_variable, lstm_hyperparameters=set, 
                    strong_predictors=strong_predictors, **kwargs)
                if verbose: print("Hyperparameters: {}; MSE: {:.3f}; WMSE: {:.3f}".format(set, mse, wmse))
                lstm_hyperparameter_results.append([set, mse, wmse])

        # return results as a list of tuples where each tuple contains the hyperparameter dict, mse, and wmse
        return pv_hyperparameter_results, lstm_hyperparameter_results


    """
    This method is used to format a dataset for lstm training. It takes a pandas dataframe with columns representing the variables to be included in the model. 
        It then converts the dataset into a Pytorch DataLoader as an iterable of B x S x (K+T) Tensors, where B is the batch size, S is the sequences size (lookback window), 
        K is the number of variables in the original dataset, and T is the number of added time encodings (current hard coded to 3). It returns DataLoaders and the 
        normalization factors needed to reproduce the original data.
    """
    def format_lstm_data(self, clean_data:pd.DataFrame, sequence_length:int, batch_size:int, proportion_validation:float=0, 
        input_scaling:tuple=None, input_time_scaling:tuple=None, **kwargs):
        """
        Parameters:
        df (pd.DataFrame): a dataframe with all variables to be included in the model. Should not contain time embeddings. Must have a datetime-like index.
        
        sequence_length
        """
        if not pd.api.types.is_datetime64_ns_dtype(clean_data.index.dtype):
            clean_data.index = pd.to_datetime(clean_data.index)

        # encode hour of the day, day of the week, and day of the year into a new dataframe
        input_time = pd.DataFrame(data={"Hour of Day":clean_data.index.hour, "Day of Week":clean_data.index.dayofweek, "Day of Year":clean_data.index.dayofyear}, index=clean_data.index)
        input_data = pd.get_dummies(clean_data, drop_first=True).astype("float32")

        # calculate normalization factors. Also save these as they will be used later to transform output back to actual values.
        if input_scaling is None:
            input_min_vals = np.min(input_data, axis=0) 
            input_max_vals = np.max(input_data, axis=0)
        else:
            input_min_vals = input_scaling[0]
            input_max_vals = input_scaling[1]
        if input_time_scaling is None:
            input_time_min_vals = np.min(input_time, axis=0)
            input_time_max_vals = np.max(input_time, axis=0)
        else:
            input_time_min_vals = input_time_scaling[0]
            input_time_max_vals = input_time_scaling[1]

        # Apply normalization to put all variables in the range [0, 1]. This helps the model learn equally from them all.
        input_data = (input_data - input_min_vals) / (input_max_vals - input_min_vals)
        input_time = (input_time - input_time_min_vals) / (input_time_max_vals - input_time_min_vals)

        x = input_data.values
        x_time = input_time.values

        # format the dataset
        K = input_data.shape[1]  # Number of features

        # Calculate the number of sequences of length S that can be produced
        num_sequences = x.shape[0] - (sequence_length + self.short_term_horizon)

        # Initialize an empty list to store the groups
        x_inputs = []
        x_time_inputs = []
        y_outputs = []

        # Iterate over the array to create groups
        for i in range(num_sequences):
            input = x[i:i+sequence_length]
            time_input = x_time[i:i+sequence_length]
            output = x[i+sequence_length+self.short_term_horizon-1,0]
            x_inputs.append(input)
            y_outputs.append(output)
            x_time_inputs.append(time_input)

        # concatenate sequences into a single tensor dataset
        x_inputs = t.Tensor(np.array(x_inputs))
        x_time_inputs = t.Tensor(np.array(x_time_inputs))
        y_outputs = t.Tensor(np.array(y_outputs))

        if proportion_validation > 0: 
            validation_size = int(np.floor(x_inputs.shape[0]*proportion_validation))
            # define train_loader for training the model
            train_dataset = t.utils.data.TensorDataset(x_inputs[:-validation_size], x_time_inputs[:-validation_size], y_outputs[:-validation_size])
            train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            # define validation_loader for early stopping validation evaluations
            validation_dataset = t.utils.data.TensorDataset(x_inputs[-validation_size:], x_time_inputs[-validation_size:], y_outputs[-validation_size:])
            validation_loader = t.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataset = t.utils.data.TensorDataset(x_inputs, x_time_inputs, y_outputs)
            train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            validation_loader = None

        return train_loader, validation_loader, (input_min_vals, input_max_vals), (input_time_min_vals, input_time_max_vals)
    

    """
    
    """
    def fit_lstm(self, model:LSTM, train_loader:t.utils.data.DataLoader, val_loader:t.utils.data.DataLoader=None, lr:float=0.0005, dropout:float=0.1, 
        patience:int=5, weight_decay:float=0, verbose:bool=False, loss_scalar:int=1000, max_epochs:int=100, overwrite_class_model:bool=True, 
        input_scaling=None, input_time_scaling=None, **kwargs):
        # copy model to avoid overwriting
        model = deepcopy(model)

        # define pytorch device as whatever device the model in on
        device = next(model.parameters()).device

        # define optimizer
        criterion = t.nn.MSELoss()
        optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if val_loader is not None: # using early stopping validation
            best_val_loss = np.inf
            best_model_state = None
            counter = 0
        for epoch in range(max_epochs):
            losses = []
            for i, (inputs, time_inputs, targets) in enumerate(train_loader):
                # put all batches on the correct device
                inputs = inputs.to(device=device)
                time_inputs = time_inputs.to(device=device)
                targets = targets.to(device=device)

                # evaluate model
                outputs = model(inputs, time_inputs, bayesian_predict=True)[:,0]

                # apply autograd
                optimizer.zero_grad()
                loss = criterion(targets, outputs) * loss_scalar
                losses.append(loss.item())
                loss.backward()

                # Print summary statistics of model parameters and gradients
                if verbose:
                    if (i == 0):
                        params = []
                        gradients = []
                        for name, param in model.named_parameters():
                            params.append(param.view(-1).detach())
                            if param.grad is not None:
                                gradients.append(param.grad.view(-1).detach())
                        params = t.abs(t.cat(params))
                        gradients = t.abs(t.cat(gradients))
                        print("Average Parameter Absolute Value: {:.5f}".format(np.mean(params.cpu().numpy())))
                        print("Average Gradient Absolute Value: {:.5f}".format(np.mean(gradients.cpu().numpy())))

                optimizer.step()
            losses = np.mean(losses)
            if verbose:
                print("Epoch {}, Training Loss: {}".format(epoch+1, losses.item()))

            if val_loader is not None:
                # here implement early stopping using validation data
                with t.no_grad():
                    val_loss = []
                    for i, (inputs, time_inputs, targets) in enumerate(val_loader):
                        inputs = inputs.to(device=device)
                        time_inputs = time_inputs.to(device=device)
                        targets = targets.to(device=device)
                        outputs = model(inputs, time_inputs, bayesian_predict=False)[:,0]
                        loss = criterion(targets, outputs) * loss_scalar
                        val_loss.append(loss.item())
                    val_loss = np.mean(val_loss)
                    if verbose:
                        print("Epoch {}, Validation Loss: {}".format(epoch+1, val_loss.item()))

                # Check for improvement in validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # Save the best model state
                    best_model_state = model.state_dict()
                else:
                    counter += 1
                    if counter >= patience:
                        if verbose: print("Early stopping!")
                        break

        # Restore the best performing model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # save scaling information to the model
        model.input_scaling = input_scaling
        model.input_time_scaling = input_time_scaling

        # save model to Forecaster class
        if overwrite_class_model:
            self.lstm = model
            self.dependent_variable = input_scaling[0].index[0]
            self.predictor_variables = input_scaling[0].index[1:]

        return model

    """
    
    """
    def format_fit_lstm(self, clean_data, sequence_length:int=10, batch_size:int=10, max_epochs:int=5, proportion_validation=0.1, lstm_device:str="cpu", overwrite_class_model:bool=True, verbose:bool=True, **kwargs):
        # format the data for lstm training
        train_loader, val_loader, input_scaling, input_time_scaling = self.format_lstm_data(clean_data, sequence_length, batch_size, proportion_validation, **kwargs)

        # define the LSTM model
        lstm = LSTM(input_size=train_loader.dataset[0][0].shape[-1]+train_loader.dataset[0][1].shape[-1], output_size=1, training_sequence_length=sequence_length, **kwargs).to(device=lstm_device)

        # fit the LSTM model
        self.fit_lstm(lstm, train_loader=train_loader, val_loader=val_loader, input_scaling=input_scaling, input_time_scaling=input_time_scaling, max_epochs=max_epochs,
            device=lstm_device, overwrite_class_model=overwrite_class_model, verbose=verbose, **kwargs)

        return lstm


