import pandas as pd
import numpy as np
import get_data as gd
import preprocessing as pp
import model_definitions as md
import yaml
from termcolor import colored
import random


"""
Conducts a series of unit tests that should run without producing errors. These tests should be evaluated everytime a pull-request is made.
"""
def main(test_data_collection:bool=True, test_data_processing:bool=True, test_forecasting_model:bool=True):
    """
    
    """
    ### unit tests for data collection component ###
    if test_data_collection:
        # open the api yaml configuration file
        with open("Configuration Files/api_keys.yml", 'r') as file:
            data = yaml.safe_load(file)
            eia_api_key = data["eia"]

        # test using random start date and random end date, run all data collection methods and check that they return non-empty dataframes
        print("Running Data Collection Test 1.")
        try:
            start_date = (np.datetime64('today') - np.timedelta64(np.random.randint(0, 10000), 'D'))
            end_date = start_date + np.timedelta64(np.random.randint(0, 10000), 'D')
            df = gd.main(start_date, end_date, eia_api_key, save_dataset=False, verbose=False)
            print(test_data_collection_df(df, 1, start_date=start_date, end_date=end_date))
        except SystemExit as e: 
            print(colored('Data Collection Test 1 Passed.', 'green'))
        except BaseException as e:
            print(e)
            print(colored('Data Collection Test 1 Failed. An error occurred.', 'red'))

        # test using too early of a start date and too late of an end date
        print("Running Data Collection Test 2.")
        start_date = np.datetime64("1970-01-01")
        end_date = np.datetime64("2030-01-01")
        df = gd.main(start_date, end_date, eia_api_key, save_dataset=False, verbose=False)
        print(test_data_collection_df(df, 2, start_date=start_date, end_date=end_date))

        # test using start date after end date
        print("Running Data Collection Test 3.")
        try:
            start_date = np.datetime64('today') - np.timedelta64(np.random.randint(0, 10000), 'D')
            end_date = start_date - np.timedelta64(np.random.randint(0, 10000), 'D')
            df = gd.main(start_date, end_date, eia_api_key, save_dataset=False, verbose=False)
        except SystemExit as e:
            print(colored('Data Collection Test 3 Passed.', 'green'))
        except BaseException as e:
            print(e)
            print(colored('Data Collection Test 3 Failed. An error occurred.', 'red'))

        # test using non-date-like start and end arguments
        print("Running Data Collection Test 4.")
        try:
            df = gd.main(1, 2, eia_api_key, save_dataset=False, verbose=False)
        except SystemExit as e: 
            print(colored('Data Collection Test 4 Passed.', 'green'))
        except BaseException as e:
            print(type(e), e)
            print(colored('Data Collection Test 4 Failed.', 'red'))


    ### unit tests for data processing component ###
    if test_data_processing: 
        # load preliminary dataset
        print("Loading Preliminary Data for Data Processing Tests")
        preliminary_dataset = pd.read_csv(r"Saved/Datasets/preliminary.csv", index_col=0)

        # insert outliers into the preliminary dataset
        means = preliminary_dataset.mean(axis=0)
        stds = preliminary_dataset.std(axis=0)
        outlier_rows = np.random.randint(0, preliminary_dataset.shape[0], 100)
        outlier_cols = np.random.randint(0, preliminary_dataset.shape[1], 100)
        outlier_indices = list(zip(outlier_rows, outlier_cols))
        oov = [preliminary_dataset.iloc[x] for x in outlier_indices] # original outliers values
        for x in outlier_indices: 
            sign = -1 ** (random.choice([1,2]))
            preliminary_dataset.iloc[x] = means.iloc[x[1]] + (sign * stds.iloc[x[1]] * 1000)

        # insert missing values into the preliminary dataset
        missing_values_rows = np.random.randint(0, preliminary_dataset.shape[0], 100)
        missing_values_cols = np.random.randint(0, preliminary_dataset.shape[1], 100)
        missing_values_indices = list(zip(missing_values_rows, missing_values_cols))
        omv = [preliminary_dataset.iloc[x] for x in missing_values_indices] # original missing values
        for x in missing_values_indices: preliminary_dataset.iloc[x] = np.nan

        # run the data processing component
        print("Running Data Processing Tests 1 & 2.")
        processor = pp.PreprocessingPipeline(save_datasets=False, produce_eda_plots=False)
        clean_data, _ = processor.process_dataset(preliminary_dataset, verbose=False)

        # check that outliers have been replaced
        nov = [clean_data.iloc[x] for x in outlier_indices] # new outliers values
        if sum((np.array(oov)==np.array(nov))) == 0:
            print(colored("Data Processing Test 1 Passed. Outliers have been identified and replaced.", "green"))
        else:
            print(colored("Data Processing Test 1 Failed. Outliers still remain in the dataset after processing.", "red"))

        # check that the missing values have been replaced
        nmv = [clean_data.iloc[x] for x in outlier_indices] # new missing values
        if sum((np.array(omv)==np.array(nmv))) == 0:
            print(colored("Data Processing Test 2 Passed. Missing values have been imputed.", "green"))
        else:
            print(colored("Data Processing Test 2 Failed. Missing values still remain in the dataaset after processing.", "red"))

        # test that the ProcessingPipeline can handle input of the wrong datatype
        print("Running Data Processing Test 3.")
        try:
            processor.process_dataset(1,2,3,4)
        except SystemExit as e: 
            print(colored("Data Processing Test 3 Passed.", "green"))
        except BaseException as e:
            print(type(e), e)
            print(colored('Data Processing Test 3 Failed. The incorrect input datatypes were not handeled correctly.', 'red'))


    ### unit tests for Forecasting model functionality ###
    if test_forecasting_model:

        # load in clean dataset
        clean_data = pd.read_csv(r"Saved/Datasets/clean_training.csv", index_col=0)
        clean_data.index = pd.to_datetime(clean_data.index)
        clean_data.loc[:,"HourlyPrecipitation"] = clean_data["HourlyPrecipitation"].replace({np.nan:"None"})

        # define model
        f = md.Forecaster()

        # fit model
        try:
            sequence_length = 24
            hyperparameters = {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01, 'point_var_lags': 10, 'minimum_error_prediction': 100, 
                'error_trend': 100, "batch_size":100, "lr":0.0005, "dropout":0.1, "num_layers":1, "hidden_size":16, "sequence_length":sequence_length, "max_epochs":5}
            f.fit(clean_training_data=clean_data, dependent_variable="Energy Demand (MWH)", hyperparameters=hyperparameters, lstm_device="cuda", verbose=False)
            print(colored("Forecasting Model Test 1 Passed.", "green"))
        except BaseException as e:
            print(type(e), e)
            print(colored('Forecasting Model Test 1 Failed. The model ran into an error while being fit.', 'red'))

        # make short-term forecasts, check format
        short_term_forecasts = f.short_term_predict(clean_data, expected_output_size=clean_data.shape[0]-f.short_term_horizon-sequence_length, 
            lstm_sequence_length=sequence_length)
        if short_term_forecasts.shape == (clean_data.shape[0]-f.short_term_horizon-sequence_length, 2):
            print(colored("Forecasting Model Test 2 Passed.", "green"))
        else:
            print(colored('Forecasting Model Test 2 Failed. The model ran into an error while making short-term predictions.', 'red'))

        # test using incorrect input for predicting short-term forecasts
        try:
            short_term_forecasts = f.short_term_predict(clean_data.iloc[:100, 0], expected_output_size=clean_data.shape[0]-f.short_term_horizon-sequence_length, 
                lstm_sequence_length=sequence_length)
            short_term_forecasts = f.short_term_predict(clean_data, expected_output_size=0, 
                lstm_sequence_length=sequence_length)
        except SystemExit:
            print(colored("Forecasting Model Test 3 Passed.", "green"))
        except BaseException as e:
            print(type(e), e)
            print(colored('Forecasting Model Test 3 Failed. The model was unable to handle badly formatted input.', 'red'))

        # make long-term forecasts, check format
        n = 1000
        long_term_forecasts = f.long_term_predict(hours_ahead=n)
        if long_term_forecasts.shape == (n, 2):
            print(colored("Forecasting Model Test 4 Passed.", "green"))
        else:
            print(colored('Forecasting Model Test 4 Failed. The model ran into an error while making long-term predictions.', 'red'))

        # test using incorrect input for predicting long-term forecasts
        try:
            long_term_forecasts = f.long_term_predict(hours_ahead=7.5)
            long_term_forecasts = f.long_term_predict(hours_ahead=-1)
        except SystemExit:
            print(colored("Forecasting Model Test 5 Passed.", "green"))
        except BaseException as e:
            print(type(e), e)
            print(colored('Forecasting Model Test 5 Failed. The model was unable to handle badly formatted input.', 'red'))



### Helper Functions ###

def test_data_collection_df(df, test_int:int, start_date, end_date):
    if len(df.columns) != 18: test_error = colored(f'Data Collection Test{test_int} Failed. There are not 18 columns present in the downloaded dataset.', 'red')
    elif not pd.api.types.is_datetime64_any_dtype(df.index): test_error = colored(f'Data Collection Test {test_int} Failed. Index was not correctly converted to datetime.', 'red')
    elif df.index.min() < start_date: test_error = colored(f'Data Collection Test {test_int} Failed. The dataset contains observations from outside the requested window.', 'red')
    elif df.index.max() > end_date: test_error = colored(f'Data Collection Test {test_int} Failed. The dataset contains observations from outside the requested window.', 'red')
    else: test_error = colored('Data Collection Test {} Passed.'.format(test_int), 'green')
    return test_error
    

if __name__ == "__main__":
    main(test_data_collection=True, test_data_processing=True, test_forecasting_model=True)