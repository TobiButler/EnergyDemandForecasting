import pandas as pd
import numpy as np
import os
import datetime
import get_data as gd
import yaml
from termcolor import colored


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
        # HERE: Taking too long. Need to add manual check of some kind.
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

        # using non-date-like start and end arguments
        print("Running Data Collection Test 4.")
        try:
            df = gd.main(1, 2, eia_api_key, save_dataset=False, verbose=False)
        except SystemExit as e: 
            print(colored('Data Collection Test 4 Passed.', 'green'))
        except BaseException as e:
            print(e)
            print(colored('Data Collection Test 4 Failed. The non-datetime start and end arguments did not raise a TypeError.', 'red'))


    ### unit tests for data processing component ###
    if test_data_processing: 
        pass

        # load preliminary dataset
        preliminary_dataset = pd.read_csv(r"Saved/Datasets/preliminary.csv", index_col=0)

        # insert outliers into the preliminary dataset


        # insert missing values into the preliminary dataset

        # run the data processing component

        # check that outliers have been replaced

        # check that the missing values have been replaced


        # unit test for missing values


    ### unit tests for Forecasting model functionality ###

    # define model

    # fit model using dataset from repository

    # make short-term forecasts, check format

    # make long-term forecasts, check format


def test_forecaster(): pass


def test_data_processing_df(df, test_int:int): pass


def test_data_collection_df(df, test_int:int, start_date, end_date):
    if len(df.columns) != 18: test_error = colored(f'Data Collection Test{test_int} Failed. There are not 18 columns present in the downloaded dataset.', 'red')
    elif not pd.api.types.is_datetime64_any_dtype(df.index): test_error = colored(f'Data Collection Test {test_int} Failed. Index was not correctly converted to datetime.', 'red')
    elif df.index.min() < start_date: test_error = colored(f'Data Collection Test {test_int} Failed. The dataset contains observations from outside the requested window.', 'red')
    elif df.index.max() > end_date: test_error = colored(f'Data Collection Test {test_int} Failed. The dataset contains observations from outside the requested window.', 'red')
    else: test_error = colored('Data Collection Test {} Passed.'.format(test_int), 'green')
    return test_error
    

if __name__ == "__main__":
    main()