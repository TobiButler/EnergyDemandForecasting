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
def main():
    """
    
    """

    # open the api yaml configuration file
    with open("Configuration Files/api_keys.yml", 'r') as file:
        data = yaml.safe_load(file)
        eia_api_key = data["eia"]

    print(colored('world', 'green'))
    # using random start date and random end date, run all data collection methods and check that they return non-empty dataframes
    start_date = np.datetime64('today') - np.timedelta64(np.random.randint(0, 10000), 'D')
    end_date = start_date + np.timedelta64(np.random.randint(0, 10000), 'D')
    df = gd.main(start_date, end_date, eia_api_key, save_dataset=False)
    print(colored('Data Collection Test 1 Passed.', 'green'))
    # print("Data Collection Test 1 Passed.", colo)

    # using too early of a start date and too late of an end date
    start_date = np.datetime64("1970-01-01")
    end_date = np.datetime64("2030-01-01")
    df = gd.main(start_date, end_date, eia_api_key, save_dataset=False)

    # using start date after end date
    start_date = np.datetime64('today') - np.timedelta64(np.random.randint(0, 10000), 'D')
    end_date = start_date - np.timedelta64(np.random.randint(0, 10000), 'D')
    df = gd.main(start_date, end_date, eia_api_key, save_dataset=False)

    # unit test for outliers


    # unit test for missing values


    # unit test for Forecasting model functionality





if __name__ == "__main__":
    main()