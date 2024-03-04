# created by Tobias Butler 
# Last Modified: 03/01/2024
"""
Description: This module contains functionality to gather hourly residential energy demand data local to New York City 
    from the U.S. Energy Information Agency (EIA), hourly weather-related data local to Central Park weather station from 
    the U.S. National Oceanic and Atmospheric Agency (NOAA), monthly energy price data local to New York State from the EIA, 
    and monthly economic indices local to New York City from the U.S. Bureau of Labor Statistics (BLS), and combine them all 
    into a single dataset with an hourly timescale.
"""

# import third party libraries
import pandas as pd
import numpy as np
import requests
import time
from copy import deepcopy
import json
import yaml
import sys
import os

"""
This main function takes starting date and ending date and retrieves data related to energy demand between those dates.
"""
def main(start:str, end:str, eia_api_key:str=None, path_to_directory:str=None):
    # create directory to hold datasets
    if path_to_directory is None: path_to_directory = r"Saved"
    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)
        subdir_path = os.path.join(path_to_directory, r"Datasets")
        os.makedirs(subdir_path)

    # convert start and end arguments to datetimes
    start = np.datetime64(start)
    end = np.datetime64(end)

    all_data = combine_data(start=start, end=end, eia_api_key=eia_api_key)

    path_to_save = r"{}/Datasets/preliminary.csv".format(path_to_directory)
    print("All data loaded successfully. Saving preliminary dataset to {}".format(path_to_save))
    all_data.to_csv(path_to_save)
    print("Preliminary dataset saved.")


"""
This function collects all hourly energy demand data for NY city between the start and end dates provided. 
    It requires an API key for the HTTP Get requests. The EIA has a 5000 value limit, so this function calls a 
    helper function if there are more than 5000 periods between the start and end dates provided.
"""
def get_energy_demand_data(start:np.datetime64, end:np.datetime64, eia_api_key:str=None, medrl:int=5000, **kwargs):
    """
    Paramters
    ----------
    start (numpy.datetime64): The starting date for the data to be retrieved

    end (numpy.datetime64): The ending date for the data to be retrieved

    eia_api_key (str): An api key obtained from the EIA website. By default, this is set to None and a key is looked for 
        in a project directory called Configuration Files.

    medrl (maximum energy demand request length) (int): The maximum number of values to request from the EIA per request. 
        The default value is 5000 because that is the uppermost limit set by the EIA API

    Returns
    ----------
    pandas.DataFrame: A dataframe containing hourly energy demand observations for New York City between the provided 
        start date and end date.
    """
    energy_demand = []
    num_days_requested = (end - start).astype("timedelta64[D]").astype(int)
    num_days_per_request = medrl//24
    num_requests = int(np.ceil(num_days_requested / num_days_per_request))
    print("Requesting energy demand data from EIA over {} requests".format(num_requests))

    # get api_key for eia
    if eia_api_key is None:
        with open("Configuration Files/api_keys.yml", "r") as file:
            eia_api_key = yaml.safe_load(file)["eia"]
    
    # request data from EIA in batches
    eia_start = deepcopy(start)
    while eia_start < end:
        eia_end = eia_start + np.timedelta64(num_days_per_request, 'D')
        
        if eia_end > end:
            df = repeated_energy_demand_request(eia_start, end, eia_api_key)
            energy_demand.append(df)
            break
        else:
            df = repeated_energy_demand_request(eia_start, eia_end, eia_api_key)
            energy_demand.append(df)
        eia_start = eia_end

        # wait for 5 seconds before continuing requests (to avoid being cutoff by API)
        time.sleep(5)

    energy_demand_data = pd.concat(energy_demand, axis=0)
    energy_demand_data = energy_demand_data[~energy_demand_data.index.duplicated(keep="first")]
    return energy_demand_data


"""
This function requests hourly energy demand data for NY city from the EIA web api. This is a helper function designed 
    to be called repeatedly since only 5000 values can be obtained from the EIA per request.
"""
def repeated_energy_demand_request(start:np.datetime64, end:np.datetime64, eia_api_key:str) -> pd.DataFrame:
    # convert start and end datetimes into hourly strings
    start = np.datetime_as_string(start, unit="h")
    end = np.datetime_as_string(end, unit="h")
    
    # define GET request
    request_url = fr"https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/?api_key={eia_api_key}" \
        fr"&frequency=hourly&data[0]=value&facets[parent][]=NYIS&facets[subba][]=ZONJ&" \
        fr"start={start}&end={end}&sort[0][column]=period&sort[0][direction]=asc&offset=0" \

    r = requests.get(request_url)
    energy_demand = r.json()["response"]["data"]
    energy_demand = pd.json_normalize(energy_demand)[["period", "value"]].rename(
        columns={"period":"Time", "value":"Energy Demand (MWH)"}).set_index(["Time"])
    return energy_demand


"""
This function requests monthly energy price data from the EIA web api for the state of New York
"""
def get_energy_price_data(start:np.datetime64, end:np.datetime64, eia_api_key:str=None) -> pd.DataFrame:
    """
    Paramters
    ----------
    start (numpy.datetime64): The starting date for the data to be retrieved

    end (numpy.datetime64): The ending date for the data to be retrieved

    eia_api_key (str): An api key obtained from the EIA website. By default, this is set to None and a key is 
        looked for in a project directory called Configuration Files.

    Returns
    ----------
    pandas.DataFrame: A dataframe containing monthly energy prices for residential energy in New York City between 
        the provided start date and end date.
    """
    # convert start and end datetimes into monthly strings
    start = np.datetime_as_string(start, unit="M")
    end = np.datetime_as_string(end, unit="M")

    # get api_key for eia
    if eia_api_key is None:
        with open("Configuration Files/api_keys.yml", "r") as file:
            eia_api_key = yaml.safe_load(file)["eia"]

    # define GET request
    print("Requesting energy price data from EIA")
    request_url = fr"https://api.eia.gov/v2/electricity/retail-sales/data/?api_key={eia_api_key}" \
        fr"&frequency=monthly&data[0]=price&facets[stateid][]=NY&facets[sectorid][]=RES&start={start}" \
        fr"&end={end}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"
    r = requests.get(request_url)
    energy_price = r.json()["response"]["data"]
    energy_price = pd.json_normalize(energy_price)[["period", "price"]].rename(columns={"period":"Time", 
        "price":"Energy Price (cents/KWH)"}).set_index(["Time"])
    return energy_price


"""
This function loads hourly historical weather data for NY city (based on the weather station at central park) 
    between the start and end times provided.
"""
def get_weather_data(start:np.datetime64, end:np.datetime64):
    """
    Paramters
    ----------
    start (numpy.datetime64): The starting time for the data to be retrieved

    end (numpy.datetime64): The ending time for the data to be retrieved

    Returns
    ----------
    pandas.DataFrame: A dataframe containing hourly weather-related observations for New York City between the 
        provided start and end times.
    """
    # define columns to keep from loaded csv files
    data_columns = [
        "DATE",
        "REPORT_TYPE",
        "SOURCE",
        "HourlyDewPointTemperature",
        "HourlyDryBulbTemperature",
        "HourlyPrecipitation",
        "HourlyRelativeHumidity",
        "HourlyStationPressure",
        "HourlyWetBulbTemperature",
        "HourlyWindSpeed"
    ]

    # download data from NOAA
    print("Requesting Weather Data from NOAA")
    start_year = np.datetime64(start, 'Y').astype(int) + 1970
    end_year = np.datetime64(end, 'Y').astype(int) + 1970
    dfs = []
    for year in range(start_year, end_year+1): # downloads one year's historical weather data one at a time
        df = pd.read_csv("https://www.ncei.noaa.gov/data/local-climatological-data/access/{}/72505394728.csv".format(year), 
            low_memory=False)[data_columns]
        dfs.append(df)
        time.sleep(1) # take a small break to avoid exceeding request limit
    weather_data = pd.concat(dfs, axis=0) # combine data from all years

    # filter downloaded data for correct values
    weather_data = weather_data[(weather_data["REPORT_TYPE"]=="FM-15") & (weather_data["SOURCE"]==7)].drop(
        columns={"REPORT_TYPE", "SOURCE"})
    
    # remove 's' characters from data values. I think they indicate "scaled" data points but need to confirm this.
    weather_data = weather_data.replace({"s":""}, regex=True)
    # '*' and 'T' values seem to represent missing data points
    weather_data = weather_data.replace({"T":np.nan, "*":np.nan})

    # filter for the start and end arguments provided
    weather_data["DATE"] = pd.to_datetime(weather_data["DATE"])
    keep_mask = (weather_data["DATE"] > start) & (weather_data["DATE"] < end)
    weather_data = weather_data[keep_mask]

    # format index for joining with other data
    weather_data["DATE"] = weather_data["DATE"].dt.strftime("%Y-%m-%dT%H")
    weather_data = weather_data.rename(columns={"DATE":"Time"}).set_index("Time")

    return weather_data


"""
This function requests several monthly economic indicators from the beruea of labor statistics (BLS) web api. 
    Names for each of the indicators are defined in the function.
"""
def get_bls_data(start_year:int, end_year:int):
    """
    Paramters
    ----------
    start_year (numpy.datetime64): The first year for data to be retrieved

    end_year (numpy.datetime64): The last year for data to be retrieved. Data will be retrieved for all months in and 
        between start_year and end_year

    Returns
    ----------
    pandas.DataFrame: A dataframe containing monthly economic-related indicator values for New York City between the 
        provided start and end years.
    """
    variable_names = {"CUURS12ASA0":"CPI-U", "LAUCT365100000000003":"Unemployment Rate", "LAUCT365100000000004":"Unemployment Level", 
        "LAUCT365100000000005":"Employment Level", "LAUCT365100000000006":"Labor Force Level","LAUCT365100000000007":"Employment Population Ratio",  
        "LAUCT365100000000008":"Labor Force Participation", "LAUCT365100000000009":"Civilian Noninstitutional Population", 
        "SMU36935610000000001":"Payroll Employment"}

    # define GET Request (really using POST)
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": ['CUURS12ASA0', 'LAUCT365100000000003', 'LAUCT365100000000004', 'LAUCT365100000000005', 
        'LAUCT365100000000006', 'LAUCT365100000000007', 'LAUCT365100000000008', 'LAUCT365100000000009', 'SMU36935610000000001'],
        "startyear":str(start_year), "endyear":str(end_year)})
    
    print("Requesting data from Bereau of Labor Statistics")
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)

    # organize response data
    bls_series = []
    for timeseries in json_data["Results"]["series"]:
        series_name = variable_names[timeseries["seriesID"]]
        series_data = pd.json_normalize(timeseries["data"])
        series_data["Time"] = series_data["year"] + "-" + series_data["period"].str.replace("M","", regex=True)
        series_data = series_data[["Time", "value"]].rename(columns={"value":series_name}).set_index("Time")
        bls_series.append(series_data)
    bls_data = pd.concat(bls_series, axis=1)
    return bls_data
    

"""
This function combines datasets obtained from each of the other functions. It is designed to be flexible, 
    so that either the individual datasets can be passed or, if not passed, then the function to obtain them will be called instead.
"""
def combine_data(energy_demand_data:pd.DataFrame=None, energy_price_data:pd.DataFrame=None, weather_data:pd.DataFrame=None, 
    bls_data:pd.DataFrame=None, start:np.datetime64=None, end:np.datetime64=None, eia_api_key:str=None):
    """
    Parameters
    ----------
    energy_demand_data (pandas.DataFrame): An optional dataframe containing hourly energy demand data from the EIA. If not provided, 
        this data will be fetched using the start and end times provided.

    energy_price_data (pandas.DataFrame):An optional dataframe containing monthly energy price data from the EIA. If not provided, 
        this data will be fetched using the start and end times provided.

    weather_data (pandas.DataFrame):An optional dataframe containing hourly weather-related data from the NOAA. If not provided, 
        this data will be fetched using the start and end times provided.

    bls_data (pandas.DataFrame):An optional dataframe containing monthly economic-related data from the BLS. If not provided, 
        this data will be fetched using the start and end times provided.

    start (numpy.datetime64): The starting time for the data to be retrieved

    end (numpy.datetime64): The ending time for the data to be retrieved

    eia_api_key (str): An api key obtained from the EIA website. By default, this is set to None and a key is 
        looked for in a project directory called Configuration Files.

    Returns
    ----------
    pandas.DataFrame: A dataframe containing hourly observations with columns that include energy demand, energy price, weather-related variables, 
        and economic-related variables.
    """
    # if any of the datasets are not provided, need to provide start, end, and possibly api_key
    if energy_demand_data is None: 
        if (any([start is None, end is None])):
            raise ValueError("If not providing an energy demand dataset, you must provide \"start\" and \"end\" arguments.")
        energy_demand_data = get_energy_demand_data(start=start, end=end, eia_api_key=eia_api_key)
    else: energy_demand_data = energy_demand_data.copy() # copy dataframe to avoid overwriting
    if energy_price_data is None: 
        if (any([start is None, end is None])):
            raise ValueError("If not providing an energy price dataset, you must provide \"start\" and \"end\" arguments.")
        energy_price_data = get_energy_price_data(start=start, end=end, eia_api_key=eia_api_key)
    else: energy_price_data = energy_price_data.copy() # copy dataframe to avoid overwriting
    if weather_data is None:
        if (any([start is None, end is None])):
            raise ValueError("If not providing a weather dataset, you must provide \"start\" and \"end\" arguments.")
        weather_data = get_weather_data(start=start, end=end)
    else: weather_data = weather_data.copy()
    if bls_data is None: 
        if (any([start is None, end is None])): 
            raise ValueError("If not providing an economic dataset, you must provide \"start\" and \"end\" arguments.")
        bls_data = get_bls_data(start_year=np.datetime64(start, 'Y').astype(int) + 1970, end_year=np.datetime64(end, 'Y').astype(int) + 1970)
    else: bls_data = bls_data.copy()

    # merge hourly energy demand data with hourly weather data
    energy_demand_data["Energy Demand Time"] = pd.to_datetime(energy_demand_data.index)
    all_data = pd.concat([energy_demand_data, weather_data], axis=1)

    # merge hourly and monthly data on month
    all_data["Month"] = all_data["Energy Demand Time"].dt.strftime("%Y-%m")
    all_data = pd.merge(all_data, energy_price_data, left_on=["Month"], right_index=True, how="outer")
    all_data = pd.merge(all_data, bls_data, left_on=["Month"], right_index=True, how="outer")

    # get first non-NA value of energy demand
    first_date = pd.to_datetime(all_data["Energy Demand Time"]).min()
    all_data = all_data[all_data["Energy Demand Time"] >= first_date]
    all_data = all_data.drop(columns={"Energy Demand Time", "Month"})
    all_data.index = pd.to_datetime(all_data.index)

    # convert all columns to float
    all_data = all_data.apply(lambda x: x.astype("float", errors="ignore"))

    return all_data


if __name__ == "__main__":
    # extract start time and end time (should be strings in YYYY-mm-dd format)
    start = sys.argv[1]
    end = sys.argv[2]

    # check for optional api_key and path arguments
    if len(sys.argv) > 3:
        eia_api_key = sys.argv[3]
        if len(sys.argv) > 4: 
            path_to_saved_directory = sys.argv[4]
        else: path_to_saved_directory = None
    else:
        eia_api_key = None
        path_to_saved_directory = None
    
    # call main function to collect and combine all data and save it
    main(start=start, end=end, eia_api_key=eia_api_key, path_to_directory=path_to_saved_directory)