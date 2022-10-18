#- Modules
import numpy as np
import xarray as xr
import pandas as pd

#- Main
def is_winter(month):
    """
    Description:
        Mask for winter season, e.g. December, January and February.

    Parameters:
        month (xr.DataArray): Containing the month of a timeseries
    
    Returns:
        Boolean mask for winter season
    """

    return (month == 1) | (month == 2) | (month == 12)

def is_autumn(month):
    """
    Description:
        Mask for autumn season, e.g. Sep, Oct and Nov.

    Parameters:
        month (xr.DataArray): Containing the month of a timeseries
    
    Returns:
        Boolean mask for autumn season
    """

    return (month == 9) | (month == 10) | (month == 11)

def get_analysis(ds):
    """
    Description: 
        Selects all values of GESLA data where use_flag == 1.
        Drops all NaN values.

    Parameters:
        ds (xr.Dataset): GESLA Dataset for several stations

    Returns:
        df (pd.Dataframe): 
    """
    ds = ds.where(ds.use_flag == 1., drop = True) # Analysis flag
    df = ds.to_dataframe().dropna(how="all") 

    return df

def detrend(df, level="station"):
    """
    Description:
        Detrends pd.Series by subtracting mean from specified index / level.
        Data is grouped by level.

    Parameters:
        df (pd.Series): Dataframe with timeseries data
        level (str): Index along which to subtract mean (Default:"station")

    Returns:
        pd.Series: Detrended dataframe for each index of level
    """
    return (df - df.groupby(level=level).mean())

def detrend_signal(gesla_predictand, type_):
    """
    Description:
        Detrend GESLA-Dataset either by subtracting mean or linear trend
    Parameters:
        gesla_predictand (pd.Series): Gesla dataset at selected station. 
        type_ (str): "constant" or "linear" for either subtracting mean or linear trend, respectively.
    Returns:
        df (pd.Series): Detrended Gesla Dataseries.
    Note: 
        Only one station can be selected. This is not grouped by stations.
        Returns in the format as expected by further modules.
    """
    from scipy import signal
    import pandas as pd
    import numpy as np

    detrended = signal.detrend(gesla_predictand, type=type_,)

    # Get date_time for index
    #---
    date_time = []
    for index in gesla_predictand.index.values:
        date = index[1]
        date_time.append(date)

    # Create new dataframe
    #---
    station_idx = np.zeros(detrended.shape).astype(int)
    df = pd.DataFrame(
        {"sea_level": detrended},
        index=[
            station_idx,
            date_time,
        ],
        
    )
    df.index.names = ["station", "date_time"]
    
    return df.squeeze()

def apply_dummies(df, percentile=0.95, level="station"):
    """
    Description:
        Applies one-hot encoding on dataseries for specified percentile along
        an index. Labels data with 1, if datapoint is in percentile.

    Parameters:
        df (pd.Series): Dataseries with timeseries data
        percentile (float): Percentile to evaluate dummies (Default: 0.95)
        level (str): Index along which to subtract mean (Default: "station")

    Returns:
        dummies (pd.Series): DataFrame with labeled data (1 if data is in percentile, 0 if not.)
    """
    dummies = df - df.groupby(level=level).quantile(percentile)
    dummies[dummies >= 0] = 1
    dummies[dummies < 0] = 0

    return dummies

def select_season(ds_gesla, season):
    """
    Description:
        Selects a season of GESLA dataset
    Parameters:
        ds_gesla (xr.Dataset): GESLA dataset 
        season (str): "winter" or "autumn"
    Returns:
        season_ds (xr.Dataset): GESLA Dataset for specific month of the season.
    """
    # Modules
    #---
    from data import gesla_preprocessing

    if season == "autumn":
        get_season = gesla_preprocessing.is_autumn
    elif season == "winter":
        get_season = gesla_preprocessing.is_winter
    else:
        raise Exception(f"season: {season} is not available in this process")

    season_ds = ds_gesla.sel(date_time=get_season(ds_gesla['date_time.month']))

    return season_ds

def station_position(gesla_predictand, station_names):
    # TODO: Put into module
    """
    Description:
        Returns positions longitudes and latitudes of GESLA Stations
    Parameters:
        gesla_predictand (xr.Dataset): GESLA Dataset loaded with data_loader.load_gesla(station_names)
        station_names (list): List of station names within GESLA dataset
    Returns:
        station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
    """
    lon = gesla_predictand.longitude.values
    lat = gesla_predictand.latitude.values

    station_positions = {}
    for station_idx, station_name in enumerate(station_names):
        station_positions[station_name] = [lon[station_idx], lat[station_idx]]

    return station_positions