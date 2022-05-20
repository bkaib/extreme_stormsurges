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
        Detrends dataframe by subtracting mean from specified index / level.
        Data is grouped by level.

    Parameters:
        df (pd.DataFrame): Dataframe with timeseries data
        level (str): Index along which to subtract mean (Default:"station")

    Returns:
        pd.DataFrame: Detrended dataframe for each index of level
    """
    return (df - df.groupby(level=level).mean())

def apply_dummies(df, percentile=0.95, level="station"):
    """
    Description:
        Applies one-hot encoding on dataframe for specified percentile along
        an index. Labels data with 1, if datapoint is in percentile.

    Parameters:
        df (pd.DataFrame): Dataframe with timeseries data
        percentile (float): Percentile to evaluate dummies (Default: 0.95)
        level (str): Index along which to subtract mean (Default: "station")

    Returns:
        dummies (pd.DataFrame): DataFrame with labeled data (1 if data is in percentile, 0 if not.)
    """
    dummies = df - df.groupby(level=level).quantile(percentile)
    dummies[dummies >= 0] = 1
    dummies[dummies < 0] = 0

    return dummies