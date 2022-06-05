#---
# Modules
#---
import numpy as np
import xarray as xr
import pandas as pd

from data import data_loader
from data import gesla_preprocessing
from data import era5_preprocessing

from gesla import GeslaDataset

#---
# Modularized functions
#--- 
def intersect_time(predictor, predictand):
    """
    Description:
        Returns data of predictor and predictand in overlapping time-intervall.

    Parameters:
        predictor (xr.DataArray): Predictor values as a timeseries with lat, lon
        predictand (xr.DataArray): Predictand values as a timeseries per station

    Returns:
        X, Y, t
        X (np.array, float): Predictor values as a field time series. Shape:(time, lat, lon)
        Y (np.array, float): Predictand at selected stations. Shape:(time, stations)
        t (np.array, datetime.date): Time-series for x, y. Shape:(time,)
    """

    #---
    # Predictor and predictand values of overlapping time series
    #
    # GESLA data is hourly. Needs to be daily, like ERA5. 
    #---
    predictor_time = pd.to_datetime(predictor.time.values).date
    predictand_time = pd.to_datetime(predictand.date_time.values).date
    predictor = predictor.values # Daily data
    predictand = predictand.values # Hourly data

    # Choose maximum per day, i.e. if one hour
    # a day indicates an extreme surge, the whole day 
    # is seen as extreme surge.
    print("Get overlapping timeseries of ERA5 and GESLA")

    predictand_dmax = []
    for date in predictor_time:
        time_idx = np.where(predictand_time==date)[0] # Intersection of timeseries'
        dmax = np.max(predictand[:, time_idx], axis=1) # Daily maximum of predictand
        predictand_dmax.append(dmax)

    predictand_dmax = np.array(predictand_dmax)

    X = predictor
    Y = predictand_dmax
    t = predictor_time
        
    return X, Y, t
    
def preprocess(season, 
predictor, 
percentile, 
era5_import, 
range_of_years, 
subregion, 
station_names=['hanko-han-fin-cmems',],
):
    """
    Description:
        Returns predictor, predictand and time-interval for model-training.
        NaN-values are not handled.

        Predictor is chosen from a fixed subregion and loaded from resources>era5_import1.
        From GESLA only "analysis" flagged data is chosen as a predictand.
        Predictand is detrended with mean. 
        One-Hot Encoding is applied to Predictand depending on percentile.
        Overlapping time-period of ERA5 and GESLA is chosen. 
        GESLA is transformed from hourly to daily data by choosing the maximum value within a day.

    Parameters:
        season (string): either "winter" or "autumn"
        predictor (string): from list ["sp", "tp", "u10", "v10"]
        percentile (float): Percentile of Extreme Sea Level in GESLA-Dataset
        era5_import (str): Flag for ERA5-Import file found in resources/era5/era5_import
        range_of_years (str): Range of years of predictors
        subregion (str): Selected subregion of predictors
        station_names (list): List of station names in GESLA dataset. (Defaults: ["hanko-han-fin-cmems"])

    Returns:
        X, Y, t
        X (np.array, float): Predictor values as a field time series. Shape:(time, lat, lon)
        Y (np.array, float): Predictand at selected stations. Shape:(time, stations)
        t (np.array, datetime.date): Time-series for x, y. Shape:(time,)
    
    """
    #---
    # Load Predictors
    #----

    print(f"Load ERA5-Predictor: {predictor} in region: {subregion} for years: {range_of_years} in season: {season}")

    # Load daily mean data for all predictors
    dmean_predictor = data_loader.load_daymean_era5(
        range_of_years=range_of_years, 
        subregion=subregion, 
        season=season, 
        predictor=predictor,
        era5_import=era5_import,  
    )

    #---
    # Load Predictand
    #---

    print("Load Predictand from GESLA")

    # Create GESLA Dataset
    meta_file = "resources/gesla/GESLA3_ALL.csv"
    data_path = "resources/gesla/GESLA3.0_ALL.zip"

    g3 = GeslaDataset(meta_file=meta_file, data_path=data_path)

    # Select Stations
    ds = g3.files_to_xarray(station_names)

    # Select a season
    if season == "autumn":
        get_season = gesla_preprocessing.is_autumn
    elif season == "winter":
        get_season = gesla_preprocessing.is_winter
    else:
        raise Exception(f"season: {season} is not available in this process")

    season_ds = ds.sel(date_time=get_season(ds['date_time.month']))

    # Select only sea_level analysis data
    analysis_data = gesla_preprocessing.get_analysis(season_ds)
    analysis_data = analysis_data["sea_level"]

    # Detrend data grouped by station
    anomalies = gesla_preprocessing.detrend(analysis_data, level="station")

    # Apply one hot encoding
    extremes = gesla_preprocessing.apply_dummies(anomalies, percentile=percentile, level="station")
    print("Applied one-hot-encoding")

    # Convert to dataset
    # nan values: no measurement at that timestamp for specific station
    extremes = extremes.to_xarray()

    #---
    # Predictor and predictand values of overlapping time series
    #
    # GESLA data is hourly. Needs to be daily, like ERA5. 
    #---
    predictor_time = pd.to_datetime(dmean_predictor.time.values).date
    predictand_time = pd.to_datetime(extremes.date_time.values).date
    predictor = dmean_predictor[predictor].values # Daily data
    sl = extremes.values # Hourly data

    # Choose maximum per day, i.e. if one hour
    # a day indicates an extreme surge, the whole day 
    # is seen as extreme surge.
    print("Get overlapping timeseries of ERA5 and GESLA")

    sl_dmax = []
    for date in predictor_time:
        time_idx = np.where(predictand_time==date)[0] # Intersection of timeseries'
        slmax = np.max(sl[:, time_idx], axis=1)
        sl_dmax.append(slmax)

    sl_dmax = np.array(sl_dmax)

    X = predictor
    Y = sl_dmax
    t = predictor_time
        
    return X, Y, t

#---
# Information about preprocessed datasets
#---
def get_lonlats(range_of_years, subregion, season, predictor, era5_import):
    """
    Description:
        Returns lats, lons of the already preprocessed ERA5 Predictor area.
    Parameters:
        range_of_years (str): Range of years, e.g. "1999-2008"
        subregion (str): Subregion of the original ERA5 data, e.g. "lon-0530_lat7040”
        season (str): winter or autumn
        predictor (str): Predictor, either ["sp", "tp", "u10", "v10"]
        era5_import (str): subfolder where data is stored, e.g. resources/era5/era5_import
    Returns:
        lats, lons
        lats (np.array): latitudes
        lons (np.array): longitudes
    """
    # Modules
    from data import data_loader
    
    #---
    # Load Predictors
    #----
    print(f"Load ERA5-Predictor: {predictor} in region: {subregion} for years: {range_of_years} in season: {season}")

    # Load daily mean data for all predictors
    dmean = data_loader.load_daymean_era5(
        range_of_years=range_of_years, 
        subregion=subregion, 
        season=season, 
        predictor=predictor,
        era5_import=era5_import,  
    )
    lats = dmean.latitude.values
    lons = dmean.longitude.values
    
    return(lats, lons)

#----------------------------------------
#---
#  Specific preprocesses (not modularized)
#---
def preprocessing1(season, predictor, percentile=0.95,):
    """
    Description:
        Returns predictor, predictand and time-interval for model-training.
        NaN-values are not handled.

        Predictor is chosen from a fixed subregion and loaded from resources>era5_import1.
        From GESLA only "analysis" flagged data is chosen as a predictand.
        Predictand is detrended with mean. 
        One-Hot Encoding is applied to Predictand depending on percentile.
        Overlapping time-period of ERA5 and GESLA is chosen. 
        GESLA is transformed from hourly to daily data by choosing the maximum value within a day.

    Parameters:
        season (string): either "winter" or "autumn"
        predictor (string): from list ["sp", "tp", "u10", "v10"]
        percentile (float): Percentile of Extreme Sea Level in GESLA-Dataset (Defaults: 0.95)

    Returns:
        x (np.array, float): Predictor values as a field time series. Shape:(time, lat, lon)
        y (np.array, float): Predictand at selected stations. Shape:(time, stations)
        t (np.array, datetime.date): Time-series for x, y. Shape:(time,)
    
    """
    #---
    # Setup: Select years and region
    #---
    range_of_years = "1999-2008" 
    subregion = "lon-0530_lat7040" 
    era5_import = "preprocessing1" # Flag for ERA5 data import

    #---
    # Load Predictors
    #----

    print(f"Load ERA5-Predictor: {predictor} in region: {subregion} for years: {range_of_years} in season: {season}")

    # Load daily mean data for all predictors
    sp_dmean = data_loader.load_daymean_era5(
        range_of_years=range_of_years, 
        subregion=subregion, 
        season=season, 
        predictor=predictor,
        era5_import=era5_import,  
    )

    #---
    # Load Predictand
    #---

    print("Load Predictand from GESLA")

    # Create GESLA Dataset
    meta_file = "resources/gesla/GESLA3_ALL.csv"
    data_path = "resources/gesla/GESLA3.0_ALL.zip"

    g3 = GeslaDataset(meta_file=meta_file, data_path=data_path)

    # Select Stations
    station_names = [
        'hanko-han-fin-cmems',
        # 'vahemadal-vah-est-cmems', just nan
        'pori-por-fin-cmems',
    ]

    ds = g3.files_to_xarray(station_names)

    # Select a season
    if season == "autumn":
        get_season = gesla_preprocessing.is_autumn
    elif season == "winter":
        get_season = gesla_preprocessing.is_winter
    else:
        raise Exception(f"season: {season} is not available in this process")

    season_ds = ds.sel(date_time=get_season(ds['date_time.month']))

    # Select only sea_level analysis data
    df = gesla_preprocessing.get_analysis(season_ds)
    df = df["sea_level"]

    # Detrend data grouped by station
    df_anom = gesla_preprocessing.detrend(df, level="station")

    # Apply one hot encoding
    df_isextreme = gesla_preprocessing.apply_dummies(df_anom, percentile=percentile, level="station")
    print("Applied one-hot-encoding")

    # Convert to dataset
    # nan values: no measurement at that timestamp for specific station
    ds_extremes = df_isextreme.to_xarray()

    #---
    # Predictor and predictand values of overlapping time series
    #
    # GESLA data is hourly. Needs to be daily, like ERA5. 
    #---
    predictor_time = pd.to_datetime(sp_dmean.time.values).date
    predictand_time = pd.to_datetime(ds_extremes.date_time.values).date
    sp = sp_dmean[predictor].values # Daily data
    sl = ds_extremes.values # Hourly data

    # Choose maximum per day, i.e. if one hour
    # a day indicates an extreme surge, the whole day 
    # is seen as extreme surge.
    print("Get overlapping timeseries of ERA5 and GESLA")

    sl_dmax = []
    for date in predictor_time:
        time_idx = np.where(predictand_time==date)[0] # Intersection of timeseries'
        slmax = np.max(sl[:, time_idx], axis=1)
        sl_dmax.append(slmax)

    sl_dmax = np.array(sl_dmax)

    x = sp
    y = sl_dmax
    t = predictor_time
        
    return x, y, t