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

#- Main
def preprocessing1(season, predictor, percentile=0.95,):
    """
    Description:
        Returns predictor, predictand and time-interval for model-training.
        NaN-values are not handled.

        Predictor is chosen from a fixed subregion and loaded from resources>preprocess1.
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
    preprocess = "preprocess1" # Flag for ERA5 data import

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
        preprocess=preprocess,  
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
    filenames = [
        'hanko-han-fin-cmems',
        # 'vahemadal-vah-est-cmems', just nan
        'pori-por-fin-cmems',
    ]

    ds = g3.files_to_xarray(filenames)

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