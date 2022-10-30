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
# Intersect time
#---
def intersect_time(predictor, predictand, is_prefilling):
    """
    Description:
        Returns data of predictor and predictand in overlapping time-intervall.

    Parameters:
        predictor (xr.DataArray): Predictor values as a timeseries with lat, lon
        predictand (xr.DataArray): Predictand values as a timeseries per station
        is_prefilling (bool): If the predictor is prefilling of Baltic Sea, e.g. Degerby SL as a proxy

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
    import pandas as pd
    import numpy as np
    predictand_time = pd.to_datetime(predictand.date_time.values).date
    predictor_values = predictor.values # Daily (ERA5) or hourly (Degerby) data
    predictand_values = predictand.values # Hourly data

    if is_prefilling:
        predictor_time = pd.to_datetime(predictor.date_time.values).date
        predictor_values = np.swapaxes(predictor_values, axis1=0, axis2=1)
    else:
        predictor_time = pd.to_datetime(predictor.time.values).date

    # Choose maximum per day, i.e. if one hour
    # a day indicates an extreme surge, the whole day 
    # is seen as extreme surge.
    print("Get overlapping timeseries of ERA5 and GESLA")

    predictand_dmax = []
    for date in predictor_time:
        time_idx = np.where(predictand_time==date)[0] # Intersection of timeseries'
        if time_idx.shape[0] == 0: # If no intersection
            time_idx = np.where(predictor_time==date)[0] # Find poistion in (updated) predictor timeseries
            predictor_values = np.delete(predictor_values, time_idx, axis=0) # Update predictor timeseries to match predictand
            predictor_time = np.delete(predictor_time, time_idx, axis=0) # Update predictor timepoints
            print(f"date:{date} at position {time_idx} was deleted as it was in predictor data but not in predictand data") 
        else:
            dmax = np.max(predictand_values[:, time_idx], axis=1) # Daily maximum of predictand
            predictand_dmax.append(dmax)

    predictand_dmax = np.array(predictand_dmax)

    X = predictor_values
    Y = predictand_dmax
    t = predictor_time
        
    return X, Y, t

def intersect_all_times(predictors, gesla_predictand, range_of_years, subregion, season, preprocess):
    """
    Description: 
        Returns an overlapping timeseries for all given predictors and the gesla predictand.
    Parameters:
        predictors (list): List of predictors
        gesla_predictand (xarray): Xarray dataset of predictand
        range_of_years (str): Either "1999-2008" or "2009-2018"
        subregion (str): Lon Lat subregion from preprocessing
        preprocess (str): Preprocessing applied to get ERA5 predictors
    Returns:
        tt (np.array): Overlapping dates of all timeseries of all predictors and predictand.
    """
    era5_counter = 0
    for pred_idx, predictor in enumerate(predictors):
        if predictor == "pf":
            is_prefilled = True
        else:
            is_prefilled = False

        # Load Predictor
        #---
        print(f"Load predictor {predictor}")
        if predictor == "pf":
            era5_predictor = data_loader.load_pf(season)
            era5_predictor_tmp = data_loader.load_daymean_era5(range_of_years, subregion, season, predictors[era5_counter-1], preprocess)

            X_, Y_, t_ = intersect_time(era5_predictor_tmp, era5_predictor, is_prefilling=False)
            
        else:
            era5_counter = era5_counter + 1
            era5_predictor = data_loader.load_daymean_era5(range_of_years, subregion, season, predictor, preprocess) # TODO: Change back to range_of_years
            era5_predictor = convert_timestamp(era5_predictor, dim="time")
        
            X_, Y_, t_ = intersect_time(era5_predictor, gesla_predictand, is_prefilled)

        # Compare to intersections done before and keep intersection of all predictors. Since one predictor was intersected with GESLA
        # This intersection leads to a time-series with available dates in all predictors and GESLA dataset
        #---
        
        if pred_idx == 0:
            print("Create tt")
            tt = np.array(t_)
            print(f"shape: {tt.shape}")
        else:
            tt = np.intersect1d(tt, t_)

    return tt

def timelag(X, Y, t, timelag):
    """
    Description: 
        Returns timelagged predictor data X_timelag for predictand Y_timelag.
        Shifts predictand data Y according to the given timelag.
    Parameters:
        X (np.array, float): Predictor values as a field time series. Shape:(time, lat, lon)
        Y (np.array, float): Predictand at selected stations. Shape:(time, stations)
        t (np.array, datetime.date): Time-series of intersected timepoints of X and Y. Shape:(time,)
        timelag (int): timelag for predictor data X. Dimension of timelag depends on timeseries-interval of X.
    Returns:
        X_timelag (np.array, float): Predictor values as a field time series (timelagged). Shape:(time, lat, lon)
        Y_timelag (np.array, float): Predictand at selected stations. Shape:(time, stations)
        t_predictor (np.array, datetime.date): Time-series of X (timelagged). Shape:(time,)
        t_predictand (np.array, datetime.date): Time-series of Y. Shape:(time,)

    """
    n_timepoints = len(t)
    # Timelag of predictor 
    # Return predictor data and corresponding timeseries
    #---
    t_predictor = t[:(n_timepoints-timelag)]
    X_timelag =  X[:(n_timepoints-timelag)]

    # Return Predictand data (not lagged) and corresponding timeseries
    #---
    t_predictand = t[timelag:]
    Y_timelag = Y[timelag:]

    return (X_timelag, Y_timelag, t_predictor, t_predictand)

def combine_timelags(X, Y, timelags):
    """
    Description:
        Returns combined timelagged predictor data X_timelag for predictand Y_timelag.
        Shifts predictand data Y according to the maximum timelag given in timelags.
        Note: Input data X, Y needs to be on the same time-interval (see preprocessing.intersect_time)
        
    Parameters:
        X (np.array, float): Predictor values as a field time series. Shape:(n_labels, lat, lon)
        Y (np.array, float): Predictand at selected stations. Shape:(n_labels, stations)

    Returns:
        X_timelag (np.array, float): Combined timelagged Predictor values in increasing order of timelags, e.g. t=0, t=1,..., Shape:(timelag, n_labels, lat, lon)
        Y_timelag (np.array, float): Timelagged Predictand at selected stations. Shape:(n_labels, stations)
    """

    # Initialize
    #---
    timelags.sort()
    max_timelag = max(timelags)

    # Get timelagged Predictand 
    #---
    Y_timelag = Y[max_timelag:]

    # Get timelagged predictors
    #---
    X_timelag = []

    for timelag_ in timelags:

        assert timelag_ >= 0, f"Timelag = {timelag_} needs to be a positive integer"

        idx = max_timelag - timelag_

        if timelag_ > 0:
            X_tmp = X[idx : - timelag_]
        if timelag_ == 0: 
            X_tmp = X[idx:]

        X_timelag.append(X_tmp)

    X_timelag = np.array(X_timelag)

    return X_timelag, Y_timelag

def add_timelag(X, Y, timelags, pred_idx):
    """
    Description:
        Returns combined timelagged predictor data X_timelag for predictand Y_timelag.
        Shifts predictand data Y according to the maximum timelag given in timelags.
        Note: Input data X, Y needs to be on the same time-interval (see preprocessing.intersect_time)
        
    Parameters:
        X (np.array, float): Predictor values as a field time series. Shape:(n_labels, lat, lon) or (n_labels, i, ..., k)
        Y (np.array, float): Predictand at selected stations. Shape:(n_labels, stations)
        timelags (list): List of all timelags of a model run (e.g. for all combinations of predictors)
        pred_idx (int): Index of the current predictor

    Returns:
        X_timelag (np.array, float): Combined timelagged Predictor values in increasing order of timelags, e.g. t=0, t=1,..., Shape:(timelag, n_labels, lat, lon)
        Y_timelag (np.array, float): Timelagged Predictand at selected stations. Shape:(n_labels, stations)
    """

    # Initialize
    #---
    pred_timelag = timelags[pred_idx]

    # timelags.sort()
    
    max_timelag = max(timelags)

    # Get timelagged Predictand 
    #---
    Y_timelag = Y[max_timelag:] # In order to have same predictand for all predictors

    # Get timelagged predictors
    #---
    X_timelag = []

    # for timelag_ in timelags:

    idx = max_timelag - pred_timelag

    if pred_timelag > 0:
        X_timelag = X[idx : - pred_timelag]
    if pred_timelag == 0: 
        X_timelag = X[idx:]

    return X_timelag, Y_timelag

def convert_timestamp(da, dim):
    """
    Description: 
        Converts timepoint to datetime64 type
    Parameters:
        da (DataArray): DataArray with timeseries
        dim (str): Flag of time dimension
    Return:
        da (DataArray): DataArray with converted timestamp values
    """
    # Ensure correct dtype of timestamp
    #---
    if "datetime64" in str(da[dim].dtype):
        print(f"timeseries is already of dtype {da[dim].dtype}")
        
    if da[dim].dtype == "float64":
        print(f"Convert timestamp from dtype: {da[dim].dtype} to datetime64")
        era5_timeflag = "T11:30:00.000000000" # Timeflag of ERA5
        t = da[dim].values 
        t_datetime = float64_to_datetime64(t, era5_timeflag) # Convert datatype 
        coords_to_replace = {"time" : t_datetime} 
        da = replace_coords(da, coords_to_replace) # Replace dimension values with new datatype

    return da

def get_timeseries(predictor, ts, is_prefilling):
    """
    Description: 
        Returns values of a predictor or dataset that only contains dates given in ts.
    Parameters:
        predictor (xarray): Values of predictor
        ts (list): List of date times
        is_prefilling (bool): Whether or not the predictor is prefilling (True) or ERA5-Data (False)
    Return:
        timeseries (np.array): Dataset that only contains dates given in ts 
    """
    import pandas as pd
    import numpy as np
    
    if is_prefilling:
        t = pd.to_datetime(predictor.date_time.values).date
    else:
        t = pd.to_datetime(predictor.time.values).date

    values = predictor.values

    timeseries = []
    for date in ts:
        time_idx = np.where(t==date)[0] # Intersection of timeseries'
        if time_idx.shape[0] == 0: # If no intersection
            pass
        else: # If time overlaps add timepoint
            if is_prefilling:
                added_value = np.max(values[:, time_idx], axis=1) # Daily maximum of prefilling
                timeseries.append(added_value)
            else:
                added_value = values[time_idx, :, :,] # Daily maximum of predictand
                timeseries.append(added_value)
                
    timeseries = np.array(timeseries)
    if not is_prefilling:
        timeseries = timeseries[:, 0, :, :] # Bring to format (time, lon, lat)
    return timeseries

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

def check_combination(predictors):
    """
    Description:
        Check whether prefilling is combined with ERA5 data or not
    Parameters:
        predictors (list): List of all predictors used in model run
    Returns
        is_pf_combined (bool): Whether or not prefilling is combined with ERA5 data or not.
    """

    if all(x in ["sp", "tp", "u10", "v10"] for x in predictors):
        is_era5 = True
    else:
        is_era5 = False

    if all(x in ["pf"] for x in predictors):
        is_pf = True
    else:
        is_pf = False

    if not is_era5 and not is_pf:
        print("ERA5 and prefilled are combined")
        is_pf_combined = True
    else:
        print("Prefilling or ERA5 is used alone")
        is_pf_combined = False

    return is_pf_combined
#---
# General transformation of data
#---
def aggregate_dimension(da, dim):
    """
    Description:
        Aggregate (mean) along a dimension of a dataset. 
    Parameters:
        da (DataArray): xr.DataArray 
        dim (str): Dimension along which mean is taken
    Return:
        aggregated_da
    """
    aggregated_da = da.mean(dim=dim)
    
    return aggregated_da

def float64_to_datetime64(t, timeflag=""):
    """
    Description: 
        Converts dates of type float64, e.g. (yyyymmdd.0) into datetime64 format.
    Parameters:
        t (np.array, float64): Values of timeseries to be converted
        timeflag (str): Timeflag of specific hours of measurement, e.g. T11:30:00.000000000.
    Returns:
        t_datetime (np.array, np.datetime64): Converted values of timeseries
    """
    #- Modules
    import numpy as np

    #- Init
    t_datetime = []

    #- Main
    for timepoint in t: 
        timepoint = str(int(timepoint))
        yyyy = timepoint[:4]
        mm = timepoint[4:6]
        dd = timepoint[6:8]
        new_date = np.datetime64(f"{yyyy}-{mm}-{dd}{timeflag}")
        t_datetime.append(new_date)

    t_datetime = np.array(t_datetime)

    return t_datetime

def replace_coords(da, coords_to_replace):
    """
    Description:
        Replaces indicated coordinates (keys) in coords_to_replace with corresponding values.
        Leaves name of dimension unchanged.
    Parameters:
        da (xr.DataArray): DataArray with coordinates and values to replace
        coords_to_replace (dict): Dictionary with coordinates as keys and new values to replace old ones with.
    Returns:
        da (xr.DataArray): DataArray with updated coordinate values
    """
    for coord, values in coords_to_replace.items():
        da.coords[coord] = values
        
    return da

def replace_dims(da, dims_to_replace):
    """
    Description:
        Replaces indicated dimensions (keys) in dims_to_replace with corresponding values.
        Leaves name of dimension unchanged.
    Parameters:
        da (xr.DataArray): DataArray with dimension and values to replace
        dims_to_replace (dict): Dictionary with dimensions as keys and new values to replace old ones with.
    Returns:
        da (xr.DataArray): DataArray with updated dimension values
    """
    import xarray as xr
    for dim, values in dims_to_replace.items():
        
        da = da.expand_dims({f"{dim}_tmp" : values}) # Expand dimensions with temporary new dimension
        da = da.drop_vars(f"{dim}") # Drop old dimension with old values
        da = da.rename({f'{dim}_tmp': dim,})

    return da

def array_to_series(arr, index, index_name, series_name):
    # Put to preprocessing
    """
    Description:
        Converts np.array to a pd.Series
    Parameters:
        arr (np.array): Values to convert
        index (list): List of index values corresponding to values in arr
        index_name (str): Name of the index of the pd.Series, e.g. "station"
        series_name (str): Name of the series of the pd.Series, e.g. "sea_level"
    Returns:
        series (pd.Series): Pandas Series with indicated index names
    """
    d = {f'{index_name}': index, f'{series_name}': arr}
    df = pd.DataFrame(d).set_index(f'{index_name}')
    series = df.squeeze()
    
    return series

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
    era5_import = "preprocess1" # Flag for ERA5 data import

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
    sp = sp_dmean.values # Daily data
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
