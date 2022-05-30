#- Modules
import numpy as np
import xarray as xr
import pandas as pd

#- Main
def load_hourly_era5(range_of_years, subregion, season, predictor, era5_import):
    """
    Description:
        Loads hourly ERA5 Data for a specified range of years, 
        subregion and season for one predictor.

    Parameters:
        range_of_years (str): range of years, "year1-year2"
        subregion (str): Lon / Lat subregion of 'lon-7050_lat8030'
        season (str): winter or autumn
        predictor (str): Predictor of Storm Surges
        era5_import (str): Folder of preprocessed data

    Returns:
        ds (xr.Dataset): Hourly xarray-Dataset for the predictor

    Note:
        Parameter-Section needs to be updated, if new data was era5_imported.
    """

    # Parameter
    available_years = [
    "1999-2008",
    "2009-2018",
    "2019-2022",
    ]

    available_predictors = [
        "sp",
        "tp",
        "u10",
        "v10",
    ]

    available_subregions = [
        "lon-0530_lat7040", # era5_importing1
    ]

    available_seasons = [
        "winter",
        "autumn",
    ]

    available_era5_imports = [
        "era5_import1",
    ]

    # Error Handling
    assert range_of_years in available_years, f'years: {range_of_years} not in available set of {available_years}'
    assert subregion in available_subregions, f'subregion: {subregion} not in available set of {available_subregions}'
    assert season in available_seasons, f'season: {season} not in available set of {available_seasons}'
    assert predictor in available_predictors, f'predictor: {predictor} not in available set of {available_predictors}'
    assert era5_import in available_era5_imports, f'era5_import: {era5_import} not in available set of {available_era5_imports}'

    # Main
    folder = f"resources\era5\{era5_import}\\"

    file = f'{folder}{predictor}_{range_of_years}_{subregion}_{season}.nc'

    ds = xr.open_dataset(file)

    return ds

def load_daymean_era5(range_of_years, subregion, season, predictor, era5_import):
    """
    Description:
        Loads daily mean of ERA5 Data for a specified range of years, 
        subregion and season for one predictor.

    Parameters:
        range_of_years (str): range of years, "year1-year2"
        subregion (str): Lon / Lat subregion of 'lon-7050_lat8030'
        season (str): winter or autumn
        predictor (str): Predictor of Storm Surges
        era5_import (str): era5_importing Folder

    Returns:
        ds (xr.Dataset): Hourly xarray-Dataset for the predictor

    Note:
        Parameter-Section needs to be updated, if new data was era5_imported.
    """

    # Parameter
    available_years = [
    "1999-2008",
    "2009-2018",
    "2019-2022",
    ]

    available_predictors = [
        "sp",
        "tp",
        "u10",
        "v10",
    ]

    available_subregions = [
        "lon-0530_lat7040", # preprocess1
    ]

    available_seasons = [
        "wdmean", # winter
        "admean", # autumn
    ]

    available_era5_imports = [
        "preprocess1",
    ]

    # Preprocessed Parameters
    if season.lower() == "winter":
        season = "wdmean"
    if season.lower() == "autumn":
        season = "admean"

    # Error Handling
    assert range_of_years in available_years, f'years: {range_of_years} not in available set of {available_years}'
    assert subregion in available_subregions, f'subregion: {subregion} not in available set of {available_subregions}'
    assert season in available_seasons, f'season: {season} not in available set of {available_seasons}'
    assert predictor in available_predictors, f'predictor: {predictor} not in available set of {available_predictors}'
    assert era5_import in available_era5_imports, f'Preprocess: {era5_import} not in available set of {available_era5_imports}'

    # Main
    folder = f"resources\era5\{era5_import}\\"

    file = f'{folder}{predictor}_{range_of_years}_{subregion}_{season}.nc'

    ds = xr.open_dataset(file)

    return ds