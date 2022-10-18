#---
# Modules
#---
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


from data import data_loader
from data import gesla_preprocessing
from data import era5_preprocessing
from data import preprocessing
from data import saver
from data import visualisation

from models import modelfit
from models import evaluation
from models import loader

def run(
model_run, 
run_id, 
season,
station_names,
detrend_type, 
predictors_of_model, 
timelags_of_model, 
models_path, 
percentile, 
clf, 
hparam_grid, 
optimizer, 
k, 
n_iter, 
random_state,
test_size,
is_optimized, 
is_scaled,
is_overlay_importance,
is_station_name,
):
    """
    Description:
        Builds a model to predict (percentile) extreme storm surges at station_names using predictors for a specified season by using
        a classifier.
    Parameters:
        season (str): Either winter or autumn
        predictors (list): List of predictors (X) used to predict Storm Surge (y)
        percentile (float): Percentile of Extreme Storm Surges
        station_names (list): List of stations used from GESLA data-set
        detrend_type (str): "constant" or "linear" for detrending GESLA Dataset with mean or linear-trend, respectively.
        clf (sklearn.modeltype): Used model to do prediction, e.g. RandomForestClassifier, LogisticRegression,...
        hparam_grid (dict): Selection of multiple Hyperparameters either used for building the model or fitted 
                            to clf via the optimizer depending on is_optimized.
                            If hparam_grid is used for optimization, values of dictionary need to be lists.
        optimizer (str): Optimizer for automatic search of hyperparameters, either GridSearchCV or RandomSearchCV
        run_id (int): Number of the run (used for saving output)
        model_run (str): Naming of the model run (used for saving output)
        k (int): Number of k-fold crossvalidations in model fit (Defaults: 3)
        n_iter (int): Number of combinations used for RandomizedSearchCV (Defaults: None)
        is_optimized (bool): Whether or not to search best combination of hparams within hparam_grid with 
                             the optimizer (Defaults: True)
                             If is_optiomized == False, hyperparameters are loaded from a model given in models_path
        is_scaled (bool): Use StandardScaler to scale data if it is on different scales (Defaults: False)
        is_overlay_importance (bool): Whether to plot predictor maps with overlayed importance positions
        is_station_name (bool): Whether to plot station name over station itself.
    Returns:
        None. Saves Output in results>random_forest>model_run folder. Saves model in models>random_forest>model_run
    """
    import sys
    #--- 
    # Initialize
    #---
    preprocess = "preprocess1" # ["preprocess1"]
    if all(x in ["daugavgriva-dau-lva-cmems", "travemuende-tra-deu-cmems"] for x in station_names): # Those stations only record from 2005-2020
        range_of_years = "2009-2018"
    else:
        range_of_years = "1999-2008" # ["1999-2008", "2009-2018", "2019-2022",] 
    unknown_range_of_years = "2009-2018"
    subregion = "lon-0530_lat7040" # ["lon-0530_lat7040"]
    lats, lons = preprocessing.get_lonlats(
        range_of_years,
        subregion,
        season,
        predictor="sp", # Does not matter which predictor. All predictors are sampled on same lon-lat field.
        era5_import=preprocess,
        )
    colorbar_range = { # vmin vmax values for colorbar of predictor maps
        'sp': np.array([ 980., 1020.,]),  # Low pressure systems <980hPa (see Theory Part)
        'tp': np.array([0.    , 0.0018]),
        'u10': np.array([-17.2,  17.2]), # Storm is defined by wind stronger than 17.2m/s
        'v10': np.array([-17.2,  17.2]),
        }
    nlevels = 10 # For contourplot of predictor maps
    orig_stdout = sys.stdout # Original Output for print etc.
    
    #---
    # Preprocess GESLA Data
    #---

    # Load Predictand
    #---
    gesla_predictand = data_loader.load_gesla(station_names)

    # Get lon/lat positions of stations
    #---
    station_positions = gesla_preprocessing.station_position(gesla_predictand, station_names)

    # Select a season
    #---
    gesla_predictand = gesla_preprocessing.select_season(gesla_predictand, season)

    # Select only sea_level analysis data
    #---
    gesla_predictand = gesla_preprocessing.get_analysis(gesla_predictand)
    gesla_predictand = gesla_predictand["sea_level"] # Select values

    # Detrend 
    #---
    gesla_predictand = gesla_preprocessing.detrend_signal(gesla_predictand, type_=detrend_type) 

    # Apply one hot encoding
    #---
    gesla_predictand = gesla_preprocessing.apply_dummies(gesla_predictand, percentile=percentile, level="station")
    
    print(f"Applied one-hot-encoding with Percentile: {percentile}")

    # Convert to DataArray
    # nan values: no measurement at that timestamp for specific station
    #---
    gesla_predictand = gesla_predictand.to_xarray()

    #---
    # Loop over all run_ids of a model_run / predictors
    #---
    for idx, predictors in enumerate(predictors_of_model): # Loops over separate modelruns 
        
        #---
        # Save printed output to file
        #---
        folder = f"results/random_forest/{model_run}/"
        saver.directory_existance(folder)
        file_path = f'{folder}output_runid{run_id}.txt'
        sys.stdout = open(file_path, "w") 

        #---
        # Load ERA5- Predictor
        #---
        
        # Initialize
        #---
        timelags = timelags_of_model[idx]

        is_pf_combined = preprocessing.check_combination(predictors)
        X = []
        Y = []
        t = []
        pred_units = []
        pred_names = []
        tic_main = time.perf_counter()

        if is_pf_combined==True:
            ts = preprocessing.intersect_all_times(predictors, gesla_predictand, range_of_years, subregion, season, preprocess)

        # Main
        #---
        era5_counter = 0
        for pred_idx, predictor in enumerate(predictors):
            print(f"Add predictor {predictor} to model input features")

            # Load data of predictor
            #---
            import xarray as xr
            if predictor == "pf":
                is_prefilled = True

                era5_predictor = data_loader.load_pf(season)

                if is_pf_combined:
                    X_ = preprocessing.get_timeseries(era5_predictor, ts, is_prefilling=True)
                    Y_ = preprocessing.get_timeseries(gesla_predictand, ts, is_prefilling=True)
                else:
                    # If pf is used without any ERA5 data, use hourly data of Degerby.
                    X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled) 

            else:
                is_prefilled = False
                era5_counter = era5_counter + 1
                era5_predictor = data_loader.load_daymean_era5(range_of_years, subregion, season, predictor, preprocess) # TODO: Change back to range_of_years
                era5_predictor = preprocessing.convert_timestamp(era5_predictor, dim="time")

                # Convert predictor sp.unit from Pa to hPA
                #---
                if predictor == "sp":
                    with xr.set_options(keep_attrs=True):
                        old_unit = era5_predictor.attrs["units"]
                        era5_predictor = era5_predictor / 100
                        era5_predictor.attrs["units"] = "hPa"
                        new_unit = era5_predictor.attrs["units"]
                        print(f"Converted units of {predictor} from {old_unit} to {new_unit}")

                if is_pf_combined:
                    X_ = preprocessing.get_timeseries(era5_predictor, ts, is_prefilling=False)
                    Y_ = preprocessing.get_timeseries(gesla_predictand, ts, is_prefilling=True)
                else:
                    X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled)

            
            #---
            # Get timelags
            #---
            print(f"Introduce timelag: {timelags[pred_idx]} for {predictor}")
            print(f"Shapes X_: {X_.shape}, Y_: {Y_.shape}")

            X_timelag, Y_timelag = preprocessing.add_timelag(X_, Y_, timelags, pred_idx) 

            X.append(X_timelag)
            Y.append(Y_timelag)
            # t.append(t_)

            # # Save unit and name of predictor
            # #---
            if predictor == "pf":
                pred_units.append("m")
                if is_pf_combined:
                    pred_names.append(f"{predictor}_tlag{timelags[pred_idx]}") # Daily values of timelags if is_pf_combined
                else:
                    pred_names.append(f"{predictor}_tlag{timelags[pred_idx] // 24}") # Hourly values converted back to daily values for interpretation
            else:
                pred_units.append(era5_predictor.units)
                pred_names.append(f"{era5_predictor.name}_tlag{timelags[pred_idx]}")
        
        #--- 
        # Convert to format needed for model fit
        #---
        if is_pf_combined:
            Y = np.array(Y) 
            Y = Y[0, :] # Assume all timeseries are the same for the predictors.
            ndim = Y.shape[0]
            max_length = len(X)
            n_pfs = max_length - era5_counter # Number of prefilling predictors

            print(era5_counter, max_length, n_pfs, ndim)

            era5_x = np.array(X[:era5_counter])
            era5_x = era5_x.swapaxes(0, 1)
            era5_x = era5_x.reshape(ndim, -1)

            print(f"ERA5 shape: {era5_x.shape}")

            #--- 
            # Insert prefilling data at the beginning
            #---
            XX = era5_x
            new_names = pred_names
            new_units = pred_units
            for i in range(era5_counter, max_length):
                print(f"Add Predictor {pred_names[i]} to end of X")
                XX = np.append(XX, X[i], axis=1)


            y = Y[:, 0] # Select one station

            #---
            # Handle NaN Values
            #---

            # Insert numerical value that is not in data.
            # ML will hopefully recognize it.
            XX[np.where(np.isnan(XX))] = -999

            X = XX
            
            del XX
            print("Data is prepared as follows")
            print(f"X.shape : {X.shape}")
            print(f"y.shape : {y.shape}")
            print(f"pred_names : {pred_names}")
            print(f"pred_units : {pred_units}")
        else:
            # Convert to format needed for model fit
            #--- 
            max_length = len(X)
            n_pfs = max_length - era5_counter # Number of prefilling predictors

            X = np.array(X)
            Y = np.array(Y) 
            Y = Y[0, :] # Assume all timeseries are the same for the predictors.
            t = np.array(t)

            # Reshape for model input
            #---
            print(f"Reshape for model input")

            ndim = Y.shape[0]

            X = X.swapaxes(0, 1) # Put time dimension to front

            print(X.shape) # (time, timelags, predictor_combination, lon?, lat?)

            X = X.reshape(ndim, -1) # Reshapes into (time, pred1_lonlats:pred2_lonlats:...:predn_lonlats)
            y = Y[:, 0] # Select one station

            #---
            # Handle NaN Values
            #---

            # Insert numerical value that is not in data.
            # ML will hopefully recognize it.
            X[np.where(np.isnan(X))] = -999

            print("Data is prepared as follows")
            print(f"X.shape : {X.shape}")
            print(f"y.shape : {y.shape}")
            print(f"pred_names : {pred_names}")
            print(f"pred_units : {pred_units}")
        
        #---
        # Apply Train-Test split 
        #---
        print("Apply train-test-split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

        #---
        # Scale data if they are on different scales
        #---
        X_test_unscaled = X_test

        if is_scaled:
            print("Scale training data")
            s = StandardScaler()
            s.fit(X_train)
            X_train = s.transform(X_train)
            X_test = s.transform(X_test)

        #---
        #  Optimize Hyperparameters
        #---
        if is_optimized:
            print(f"Optimize Hyperparameters using {optimizer}")
            print(f"Tested Hyperparameters: {hparam_grid}")
            
            hparam_grid_opt = modelfit.optimize_hyperparameter(X_train, y_train, clf(), optimizer, hparam_grid, k, n_iter, n_jobs=-1)

        if not is_optimized:
            model_path = models_path[idx]
            print(f"Hyperparameters are loaded from model: {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            hparam_grid_opt = model.get_params()
            
        #---
        #  Save hyperparameters
        #---
        folder = f"models/random_forest/{model_run}/" 
        saver.directory_existance(folder)
        saver.save_hpdict(hparam_grid_opt, run_id, model_run, percentile, folder)

        print("Saved Hyperparameters")

        #---
        # Fit the model
        #---
        print(f"Fit model with hyperparameters {hparam_grid_opt}")

        model = clf(**hparam_grid_opt) # One can set parameters afterwards via model.set_params() 

        model.fit(X_train, y_train)

        #---
        # Saving the model
        #---
        print(f"Save model: {model}")
        filename = f'{model_run}_{optimizer}_{run_id}.sav'
        pickle.dump(model, open(f'{folder}{filename}', 'wb'))

        #---
        # Plot Predictor Maps
        #---
        tic_predmap = time.perf_counter()

        if era5_counter != 0: # Do not print predictor maps if only prefilling is used as predictor. Otherwise print predictor maps
                print(f"Plot predictor maps")
                visualisation.predictor_maps( # TODO: If only pf is used this should be skipped, e.g. no plotting of importance maps
                        model, X_test, y_test,
                        X_test_unscaled,
                        ndim, n_pfs, is_pf_combined, # Needed to separate ERA5 importance from Prefilling importance
                        lons, lats, pred_units, pred_names, 
                        station_positions, station_names, is_station_name, 
                        is_overlay_importance, 
                        run_id, model_run, 
                        percentile=99., markersize=5, alpha=0.08, color="k", colorbar_range=colorbar_range, nlevels=nlevels,
                        )
        toc_predmap = time.perf_counter()
        time_predmap = toc_predmap - tic_predmap # Save time of this separately as it is not necessary for model predictions

        #---
        # Evaluate model / Diagnostic
        #--- 
        print("Evaluate Model \n")

        # Score & Importance
        #---
        test_score = model.score(X_test, y_test)
        train_score = model.score(X_train, y_train)
        relative_score = evaluation.relative_scores(train_score, test_score)
        importance = model.feature_importances_

        # Save Scores & Importance
        #---
        folder = f"results/random_forest/{model_run}/"
        saver.directory_existance(folder)

        fname = f"importance_{str(percentile)[-2:]}_{run_id}"
        np.save(f"{folder}{fname}", importance)
        print(f"saved importance to : {folder}{fname}")

        fname = f"testscore_{str(percentile)[-2:]}_{run_id}"
        np.save(f"{folder}{fname}", test_score)
        print(f"saved testscore to : {folder}{fname}")

        fname = f"trainscore_{str(percentile)[-2:]}_{run_id}"
        np.save(f"{folder}{fname}", train_score)
        print(f"saved trainscore to : {folder}{fname}")

        fname = f"relativescore_{str(percentile)[-2:]}_{run_id}"
        np.save(f"{folder}{fname}", relative_score)
        print(f"saved relativescore to : {folder}{fname}")
        

        # Plot importance of each predictor from combination
        # Goal:
        # 1. Separate importance per predictor
        # 2. Plot importance of each predictor on lon lat map
        #---
        n_pred_features = len(lons) * len(lats) # Features per predictor (lon/lat Input-Field). Needed for importance separation

        if is_pf_combined:
            pf_importance = importance[-n_pfs:] 
            era5_importance = importance[:-n_pfs]
            predictor_importances = evaluation.separate_predictor_importance(era5_importance, n_pred_features) # Plot only importance map of era5 data
        else:
            if era5_counter != 0: # Only plot predictor importance if at least one predictor is ERA5
                predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features)
        
        if era5_counter !=0: # Only plot predictor importance if at least one predictor is ERA5
            for pred_idx, pred_importance in enumerate(predictor_importances): # TODO: If only pf is used this should be skipped, e.g. no plotting of importance maps
                # Plot importance map and save it
                predictor = pred_names[pred_idx] 
                tflag = f"{predictor}_{str(percentile)[-2:]}"
                
                if predictor == "pf": # Dont plot predictor importance if predictor is "pf"
                    pass
                else: # Only plot predictor importance on map if predictor is ERA5
                    fig, ax = evaluation.importance_map(pred_importance, lons, lats, tflag)

                    # Add position of station to map
                    #---
                    for station_name in station_names:
                        visualisation.plot_station(ax, station_positions, station_name, is_station_name)
                        
                    fname = f"importance_{predictor}_{str(percentile)[-2:]}_{run_id}"
                    fig.savefig(f"{folder}{fname}.pdf")

        #---
        # Confusion matrix
        #---
        # Format: 
        # Reality / Model: Negative, Positive
        # Negative    Right Negative, False Positive 
        # Positive    False Negative, Right Positive

        print("Show Confusion Matrix on testdata \n")
        cfm_fig1 = evaluation.plot_cf(model, X_test, y_test)
        cfm_fig1.show()

        print("Show Confusion Matrix on traindata \n")
        cfm_fig2 = evaluation.plot_cf(model, X_train, y_train)
        cfm_fig2.show()

        # Save CFM
        fname = f"{folder}testcf_matrix_{str(percentile)[-2:]}_{run_id}.pdf"
        cfm_fig1.savefig(fname)
        print(f"saved cf matrix to : {fname}")

        fname = f"{folder}traincf_matrix_{str(percentile)[-2:]}_{run_id}.pdf"
        cfm_fig2.savefig(fname)
        print(f"saved cf matrix to : {fname}")

        # Calculate CFM-Metrics
        print("Evaluate CFM Metrics of test data")
        metrics1 = evaluation.cfm_metrics(model, X_test, y_test)
        
        print("Evaluate CFM Metrics of train data")
        metrics2 = evaluation.cfm_metrics(model, X_train, y_train)

        fname = f"testcf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
        with open(f"{folder}{fname}", 'wb') as f:
            pickle.dump(metrics1, f)

        fname = f"traincf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
        with open(f"{folder}{fname}", 'wb') as f:
            pickle.dump(metrics2, f)

        print(f"saved cf metrics to : {fname}")

        #---
        # AUROC
        # Receiver Operating Characteristics & Area Under the Curve
        #---

        print("Show AUROC \n")

        y_test_proba = model.predict_proba(X_test)[:, 1] # Prob. for predicting 0 or 1, we only need second col

        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        auc = roc_auc_score(y_test, y_test_proba)

        print(f'AUC: {auc}')

        fig, ax = plt.subplots(tight_layout=True)

        ax.plot(fpr, tpr)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"AUROC with AUC = {auc}")

        fname = f"{folder}AUROC_{str(percentile)[-2:]}_{run_id}.pdf"
        fig.savefig(fname)
        print(f"saved AUROC to : {fname}")

        #---
        # Test on data completely unknown to model
        #---
        if range_of_years != unknown_range_of_years: # For stations that go from 2005-2020 no unknown range of years exists, e.g. No evaluation has to be made
            # Initialize
            #---
            X = []
            Y = []
            t = []

            if is_pf_combined==True:
                ts = preprocessing.intersect_all_times(predictors, gesla_predictand, unknown_range_of_years, subregion, season, preprocess)

            # Main
            #---
            era5_counter = 0
            for pred_idx, predictor in enumerate(predictors):
                print(f"Add predictor {predictor} to model input features")

                # Load data of predictor
                #---
                import xarray as xr
                if predictor == "pf":
                    is_prefilled = True

                    era5_predictor = data_loader.load_pf(season)

                    if is_pf_combined:
                        X_ = preprocessing.get_timeseries(era5_predictor, ts, is_prefilling=True)
                        Y_ = preprocessing.get_timeseries(gesla_predictand, ts, is_prefilling=True)
                    else:
                        # If pf is used without any ERA5 data, use hourly data of Degerby.
                        X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled) 

                else:
                    is_prefilled = False
                    era5_counter = era5_counter + 1
                    era5_predictor = data_loader.load_daymean_era5(unknown_range_of_years, subregion, season, predictor, preprocess) 
                    era5_predictor = preprocessing.convert_timestamp(era5_predictor, dim="time")

                    # Convert predictor sp.unit from Pa to hPA
                    #---
                    if predictor == "sp":
                        with xr.set_options(keep_attrs=True):
                            old_unit = era5_predictor.attrs["units"]
                            era5_predictor = era5_predictor / 100
                            era5_predictor.attrs["units"] = "hPa"
                            new_unit = era5_predictor.attrs["units"]
                            print(f"Converted units of {predictor} from {old_unit} to {new_unit}")
                            
                    if is_pf_combined:
                        X_ = preprocessing.get_timeseries(era5_predictor, ts, is_prefilling=False)
                        Y_ = preprocessing.get_timeseries(gesla_predictand, ts, is_prefilling=True)
                    else:
                        X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled)

                
                #---
                # Get timelags
                #---
                print(f"Introduce timelag: {timelags[pred_idx]} for {predictor}")
                print(f"Shapes X_: {X_.shape}, Y_: {Y_.shape}")

                X_timelag, Y_timelag = preprocessing.add_timelag(X_, Y_, timelags, pred_idx) 

                X.append(X_timelag)
                Y.append(Y_timelag)
                # t.append(t_)

            #--- 
            # Convert to format needed for model fit
            #---
            if is_pf_combined:
                Y = np.array(Y) 
                Y = Y[0, :] # Assume all timeseries are the same for the predictors.
                ndim = Y.shape[0]
                max_length = len(X)
                n_pfs = max_length - era5_counter # Number of prefilling predictors

                print(era5_counter, max_length, n_pfs, ndim)

                era5_x = np.array(X[:era5_counter])
                era5_x = era5_x.swapaxes(0, 1)
                era5_x = era5_x.reshape(ndim, -1)

                print(f"ERA5 shape: {era5_x.shape}")

                #--- 
                # Insert prefilling data at the beginning
                #---
                XX = era5_x
                for i in range(era5_counter, max_length):
                    print(f"Add Predictor {pred_names[i]} to end of X")
                    XX = np.append(XX, X[i], axis=1)


                y = Y[:, 0] # Select one station

                #---
                # Handle NaN Values
                #---

                # Insert numerical value that is not in data.
                # ML will hopefully recognize it.
                XX[np.where(np.isnan(XX))] = -999

                X = XX
                
                del XX
                print("Data is prepared as follows")
                print(f"X.shape : {X.shape}")
                print(f"y.shape : {y.shape}")
                print(f"pred_names : {pred_names}")
                print(f"pred_units : {pred_units}")
            else:
                # Convert to format needed for model fit
                #---      
                X = np.array(X)
                Y = np.array(Y) 
                Y = Y[0, :] # Assume all timeseries are the same for the predictors.
                # t = np.array(t)

                # Reshape for model input
                #---
                print(f"Reshape for model input")

                ndim = Y.shape[0]

                X = X.swapaxes(0, 1) # Put time dimension to front

                print(X.shape) # (time, timelags, predictor_combination, lon?, lat?)

                X = X.reshape(ndim, -1) # Reshapes into (time, pred1_lonlats:pred2_lonlats:...:predn_lonlats)
                y = Y[:, 0] # Select one station

                #---
                # Handle NaN Values
                #---

                # Insert numerical value that is not in data.
                # ML will hopefully recognize it.
                X[np.where(np.isnan(X))] = -999

                print("Data is prepared as follows")
                print(f"X.shape : {X.shape}")
                print(f"y.shape : {y.shape}")
                print(f"pred_names : {pred_names}")
                print(f"pred_units : {pred_units}")

            if is_scaled:
                print("Scale unknown test data")
                s = StandardScaler()
                s.fit(X)
                X = s.transform(X)

            # Evaluate Prediction on unknown data
            #---
            print("Show Confusion Matrix on unknown testdata \n")
            cfm_fig = evaluation.plot_cf(model, X, y)

            # Save CFM
            fname = f"{folder}unknowncf_matrix_{str(percentile)[-2:]}_{run_id}.pdf"
            cfm_fig.savefig(fname)
            print(f"saved cf matrix to : {fname}")

            # Calculate CFM-Metrics
            print("Evaluate CFM Metrics of unknown test data")
            metrics = evaluation.cfm_metrics(model, X, y)
            print(metrics)
            fname = f"unknowncf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
            with open(f"{folder}{fname}", 'wb') as f:
                pickle.dump(metrics, f)

        #--- 
        # Calculate runtime
        #---
        sec2min = 1 / 60
        toc_main = time.perf_counter()

        total_time = ((toc_main - tic_main) - time_predmap) * sec2min # Time of the whole model run excluding the computing time for predictor maps

        fname = f"clock_{run_id}"
        np.save(f"{folder}{fname}", total_time)
        print(f"saved runtime of {round(total_time, ndigits=2)} minutes to : {folder}{fname}")

        #---
        # Update run_id
        #---
        run_id = run_id + 1
        sys.stdout = orig_stdout # Restore original system output


#--- 
# Older version
# Error was only in evaluating unknown data. 
# When loading data, the time interval overlap was chosen poorly. 
#---
# def run(
# model_run, 
# run_id, 
# season,
# station_names,
# detrend_type, 
# predictors_of_model, 
# timelags_of_model, 
# models_path, 
# percentile, 
# clf, 
# hparam_grid, 
# optimizer, 
# k, 
# n_iter, 
# random_state,
# test_size,
# is_optimized, 
# is_scaled,
# is_overlay_importance,
# is_station_name,
# ):
#     """
#     Description:
#         Builds a model to predict (percentile) extreme storm surges at station_names using predictors for a specified season by using
#         a classifier.
#     Parameters:
#         season (str): Either winter or autumn
#         predictors (list): List of predictors (X) used to predict Storm Surge (y)
#         percentile (float): Percentile of Extreme Storm Surges
#         station_names (list): List of stations used from GESLA data-set
#         detrend_type (str): "constant" or "linear" for detrending GESLA Dataset with mean or linear-trend, respectively.
#         clf (sklearn.modeltype): Used model to do prediction, e.g. RandomForestClassifier, LogisticRegression,...
#         hparam_grid (dict): Selection of multiple Hyperparameters either used for building the model or fitted 
#                             to clf via the optimizer depending on is_optimized.
#                             If hparam_grid is used for optimization, values of dictionary need to be lists.
#         optimizer (str): Optimizer for automatic search of hyperparameters, either GridSearchCV or RandomSearchCV
#         run_id (int): Number of the run (used for saving output)
#         model_run (str): Naming of the model run (used for saving output)
#         k (int): Number of k-fold crossvalidations in model fit (Defaults: 3)
#         n_iter (int): Number of combinations used for RandomizedSearchCV (Defaults: None)
#         is_optimized (bool): Whether or not to search best combination of hparams within hparam_grid with 
#                              the optimizer (Defaults: True)
#                              If is_optiomized == False, hyperparameters are loaded from a model given in models_path
#         is_scaled (bool): Use StandardScaler to scale data if it is on different scales (Defaults: False)
#         is_overlay_importance (bool): Whether to plot predictor maps with overlayed importance positions
#         is_station_name (bool): Whether to plot station name over station itself.
#     Returns:
#         None. Saves Output in results>random_forest>model_run folder. Saves model in models>random_forest>model_run
#     """
#     import sys
#     #--- 
#     # Initialize
#     #---
#     preprocess = "preprocess1" # ["preprocess1"]
#     range_of_years = "1999-2008" # ["1999-2008", "2009-2018", "2019-2022",] 
#     unknown_range_of_years = "2009-2018"
#     subregion = "lon-0530_lat7040" # ["lon-0530_lat7040"]
#     lats, lons = preprocessing.get_lonlats(
#         range_of_years,
#         subregion,
#         season,
#         predictor="sp", # Does not matter which predictor. All predictors are sampled on same lon-lat field.
#         era5_import=preprocess,
#         )
#     colorbar_range = { # vmin vmax values for colorbar of predictor maps
#         'sp': np.array([ 980., 1020.,]),  # Low pressure systems <980hPa (see Theory Part)
#         'tp': np.array([0.    , 0.0018]),
#         'u10': np.array([-17.2,  17.2]), # Storm is defined by wind stronger than 17.2m/s
#         'v10': np.array([-17.2,  17.2]),
#         }
#     nlevels = 10 # For contourplot of predictor maps
#     orig_stdout = sys.stdout # Original Output for print etc.
    
#     #---
#     # Preprocess GESLA Data
#     #---

#     # Load Predictand
#     #---
#     gesla_predictand = data_loader.load_gesla(station_names)

#     # Get lon/lat positions of stations
#     #---
#     station_positions = gesla_preprocessing.station_position(gesla_predictand, station_names)

#     # Select a season
#     #---
#     gesla_predictand = gesla_preprocessing.select_season(gesla_predictand, season)

#     # Select only sea_level analysis data
#     #---
#     gesla_predictand = gesla_preprocessing.get_analysis(gesla_predictand)
#     gesla_predictand = gesla_predictand["sea_level"] # Select values

#     # Detrend 
#     #---
#     gesla_predictand = gesla_preprocessing.detrend_signal(gesla_predictand, type_=detrend_type) 

#     # Apply one hot encoding
#     #---
#     gesla_predictand = gesla_preprocessing.apply_dummies(gesla_predictand, percentile=percentile, level="station")
#     print(f"Applied one-hot-encoding with Percentile: {percentile}")

#     # Convert to DataArray
#     # nan values: no measurement at that timestamp for specific station
#     #---
#     gesla_predictand = gesla_predictand.to_xarray()

#     #---
#     # Loop over all run_ids of a model_run / predictors
#     #---
#     for idx, predictors in enumerate(predictors_of_model): # Loops over separate modelruns 
        
#         #---
#         # Save printed output to file
#         #---
#         folder = f"results/random_forest/{model_run}/"
#         saver.directory_existance(folder)
#         file_path = f'{folder}output_runid{run_id}.txt'
#         sys.stdout = open(file_path, "w") 

#         #---
#         # Load ERA5- Predictor
#         #---
        
#         # Initialize
#         #---
#         timelags = timelags_of_model[idx]
#         X = []
#         Y = []
#         t = []
#         pred_units = []
#         pred_names = []
#         if ("pf" in predictors) and (("sp" or "tp" or "u10" or "v10") in predictors):
#             is_pf_combined = True
#             print("Prefilling is combined with ERA5 Predictors")
#         else:
#             is_pf_combined = False
#             print("Prefilling is either not used or used alone as a predictor.")
            
#         tic_main = time.perf_counter()

#         # Main
#         #---
#         era5_counter = 0
#         for pred_idx, predictor in enumerate(predictors):
#             print(f"Add predictor {predictor} to model input features")

#             # Load data of predictor
#             #---
#             import xarray as xr
#             if predictor == "pf":
#                 is_prefilled = True

#                 era5_predictor = data_loader.load_pf(season)

#                 if is_pf_combined:
#                     # Set predictand to degerby proxy with intersected time. Timescale is reduced from hourly to daily like ERA5
#                     era5_predictor_tmp = data_loader.load_daymean_era5(range_of_years, subregion, season, predictors[era5_counter-1], preprocess) # Load any ERA5 predictor that is in combination
#                     era5_predictor_tmp = preprocessing.convert_timestamp(era5_predictor_tmp, dim="time")
#                     tmp, era5_predictor, tmp = preprocessing.intersect_time(era5_predictor_tmp, era5_predictor, is_prefilling=False) # Reduces hourly data at Degerby to daily data (max per day of sea level is chosen)
                    
#                     tmp, Y_, t_ = preprocessing.intersect_time(era5_predictor_tmp, gesla_predictand, is_prefilling=False)
                    
#                     X_ = era5_predictor 

#                     del tmp, era5_predictor_tmp
#                 else:
#                     # If pf is used without any ERA5 data, use hourly data of Degerby.
#                     X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled) 
                
#             else:
#                 is_prefilled = False
#                 era5_counter = era5_counter + 1
#                 era5_predictor = data_loader.load_daymean_era5(range_of_years, subregion, season, predictor, preprocess)
#                 era5_predictor = preprocessing.convert_timestamp(era5_predictor, dim="time")

#                 # Convert predictor sp.unit from Pa to hPA
#                 #---
#                 if predictor == "sp":
#                     with xr.set_options(keep_attrs=True):
#                         old_unit = era5_predictor.attrs["units"]
#                         era5_predictor = era5_predictor / 100
#                         era5_predictor.attrs["units"] = "hPa"
#                         new_unit = era5_predictor.attrs["units"]
#                         print(f"Converted units of {predictor} from {old_unit} to {new_unit}")
                
#                 X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled)

#             #---
#             # Get timelags
#             #---
#             print(f"Introduce timelag: {timelags[pred_idx]} for {predictor}")

#             X_timelag, Y_timelag = preprocessing.add_timelag(X_, Y_, timelags, pred_idx) 

#             X.append(X_timelag)
#             Y.append(Y_timelag)
#             t.append(t_)

#             # # Save unit and name of predictor
#             # #---
#             if predictor == "pf":
#                 pred_units.append("m")
#                 if is_pf_combined:
#                     pred_names.append(f"{predictor}_tlag{timelags[pred_idx]}") # Daily values of timelags if is_pf_combined
#                 else:
#                     pred_names.append(f"{predictor}_tlag{timelags[pred_idx] // 24}") # Hourly values converted back to daily values for interpretation
#             else:
#                 pred_units.append(era5_predictor.units)
#                 pred_names.append(f"{era5_predictor.name}_tlag{timelags[pred_idx]}")
        
#         #--- 
#         # Convert to format needed for model fit
#         #---
#         if is_pf_combined:
#             Y = np.array(Y) 
#             Y = Y[0, :] # Assume all timeseries are the same for the predictors.
#             ndim = Y.shape[0]
#             max_length = len(X)
#             n_pfs = max_length - era5_counter # Number of prefilling predictors

#             print(era5_counter, max_length, n_pfs, ndim)

#             era5_x = np.array(X[:era5_counter])
#             era5_x = era5_x.swapaxes(0, 1)
#             era5_x = era5_x.reshape(ndim, -1)

#             print(f"ERA5 shape: {era5_x.shape}")

#             #--- 
#             # Insert prefilling data at the beginning
#             #---
#             XX = era5_x
#             new_names = pred_names
#             new_units = pred_units
#             for i in range(era5_counter, max_length):
#                 print(f"Add Predictor {pred_names[i]} to end of X")
#                 XX = np.append(XX, X[i], axis=1)


#             y = Y[:, 0] # Select one station

#             #---
#             # Handle NaN Values
#             #---

#             # Insert numerical value that is not in data.
#             # ML will hopefully recognize it.
#             XX[np.where(np.isnan(XX))] = -999

#             X = XX
            
#             del XX
#             print("Data is prepared as follows")
#             print(f"X.shape : {X.shape}")
#             print(f"y.shape : {y.shape}")
#             print(f"pred_names : {pred_names}")
#             print(f"pred_units : {pred_units}")
#         else:
#             # Convert to format needed for model fit
#             #--- 
#             max_length = len(X)
#             n_pfs = max_length - era5_counter # Number of prefilling predictors

#             X = np.array(X)
#             Y = np.array(Y) 
#             Y = Y[0, :] # Assume all timeseries are the same for the predictors.
#             t = np.array(t)

#             # Reshape for model input
#             #---
#             print(f"Reshape for model input")

#             ndim = Y.shape[0]

#             X = X.swapaxes(0, 1) # Put time dimension to front

#             print(X.shape) # (time, timelags, predictor_combination, lon?, lat?)

#             X = X.reshape(ndim, -1) # Reshapes into (time, pred1_lonlats:pred2_lonlats:...:predn_lonlats)
#             y = Y[:, 0] # Select one station

#             #---
#             # Handle NaN Values
#             #---

#             # Insert numerical value that is not in data.
#             # ML will hopefully recognize it.
#             X[np.where(np.isnan(X))] = -999

#             print("Data is prepared as follows")
#             print(f"X.shape : {X.shape}")
#             print(f"y.shape : {y.shape}")
#             print(f"pred_names : {pred_names}")
#             print(f"pred_units : {pred_units}")
        
#         #---
#         # Apply Train-Test split 
#         #---
#         print("Apply train-test-split")
#         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

#         #---
#         # Scale data if they are on different scales
#         #---
#         X_test_unscaled = X_test

#         if is_scaled:
#             print("Scale training data")
#             s = StandardScaler()
#             s.fit(X_train)
#             X_train = s.transform(X_train)
#             X_test = s.transform(X_test)

#         #---
#         #  Optimize Hyperparameters
#         #---
#         if is_optimized:
#             print(f"Optimize Hyperparameters using {optimizer}")
#             print(f"Tested Hyperparameters: {hparam_grid}")
            
#             hparam_grid_opt = modelfit.optimize_hyperparameter(X_train, y_train, clf(), optimizer, hparam_grid, k, n_iter, n_jobs=-1)

#         if not is_optimized:
#             model_path = models_path[idx]
#             print(f"Hyperparameters are loaded from model: {model_path}")
#             with open(model_path, 'rb') as f:
#                 model = pickle.load(f)

#             hparam_grid_opt = model.get_params()
            
#         #---
#         #  Save hyperparameters
#         #---
#         folder = f"models/random_forest/{model_run}/" 
#         saver.directory_existance(folder)
#         saver.save_hpdict(hparam_grid_opt, run_id, model_run, percentile, folder)

#         print("Saved Hyperparameters")

#         #---
#         # Fit the model
#         #---
#         print(f"Fit model with hyperparameters {hparam_grid_opt}")

#         model = clf(**hparam_grid_opt) # One can set parameters afterwards via model.set_params() 

#         model.fit(X_train, y_train)

#         #---
#         # Saving the model
#         #---
#         print(f"Save model: {model}")
#         filename = f'{model_run}_{optimizer}_{run_id}.sav'
#         pickle.dump(model, open(f'{folder}{filename}', 'wb'))

#         #---
#         # Plot Predictor Maps
#         #---
#         tic_predmap = time.perf_counter()

#         if era5_counter != 0: # Do not print predictor maps if only prefilling is used as predictor. Otherwise print predictor maps
#                 print(f"Plot predictor maps")
#                 visualisation.predictor_maps( # TODO: If only pf is used this should be skipped, e.g. no plotting of importance maps
#                         model, X_test, y_test,
#                         X_test_unscaled,
#                         ndim, n_pfs, is_pf_combined, # Needed to separate ERA5 importance from Prefilling importance
#                         lons, lats, pred_units, pred_names, 
#                         station_positions, station_names, is_station_name, 
#                         is_overlay_importance, 
#                         run_id, model_run, 
#                         percentile=99., markersize=5, alpha=0.08, color="k", colorbar_range=colorbar_range, nlevels=nlevels,
#                         )
#         toc_predmap = time.perf_counter()
#         time_predmap = toc_predmap - tic_predmap # Save time of this separately as it is not necessary for model predictions

#         #---
#         # Evaluate model / Diagnostic
#         #--- 
#         print("Evaluate Model \n")

#         # Score & Importance
#         #---
#         test_score = model.score(X_test, y_test)
#         train_score = model.score(X_train, y_train)
#         relative_score = evaluation.relative_scores(train_score, test_score)
#         importance = model.feature_importances_

#         # Save Scores & Importance
#         #---
#         folder = f"results/random_forest/{model_run}/"
#         saver.directory_existance(folder)

#         fname = f"importance_{str(percentile)[-2:]}_{run_id}"
#         np.save(f"{folder}{fname}", importance)
#         print(f"saved importance to : {folder}{fname}")

#         fname = f"testscore_{str(percentile)[-2:]}_{run_id}"
#         np.save(f"{folder}{fname}", test_score)
#         print(f"saved testscore to : {folder}{fname}")

#         fname = f"trainscore_{str(percentile)[-2:]}_{run_id}"
#         np.save(f"{folder}{fname}", train_score)
#         print(f"saved trainscore to : {folder}{fname}")

#         fname = f"relativescore_{str(percentile)[-2:]}_{run_id}"
#         np.save(f"{folder}{fname}", relative_score)
#         print(f"saved relativescore to : {folder}{fname}")
        

#         # Plot importance of each predictor from combination
#         # Goal:
#         # 1. Separate importance per predictor
#         # 2. Plot importance of each predictor on lon lat map
#         #---
#         n_pred_features = len(lons) * len(lats) # Features per predictor (lon/lat Input-Field). Needed for importance separation

#         if is_pf_combined:
#             pf_importance = importance[-n_pfs:] 
#             era5_importance = importance[:-n_pfs]
#             predictor_importances = evaluation.separate_predictor_importance(era5_importance, n_pred_features) # Plot only importance map of era5 data
#         else:
#             if era5_counter != 0: # Only plot predictor importance if at least one predictor is ERA5
#                 predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features)
        
#         if era5_counter !=0: # Only plot predictor importance if at least one predictor is ERA5
#             for pred_idx, pred_importance in enumerate(predictor_importances): # TODO: If only pf is used this should be skipped, e.g. no plotting of importance maps
#                 # Plot importance map and save it
#                 predictor = pred_names[pred_idx] 
#                 tflag = f"{predictor}_{str(percentile)[-2:]}"
                
#                 if predictor == "pf": # Dont plot predictor importance if predictor is "pf"
#                     pass
#                 else: # Only plot predictor importance on map if predictor is ERA5
#                     fig, ax = evaluation.importance_map(pred_importance, lons, lats, tflag)

#                     # Add position of station to map
#                     #---
#                     for station_name in station_names:
#                         visualisation.plot_station(ax, station_positions, station_name, is_station_name)
                        
#                     fname = f"importance_{predictor}_{str(percentile)[-2:]}_{run_id}"
#                     fig.savefig(f"{folder}{fname}.pdf")

#         #---
#         # Confusion matrix
#         #---
#         # Format: 
#         # Reality / Model: Negative, Positive
#         # Negative    Right Negative, False Positive 
#         # Positive    False Negative, Right Positive

#         print("Show Confusion Matrix on testdata \n")
#         cfm_fig1 = evaluation.plot_cf(model, X_test, y_test)
#         cfm_fig1.show()

#         print("Show Confusion Matrix on traindata \n")
#         cfm_fig2 = evaluation.plot_cf(model, X_train, y_train)
#         cfm_fig2.show()

#         # Save CFM
#         fname = f"{folder}testcf_matrix_{str(percentile)[-2:]}_{run_id}.pdf"
#         cfm_fig1.savefig(fname)
#         print(f"saved cf matrix to : {fname}")

#         fname = f"{folder}traincf_matrix_{str(percentile)[-2:]}_{run_id}.pdf"
#         cfm_fig2.savefig(fname)
#         print(f"saved cf matrix to : {fname}")

#         # Calculate CFM-Metrics
#         print("Evaluate CFM Metrics of test data")
#         metrics1 = evaluation.cfm_metrics(model, X_test, y_test)
        
#         print("Evaluate CFM Metrics of train data")
#         metrics2 = evaluation.cfm_metrics(model, X_train, y_train)

#         fname = f"testcf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
#         with open(f"{folder}{fname}", 'wb') as f:
#             pickle.dump(metrics1, f)

#         fname = f"traincf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
#         with open(f"{folder}{fname}", 'wb') as f:
#             pickle.dump(metrics2, f)

#         print(f"saved cf metrics to : {fname}")

#         #---
#         # AUROC
#         # Receiver Operating Characteristics & Area Under the Curve
#         #---

#         print("Show AUROC \n")

#         y_test_proba = model.predict_proba(X_test)[:, 1] # Prob. for predicting 0 or 1, we only need second col

#         fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
#         auc = roc_auc_score(y_test, y_test_proba)

#         print(f'AUC: {auc}')

#         fig, ax = plt.subplots(tight_layout=True)

#         ax.plot(fpr, tpr)
#         ax.set_xlabel("FPR")
#         ax.set_ylabel("TPR")
#         ax.set_title(f"AUROC with AUC = {auc}")

#         fig.show()

#         fname = f"{folder}AUROC_{str(percentile)[-2:]}_{run_id}.pdf"
#         fig.savefig(fname)
#         print(f"saved AUROC to : {fname}")

#         #---
#         # Test on data completely unknown to model
#         # TODO: This can also be done if pf is in predictors and is_pf_combined is switched on. Then I need to load predictor values as before.
#         #---
#         # Initialize
#         #---

#         X = []
#         Y = []
#         t = []

#         # Load unknown data
#         #---
#         era5_counter = 0
#         for pred_idx, predictor in enumerate(predictors):
#             print(f"Add predictor {predictor} to model input features")

#             # Load data of predictor
#             #---
#             import xarray as xr
#             if predictor == "pf":
#                 is_prefilled = True

#                 era5_predictor = data_loader.load_pf(season)

#                 if is_pf_combined:
#                     # Set predictand to degerby proxy with intersected time. Timescale is reduced from hourly to daily like ERA5
#                     era5_predictor_tmp = data_loader.load_daymean_era5(unknown_range_of_years, subregion, season, predictors[era5_counter-1], preprocess) # Load any ERA5 predictor that is in combination
#                     era5_predictor_tmp = preprocessing.convert_timestamp(era5_predictor_tmp, dim="time")
#                     tmp, era5_predictor, tmp = preprocessing.intersect_time(era5_predictor_tmp, era5_predictor, is_prefilling=False) # Reduces hourly data at Degerby to daily data (max per day of sea level is chosen)
                    
#                     tmp, Y_, t_ = preprocessing.intersect_time(era5_predictor_tmp, gesla_predictand, is_prefilling=False)
                    
#                     X_ = era5_predictor 

#                     del tmp, era5_predictor_tmp
#                 else:
#                     # If pf is used without any ERA5 data, use hourly data of Degerby.
#                     X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled) 
                
#             else:
#                 is_prefilled = False
#                 era5_counter = era5_counter + 1
#                 era5_predictor = data_loader.load_daymean_era5(unknown_range_of_years, subregion, season, predictor, preprocess)
#                 era5_predictor = preprocessing.convert_timestamp(era5_predictor, dim="time")

#                 # Convert predictor sp.unit from Pa to hPA
#                 #---
#                 if predictor == "sp":
#                     with xr.set_options(keep_attrs=True):
#                         old_unit = era5_predictor.attrs["units"]
#                         era5_predictor = era5_predictor / 100
#                         era5_predictor.attrs["units"] = "hPa"
#                         new_unit = era5_predictor.attrs["units"]
#                         print(f"Converted units of {predictor} from {old_unit} to {new_unit}")
                
#                 X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand, is_prefilled)

#             #---
#             # Get timelags
#             #---
#             print(f"Introduce timelag: {timelags[pred_idx]} for {predictor}")

#             print(f"with shape {X_.shape}")
#             X_timelag, Y_timelag = preprocessing.add_timelag(X_, Y_, timelags, pred_idx) 

#             X.append(X_timelag)
#             Y.append(Y_timelag)
#             t.append(t_)

#         #--- 
#         # Convert to format needed for model fit
#         #---
#         if is_pf_combined:
#             Y = np.array(Y) 
#             Y = Y[0, :] # Assume all timeseries are the same for the predictors.
#             ndim = Y.shape[0]
#             max_length = len(X)
#             n_pfs = max_length - era5_counter # Number of prefilling predictors

#             print(era5_counter, max_length, n_pfs, ndim)

#             era5_x = np.array(X[:era5_counter])
#             era5_x = era5_x.swapaxes(0, 1)
#             era5_x = era5_x.reshape(ndim, -1)

#             print(f"ERA5 shape: {era5_x.shape}")

#             #--- 
#             # Insert prefilling data at the beginning
#             #---
#             XX = era5_x
#             for i in range(era5_counter, max_length):
#                 print(f"Add Predictor {pred_names[i]} to end of X")
#                 XX = np.append(XX, X[i], axis=1)


#             y = Y[:, 0] # Select one station

#             #---
#             # Handle NaN Values
#             #---

#             # Insert numerical value that is not in data.
#             # ML will hopefully recognize it.
#             XX[np.where(np.isnan(XX))] = -999

#             X = XX
            
#             del XX
#             print("Data is prepared as follows")
#             print(f"X.shape : {X.shape}")
#             print(f"y.shape : {y.shape}")
#             print(f"pred_names : {pred_names}")
#             print(f"pred_units : {pred_units}")
#         else:
#             # Convert to format needed for model fit
#             #---      
#             X = np.array(X)
#             Y = np.array(Y) 
#             Y = Y[0, :] # Assume all timeseries are the same for the predictors.
#             t = np.array(t)

#             # Reshape for model input
#             #---
#             print(f"Reshape for model input")

#             ndim = Y.shape[0]

#             X = X.swapaxes(0, 1) # Put time dimension to front

#             print(X.shape) # (time, timelags, predictor_combination, lon?, lat?)

#             X = X.reshape(ndim, -1) # Reshapes into (time, pred1_lonlats:pred2_lonlats:...:predn_lonlats)
#             y = Y[:, 0] # Select one station

#             #---
#             # Handle NaN Values
#             #---

#             # Insert numerical value that is not in data.
#             # ML will hopefully recognize it.
#             X[np.where(np.isnan(X))] = -999

#             print("Data is prepared as follows")
#             print(f"X.shape : {X.shape}")
#             print(f"y.shape : {y.shape}")
#             print(f"pred_names : {pred_names}")
#             print(f"pred_units : {pred_units}")

#         if is_scaled:
#             print("Scale unknown test data")
#             s = StandardScaler()
#             s.fit(X)
#             X = s.transform(X)

#         # Evaluate Prediction on unknown data
#         #---
#         print("Show Confusion Matrix on unknown testdata \n")
#         cfm_fig = evaluation.plot_cf(model, X, y)
#         cfm_fig.show()

#         # Save CFM
#         fname = f"{folder}unknowncf_matrix_{str(percentile)[-2:]}_{run_id}.pdf"
#         cfm_fig.savefig(fname)
#         print(f"saved cf matrix to : {fname}")

#         # Calculate CFM-Metrics
#         print("Evaluate CFM Metrics of unknown test data")
#         metrics = evaluation.cfm_metrics(model, X, y)
#         print(metrics)
#         fname = f"unknowncf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
#         with open(f"{folder}{fname}", 'wb') as f:
#             pickle.dump(metrics, f)

#         #--- 
#         # Calculate runtime
#         #---
#         sec2min = 1 / 60
#         toc_main = time.perf_counter()

#         total_time = ((toc_main - tic_main) - time_predmap) * sec2min # Time of the whole model run excluding the computing time for predictor maps

#         fname = f"clock_{run_id}"
#         np.save(f"{folder}{fname}", total_time)
#         print(f"saved runtime of {round(total_time, ndigits=2)} minutes to : {folder}{fname}")

#         #---
#         # Update run_id
#         #---
#         run_id = run_id + 1
#         sys.stdout = orig_stdout # Restore original system output

