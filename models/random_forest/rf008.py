# Modules
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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

from models import modelfit
from models import evaluation

#---
# Main
#---
def run(season, predictors, percentile, station_names, clf, hparam_grid, optimizer, run_id, model_run, k=3, n_iter=None, is_optimized=True, is_scaled=False):
    """
    Description:
        Builds a model to predict (percentile) extreme storm surges at station_names using predictors for a specified season by using
        a classifier.
    Parameters:
        season (str): Either winter or autumn
        predictors (list): List of predictors (X) used to predict Storm Surge (y)
        percentile (float): Percentile of Extreme Storm Surges
        station_names (list): List of stations used from GESLA data-set
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
        is_scaled (bool): Use StandardScaler to scale data if it is on different scales (Defaults: False)
    Returns:
        None. Saves Output in results>random_forest>model_run folder. Saves model in models>random_forest>model_run
    """
    #---
    # Modularize Preprocessing
    #---

    # Get timeseries of predictor and predictand
    preprocess = "preprocess1" # ["preprocess1"]
    range_of_years = "1999-2008" # ["1999-2008", "2009-2018", "2019-2022",]
    subregion = "lon-0530_lat7040" # ["lon-0530_lat7040"]

    #---
    # Preprocess GESLA Data
    #---

    # Load Predictand
    #---
    gesla_predictand = data_loader.load_gesla(station_names)

    # Select a season
    #---
    gesla_predictand = gesla_preprocessing.select_season(gesla_predictand, season)

    # Select only sea_level analysis data
    #---
    gesla_predictand = gesla_preprocessing.get_analysis(gesla_predictand)

    # Subtract mean of data grouped by station
    #---
    gesla_predictand = gesla_predictand["sea_level"] # Detrend expects pd.Series
    gesla_predictand = gesla_preprocessing.detrend(gesla_predictand, level="station")

    # Apply one hot encoding
    gesla_predictand = gesla_preprocessing.apply_dummies(gesla_predictand, percentile=percentile, level="station")
    print(f"Applied one-hot-encoding with Percentile: {percentile}")

    # Convert to DataArray
    # nan values: no measurement at that timestamp for specific station
    gesla_predictand = gesla_predictand.to_xarray()

    # Load already preprocessed Era5 Data
    # Preprocessing done with cdo
    #---
    X = []
    Y = []
    t = []
    for predictor in predictors:
        print(f"Add predictor {predictor} to model input features")

        era5_predictor = data_loader.load_daymean_era5(range_of_years, subregion, season, predictor, preprocess)
        X_, Y_, t_ = preprocessing.intersect_time(era5_predictor, gesla_predictand)

        X.append(X_)
        Y.append(Y_)
        t.append(t_)

    X = np.array(X)
    Y = np.array(Y)
    t = np.array(t)

    #---
    # Assert that timeinterval of all predictors are the same
    #---
    def assert_timeintervals(t1, t2):
        """
        Description: Checks if time intervals are the same. Throws assertion error if not.
        Parameters:
            t1, t2 (np.array): Arrays of time points, shape:(timepoints,)
        """

        if len(np.where(t1 != t2)[0]) == 0:
            is_equal = True
        else:
            is_equal = False

        assert is_equal, "Timeinterval of predictand is not equal"

        print("Time-interval is the same")

    # Check if time interval is the same for all timeseries of predictors 
    print("Assert that timeinterval of all predictors are the same")  
    for i, timeseries in enumerate(t):
        assert_timeintervals(t[i], timeseries)

    # If all timeseries are equal, reduce to one timeseries for all predictors
    t = t[0, :]
    Y = Y[0] # Choose entries of first predictor, since Y[i] == Y[j] for all i != j.

    print("All Time-intervals are the same")  

    #---
    #  Get multiple predictor in shape for model input
    #---
    print("Prepare input data for model training")

    ndim = t.shape[0]
    X = np.swapaxes(X, axis1=0, axis2=1,) # Swap predictor dimension with time dimension

    # Reshape hängt die LonLat Werte für alle Prädiktoren aneinander durch .flatten().
    X = X.reshape(ndim, -1) # shape:(time, pred1_lonlats:pred2_lonlats:...), i.e. (time, features)
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

    #---
    # Train Model
    #---
    print("Start Model Training")

    # Train-Test Split
    print("Do Train-Test-Split")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    # Scale data if they are on different scales
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
        hparam_grid = modelfit.optimize_hyperparameter(X_train, y_train, clf(), optimizer, hparam_grid, k, n_iter, n_jobs=-1)

    #---
    #  Save hyperparameters
    #---
    folder = f"models/random_forest/{model_run}/" 
    saver.directory_existance(folder)
    saver.save_hpdict(hparam_grid, run_id, model_run, percentile, folder)

    print("Saved Hyperparameters")

    #---
    # Fit the model
    #---
    print(f"Fit model with hyperparameters {hparam_grid}")

    model = clf(**hparam_grid) # One can set parameters afterwards via model.set_params()

    model.fit(X_train, y_train)

    #---
    # Saving the model
    #---
    print("Save model")
    filename = f'{model_run}_{optimizer}_{run_id}.sav'
    pickle.dump(model, open(f'{folder}{filename}', 'wb'))

    #---
    # Evaluate model / Diagnostic
    #--- 
    print("Evaluate Model \n")

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
    # Get lat,lons of this model run
    lats, lons = preprocessing.get_lonlats(
        range_of_years,
        subregion,
        season,
        predictor="sp", # Does not matter which predictor. All predictors are sampled on same lon-lat field.
        era5_import=preprocess,
    )
    n_pred_features = len(lons) * len(lats) # Features per predictor (lon/lat Input-Field). Needed for importance separation

    predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features)

    for pred_idx, pred_importance in enumerate(predictor_importances):
        # Plot importance map and save it
        predictor = predictors[pred_idx]
        tflag = f"{predictor}_{str(percentile)[-2:]}"

        fig = evaluation.importance_map(pred_importance, lons, lats, tflag)
        
        fname = f"importance_{predictor}_{str(percentile)[-2:]}_{run_id}"
        fig.savefig(f"{folder}{fname}.pdf")
        
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
    metrics1 = evaluation.cfm_metrics(model, X_test, y_test)
    metrics2 = evaluation.cfm_metrics(model, X_train, y_train)

    fname = f"testcf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
    with open(f"{folder}{fname}", 'wb') as f:
        pickle.dump(metrics1, f)

    fname = f"traincf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"
    with open(f"{folder}{fname}", 'wb') as f:
        pickle.dump(metrics2, f)

    print(f"saved cf metrics to : {fname}")


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

    fig.show()

    fname = f"{folder}AUROC_{str(percentile)[-2:]}_{run_id}.pdf"
    fig.savefig(fname)
    print(f"saved AUROC to : {fname}")