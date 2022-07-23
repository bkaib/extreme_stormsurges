# Modules
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

from data import data_loader
from data import gesla_preprocessing
from data import era5_preprocessing
from data import preprocessing
from data import saver

from models import modelfit
from models import evaluation
from models import evaluation

def run(season, predictor, optimizer, clf, k, n_iter, param_grid, percentile, station_names, run_id, model_run):

    #---
    # Modularize Preprocessing
    #---

    # Get timeseries of predictor and predictand
    preprocess = "preprocess1" # ["preprocess1"]
    range_of_years = "1999-2008" # ["1999-2008", "2009-2018", "2019-2022",]
    subregion = "lon-0530_lat7040" # ["lon-0530_lat7040"]

    # Load already preprocessed Era5 Data
    # Preprocessing done with cdo
    #---
    era5_predictor = data_loader.load_daymean_era5(range_of_years, subregion, season, predictor, preprocess)

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

    #---
    # Get overlapping time-series
    #---
    X, Y, t = preprocessing.intersect_time(era5_predictor, gesla_predictand)

    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")

    # Save number of lat/lon for interpreting model output later
    ndim = X.shape[0]
    nlat = X.shape[1]
    nlon = X.shape[2]

    # Prepare shape for model
    X = X.reshape(ndim, -1) # (ndim, nclasses)
    y = Y[:, 0] # Select only one station

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

    #---
    #  Optimize Hyperparameters
    #---

    print(f"Optimize Hyperparameters from grid: {param_grid} with {optimizer}")
    best_params = modelfit.optimize_hyperparameter(X_train, y_train, clf, optimizer, param_grid, k, n_iter=n_iter, n_jobs=-1)
    
    #---
    #  Save hyperparameters
    #---

    folder = f"models/random_forest/{model_run}/" 
    saver.directory_existance(folder)
    saver.save_hpdict(best_params, run_id, model_run, percentile, folder)

    print("Saved Hyperparameters")

    #---
    # Fit the model
    #---
    print("Fit model")

    model = RandomForestClassifier(criterion='gini',
    n_estimators=best_params["n_estimators"], #- nTrees 
    max_depth=best_params["max_depth"], 
    min_samples_leaf=best_params["min_samples_leaf"],
    min_samples_split=best_params["min_samples_split"],
    random_state=0, # To compare results when changing hyperparameters
    class_weight="balanced",
    oob_score=True,
    )

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

    # Plot importance
    #---
    # Get lat,lons of this model run
    lats, lons = preprocessing.get_lonlats(
        range_of_years,
        subregion,
        season,
        predictor,
        era5_import=preprocess,
    )

    n_pred_features = len(lons) * len(lats) # Features per predictor (lon/lat Input-Field). Needed for importance separation

    folder = f"results/random_forest/{model_run}/"
    saver.directory_existance(folder)

    predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features)
    for subclass_idx, pred_importance in enumerate(predictor_importances):
        # Plot importance map and save it
        tflag = f"{predictor}_subclass{subclass_idx}_{str(percentile)[-2:]}"

        fig = evaluation.importance_map(pred_importance, lons, lats, tflag)
        
        fname = f"importance_{predictor}_subclass{subclass_idx}_{str(percentile)[-2:]}_{run_id}"
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