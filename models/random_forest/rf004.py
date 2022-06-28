def run(season, predictors, percentile, station_names, run_id):
    # Modules
    import numpy as np
    import xarray as xr
    import pandas as pd

    from data import data_loader
    from data import gesla_preprocessing
    from data import era5_preprocessing
    from data import preprocessing

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

        assert is_equal, "Timeinterval is not equal"

        print("Time-interval is the same")

    # Check if time interval is the same for all timeseries of predictors 
    print("Assert that timeinterval of all predictors are the same")  
    for i, timeseries in enumerate(t):
        assert_timeintervals(t[i], timeseries)

    # If all timeseries are equal, reduce to one timeseries for all predictors
    t = t[0, :]
    Y = Y[0]

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
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    print("Start Model Training")

    # Train-Test Split
    print("Do Train-Test-Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    #---
    #  Optimize Hyperparameters
    #---
    from models import modelfit

    print("Optimize Hyperparameters")

    clf = RandomForestClassifier()
    optimizer = "RandomSearchCV"
    k = 3 # k-fold cross-validation
    n_iter = 100 # number of combinations

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 0, stop = 1000, num = 10)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 55, num = 5)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                }

    print(f"Tested Hyperparameters: {param_grid}")

    best_params = modelfit.optimize_hyperparameter(X_train, y_train, clf, optimizer, param_grid, k, n_iter=n_iter, n_jobs=-1)

    #---
    #  Save hyperparameters
    #---
    from data import saver

    model_run = "rf004"
    folder = f"models/random_forest/{model_run}/" 
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
    filename = f'{model_run}{optimizer}_{run_id}.sav'
    pickle.dump(model, open(f'{folder}{filename}', 'wb'))

    #---
    # Evaluate model / Diagnostic
    #--- 
    print("Evaluate Model \n")

    # Score & Importance
    #---
    test_score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    importance = model.feature_importances_

    print(f"test_score: {test_score}")
    print(f"train_score: {train_score}")
    print(f"importance: {importance}")

    folder = f"results/random_forest/{model_run}/"
    fname = f"importance_{str(percentile)[-2:]}_{run_id}"
    np.save(f"{folder}{fname}", importance)
    print(f"saved importance to : {folder}{fname}")

    # Plot importance of each predictor from combination
    # Goal:
    # 1. Separate importance per predictor
    # 2. Plot importance of each predictor on lon lat map
    #---
    from models import evaluation
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
        
        folder = f"results/random_forest/{model_run}/"
        fname = f"importance_{predictor}_{str(percentile)[-2:]}_{run_id}"
        fig.savefig(f"{folder}{fname}.pdf")
        
    # Confusion matrix
    #---
    # Format: 
    # Reality / Model: Negative, Positive
    # Negative    Right Negative, False Positive 
    # Positive    False Negative, Right Positive

    from sklearn.metrics import confusion_matrix
    from models import evaluation

    print("Show Confusion Matrix \n")

    cfm_fig = evaluation.plot_cf(model, X_test, y_test)
    cfm_fig.show()

    # Save CFM
    fname = f"{folder}cf_matrix_{str(percentile)[-2:]}_{run_id}.jpg"
    cfm_fig.savefig(fname)
    print(f"saved cf matrix to : {fname}")

    # Calculate CFM-Metrics
    metrics = evaluation.cfm_metrics(model, X_test, y_test)
    fname = f"cf_metrics_{str(percentile)[-2:]}_{run_id}.pkl"

    with open(f"{folder}{fname}", 'wb') as f:
        pickle.dump(metrics, f)

    print(f"saved cf metrics to : {fname}")


    # AUROC
    # Receiver Operating Characteristics & Area Under the Curve
    #---
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

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

    fname = f"{folder}AUROC_{str(percentile)[-2:]}_{run_id}.jpg"
    fig.savefig(fname)
    print(f"saved AUROC to : {fname}")

#---
# Main
#---
season = "winter" # ["winter", "autumn",] 
percentile = 0.95 # [0.95, 0.99,] 
station_names = ["hanko-han-fin-cmems",]

preds = [
    ["sp", "u10",], # run_id 0
    ["sp", "tp", "u10",], # run_id 1
    ["sp", "tp",],
    ["tp","u10"],
    ["sp", "tp", "u10", "v10"],
]

for run_id, predictors in enumerate(preds):
    run(season, predictors, percentile, station_names, run_id)