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
season = "winter" # ["winter", "autumn",] 
predictors = ["sp", "tp", "u10", "v10",]
percentile = 0.99 # [0.95, 0.99,] 
era5_import = "preprocess1" # ["preprocess1"]
range_of_years = "1999-2008" # ["1999-2008", "2009-2018", "2019-2022",]
subregion = "lon-0530_lat7040" # ["lon-0530_lat7040"]
station_names = ["hanko-han-fin-cmems",]

for predictor in predictors:
    # Load already preprocessed Era5 Data
    # Preprocessing done with cdo
    #---
    era5_predictor = data_loader.load_daymean_era5(range_of_years, subregion, season, predictor, era5_import)

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
    print(f"t: {t.shape}")

    # Reshape for model input
    #---

    ndim = t.shape[0]
    X = X.reshape(ndim, -1) # (ndim, nclasses)
    y = Y[:, 0] # Select one station
    print(X.shape)
    print(y.shape)

    #---
    # Handle NaN Values
    #---

    # Insert numerical value that is not in data.
    # ML will hopefully recognize it.
    X[np.where(np.isnan(X))] = -999

    #---
    # Train Model
    #---
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    #---
    #  Optimize Hyperparameters
    #---
    def optimize_hyperparameter(X_train, y_train, clf, optimizer, param_grid, k, n_iter=None, n_jobs=-1):
        """
        Description: 
            Return best hyperparameters for a model based on chosen optimizer
        Parameters:
            X_train (): Predictor train data
            y_train (): Predictand train data

            clf (): Base Model
            optimizer (): GridSearchCV or RandomizedSearchCV
            param_grid (dict): Dictionary with hyperparameter ranges
            k (int): k-fold Cross-Validation
            n_iter (int): Number of combinations used for RandomizedSearchCV (Defaults:None)
            n_jobs (int): Number of processor used. (Defaults:-1, e.g. all processors)
        """
        # Modules
        #---
        from sklearn.model_selection import GridSearchCV 
        from sklearn.model_selection import RandomizedSearchCV
        # RandomSearchCV
        #---
        print(f"Optimize Hyperparameters using {optimizer}")

        if optimizer == "RandomSearchCV":
            assert n_iter != None, f"{optimizer} needs number of combinations."
            opt_model = RandomizedSearchCV(estimator=clf, 
            param_distributions = param_grid, 
            n_iter = n_iter, 
            cv = k, 
            verbose = 2, 
            random_state = 0, 
            n_jobs = n_jobs,
            )

        # GridSearchCV
        #---
        if optimizer == "GridSearchCV":
            # Instantiate the grid search model
            opt_model = GridSearchCV(estimator=clf, 
            param_grid = param_grid, 
            cv = k,
            verbose = 2, 
            n_jobs = n_jobs, 
            )

        # Fit the random search model
        opt_model.fit(X_train, y_train)

        # Best Params
        #---
        return opt_model.best_params_

    #---
    #  Optimize Hyperparameters
    #---
    clf = RandomForestClassifier()
    optimizer = "RandomSearchCV"
    k = 3 # k-fold cross-validation
    n_iter = 100 # number of combinations

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True,]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    print(f"Tested Hyperparameters: {param_grid}")

    best_params = optimize_hyperparameter(X_train, y_train, clf, optimizer, param_grid, k, n_iter=n_iter, n_jobs=-1)

    # Save hyperparameters
    from data import saver
    model_run = f"RandomForest_{optimizer}"
    folder = "models/random_forest/rf003/" 
    saver.save_hpdict(best_params, predictor, model_run, percentile, folder)

    # Fit the model
    model = RandomForestClassifier(criterion='gini',
    n_estimators=best_params["n_estimators"], #- nTrees 
    max_depth=best_params["max_depth"], 
    max_features=best_params["max_features"],
    min_samples_leaf=best_params["min_samples_leaf"],
    min_samples_split=best_params["min_samples_split"],
    bootstrap=best_params["bootstrap"],
    random_state=0, # To compare results when changing hyperparameters
    class_weight="balanced",
    oob_score=True,
    )

    model.fit(X_train, y_train)

    # Saving the model
    pflag = str(percentile)[-2:]
    folder = "models/random_forest/rf003/"
    filename = f'RandomForest_{optimizer}_{predictor}{pflag}.sav'
    pickle.dump(model, open(f'{folder}{filename}', 'wb'))

    #---
    # Evaluate model / Diagnostic
    #--- 
    print("Evaluate Model \n")

    # Score & Importance
    #---
    test_score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    importances = model.feature_importances_
    folder = "results/random_forest/rf003/"
    print(f"test_score: {test_score}")
    print(f"train_score: {train_score}")
    print(f"importances: {importances}")

    fname = f"importances_{predictor}{str(percentile)[-2:]}"
    np.save(f"{folder}{fname}", importances)
    print(f"saved importances to : {folder}{fname}")

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
    fname = f"{folder}cf_matrix_{predictor}{str(percentile)[-2:]}.jpg"
    cfm_fig.savefig(fname)
    print(f"saved cf matrix to : {fname}")

    # Calculate CFM-Metrics
    metrics = evaluation.cfm_metrics(model, X_test, y_test)
    fname = f"cf_metrics_{predictor}{str(percentile)[-2:]}.pkl"

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

    fname = f"{folder}AUROC_{predictor}{str(percentile)[-2:]}.jpg"
    fig.savefig(fname)
    print(f"saved AUROC to : {fname}")

    #---
    # Visualization
    #---

    # Metric: Importance
    #---
    lats, lons = preprocessing.get_lonlats(
        range_of_years,
        subregion,
        season,
        predictor,
        era5_import,
    )

    # Plot importance-map
    from models import evaluation
    tflag = f"{predictor}{str(percentile)[-2:]}"
    fig = evaluation.importance_map(importances, lons, lats, tflag=tflag)

    # Save importance-map
    fname = f"importance_map_{predictor}{str(percentile)[-2:]}"
    fig.savefig(f"{folder}{fname}.jpg")