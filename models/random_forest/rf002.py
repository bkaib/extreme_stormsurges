def run(predictor="sp", percentile=0.95): 
    #---
    #  Modules
    #---
    print("Import Modules\n")
    import numpy as np
    from data import preprocessing

    #---
    # Initialize
    #---
    folder = "results/random_forest/rf002/" # Where to save results

    # ---
    # Preprocessing
    # ---

    # Get timeseries of predictor and predictand
    season = "winter"
    
    # Description of data
    print("Description \n")
    print("Model: Random Forest")
    print("Predictand: Classification (0,1)")
    print(f"Percentile Predictand: {percentile}")
    print(f"Predictors: {predictor} ")
    print(f"Seaon: {season}")
    print("Preprocessing / Region: preprocessing1, only one station selected")

    print("Start Preprocessing of Data\n\n")

    X, Y, t = preprocessing.preprocessing1(season, predictor, percentile)

    # Handle NaN values: 
    # Insert numerical value that is not in data.
    # ML will hopefully recognize it.
    X[np.where(np.isnan(X))] = -999

    # Save number of lat/lon for interpreting model output later
    ndim = X.shape[0]
    nlat = X.shape[1]
    nlon = X.shape[2]

    # Prepare shape for model
    X = X.reshape(ndim, -1) # (ndim, nclasses)
    y = Y[:, 0] # Select only one station

    #---
    # Train Model
    #---
    print("Start model training\n")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

    # Setup Model

    model = RandomForestClassifier(criterion='gini',
    n_estimators=91, #- nTrees 
    max_depth=9, 
    random_state=0, # To compare results when changing hyperparameters
    class_weight="balanced",
    oob_score=True,
    )

    model.fit(X_train, y_train)

    #---
    # Evaluate model / Diagnostic
    #--- 
    print("Evaluate Model \n")

    # Score & Importance
    #---
    test_score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)
    importances = model.feature_importances_

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
    import pickle
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


    print("End of Model run")

def optimization(percentile=0.95):
    """
    Description:
        Optimization of Hyperparameters applied to this run in order to get chosen hyperparameters in run().
        For RandomForestClassifier Parameters:
            n_estimators
            max_depth
            min_samples_leaf
            min_samples_split
    
    Parameters:
        percentile (float): Percentile of predictand data. (Defaults=0.95)

    Returns:
        Saves plots of validation curves for all predictors.
    """
    #---
    # Optimization: Hyperparameters
    #---
    from data import preprocessing
    import numpy as np
    # Get timeseries of predictor and predictand
    predictors = ["sp", "tp", "u10", "v10",]
    season = "winter"

    # Hyperparameters to optimize
    param_name = ('min_samples_leaf', "max_depth", "min_samples_split", "n_estimators",)
    param_range = (np.arange(0, 10, 2), np.arange(5, 30, 5), np.arange(0, 25, 5), np.arange(100, 1200, 200),)
    hparams = dict(zip(param_name, param_range))

    for predictor in predictors:
        X, Y, t = preprocessing.preprocessing1(season, predictor, percentile)

        # Handle NaN values: 
        # Insert numerical value that is not in data.
        # ML will hopefully recognize it.
        X[np.where(np.isnan(X))] = -999

        # Save number of lat/lon for interpreting model output later
        ndim = X.shape[0]
        nlat = X.shape[1]
        nlon = X.shape[2]

        # Prepare shape for model
        X = X.reshape(ndim, -1) # (ndim, nclasses)
        y = Y[:, 0] # Select only one station

        #---
        # Optimization: Hyperparameters with Validation Curve
        #---

        # Train-Test Split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

        from sklearn.model_selection import validation_curve
        import numpy as np

        for param_name, param_range in hparams.items():
            train_scores, test_scores = validation_curve(
            estimator=RandomForestClassifier(criterion="gini",
            class_weight="balanced",
            ),
            X=X_train,
            y=y_train,
            param_name=param_name, #- varying for this parameter
            param_range=param_range, #- for this parameters
            cv=3,
            )
            
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(tight_layout=True)

            ax.plot(param_range, np.mean(train_scores, axis=1), label='train')
            ax.plot(param_range, np.mean(test_scores, axis=1), label='test-cv')
            ax.set_title(f"Validation Curve for: {param_name}, Predictor: {predictor}")
            ax.legend()

            fig.show()

            folder = "results/random_forest/rf002/"
            fname = f"vc_{param_name}_{predictor}{str(percentile)[-2:]}"
            fig.savefig(f"{folder}{fname}.jpg")

        print("All Validation Curves saved")

def save_hp(hparams, predictor, model_run="rf002", percentile=0.95,):
    """
    Description: 
        Saves Hyperparameters manually read from validation curves in 
        master_thesis/results/random_forest/rf002.
        Saves to "models/random_forest/hyperparameter/" 

    Parameters:
        hparams (tuple): Fixed values of hyperparameters ('min_samples_leaf', "max_depth", "min_samples_split", "n_estimators",)
        predictor (str): Flag of predictor ["sp", "tp", "u10", "v10"]
        model_run (str): Flag for model run. (Default:"rf002")
        percentile (float): percentile of preprocessing used for validation curves (Defaults: 0.95)
    
    Returns:
        None
    """
    import pickle
    param_name = ('min_samples_leaf', "max_depth", "min_samples_split", "n_estimators",)
    params = dict(zip(param_name, hparams))

    folder = "models/random_forest/hyperparameter/"
    fname = f"{model_run}_{predictor}{str(percentile)[-2:]}.pkl"

    with open(f"{folder}{fname}", 'wb') as f:
        pickle.dump(params, f)