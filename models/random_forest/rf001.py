from matplotlib.pyplot import tight_layout


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
    folder = "results/random_forest/rf001/" # Where to save results

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

def optimization(X_train, y_train):
    """
    Description:
        Optimization of Hyperparameters applied to this run in order to get chosen hyperparameters in run().
        Runs approx. 15min.
        Optimized for RandomForestClassifier Parameters:
            n_estimators
            max_depth
    
    Parameters:
        X_train (np.array): Predictor training data from run(). Shape:(time, classes,)
        y_train (np.array): Predictand training data from run(). Shape:(time,)

    Returns:
        best_params (dict): best values for hyperparameters


    """
    #---
    # Optimization: Hyperparameters
    #---

    # Build Pipeline
    #---
    from sklearn.pipeline import Pipeline
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    max_depth = 9

    pipeline = Pipeline([
        ("rf", RandomForestClassifier(criterion='gini',
    n_estimators=91, #- nTrees 
    max_depth=max_depth, 
    random_state=0, # To compare results when changing hyperparameters
    class_weight="balanced",
    oob_score=True,)),
    ])

    # pipeline.set_params(knn__n_neighbors = 1) # To change parameters

    # Apply GridSearch
    #---
    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(pipeline, param_grid = {
        "rf__max_depth": np.arange(1, 10),
        "rf__n_estimators": np.arange(90, 100),
    })

    clf.fit(X_train, y_train)

    # Display params
    #---
    print(clf.best_params_)

    print(clf.score(X_test, y_test)) # Accuracy on basis of test data

    print(clf.best_score_) # Accuracy based on k-fold cross-validation

    return clf.best_params_