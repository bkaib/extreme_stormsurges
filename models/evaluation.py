#---
# Modules
#---
import numpy as np

#---
# Graphical evaluation
#---
def plot_cf(model, X_test, y_test,):
    """
    Description:
        Plots a 2x2 Confusion Matrix labeled with rates and absolute values

    Parameters:
        model (RandomForestClassifier): Fitted model (RF).
        X_test (np.array): Predictors of test data. Shape(time, features)
        y_test (np.array): Predictand of test data. Shape(time,)

    Returns:
        fig (matplotlib.figure.Figure): Figure of labeled confusion matrix. Percentages are rates (False Positive Rate etc.)
    
    Source (adjusted):
        "https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/"
    """
    
    # Plot Confusion Matrix
    #---
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Get confusion matrix
    y_test_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_test_pred)

    # Labels for quadrants in matrix
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    # Calculate Rates of Matrix
    tn, fp, fn, tp = cf_matrix.ravel()
    tnr = tn / (tn + fp) # True negative rate
    fpr = 1 - tnr # False Positive Rate
    tpr = tp / (tp + fn) # True positive rate
    fnr = 1 - tpr # False negative rate
    values = [tnr, fpr, fnr, tpr,]

    rates = ["{0:.2%}".format(value) for value in values]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names, group_counts, rates)]

    labels = np.asarray(labels).reshape(2,2)

    # Main Figure
    fig, ax = plt.subplots(tight_layout=True)
    
    ax = sns.heatmap(cf_matrix,
    annot=labels, 
    fmt='', 
    cmap='Blues',
    )

    # Axis labels
    ax.set_title('Confusion Matrix \n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Axis ticks
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])

    return fig

def importance_map(importance, lons, lats, tflag=""):
    """
    Description:
        Plots importance of a predictor on a lat lon map.
    Parameters:
        importance (np.array): Importances from model classification. Shape:(features,)
        lons (np.array): Longitudes of predictor importance
        lats (np.array): Latitudes of predictor importance
        tflag (str): Additional Title information
    Returns:
        fig: Figure of importance
    """
    # Modules
    import matplotlib.pyplot as plt 
    import cartopy.crs as ccrs

    # Reshape importance to lat/lon
    nlat = len(lats)
    nlon = len(lons)
    importance = importance.reshape(nlat, nlon)

    # Adjust colorbar extremes
    vmax = 0.5 * np.max(importance)

    # Plot importance on lat/lon map
    fig = plt.figure(tight_layout=True,)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plot = ax.contourf(lons, lats, importance, 
    transform=ccrs.PlateCarree(),
    vmax=vmax,
    )

    ax.coastlines()

    plt.colorbar(plot,)

    ax.set_title(f"Importance {tflag}")

    return fig

#---
# Metrics
#---
def cfm_metrics(model, X_test, y_test, beta=0.5):
    """
    Description:
        Use standard metrics for random forest classification with confusion matrix.
    Parameters:
        model (RandomForestClassifier): Fitted model (RF).
        X_test (np.array): Predictors of test data. Shape(time, features)
        y_test (np.array): Predictand of test data. Shape(time,)
        beta (float): Weight for measure of weighted accuracy. (Default:0.5) 
    Returns:
        metrics (dict): Set of metrics.
    """
    # Modules
    from sklearn.metrics import confusion_matrix

    # Confusion Matrix
    y_test_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_test_pred)

    # Calculate Metrics
    tn, fp, fn, tp = cf_matrix.ravel()
    tnr = tn / (tn + fp) # True negative rate
    tpr = tp / (tp + fn) # True positive rate
    gmean = np.sqrt(tnr * tpr) # G-Mean
    beta = 0.5 # Weight
    wacc = beta * tpr + (1 - beta) * tnr # Weighted Accuracy
    precision = tp / (tp + fp) 
    recall = tpr
    fmeasure = (2 * precision * recall) / (precision + recall) # F-measure

    # Display metrics
    names = ("tnr", "tpr", "gmean", "wacc", "precision", "recall", "fmeasure")
    values = (tnr, tpr, gmean, wacc, precision, recall, fmeasure)
    metrics = dict(zip(names, values))

    print("Metric values \n")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics