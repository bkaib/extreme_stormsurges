def save_hp(hparams, hparam_names, predictor, model_run, percentile, folder):
    """
    Description: 
        Saves Hyperparameters manually read from validation curves in 
        master_thesis/results/random_forest/rf002.
        Saves to "models/random_forest/hyperparameter/" 

    Parameters:
        hparams (tuple): Fixed values of hyperparameters
        hparam_names (tuple): Names of hyperparameters in hparams
        predictor (str): Flag of predictor ["sp", "tp", "u10", "v10"]
        model_run (str): Flag for model run.
        percentile (float): percentile of preprocessing predictand.
        folder (str): Folder where data is saved.
    
    Returns:
        None
    """
    import pickle
    params = dict(zip(hparam_names, hparams))
    fname = f"{model_run}_{predictor}{str(percentile)[-2:]}.pkl"

    with open(f"{folder}{fname}", 'wb') as f:
        pickle.dump(params, f)
