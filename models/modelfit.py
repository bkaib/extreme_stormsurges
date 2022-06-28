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