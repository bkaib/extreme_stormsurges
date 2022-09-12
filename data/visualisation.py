#--- 
# Modules
#---
import numpy as np
from models import evaluation
from data import saver

def map(data, lons, lats, tflag="", vmin=None, vmax=None, unit=None, is_clb_label=True, nlevels=6, cmap="viridis"):
    """
    Description:
        Plots predictor-data on a lat-lon map.
    Parameters:
        data (np.array): dataset from predictors. Shape:(lat, lon).flatten()
        lons (np.array): Longitudes of predictor data
        lats (np.array): Latitudes of predictor data
        tflag (str): Additional Title information, (Defaults: "")
        vmin (float): Minimum value for colorbar, (Defaults: None)
        vmax (float): Maximum value for colorbar, (Defaults: None)
        unit (str): Unit of data
        is_clb_label (bool): Whether to label the colorbar with unit or not.
        nlevels (int): Number of levels in contour plot (e.g. sections of the colorbar)
        cmap (str): Colormap of the plot
    Returns:
        fig: Figure of data
    """
    # Modules
    import matplotlib.pyplot as plt 
    import cartopy.crs as ccrs
    import numpy as np

    # Reshape data to lat/lon
    nlat = len(lats)
    nlon = len(lons)
    data = data.reshape(nlat, nlon)

    # Plot data on lat/lon map
    fig = plt.figure(tight_layout=True,)
    ax = plt.axes(projection=ccrs.PlateCarree())

    plot = ax.contourf(lons, lats, data,
    levels=np.linspace(vmin, vmax, num=nlevels),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    extend='both',
    )

    ax.coastlines()

    if vmin == None or vmax == None:
        cbarticks = None
    else:
        cbarticks = np.linspace(vmin, vmax, num=nlevels)

    # clb = plt.colorbar(plot, shrink=.62, ticks=cbarticks,)
    clb = plt.colorbar(plot, ticks=cbarticks)

    if is_clb_label:
        clb.set_label(f"{unit}", rotation=90, labelpad=1)

    ax.set_title(f"{tflag}")

    return fig, ax

def create_gif(figures, path, fps=1):
    """
    Description: 
        Creates and saves a gif from a list of figures
    Parameters:
        figures (list): List of Figures with 2 Axes.
        path (str): Path to save gif to.
        fps (int): Frames per second for gif. (Defaults: 1)
    Returns: 
        None
    """
    import imageio
    import numpy as np
    images = []
    for fig in figures:
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8') # Convert to RGB values
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Reshape for gif

        images.append(image)

    imageio.mimsave(f'{path}.gif', images, fps=fps)

def predictor_maps(model, X_test, y_test, X_test_unscaled, ndim, n_pfs, is_pf_combined, lons, lats, pred_units, pred_names, station_positions, station_names, is_station_name, is_overlay_importance, run_id, model_run, percentile, markersize, alpha, color, colorbar_range, nlevels=10):
    """
    Description:
        Plots values of all predictors used for model training. Selects timepoints where Storm Surges were in original data and indicates whether 
        the prediction was true or not in the filename ("isfalse", "istrue"). The file naming "predss" is for situations, where a storm surge was predicted
        but the original data has no storm surge.
    Parameters:
        model (clf): Model that was fitted to X_test, y_test.
        X_test (): Test set of predictor data used for model fit
        y_test (): Test set of predictand data used for model fit
        lons (): Values of longitudes of predictors
        lats (): Values of latitudes of predictors
        pred_units (list): Units of all predictors used, e.g. ms**-1
        pred_names (list): Names of all predictors used, e.g. sp_tlag0
        station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
        station_name (str): Name of the station given in GESLA dataset.
        is_station_name (bool): Whether to indicate station name in plot or not (Defaults: False)
        is_overlay_importance (bool): Overlay values of importance of that predictor or not
        run_id (int): Number of the model run
        model_run (str): Name of the current model run
        percentile (float): Percentile (0-100) for selecting position of overlayed importance
        markersize (int): Size of square-marked positions of importance
        alpha (float): Transparency of square-markers for importance
        color (str): Color of square-marker for importance
        colorbar_range (dict): Keys are predictors (e.g. "sp"), values are list of [vmin, vmax] values for colorbar
        nlevels (int): Number of levels in contour plot (e.g. sections of the colorbar) (Defaults: 10)

    Returns:
        None
    """
    #---
    # Make a prediction
    #---
    nlat = lats.size
    nlon = lons.size
    y_test_pred = model.predict(X_test) # Predicted data
    importance = model.feature_importances_
    n_pred_features = nlon * nlat # Features per predictor (lon/lat Input-Field). Needed for importance separation

    # Get importance per predictor of model
    #---
    if is_pf_combined:
        importance = importance[:-n_pfs] # Select only values that contain era5 as predictor
        X_test = X_test[:, :-n_pfs]
        X_test_unscaled = X_test_unscaled[:, :-n_pfs]

    predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features) #

    #---
    # Select data for plotting original storm surge events
    #---
    ss_idx = np.where(y_test == 1) # Timepoints of storm surges in original data

    y_test_ss = y_test[ss_idx] # Original data storm surges only
    y_pred_ss = y_test_pred[ss_idx] # Predictions at timepoints of SS in original data

    ntime = X_test.shape[0]
    X_pred = X_test_unscaled.reshape(ntime, -1, nlat, nlon) # Reshape to fit format for plotting predictor values on a map TODO: Needs to be adjusted to unscaled values?

    X_pred_plot = X_pred[ss_idx] # Select only predictor values at timepoints of storm surges

    #---
    # Plot & Save predictor map at original storm surge events
    #---
    n_time = X_pred_plot.shape[0]
    n_pred = X_pred_plot.shape[1]

    time_idx = 0
    for time in range(n_time):

        is_correct_prediction = (y_test_ss[time] == y_pred_ss[time])

        for pred_idx in range(n_pred):
            pred_flag = pred_names[pred_idx].split("_")[0] # Split because string is "tp_tlag0" etc.
            # Convert unit of colorbar
            #---
            unit = pred_units[pred_idx]
            if (unit == "m s**-1"): 
                unit = "m/s"

            # Create Figure
            #---
            data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

            if is_correct_prediction:
                tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 1"
                fname = f"{pred_names[pred_idx]}_{time_idx}_istrue_{run_id}"
            else:
                tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 0" 
                fname = f"{pred_names[pred_idx]}_{time_idx}_isfalse_{run_id}"
            
            # Get colorbar vmin vmax
            #---
            vmin = colorbar_range[pred_flag][0] 
            vmax = colorbar_range[pred_flag][1]

            # Choose colormap
            #---
            if pred_flag == "sp":
                cmap = "coolwarm"
            elif pred_flag == "tp":
                cmap = "Blues"
            elif (pred_flag == "u10" or pred_flag == "v10"):
                cmap= "seismic"

            # Plot figure
            #---
            fig, ax = map(data, lons, lats, tflag=tflag, unit=unit, vmin=vmin, vmax=vmax, nlevels=nlevels, cmap=cmap)
            
            # Add position of station to map
            #---
            for station_name in station_names:
                plot_station(ax, station_positions, station_name, is_station_name)

            # Add importance to map
            #---
            if is_overlay_importance:
                pred_importance = predictor_importances[pred_idx]
                evaluation.overlay_importance(ax, pred_importance, lats, lons, percentile=percentile, alpha=alpha, markersize=markersize, color=color)

            # Save plot
            #---
            folder1 = f"results/random_forest/{model_run}/predictor_maps/"
            saver.directory_existance(folder1)

            fig.savefig(f"{folder1}{fname}.pdf")

        time_idx = time_idx + 1

    #---
    # Plot & Save predictor map of predicted storm surge events where original data has no storm surge 
    #---
    idx2 = np.where(y_test_pred == 1) # Select all occurences where prediction has SS
    y_test_idx2 = y_test[idx2]
    X_pred_plot = X_pred[idx2]
    idx3 = np.where(y_test_idx2 == 0) # Subselect all occurences where prediction has SS and original data has no SS
    X_pred_plot = X_pred_plot[idx3] # Choose this selection as a plot
    n_time = X_pred_plot.shape[0]
    n_pred = X_pred_plot.shape[1]

    for time in range(n_time):
        for pred_idx in range(n_pred):
            pred_flag = pred_names[pred_idx].split("_")[0] # Split because string is "tp_tlag0" etc.
            # Convert unit of colorbar
            #---
            unit = pred_units[pred_idx]
            if (unit == "m s**-1"): 
                unit = "m/s"

            # Create Figure
            #---
            data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

            tflag = f"{pred_names[pred_idx]},  y_orig = 0, y_pred = 1"
            
            # Get colorbar vmin vmax
            #---
            vmin = colorbar_range[pred_flag][0] # Split because string is "tp_tlag0" etc.
            vmax = colorbar_range[pred_flag][1]

            # Choose colormap
            #---
            if pred_flag == "sp":
                cmap = "coolwarm"
            elif pred_flag == "tp":
                cmap = "Blues"
            elif (pred_flag == "u10" or pred_flag == "v10"):
                cmap= "seismic"
                
            # Plot figure
            #---
            fig, ax = map(data, lons, lats, tflag=tflag, unit=unit, vmin=vmin, vmax=vmax, nlevels=nlevels, cmap=cmap)

            # Add position of station to plot
            #---
            for station_name in station_names:
                plot_station(ax, station_positions, station_name, is_station_name)

            # Add importance to map
            if is_overlay_importance:
                pred_importance = predictor_importances[pred_idx]
                evaluation.overlay_importance(ax, pred_importance, lats, lons, percentile=percentile, alpha=alpha, markersize=markersize, color=color)
            
            # Save figure
            #---
            folder1 = f"results/random_forest/{model_run}/predictor_maps/"
            saver.directory_existance(folder1)

            fname = f"{pred_names[pred_idx]}_{time_idx}_predss_{run_id}"

            fig.savefig(f"{folder1}{fname}.pdf")

        time_idx = time_idx + 1

def plot_station(ax, station_positions, station_name, is_station_name=False, markersize=8, fontsize=12, color="k",):
    """
    Description:
        Plots the position of a station into given axis
    Parameters:
        ax (GeoAxesSubplot): Axis to plot station into
        station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
        station_name (str): Name of the station given in GESLA dataset.
        is_station_name (bool): Whether to indicate station name in plot or not (Defaults: False)
        
        kwargs**
        markersize (float): Size of the cross used to indicate position of station (Defaults: 10)
        fontsize (float): Fontsize of the name of station (Defaults: 12)
        color (str): Color code (Defaults: "r")
    Returns:
        None
    """
    # import matplotlib.pyplot as plt 
    import matplotlib.lines as mlines
    import cartopy.crs as ccrs

    lons = station_positions[station_name][0]
    lats = station_positions[station_name][1]

    # Mark position of station
    ax.plot(lons, lats, 'X', markersize=markersize, color=color, transform=ccrs.PlateCarree())

    # Add legend
    black_cross = mlines.Line2D([], [], color=color, marker='X', linestyle='None',
                          markersize=markersize, label=station_name)

    ax.legend(handles=[black_cross,], loc="lower center", bbox_to_anchor=(0., 0., 0.5, 0.5), fontsize=8,)

    if is_station_name:
        ax.text(1.001*lons, 1.001*lats, station_name, fontsize=fontsize, transform=ccrs.PlateCarree())


#---
# Versions up to rf011
#--

# def map(data, lons, lats, tflag="", vmin=None, vmax=None, unit=None, is_clb_label=True):
#     """
#     Description:
#         Plots predictor-data on a lat-lon map.
#     Parameters:
#         data (np.array): dataset from predictors. Shape:(lat, lon).flatten()
#         lons (np.array): Longitudes of predictor data
#         lats (np.array): Latitudes of predictor data
#         tflag (str): Additional Title information, (Defaults: "")
#         vmin (float): Minimum value for colorbar, (Defaults: None)
#         vmax (float): Maximum value for colorbar, (Defaults: None)
#         unit (str): Unit of data
#         is_clb_label (bool): Whether to label the colorbar with unit or not.
#     Returns:
#         fig: Figure of data
#     """
#     # Modules
#     import matplotlib.pyplot as plt 
#     import cartopy.crs as ccrs
#     import numpy as np

#     # Reshape data to lat/lon
#     nlat = len(lats)
#     nlon = len(lons)
#     data = data.reshape(nlat, nlon)

#     # Plot data on lat/lon map
#     fig = plt.figure(tight_layout=True,)
#     ax = plt.axes(projection=ccrs.PlateCarree())

#     plot = ax.contourf(lons, lats, data, 
#     transform=ccrs.PlateCarree(),
#     vmin=vmin,
#     vmax=vmax,
#     )

#     ax.coastlines()

#     if vmin == None or vmax == None:
#         cbarticks = None
#     else:
#         cbarticks = np.arange(vmin, vmax, (vmax-vmin) / 5) # 5 ticks

#     clb = plt.colorbar(plot, shrink=.62, ticks=cbarticks)

#     if is_clb_label:
#         clb.set_label(f"{unit}", rotation=90, labelpad=1)

#     ax.set_title(f"{tflag}")

#     return fig

# def predictor_maps(model, X_test, y_test, lons, lats, pred_units, pred_names, run_id, model_run,):
#     """
#     Description:
#         Plots values of all predictors used for model training. Selects timepoints where Storm Surges were in original data and indicates whether 
#         the prediction was true or not in the filename ("isfalse", "istrue"). The file naming "predss" is for situations, where a storm surge was predicted
#         but the original data has no storm surge.
#     Parameters:
#         model (clf): Model that was fitted to X_test, y_test.
#         X_test (): Test set of predictor data used for model fit
#         y_test (): Test set of predictand data used for model fit
#         lons (): Values of longitudes of predictors
#         lats (): Values of latitudes of predictors
#         pred_units (list): Units of all predictors used, e.g. ms**-1
#         pred_names (list): Names of all predictors used, e.g. sp_tlag0
#         run_id (int): Number of the model run
#         model_run (str): Name of the current model run
#     Returns:
#         None
#     """
#     import numpy as np
#     from data import saver

#     #---
#     # Make a prediction
#     #---
#     nlat = lats.size
#     nlon = lons.size
#     y_test_pred = model.predict(X_test) # Predicted data

#     #---
#     # Select data for plotting original storm surge events
#     #---
#     ss_idx = np.where(y_test == 1) # Timepoints of storm surges in original data

#     y_test_ss = y_test[ss_idx] # Original data storm surges only
#     y_pred_ss = y_test_pred[ss_idx] # Predictions at timepoints of SS in original data

#     ntime = X_test.shape[0]
#     X_pred = X_test.reshape(ntime, -1, nlat, nlon) # Reshape to fit format for plotting predictor values on a map

#     X_pred_plot = X_pred[ss_idx] # Select only predictor values at timepoints of storm surges

#     #---
#     # Plot & Save predictor map at original storm surge events
#     #---
#     n_time = X_pred_plot.shape[0]
#     n_pred = X_pred_plot.shape[1]

#     time_idx = 0
#     for time in range(n_time):

#         is_correct_prediction = (y_test_ss[time] == y_pred_ss[time])

#         for pred_idx in range(n_pred):

#             # Convert unit of colorbar
#             #---
#             unit = pred_units[pred_idx]
#             if (unit == "m s**-1"): 
#                 unit = "m/s"

#             # Create Figure
#             #---
#             data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

#             if is_correct_prediction:
#                 tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 1"
#                 fname = f"{pred_names[pred_idx]}_{time_idx}_istrue_{run_id}"
#             else:
#                 tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 0" 
#                 fname = f"{pred_names[pred_idx]}_{time_idx}_isfalse_{run_id}"
            
#             fig = map(data, lons, lats, tflag=tflag, unit=unit,)

#             folder1 = f"results/random_forest/{model_run}/predictor_maps/"
#             saver.directory_existance(folder1)

#             fig.savefig(f"{folder1}{fname}.pdf")

#         time_idx = time_idx + 1

#     #---
#     # Plot & Save predictor map of predicted storm surge events where original data has no storm surge 
#     #---
#     idx2 = np.where(y_test_pred == 1) # Select all occurences where prediction has SS
#     y_test_idx2 = y_test[idx2]
#     X_pred_plot = X_pred[idx2]
#     idx3 = np.where(y_test_idx2 == 0) # Subselect all occurences where prediction has SS and original data has no SS
#     X_pred_plot = X_pred_plot[idx3] # Choose this selection as a plot
#     n_time = X_pred_plot.shape[0]
#     n_pred = X_pred_plot.shape[1]

#     for time in range(n_time):
#         for pred_idx in range(n_pred):

#             # Convert unit of colorbar
#             #---
#             unit = pred_units[pred_idx]
#             if (unit == "m s**-1"): 
#                 unit = "m/s"

#             # Create Figure
#             #---
#             data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

#             tflag = f"{pred_names[pred_idx]},  y_orig = 0, y_pred = 1"
            
#             fig = map(data, lons, lats, tflag=tflag, unit=unit,)
            
#             folder1 = f"results/random_forest/{model_run}/predictor_maps/"
#             saver.directory_existance(folder1)

#             fname = f"{pred_names[pred_idx]}_{time_idx}_predss_{run_id}"

#             fig.savefig(f"{folder1}{fname}.pdf")

#         time_idx = time_idx + 1
# # def predictor_maps(model, X_test, y_test, lons, lats, pred_units, pred_names, station_positions, station_names, is_station_name, is_overlay_importance, run_id, model_run, percentile, markersize, alpha, color, colorbar_range, nlevels=10):
#     """
#     Description:
#         Plots values of all predictors used for model training. Selects timepoints where Storm Surges were in original data and indicates whether 
#         the prediction was true or not in the filename ("isfalse", "istrue"). The file naming "predss" is for situations, where a storm surge was predicted
#         but the original data has no storm surge.
#     Parameters:
#         model (clf): Model that was fitted to X_test, y_test.
#         X_test (): Test set of predictor data used for model fit
#         y_test (): Test set of predictand data used for model fit
#         lons (): Values of longitudes of predictors
#         lats (): Values of latitudes of predictors
#         pred_units (list): Units of all predictors used, e.g. ms**-1
#         pred_names (list): Names of all predictors used, e.g. sp_tlag0
#         station_positions (dict,): Dicitionary with station name (key) and a list of [lon, lat] (values)
#         station_name (str): Name of the station given in GESLA dataset.
#         is_station_name (bool): Whether to indicate station name in plot or not (Defaults: False)
#         is_overlay_importance (bool): Overlay values of importance of that predictor or not
#         run_id (int): Number of the model run
#         model_run (str): Name of the current model run
#         percentile (float): Percentile (0-100) for selecting position of overlayed importance
#         markersize (int): Size of square-marked positions of importance
#         alpha (float): Transparency of square-markers for importance
#         color (str): Color of square-marker for importance
#         colorbar_range (dict): Keys are predictors (e.g. "sp"), values are list of [vmin, vmax] values for colorbar
#         nlevels (int): Number of levels in contour plot (e.g. sections of the colorbar) (Defaults: 10)

#     Returns:
#         None
#     """
#     #---
#     # Make a prediction
#     #---
#     nlat = lats.size
#     nlon = lons.size
#     y_test_pred = model.predict(X_test) # Predicted data
#     importance = model.feature_importances_
#     n_pred_features = nlon * nlat # Features per predictor (lon/lat Input-Field). Needed for importance separation

#     # Get importance per predictor of model
#     #---
#     predictor_importances = evaluation.separate_predictor_importance(importance, n_pred_features) #

#     #---
#     # Select data for plotting original storm surge events
#     #---
#     ss_idx = np.where(y_test == 1) # Timepoints of storm surges in original data

#     y_test_ss = y_test[ss_idx] # Original data storm surges only
#     y_pred_ss = y_test_pred[ss_idx] # Predictions at timepoints of SS in original data

#     ntime = X_test.shape[0]
#     X_pred = X_test.reshape(ntime, -1, nlat, nlon) # Reshape to fit format for plotting predictor values on a map

#     X_pred_plot = X_pred[ss_idx] # Select only predictor values at timepoints of storm surges

#     #---
#     # Plot & Save predictor map at original storm surge events
#     #---
#     n_time = X_pred_plot.shape[0]
#     n_pred = X_pred_plot.shape[1]

#     time_idx = 0
#     for time in range(n_time):

#         is_correct_prediction = (y_test_ss[time] == y_pred_ss[time])

#         for pred_idx in range(n_pred):
#             pred_flag = pred_names[pred_idx].split("_")[0] # Split because string is "tp_tlag0" etc.
#             # Convert unit of colorbar
#             #---
#             unit = pred_units[pred_idx]
#             if (unit == "m s**-1"): 
#                 unit = "m/s"

#             # Create Figure
#             #---
#             data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

#             if is_correct_prediction:
#                 tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 1"
#                 fname = f"{pred_names[pred_idx]}_{time_idx}_istrue_{run_id}"
#             else:
#                 tflag = f"{pred_names[pred_idx]}, y_orig = 1, y_pred = 0" 
#                 fname = f"{pred_names[pred_idx]}_{time_idx}_isfalse_{run_id}"
            
#             # Get colorbar vmin vmax
#             #---
#             vmin = colorbar_range[pred_flag][0] 
#             vmax = colorbar_range[pred_flag][1]

#             # Choose colormap
#             #---
#             if pred_flag == "sp":
#                 cmap = "coolwarm"
#             elif pred_flag == "tp":
#                 cmap = "Blues"
#             elif (pred_flag == "u10" or pred_flag == "v10"):
#                 cmap= "seismic"

#             # Plot figure
#             #---
#             fig, ax = map(data, lons, lats, tflag=tflag, unit=unit, vmin=vmin, vmax=vmax, nlevels=nlevels, cmap=cmap)
            
#             # Add position of station to map
#             #---
#             for station_name in station_names:
#                 plot_station(ax, station_positions, station_name, is_station_name)

#             # Add importance to map
#             #---
#             if is_overlay_importance:
#                 pred_importance = predictor_importances[pred_idx]
#                 evaluation.overlay_importance(ax, pred_importance, lats, lons, percentile=percentile, alpha=alpha, markersize=markersize, color=color)

#             # Save plot
#             #---
#             folder1 = f"results/random_forest/{model_run}/predictor_maps/"
#             saver.directory_existance(folder1)

#             fig.savefig(f"{folder1}{fname}.pdf")

#         time_idx = time_idx + 1

#     #---
#     # Plot & Save predictor map of predicted storm surge events where original data has no storm surge 
#     #---
#     idx2 = np.where(y_test_pred == 1) # Select all occurences where prediction has SS
#     y_test_idx2 = y_test[idx2]
#     X_pred_plot = X_pred[idx2]
#     idx3 = np.where(y_test_idx2 == 0) # Subselect all occurences where prediction has SS and original data has no SS
#     X_pred_plot = X_pred_plot[idx3] # Choose this selection as a plot
#     n_time = X_pred_plot.shape[0]
#     n_pred = X_pred_plot.shape[1]

#     for time in range(n_time):
#         for pred_idx in range(n_pred):
#             pred_flag = pred_names[pred_idx].split("_")[0] # Split because string is "tp_tlag0" etc.
#             # Convert unit of colorbar
#             #---
#             unit = pred_units[pred_idx]
#             if (unit == "m s**-1"): 
#                 unit = "m/s"

#             # Create Figure
#             #---
#             data = X_pred_plot[time, pred_idx, :, :].flatten() # Predictor data

#             tflag = f"{pred_names[pred_idx]},  y_orig = 0, y_pred = 1"
            
#             # Get colorbar vmin vmax
#             #---
#             vmin = colorbar_range[pred_flag][0] # Split because string is "tp_tlag0" etc.
#             vmax = colorbar_range[pred_flag][1]

#             # Choose colormap
#             #---
#             if pred_flag == "sp":
#                 cmap = "coolwarm"
#             elif pred_flag == "tp":
#                 cmap = "Blues"
#             elif (pred_flag == "u10" or pred_flag == "v10"):
#                 cmap= "seismic"
                
#             # Plot figure
#             #---
#             fig, ax = map(data, lons, lats, tflag=tflag, unit=unit, vmin=vmin, vmax=vmax, nlevels=nlevels, cmap=cmap)

#             # Add position of station to plot
#             #---
#             for station_name in station_names:
#                 plot_station(ax, station_positions, station_name, is_station_name)

#             # Add importance to map
#             if is_overlay_importance:
#                 pred_importance = predictor_importances[pred_idx]
#                 evaluation.overlay_importance(ax, pred_importance, lats, lons, percentile=percentile, alpha=alpha, markersize=markersize, color=color)
            
#             # Save figure
#             #---
#             folder1 = f"results/random_forest/{model_run}/predictor_maps/"
#             saver.directory_existance(folder1)

#             fname = f"{pred_names[pred_idx]}_{time_idx}_predss_{run_id}"

#             fig.savefig(f"{folder1}{fname}.pdf")

#         time_idx = time_idx + 1
