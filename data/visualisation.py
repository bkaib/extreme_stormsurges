def map(data, lons, lats, tflag="", vmin=None, vmax=None, unit=None, is_clb_label=True):
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
    transform=ccrs.PlateCarree(),
    vmin=vmin,
    vmax=vmax,
    )

    ax.coastlines()

    if vmin == None or vmax == None:
        cbarticks = None
    else:
        cbarticks = np.arange(vmin, vmax, (vmax-vmin) / 5) # 5 ticks

    clb = plt.colorbar(plot, shrink=.62, ticks=cbarticks)

    if is_clb_label:
        clb.set_label(f"{unit}", rotation=90, labelpad=1)

    ax.set_title(f"{tflag}")

    return fig

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