# Estimating Extreme Storm Surges in the Baltic Sea via Machine Learning

# Short Description
The aim of this project is to estiamte extreme 
storm surges (>90%-percentiles) in the Baltic Sea via Machine Learning Algorithms like Random Forests, 
simple Neural Networks and eventually Generative Adversarial Networks. 
We aim for not only a classification of storm surges but also a continous forecast of sea level height 
depending on several predictors for storm surges in the Baltic Sea, like 10m-windfield, sea level pressure and total precipitation.

The masterthesis is conducted together with the Helmholtz-Zentrum-Hereon and the University of Hamburg 
and is embedded in the CLimate INTelligence project.

# Structure
data (package): Modules for preprocessing and manipulating data including bash-scripts

notebooks (folder): Contains all notebooks used while developing the packages. To run notebooks, place them to master_thesis folder.

models (folder/package): Model runs and corresponding hyperparamters as well as model-evaluation.

resources (folder): Preprocessed data of predictors and predictand.

results (folder): Visualisations of data, model-output and diagnostics.


# Note
This repository does not include the "resources" folder. In order for everything to run smoothly, 
include the resource folder containing ERA5 and GESLA Dataset. This folder can be sent to you upon request.
You can download GESLA and ERA5 from their websites. 

# Predictors
ERA5: Reanalysis data of Surface Pressure (SP\Pa), Total Precipitation (TP\) and zonal / meridional horizontal wind components at 10m height (u10\ms^-1, v10\ms^-1)


