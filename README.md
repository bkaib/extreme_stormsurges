# Estimating Extreme Storm Surges in the Baltic Sea via Machine Learning

# Short Description
The aim of this project is to estimate extreme 
storm surges (>95%-percentiles) in the Baltic Sea via a Random Forest. 
For now only binary classification is applied. A future extension aims to also supply a full regression of extreme storm surge height using Random Regression Forests.
The masterthesis is conducted together with the Helmholtz-Zentrum-Hereon and the University of Hamburg 
and is embedded in the CLimate INTelligence project.

# Structure
data (package): Modules for preprocessing and manipulating data
model (package): 
main (ipynb):
masterthesis (pdf): Theoretical background and model description

# Note
This repository does not include the resource folder. In order for everything to run smoothly, 
include the resource folder containing ERA5 and GESLA Dataset and check the naming conventions of the files. 
The data package uses fixed naming conventions to load data.
You can download GESLA and ERA5 from their corresponding websites. 

