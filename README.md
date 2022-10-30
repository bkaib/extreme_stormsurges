# Estimating Extreme Storm Surges in the Baltic Sea via Machine Learning

# Short Description
The aim of this project is to estimate extreme 
storm surges (>95%-percentiles) in the Baltic Sea via a Random Forest. 
For now only binary classification is applied. A future extension aims to also supply a full regression of extreme storm surge height using Random Regression Forests.
The masterthesis is conducted together with the Helmholtz-Zentrum-Hereon and the University of Hamburg 
and is embedded in the CLimate INTelligence project.

# Structure
data (package): Modules for preprocessing and manipulating data.

model (package): Modules for loading, fitting and evaluating a model.

resources (folder): Containing sample folders for resources (ERA5 and GESLA datasets)

main (ipynb): The main call to the software.

masterthesis (pdf): Theoretical background and model description.

# Note
This repository does not include the resource folder. In order for everything to run smoothly, 
include the resource folder containing ERA5 and GESLA Dataset and check the naming conventions of the files. 
The data package uses fixed naming conventions to load data.
You can download GESLA and ERA5 from their corresponding websites. Place the downloaded datasets into corresponding folders. You can just unpack the GESLA3.0_ALL.zip into the resources/gesla/ folder and delete the gesla_data_placeholder.txt
Keep in mind to check the naming convention of the ERA5 data after using CDO or change the naming conventions in the corrensponding data package and its loader modules.

