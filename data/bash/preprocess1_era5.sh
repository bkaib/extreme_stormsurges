#!/bin/bash

# Description:
# ----------------
# Apply this script on data stored in D://Ungesicherte_Daten/Masterarbeit/Daten/original.
# Output of this scipt was stored in resources/preprocess1

# Pseudo-Code:
# ----------------
# For all files in the folder do;
# Select a subregion 
# Select specified month
# Calculate daily averages

# Main: 
# -----------------

# Select a subregion (applied on original data)
lon1=-5
lon2=30
lat1=40
lat2=70

# $ for f in *lon-7050*.nc; do cdo sellonlatbox,$lon1,$lon2,$lat1,$lat2 $f "${f%.nc}"_new.nc; done

# Rename subregion file
fname="lon0${lon1}${lon2}_lat${lat2}${lat1}"

# for f in *_new*.nc; do mv -v "$f" "${f/lon-7050_lat8030_new/${fname}}"; done;

# Select month from subregion file
season="autumn"
month1=9
month2=10
month3=11
for f in *${fname}.nc; do cdo selmon,$month1,$month2,$month3 $f "${f%.nc}"_${season}.nc; done

# Calculate daily average
sidx="a" # Season index
for f in *${season}*.nc; do cdo daymean $f "${f%${season}.nc}"${sidx}dmean.nc; done

