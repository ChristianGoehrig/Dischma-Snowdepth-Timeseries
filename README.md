# Preprocessing Dischma Snowdepth Timeseries

## Table of Contents

- [Introduction](#introduction)
- [Dataset Download](#dataset-download)
- [Workflow Steps](#workflow-steps)
- [References](#references)

# Introduction

Project description...

# Dataset Download

Links and instructions...

# Workflow Steps

1. Download datasets...
2. Set up configuration...
3. Run script...

# References

Links, credits, etc.




Datset harmonizer

This repository contains all necessary code and data to harmonize and standardize the snow depth maps from the ADS and the UltraCam sensor systems for the Dischma catchment.
The workflow establishes a high-resolution (2m) spatial continous timeseries of snow depth maps over 10 years. Limitation have to be acknowledged for 2011, where no acqusisition took place. The acquisition at 2018 covers only a small area of the Dischma catchment due to technical problems. 

Snow depth maps of each dataset can be downloaded on Envidat.

UltraCam: https://www.envidat.ch/#/metadata/snow-depth-mapping-by-airplane-photogrammetry-2017-ongoing?search=snow+depth+maps&isAuthorSearch=false

ADS: https://www.envidat.ch/#/metadata/snow-depth-mapping?search=snow%20depth%20ads



For the harmonization of the dataset follow these steps:

NOTE: Because 2018 is of much smaller coverage, consider not integrating this year to gain larger spatial coverage.

1. Download both datasets and save in common directory
2. Download from GitHub:
   - preprocessing_base (library of functions)
   - preprocessing_config.yaml (Configuration file for processing settings)
   - preprocessing.py (Runfile for python console)
   - reference_raster_mask_2m.tif (Reference Raster for direct matching, manual configuration also possible)
3. Set up Configuration file
4. Run preprocessing script and install dependencies (packages)
5. Retrieve harmonized files in output_folder
