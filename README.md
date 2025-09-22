# Preprocessing Dischma Snowdepth Timeseries

## Table of Contents

- [Introduction](#introduction)
- [Dataset Download](#dataset-download)
- [Installation & Usage](#installation--usage)
- [Workflow Steps](#workflow-steps)
- [References](#references)

## Introduction

This repository contains all necessary code and data to harmonize and standardize the snow depth maps from the ADS and the UltraCam sensor systems for the Dischma catchment.  
The workflow establishes a high-resolution (2m) spatial continuous timeseries of snow depth maps over 10 years.  

# Dataset Download

Snow depth maps of each dataset can be downloaded from [Envidat](https://www.envidat.ch/).

- [UltraCam Dataset](https://www.envidat.ch/#/metadata/snow-depth-mapping-by-airplane-photogrammetry-2017-ongoing?search=snow+depth+maps&isAuthorSearch=false)
- [ADS Dataset](https://www.envidat.ch/#/metadata/snow-depth-mapping?search=snow%20depth%20ads)

> **Limitation:** No acquisitions exist for 2011.

## Installation & Usage

This guide assumes you have a Mamba/Conda installation. For a minimal, open-source setup, we recommend installing Miniforge from the [official repository](https://github.com/conda-forge/miniforge).  
Miniforge comes with `mamba`, a fast, parallel replacement for `conda`.

**Steps:**

1. **Clone the repository & navigate into its directory**
    ```sh
    git clone ["https://github.com/ChristianGoehrig/Dischma-Snowdepth-Timeseries")
    cd Dischma-Snowdepth-Timeseries
    ```

2. **Create and activate the environment with Mamba (or Conda)**
    ```sh
    mamba env create -f environment.yml
    conda activate Dischma-Snowdepth-Timeseries
    ```

3. **Download required datasets and configuration:**
    - Download both UltraCam and ADS datasets and save them in a common directory (see [Dataset Download](#dataset-download)).
    - Download the following from this repository:
        - `preprocessing_base` (library of functions)
        - `preprocessing_config.yaml` (configuration file for processing settings)
        - `preprocessing.py` (main script for preprocessing)
        - `reference_raster_mask_2m.tif` (reference raster for direct matching)

4. **Set up the configuration file**
    - Edit `preprocessing_config.yaml` to adjust processing settings according to your data and needs.

5. **Run the preprocessing script**
    ```sh
    python preprocessing.py
    ```
    - Ensure all dependencies are installed. If any are missing, install them as needed (see `environment.yml`).

6. **Retrieve harmonized output**
    - The harmonized files will be generated in your configured `output_folder`.
    - 
> **Note:** Because 2018 is of much smaller coverage, consider not integrating this year to gain larger spatial coverage.
