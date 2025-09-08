#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script harmonizes and standardizes snow cover data from two different sensore systems (ADS / UltraCam)
"""

import os
import sys
import glob
import re
import datetime
import warnings
import yaml
import argparse
from pathlib import Path
import numpy as np
from rasterio.enums import Resampling
import preprocessing_base as pb


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dataset harmonization and standardization script')
    parser.add_argument('--config_preprocess', type=str, default='config_preprocess.yaml', 
                        help='Path to YAML config_preprocess file')
    args = parser.parse_args()
    
    # Load config_preprocessuration
    config_preprocess = pb.load_config(args.config_preprocess)
    
    # Record start time
    start = datetime.datetime.now()
    print(f"Script started at {start}")
    
    # Validate paths if enabled in config_preprocess
    if config_preprocess.get('validate_paths', True):
        if not pb.validate_paths(config_preprocess):
            print("Path validation failed. Exiting.")
            return

    # Create case folder if it doesn't exist
    case_folder = config_preprocess['paths']['output_folder']
    os.makedirs(case_folder, exist_ok=True)
    
    # Select years to process from config_preprocess definition or dataset files
    years = config_preprocess.get('analysis', {}).get('years', [])
    if not years:
        # If years not defined in config_preprocess, extract them from the file names in dataset dir
        dataset_files = glob.glob(os.path.join(config_preprocess['paths']['dataset_dir'], "*.tif"))
        years = set()
        for file in dataset_files:
            try:
                match = re.search(r'\d{4}', os.path.basename(file))
                if match:
                    years.add(int(match.group()))
            except:
                pass
        years = sorted(list(years))
    
    print(f"Processing years: {years}")
    
    # Select desired years from datapool
    input_data = pb.select_files_by_year(config_preprocess['paths'].get('dataset_dir', case_folder), years)
    
    if not input_data:
        print("No matching data files found. Exiting.")
        return

    #################################################################################################################
    ############### Main Processing Steps ############################################################################
    #################################################################################################################

    # 1. Reproject and align rasters
    reference_raster = config_preprocess['paths'].get('reference_raster')
    resampling_method_str = config_preprocess.get('analysis', {}).get('resampling_method',"average") #default average unless defined in config
    
    # Map string to Resampling enum
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos,
        'average': Resampling.average,
        'mode': Resampling.mode
    }
    resampling_method = resampling_methods.get(resampling_method_str, Resampling.average)
    
    print("Reprojecting and aligning rasters...")
    pb.reproject_and_align_rasters(
        input_data,
        case_folder,
        target_crs=config_preprocess.get('analysis', {}).get('crs'),
        reference_raster=reference_raster, 
        resolution=config_preprocess.get('analysis', {}).get('resolution'),
        resampling_method=resampling_method,
        set_negative_to_nodata=config_preprocess.get('analysis', {}).get('set_negative_to_nodata', True), #default True = set negative values to nodata
        outlier_threshold=config_preprocess.get('analysis', {}).get('outlier_threshold', None), #default None = no outlier removal
    )

    # Record end time and show elapsed time
    end = datetime.datetime.now()
    print(f"Script ended at {end}")
    time_diff = end - start
    print(f"Total computation time: {time_diff}")

if __name__ == "__main__":
    main()