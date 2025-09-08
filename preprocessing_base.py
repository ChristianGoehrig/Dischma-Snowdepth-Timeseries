import numpy as np
from scipy import stats
import os
import re
import rasterio
from rasterio.enums import Resampling, Compression
from rasterio.warp import reproject, calculate_default_transform
import glob
import yaml
import sys

# Handle YAML config file loading
def load_config(config_path):
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data as a Python dictionary.

    Raises:
        SystemExit: If the file cannot be parsed due to invalid YAML syntax.

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config["database"]["host"])
    """

    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            print(f"config file loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config file: {e}")
            sys.exit(1)

# Validate paths in config
def validate_paths(config_file):
    """
    Validate and prepare file system paths defined in the configuration.

    Args:
        config_file (dict): Parsed configuration dictionary containing
            a 'paths' key with at least:
                - 'dataset_dir' (str): Path to the dataset directory.
                - 'output_folder' (str): Path to the output folder.

    Returns:
        bool:
            - True if all required paths exist (or are successfully created).
            - False if any required input path does not exist.

    Example:
        >>> config = {"paths": {"dataset_dir": "./data", "output_folder": "./output"}}
        >>> if not validate_paths(config):
        ...     raise FileNotFoundError("Dataset directory is missing.")
    """

    required_paths = [
        config_file['paths']['dataset_dir'],
    ]

    # Check for required paths
    for path in required_paths:
        if not os.path.exists(path):
            print(f"ERROR: Input data path does not exist: {path}")
            return False

    # Create output directory if it doesn't exist
    os.makedirs(config_file['paths']['output_folder'], exist_ok=True)

    return True


def select_files_by_year(folder_path, years):
    """
    Select files from a folder that contain specified years in their filenames.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing files to search
    years : list
        List of years (int or str) to search for in filenames

    Returns
    -------
    list
        List of full file paths for files containing any of the specified years

    Notes
    -----
    The function performs a simple string match, checking if the string
    representation of each year appears anywhere in the filename

    Examples
    --------
    >>> select_files_by_year('/data/rainfall', [2020, 2021])
    ['/data/rainfall/data_2020.csv', '/data/rainfall/rainfall_2021.txt']
    """
    selected_files = []

    # Iterate over files in the folder using glob
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        file_name = os.path.basename(file_path)
        if any(str(year) in file_name for year in years):
            selected_files.append(file_path)  # Append the full path

    print("Years of Data selected for processing")

    return selected_files

def create_valid_data_mask(array1, array2, nodata_values=[-999, -99, -9999, -3.40282e+38],
                           outlier_method=None, outlier_threshold=None, no_negatives_array_1=True):
    """
    Create mask for valid data points with outlier detection.

    Parameters:
    -----------
    array1, array2 : numpy arrays
        Input arrays to validate
    nodata_values : list
        Values to treat as missing data
    outlier_method : str
        Method for outlier detection: 'iqr', 'zscore', 'percentile', 'mad'
    outlier_threshold : float or tuple
        Threshold for outlier detection (method-specific)
    """
    # Start with non-NaN values
    mask = ~np.isnan(array1) & ~np.isnan(array2)

    # Remove common nodata values
    for nodata in nodata_values:
        mask = mask & (array1 != nodata) & (array2 != nodata)

    # Remove negative snow depths (if array1 is snow depth)
    if no_negatives_array_1 == True:
        mask = mask & (array1 >= 0)

    # Apply outlier detection
    if outlier_method is not None and np.any(mask):
        outlier_mask = detect_outliers(array1, array2, mask, outlier_method, outlier_threshold)
        mask = mask & ~outlier_mask

    return mask


def detect_outliers(array1, array2, valid_mask, method='None', threshold=None):
    """
    Detect outliers using various methods.

    Returns:
    --------
    outlier_mask : numpy array
        Boolean mask where True indicates outlier
    """
    outlier_mask = np.zeros_like(array1, dtype=bool)

    if method == 'iqr':
        return detect_outliers_iqr(array1, array2, valid_mask, threshold)
    elif method == 'zscore':
        return detect_outliers_zscore(array1, array2, valid_mask, threshold)
    elif method == 'percentile':
        return detect_outliers_percentile(array1, array2, valid_mask, threshold)
    elif method == 'mad':
        return detect_outliers_mad(array1, array2, valid_mask, threshold)
    elif method == 'bivariate':
        return detect_outliers_bivariate(array1, array2, valid_mask, threshold)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def detect_outliers_iqr(array1, array2, valid_mask, threshold=1.5):
    """Interquartile Range method - good for non-normal data"""
    if threshold is None:
        threshold = 1.5

    outlier_mask = np.zeros_like(array1, dtype=bool)

    for arr in [array1, array2]:
        valid_data = arr[valid_mask]
        if len(valid_data) == 0:
            continue

        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outlier_mask |= (arr < lower_bound) | (arr > upper_bound)

    return outlier_mask


def detect_outliers_zscore(array1, array2, valid_mask, threshold=3.0):
    """Z-score method - good for normally distributed data"""
    if threshold is None:
        threshold = 3.0

    outlier_mask = np.zeros_like(array1, dtype=bool)

    for arr in [array1, array2]:
        valid_data = arr[valid_mask]
        if len(valid_data) == 0:
            continue

        z_scores = np.abs(stats.zscore(valid_data))
        valid_indices = np.where(valid_mask)[0]

        # Map z-scores back to original array positions
        temp_outliers = np.zeros_like(arr, dtype=bool)
        temp_outliers[valid_indices] = z_scores > threshold
        outlier_mask |= temp_outliers

    return outlier_mask


def detect_outliers_percentile(array1, array2, valid_mask, threshold=(1, 99)):
    """Percentile method - removes extreme percentiles"""
    if threshold is None:
        threshold = (1, 99)

    lower_pct, upper_pct = threshold
    outlier_mask = np.zeros_like(array1, dtype=bool)

    for arr in [array1, array2]:
        valid_data = arr[valid_mask]
        if len(valid_data) == 0:
            continue

        lower_bound = np.percentile(valid_data, lower_pct)
        upper_bound = np.percentile(valid_data, upper_pct)

        outlier_mask |= (arr < lower_bound) | (arr > upper_bound)

    return outlier_mask


def detect_outliers_mad(array1, array2, valid_mask, threshold=3.5):
    """Modified Z-score using Median Absolute Deviation - robust to outliers"""
    if threshold is None:
        threshold = 3.5

    outlier_mask = np.zeros_like(array1, dtype=bool)

    for arr in [array1, array2]:
        valid_data = arr[valid_mask]
        if len(valid_data) == 0:
            continue

        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))

        # Avoid division by zero
        if mad == 0:
            mad = np.mean(np.abs(valid_data - median))

        if mad > 0:
            modified_z_scores = 0.6745 * (arr - median) / mad
            outlier_mask |= np.abs(modified_z_scores) > threshold

    return outlier_mask


def detect_outliers_bivariate(array1, array2, valid_mask, threshold=3.0):
    """Bivariate outlier detection using Mahalanobis distance"""
    if threshold is None:
        threshold = 3.0

    outlier_mask = np.zeros_like(array1, dtype=bool)

    # Get valid data points
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return outlier_mask

    valid_data = np.column_stack([array1[valid_mask], array2[valid_mask]])

    try:
        # Calculate covariance matrix
        cov_matrix = np.cov(valid_data.T)
        if np.linalg.det(cov_matrix) == 0:
            return outlier_mask

        # Calculate Mahalanobis distance
        inv_cov = np.linalg.inv(cov_matrix)
        mean_data = np.mean(valid_data, axis=0)

        # Calculate distances for all valid points
        diff = valid_data - mean_data
        mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        # Determine outliers
        outlier_threshold = np.percentile(mahal_dist, 95)  # or use threshold parameter
        outliers_in_valid = mahal_dist > max(outlier_threshold, threshold)

        # Map back to original array
        outlier_mask[valid_indices] = outliers_in_valid

    except np.linalg.LinAlgError:
        # Fallback to univariate if covariance matrix is singular
        return detect_outliers_iqr(array1, array2, valid_mask, 1.5)

    return outlier_mask


# Example usage and comparison
def compare_outlier_methods(array1, array2, nodata_values=[-999, -99, -9999, -3.40282e+38]):
    """Compare different outlier detection methods"""
    methods = ['iqr', 'zscore', 'percentile', 'mad', 'bivariate']
    results = {}

    for method in methods:
        try:
            mask = create_valid_data_mask(array1, array2, nodata_values,
                                          outlier_method=method)
            valid_count = np.sum(mask)
            outlier_count = np.sum(~np.isnan(array1) & ~np.isnan(array2)) - valid_count

            results[method] = {
                'valid_points': valid_count,
                'outliers_removed': outlier_count,
                'percentage_kept': (valid_count / len(array1)) * 100
            }
        except Exception as e:
            results[method] = {'error': str(e)}

    return results


def reproject_and_align_rasters(src_paths, case_folder, target_crs=None, reference_raster=None,
                            resolution=None, resampling_method=Resampling.average,
                            apply_ref_mask=True, set_negative_to_nodata=True, outlier_threshold=None):
    """
    Reproject and align multiple rasters to a target CRS and/or match the extent and resolution of a reference raster.

    Parameters:
    -----------
    src_paths : list of str
        Paths to the source raster files to be processed
    dst_paths : list of str, optional
        Paths where the reprojected and aligned rasters will be saved
        If None, will generate output paths based on source paths
    target_crs : dict or str, optional
        Target coordinate reference system as a PROJ4 string, EPSG code, or WKT string
        If None and reference_raster is provided, the CRS from reference_raster will be used
    reference_raster : str, optional
        Path to a reference raster to match the extent and resolution
        If provided, the output will align with this raster
    resolution : tuple (float, float), optional
        (x_res, y_res) target resolution in target CRS units
        If None and reference_raster is provided, the resolution from reference_raster will be used
    resampling_method : rasterio.warp.Resampling, optional
        Resampling algorithm to use for reprojection (default: Resampling.average)
    apply_ref_mask : bool, optional
        If True and reference_raster is provided, will set output pixels to NoData where reference has NaN
    outlier_threshold : float, optional

    Returns:
    --------
    list of bool
        List of boolean values indicating success for each processed raster
    """

    # Ensure src_paths is a list
    if isinstance(src_paths, str):
        src_paths = [src_paths]

    #   define output folder
    output_folder = os.path.join(case_folder, "uniform")

    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Prepare reference raster mask (read only once)
    ref_mask = None
    ref_transform = None
    ref_width = None
    ref_height = None
    ref_crs = None
    common_nodata_values = [-999, -99, -9999, -3.40282e+38, ]

    # Process reference raster if provided
    if reference_raster:
        with rasterio.open(reference_raster) as ref:
            target_crs = target_crs or ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_crs = ref.crs
            ref_nodata = ref.nodata
            ref_compression = ref.profile.get("compress", "lzw")

            # Read reference mask (NaN values) if apply_ref_mask is True
            if apply_ref_mask:
                ref_data = ref.read(1)  # Read first band, assuming mask applies to all bands

                # Create initial mask from NaN values
                ref_mask = np.isnan(ref_data)

                # Add NoData values to mask if defined
                if ref_nodata is not None:
                    ref_mask = np.logical_or(ref_mask, ref_data == ref_nodata)

                # Try common NoData values if no mask was created
                if not np.any(ref_mask):
                    for value in common_nodata_values:
                        potential_mask = (ref_data == value)
                        if np.any(potential_mask):
                            ref_mask = np.logical_or(ref_mask, potential_mask)
                            print(f"Using {value} as NoData in reference raster")
    elif target_crs is None:
        raise ValueError("Either target_crs or reference_raster must be provided")

    # List to store processing results
    processing_results = []

    # Process each raster
    for src_path in src_paths:

        #   Extract name for identification and prints
        date = re.search(r'\d{4}', src_path).group()

        dst_path = os.path.join(output_folder, f"{date}_uniform.tif")

        try:
            with rasterio.open(src_path) as src:
                src_crs = src.crs
                src_transform = src.transform

                # Read first band to change no data values
                src_data = src.read(1)

                # Replace common NoData values with reference nodata value
                for value in common_nodata_values:
                    src_data = np.where(src_data == value, ref_nodata, src_data)

                # Determine transform and dimensions
                if reference_raster:
                    dst_transform = ref_transform
                    dst_width = ref_width
                    dst_height = ref_height
                    dst_crs = ref_crs
                    dst_nodata = ref_nodata
                    dst_compression = ref_compression
                else:
                    # Calculate the optimal transform for the new CRS
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src_crs, target_crs, src.width, src.height,
                        left=src.bounds.left, bottom=src.bounds.bottom,
                        right=src.bounds.right, top=src.bounds.top,
                        resolution=resolution
                    )
                    dst_crs = target_crs

                # Create destination dataset
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'nodata': dst_nodata,
                    "compress": dst_compression,
                    "predictor": 2
                })

                with rasterio.open(dst_path, 'w', **dst_kwargs) as dst:
                    # Initialize destination arrays for each band
                    # dst_data = np.zeros((src.count, dst_height, dst_width), dtype=dst_kwargs['dtype'])
                    dst_data = np.full((src.count, dst_height, dst_width), dst_nodata, dtype=dst_kwargs['dtype'])

                    # Reproject each band
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=dst_data[i - 1],
                            src_transform=src_transform,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=resampling_method,
                            src_nodata=src.nodata or -999,
                            dst_nodata=dst_nodata,
                            dst_width=ref_width,
                            dst_height=ref_height,
                            compress=ref_compression
                        )

                        # Apply negative to NoData conversion if enabled
                        if set_negative_to_nodata:
                            dst_data[i - 1][dst_data[i - 1] < 0] = dst_nodata
                            print("Negative values set to NoData")

                        # Apply reference mask if available
                        if ref_mask is not None and apply_ref_mask:
                            dst_data[i - 1][ref_mask] = dst_nodata
                            print("Applied reference raster mask")

                        # Remove outliers
                        if outlier_threshold is not None:
                            dst_data[dst_data > outlier_threshold] = dst_nodata
                            print(f"Applied outlier removal with threshold {outlier_threshold}")

                    # Write all bands at once
                    dst.write(dst_data)
                    dst.nodata = dst_nodata or -999

                processing_results.append(True)
                print(f"Successfully processed {os.path.basename(src_path)}")

        except Exception as e:
            print(f"Error processing {os.path.basename(src_path)}: {e}")
            processing_results.append(False)

    return output_folder
