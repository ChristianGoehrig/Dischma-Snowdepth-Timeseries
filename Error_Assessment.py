import os
import numpy as np
import rasterio
from numpy.ma.core import compress
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
import glob

# ============================================================================
# CONFIGURATION - SET YOUR PATHS HERE
# ============================================================================
folder_05m = r"E:\manned_aircraft\christiangoehrig\data\original_data"
folder_2m = r"E:\manned_aircraft\christiangoehrig\data\datasets\max_common_acquisition_extent\HS"
output_folder = r"E:\manned_aircraft\christiangoehrig\data\Resampling_Errors3"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("=" * 70)
print("RESAMPLING ERROR ANALYSIS (GDAL/Rasterio)")
print("=" * 70)

# ============================================================================
# PART 1: CALCULATE ERROR RASTERS FOR EACH PAIR
# ============================================================================
print("\n--- PART 1: Calculating Error Rasters ---\n")

# Get list of 0.5m rasters
rasters_05m = glob.glob(os.path.join(folder_05m, "*.tif"))
print(f"Found {len(rasters_05m)} rasters in 0.5m folder\n")

error_rasters_list = []
processed_count = 0
skipped_count = 0

for path_05m in rasters_05m:
    try:
        # Get base name
        base_name = os.path.splitext(os.path.basename(path_05m))[0]
        print(f"Processing {base_name}...")

        # Build path to corresponding 2m raster
        raster_filename = os.path.basename(path_05m)
        path_2m = os.path.join(folder_2m, raster_filename)

        # Check if 2m raster exists
        if not os.path.exists(path_2m):
            print(f"  WARNING: {raster_filename} not found in 2m folder, skipping...")
            skipped_count += 1
            continue

        # Read 0.5m raster (original)
        print(f"  Reading 0.5m raster...")
        with rasterio.open(path_05m) as src_05m:
            data_05m = src_05m.read(1, masked=True)  # Read as masked array
            profile_05m = src_05m.profile
            transform_05m = src_05m.transform
            crs_05m = src_05m.crs
            nodata_05m = src_05m.nodata

        # Read 2m raster
        print(f"  Reading 2m raster...")
        with rasterio.open(path_2m) as src_2m:
            data_2m = src_2m.read(1, masked=True)
            transform_2m = src_2m.transform
            crs_2m = src_2m.crs
            bounds_2m = src_2m.bounds
            nodata_2m = src_2m.nodata

        # Step 1: Resample 2m to 0.5m resolution (nearest neighbor)
        print(f"  Resampling 2m to 0.5m (nearest neighbor)...")

        # Create destination array matching 0.5m dimensions
        resampled_2m = np.empty(data_05m.shape, dtype=data_2m.dtype)

        # Reproject 2m data to match 0.5m grid
        reproject(
            source=data_2m,
            destination=resampled_2m,
            src_transform=transform_2m,
            src_crs=crs_2m,
            dst_transform=transform_05m,
            dst_crs=crs_05m,
            resampling=WarpResampling.nearest,
            compress="lzw"
        )

        # Step 2: Create mask from 2m NoData areas
        print(f"  Creating mask from 2m NoData...")

        # Identify NoData in the resampled 2m
        if nodata_2m is not None:
            mask_2m = (resampled_2m == nodata_2m)
        else:
            mask_2m = np.zeros(resampled_2m.shape, dtype=bool)

        # If original 2m had a mask, apply it
        if hasattr(data_2m, 'mask') and data_2m.mask is not np.ma.nomask:
            # Reproject the mask
            mask_2m_original = data_2m.mask.astype(np.uint8)
            mask_2m_resampled = np.empty(data_05m.shape, dtype=np.uint8)

            reproject(
                source=mask_2m_original,
                destination=mask_2m_resampled,
                src_transform=transform_2m,
                src_crs=crs_2m,
                dst_transform=transform_05m,
                dst_crs=crs_05m,
                resampling=WarpResampling.nearest,
                compress="lzw"
            )

            mask_2m = mask_2m | (mask_2m_resampled > 0)

        # Combine with original 0.5m mask
        combined_mask = mask_2m | data_05m.mask

        # Apply combined mask to both arrays
        resampled_2m_masked = np.ma.masked_array(resampled_2m, mask=combined_mask)
        data_05m_masked = np.ma.masked_array(data_05m, mask=combined_mask)

        # Step 3: Calculate error (aggregated - original)
        print(f"  Calculating error...")
        error = resampled_2m_masked - data_05m_masked

        # Step 4: Clip to 2m extent
        print(f"  Clipping to 2m extent...")

        # Calculate pixel window for the 2m bounds in the 0.5m grid
        window = rasterio.windows.from_bounds(
            bounds_2m.left, bounds_2m.bottom,
            bounds_2m.right, bounds_2m.top,
            transform=transform_05m
        )

        # Round window to integer pixels
        window = window.round_offsets().round_lengths()

        # Extract the window as row/col slices
        row_start = int(window.row_off)
        row_end = int(window.row_off + window.height)
        col_start = int(window.col_off)
        col_end = int(window.col_off + window.width)

        # Ensure indices are within bounds
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(error.shape[0], row_end)
        col_end = min(error.shape[1], col_end)

        # Clip the error array
        error_clipped = error[row_start:row_end, col_start:col_end]

        # Update transform for clipped extent
        transform_clipped = rasterio.windows.transform(window, transform_05m)

        # Count valid pixels
        valid_pixels = np.sum(~error_clipped.mask)
        total_pixels = error_clipped.size
        print(
            f"  Valid pixels after clipping: {valid_pixels:,} / {total_pixels:,} ({100 * valid_pixels / total_pixels:.1f}%)")

        # Step 5: Save clipped error raster
        error_output = os.path.join(output_folder, f"error_{base_name}.tif")

        profile_error = profile_05m.copy()
        profile_error.update(
            dtype=rasterio.float32,
            nodata=-999,
            width=error_clipped.shape[1],
            height=error_clipped.shape[0],
            transform=transform_clipped,
            compress="lzw"
        )

        with rasterio.open(error_output, 'w', **profile_error) as dst:
            # Convert masked array to regular array with nodata
            error_filled = error_clipped.filled(-999)
            dst.write(error_filled, 1)

        error_rasters_list.append(error_output)

        print(f"  ✓ Completed: {error_output}")
        processed_count += 1

    except Exception as e:
        print(f"  ✗ ERROR processing {base_name}: {str(e)}")
        skipped_count += 1

print(f"\n--- Error Raster Summary ---")
print(f"Successfully processed: {processed_count}")
print(f"Skipped/Failed: {skipped_count}")
print(f"Total error rasters created: {len(error_rasters_list)}")

if len(error_rasters_list) == 0:
    print("\nNo error rasters were created. Exiting.")
    exit()

# ============================================================================
# PART 2: CALCULATE PIXEL-WISE RMSE
# ============================================================================
print("\n" + "=" * 70)
print("--- PART 2: Calculating Pixel-wise RMSE ---")
print("=" * 70 + "\n")

print("Loading all error rasters into memory...")

# Read all error rasters and stack them
error_arrays = []
template_profile = None
template_shape = None

for i, error_path in enumerate(error_rasters_list):
    print(f"  Loading {os.path.basename(error_path)}...")
    with rasterio.open(error_path) as src:
        data = src.read(1, masked=True)

        if template_profile is None:
            template_profile = src.profile
            template_shape = data.shape

        # Check if shapes match (they should after clipping to same extent)
        if data.shape != template_shape:
            print(f"    WARNING: Shape mismatch! Expected {template_shape}, got {data.shape}")
            print(f"    Skipping this raster...")
            continue

        error_arrays.append(data.filled(np.nan))  # Fill masked values with NaN

print(f"✓ Loaded {len(error_arrays)} error rasters\n")

if len(error_arrays) == 0:
    print("No valid error rasters to process. Exiting.")
    exit()

# Stack into 3D array (n_rasters, rows, cols)
print("Stacking arrays...")
error_stack = np.array(error_arrays)

# Calculate pixel-wise RMSE
# RMSE = sqrt(mean(error²))
print("Calculating pixel-wise RMSE...")
with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
    squared_errors = error_stack ** 2
    mean_squared = np.nanmean(squared_errors, axis=0)
    rmse_array = np.sqrt(mean_squared)

print("✓ RMSE calculation complete\n")

# Save pixel-wise RMSE raster
rmse_output = os.path.join(output_folder, "rmse_pixelwise.tif")

rmse_profile = template_profile.copy()
rmse_profile.update(
    dtype=rasterio.float32,
    nodata=-999
)

# Replace NaN with nodata value for saving
rmse_save = np.where(np.isnan(rmse_array), -999, rmse_array)

with rasterio.open(rmse_output, 'w', **rmse_profile) as dst:
    dst.write(rmse_save.astype(rasterio.float32), 1)

print(f"✓ Pixel-wise RMSE raster saved: {rmse_output}\n")

# ============================================================================
# PART 3: CALCULATE OVERALL STATISTICS
# ============================================================================
print("=" * 70)
print("--- PART 3: Calculating Overall RMSE Statistics ---")
print("=" * 70 + "\n")

print("Extracting statistics...")

# Calculate statistics (ignoring NaN values)
overall_mean_rmse = np.nanmean(rmse_array)
overall_max_rmse = np.nanmax(rmse_array)
overall_min_rmse = np.nanmin(rmse_array)
overall_std_rmse = np.nanstd(rmse_array)

# Count valid pixels
valid_pixels = np.sum(~np.isnan(rmse_array))
total_pixels = rmse_array.size

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS - RESAMPLING ERROR ANALYSIS")
print("=" * 70)
print(f"\nNumber of raster pairs analyzed: {len(error_rasters_list)}")
print(f"Valid pixels analyzed: {valid_pixels:,} / {total_pixels:,}")
print(f"\nRMSE Statistics:")
print(f"  Average RMSE (mean):     {overall_mean_rmse:.4f}")
print(f"  Maximum RMSE:            {overall_max_rmse:.4f}")
print(f"  Minimum RMSE:            {overall_min_rmse:.4f}")
print(f"  Standard Deviation:      {overall_std_rmse:.4f}")
print(f"\nOutputs saved to: {output_folder}")
print(f"  - Individual error rasters: error_*.tif")
print(f"  - Pixel-wise RMSE raster: rmse_pixelwise.tif")
print("=" * 70)

print("\n✓ Analysis complete!")