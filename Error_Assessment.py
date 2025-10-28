import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
import glob

# ============================================================================
# CONFIGURATION - SET YOUR PATHS HERE
# ============================================================================
folder_05m = r"C:\path\to\05m_rasters"
folder_2m = r"C:\path\to\2m_rasters"
output_folder = r"C:\path\to\error_rasters"

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

        # Read 2m raster
        print(f"  Reading 2m raster...")
        with rasterio.open(path_2m) as src_2m:
            data_2m = src_2m.read(1, masked=True)
            transform_2m = src_2m.transform
            crs_2m = src_2m.crs

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
            resampling=WarpResampling.nearest
        )

        # Convert to masked array with same mask as original
        resampled_2m = np.ma.masked_array(resampled_2m, mask=data_05m.mask)

        # Step 2: Calculate error (aggregated - original)
        print(f"  Calculating error...")
        error = resampled_2m - data_05m

        # Step 3: Save error raster
        error_output = os.path.join(output_folder, f"error_{base_name}.tif")

        profile_error = profile_05m.copy()
        profile_error.update(
            dtype=rasterio.float32,
            nodata=-9999
        )

        with rasterio.open(error_output, 'w', **profile_error) as dst:
            # Convert masked array to regular array with nodata
            error_filled = error.filled(-9999)
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

for i, error_path in enumerate(error_rasters_list):
    print(f"  Loading {os.path.basename(error_path)}...")
    with rasterio.open(error_path) as src:
        data = src.read(1, masked=True)
        error_arrays.append(data.filled(np.nan))  # Fill masked values with NaN

        if template_profile is None:
            template_profile = src.profile

print(f"✓ Loaded {len(error_arrays)} error rasters\n")

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
    nodata=-9999
)

# Replace NaN with nodata value for saving
rmse_save = np.where(np.isnan(rmse_array), -9999, rmse_array)

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