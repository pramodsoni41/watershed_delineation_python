import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from datetime import datetime

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
data_folder = r"F:\DATA\Global_Water_data\DATA"

center_lat = 25.71236389
center_lon = 80.58377778

SPAN_KM = 7
deg_per_km = 1 / 111.32
half_span_deg = (SPAN_KM / 2) * deg_per_km

# --------------------------------------------------
# 1️⃣  LIST FILES
# --------------------------------------------------
pattern = os.path.join(
    data_folder,
    "*Lat_20-30_Lon_80-90.tif"
)

water_files = sorted(glob.glob(pattern))

print(f"Found {len(water_files)} files")

# --------------------------------------------------
# 2️⃣  LOOP THROUGH FILES
# --------------------------------------------------
for tif_path in water_files:

    filename = os.path.basename(tif_path)

    # ---- Extract year + month from filename ----
    # Example:
    # Year_1988_month_02_Lat_10-20_Lon_80-90.tif
    parts = filename.split("_")
    year = parts[1]
    month = parts[3]

    date_label = f"{year}-{month}"

    print(f"Processing: {date_label}")

    # --------------------------------------------------
    # Read & Clip
    # --------------------------------------------------
    with rasterio.open(tif_path) as src:

        xmin = center_lon - half_span_deg
        xmax = center_lon + half_span_deg
        ymin = center_lat - half_span_deg
        ymax = center_lat + half_span_deg

        window = from_bounds(xmin, ymin, xmax, ymax, src.transform)

        water_clip = src.read(1, window=window)
        transform_clip = src.window_transform(window)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(6,6))

    plt.imshow(water_clip, cmap="Blues", vmin=0, vmax=2)
    plt.title(f"Water Surface - {date_label}")
    plt.colorbar(label="Water Class")

    # Mark centre
    row, col = ~transform_clip * (center_lon, center_lat)
    plt.plot(col, row, 'ro', markersize=6)

    plt.tight_layout()

    # Save
    out_name = f"Water_{date_label}.png"
    plt.savefig(out_name, dpi=300)
    plt.close()

print("✅ All plots generated successfully.")
