# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:29:33 2026

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:26:14 2026

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Plot all year-wise DEMs together
"""

import os
import glob
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask

# ---------------------------
# INPUTS
# ---------------------------
folder = r"E:\Suryansh\dems kenjor (2)\dems kenjor\yearwise"
watershed_path = r"E:/Suryansh/dems kenjor (2)/dems kenjor/Archive/basin_2000.shp"

# Load watershed
watershed = gpd.read_file(watershed_path)

# Get all tif files
tif_files = sorted(glob.glob(os.path.join(folder, "*.tif")))

print(f"Found {len(tif_files)} DEM files")

# ---------------------------
# CLIP FUNCTION
# ---------------------------
def clip_dem(dem_path):

    with rasterio.open(dem_path) as src:

        watershed_proj = watershed.to_crs(src.crs)

        out_image, out_transform = mask(
            src,
            watershed_proj.geometry,
            crop=True
        )

        dem = out_image[0].astype(float)

        dem[dem == src.nodata] = np.nan
        dem[dem > 65000] = np.nan
        dem[dem < 0] = np.nan

        return dem

# ---------------------------
# LOAD ALL DEMs
# ---------------------------
dems = []
names = []

for f in tif_files:
    dem = clip_dem(f)
    dems.append(dem)
    
    # Extract year from filename
    name = os.path.basename(f)
    names.append(name)

# ---------------------------
# FIND GLOBAL SCALE
# ---------------------------
all_values = np.concatenate([d.flatten() for d in dems])

global_min = np.nanpercentile(all_values, 2)
global_max = np.nanpercentile(all_values, 98)

print("Global elevation range:", global_min, "to", global_max)

# ---------------------------
# PLOT WITH SAME SCALE
# ---------------------------
n = len(dems)
cols = 3
rows = int(np.ceil(n / cols))

plt.figure(figsize=(5*cols, 4*rows))

for i in range(n):

    plt.subplot(rows, cols, i+1)

    plt.imshow(
        dems[i],
        cmap="terrain",
        vmin=global_min,
        vmax=global_max
    )

    plt.title(names[i])
    plt.colorbar(label="Elevation (m)")

plt.tight_layout()
plt.show()
