# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:32:17 2026

@author: acer
"""

import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from rasterio.windows import from_bounds
from rasterio.sample import sample_gen
from pyproj import Transformer
from datetime import datetime

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
dem_folder = r"E:/Project 91/"
center_lat = 25.71236389
center_lon = 80.58377778
SPAN_KM = 20
half_span_m = (SPAN_KM * 1000) / 2

# Same two pixel points (from clipped DEM)
pts = [(330.6962365591397, 273.2231182795698),
       (290.3064516129032, 336.2311827956989)]

# --------------------------------------------------
# Find all DEMs
# --------------------------------------------------
dem_files = sorted(glob.glob(os.path.join(dem_folder, "AST14DEM_*.tif")))

plt.figure(figsize=(10,5))

for dem_path in dem_files:

    filename = os.path.basename(dem_path)

    # ---------------- Extract date ----------------
    # Example: AST14DEM_00402082004052530_20250321065142.tif
    # date part = 02082004
    date_str = filename.split("_")[1][3:11]  # remove first 2 digits (tile code)

    date_label = date_str
    

    # ---------------- Read DEM ----------------
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        transform = src.transform

        transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        cx, cy = transformer.transform(center_lon, center_lat)

        xmin = cx - half_span_m
        xmax = cx + half_span_m
        ymin = cy - half_span_m
        ymax = cy + half_span_m

        window = from_bounds(xmin, ymin, xmax, ymax, transform)
        dem_clip = src.read(1, window=window)
        transform_clip = src.window_transform(window)

    dem_clip = dem_clip.astype(float)
    dem_clip[dem_clip < 0] = np.nan

    # ---------------- Build same line ----------------
    x1, y1 = transform_clip * pts[0]
    x2, y2 = transform_clip * pts[1]
    line = LineString([(x1, y1), (x2, y2)])

    # ---------------- Sample elevations ----------------
    n_points = 200
    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(d) for d in distances]
    coords = [(p.x, p.y) for p in points]

    with rasterio.open(dem_path) as src:
        elevations = np.array([val[0] for val in sample_gen(src, coords)])

    # ---------------- Plot profile ----------------
    plt.plot(distances, elevations, linewidth=2, label=date_label)

# --------------------------------------------------
# Final Plot
# --------------------------------------------------
plt.xlabel("Distance (m)")
plt.ylabel("Elevation (m)")
plt.title("Multi-Year Cross Section Profile")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("CrossSection_MultiYear.png", dpi=300)
plt.show()
