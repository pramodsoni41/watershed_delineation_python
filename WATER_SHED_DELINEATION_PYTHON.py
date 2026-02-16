# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:24:21 2026

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Watershed delineation with PySheds (Lat/Lon input + clean plots)
Dr. Pramod Soni (modified)

What this script does:
1) Reads DEM
2) Hydrologic conditioning: fill pits -> fill depressions -> resolve flats
3) Flow direction + accumulation
4) Takes outlet as Lat/Lon (WGS84) OR click point and prints Lat/Lon
5) Snaps outlet to stream mask (percentile or area threshold)
6) Delineates catchment
7) Produces "beautiful" maps: DEM, flowdir, accumulation, catchment+streams
"""
# --- ADD THESE IMPORTS AT TOP (after existing imports) ---
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon
from shapely.ops import unary_union
import rasterio.features

# --- ADD THESE USER SETTINGS (near USER SETTINGS) ---
OUT_DIR = r"E:/QUALITY_1/outputs_pysheds"
BASIN_SHP = f"{OUT_DIR}/basin.shp"
STREAMS_SHP = f"{OUT_DIR}/streams.shp"

# "Fine" streams: lower percentile => denser network (try 90–97)
FINE_STREAM_PERCENTILE = 93

# Optional: keep only streams inside basin in output shapefile
CLIP_STREAMS_TO_BASIN = True


# =========================================================
# ========= ADD THESE HELPERS BELOW YOUR HELPERS ===========
# =========================================================
def ensure_outdir(path):
    import os
    os.makedirs(path, exist_ok=True)

def mask_to_polygon(grid, mask_bool):
    """
    Convert a boolean Raster/array mask to a (multi)polygon in DEM CRS.
    Uses rasterio.features.shapes with the grid affine transform.
    """
    mask_arr = np.asarray(mask_bool).astype(np.uint8)
    transform = grid.affine  # affine transform for raster coords
    shapes_gen = rasterio.features.shapes(mask_arr, mask=mask_arr.astype(bool), transform=transform)

    polys = []
    for geom, val in shapes_gen:
        if val == 1:
            polys.append(Polygon(geom["coordinates"][0]))

    if not polys:
        return None
    return unary_union(polys)  # Polygon or MultiPolygon

def branches_to_gdf(branches, crs_wkt):
    """
    Convert pysheds extract_river_network GeoJSON-like dict to GeoDataFrame of LineStrings.
    """
    lines = []
    for feat in branches["features"]:
        coords = feat["geometry"]["coordinates"]
        if coords and len(coords) >= 2:
            lines.append(LineString(coords))
    gdf = gpd.GeoDataFrame({"id": range(1, len(lines) + 1)}, geometry=lines, crs=crs_wkt)
    return gdf

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pyproj import CRS, Transformer

# -----------------------------
# USER SETTINGS
# -----------------------------
DEM_PATH = r"E:/QUALITY_1/Terrain/Terrain.ASTGTMV003_N26E083_dem.tif"

# Outlet input mode:
#   "latlon"  -> use OUTLET_LAT/OUTLET_LON below
#   "click"   -> click on accumulation plot; it prints lat/lon and uses that
OUTLET_MODE = "latlon"   # "latlon" or "click"

OUTLET_LAT = 26.275
OUTLET_LON = 83.71

# Stream mask method:
#   "percentile" -> uses STREAM_PERCENTILE of accumulation
#   "area_km2"   -> uses STREAM_AREA_KM2 converted to cells (works best for projected CRS)
STREAM_MASK_METHOD = "percentile"   # "percentile" or "area_km2"
STREAM_PERCENTILE = 97              # 95 (more streams) ... 99 (fewer streams)
STREAM_AREA_KM2 = 20                # only used if STREAM_MASK_METHOD="area_km2"

# Snap distance control (like TauDEM "snap threshold in meters")
MAX_SNAP_DISTANCE_M = 3000  # for EPSG:4326 this will be treated approximately (see note)

# Plot tuning
FIGSIZE = (9, 7)
INTERP = "nearest"  # avoids washing out thin tributaries
ACC_VMIN_PCT = 1
ACC_VMAX_PCT = 99.5

# -----------------------------
# HELPERS
# -----------------------------
def get_transformers(grid):
    """Create transformers between DEM CRS and WGS84 lat/lon."""
    dem_crs = CRS.from_user_input(grid.crs) if grid.crs else CRS.from_epsg(4326)
    wgs84 = CRS.from_epsg(4326)
    to_ll = Transformer.from_crs(dem_crs, wgs84, always_xy=True)
    to_dem = Transformer.from_crs(wgs84, dem_crs, always_xy=True)
    return dem_crs, to_ll, to_dem

def finite(arr):
    a = np.asarray(arr, float)
    a[~np.isfinite(a)] = 0.0
    return a

def set_latlon_ticks(ax, grid, nx=5, ny=5):
    """
    Keep data plotted in DEM CRS, but label ticks as Lon/Lat.
    Works for both projected and geographic DEM.
    """
    _, to_ll, _ = get_transformers(grid)
    xmin, ymin, xmax, ymax = grid.bbox

    xticks = np.linspace(xmin, xmax, nx)
    yticks = np.linspace(ymin, ymax, ny)

    lon_labels = [to_ll.transform(x, ymin)[0] for x in xticks]
    lat_labels = [to_ll.transform(xmin, y)[1] for y in yticks]

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.4f}" for v in lon_labels])
    ax.set_yticklabels([f"{v:.4f}" for v in lat_labels])

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

def pretty_axes(ax, title):
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(True, alpha=0.25)

def build_stream_mask(grid, acc, method="percentile", percentile=97, area_km2=20):
    """
    Stream mask from flow accumulation.
    - percentile: robust across DEM sizes
    - area_km2: converts to cells if DEM CRS is projected in meters (UTM etc.)
    """
    acc_arr = finite(acc)
    pos = acc_arr[acc_arr > 0]
    if pos.size == 0:
        raise ValueError("Accumulation has no positive values. Check DEM/flowdir.")

    if method == "area_km2":
        # Only meaningful if DEM CRS is projected (meters). If geographic, fall back to percentile.
        dem_crs, _, _ = get_transformers(grid)
        if dem_crs.is_projected:
            cell = abs(grid.affine.a)  # meters
            thr = (area_km2 * 1e6) / (cell * cell)
        else:
            thr = np.percentile(pos, percentile)
        note = f"area_km2={area_km2} -> thr≈{thr:.1f} cells" if dem_crs.is_projected else f"percentile={percentile} (geo CRS)"
    else:
        thr = np.percentile(pos, percentile)
        note = f"percentile={percentile} -> thr≈{thr:.1f} cells"

    mask = grid.view(acc) >= float(thr)
    return mask, float(thr), note

def snap_with_guard(grid, mask, x, y, max_dist_m=3000):
    """Snap to mask and guard snapping distance."""
    xs, ys = grid.snap_to_mask(mask, (x, y))

    # If CRS is geographic, distance is degrees -> approximate meters using 111km/deg (rough)
    dem_crs, _, _ = get_transformers(grid)
    if dem_crs.is_geographic:
        dx = (xs - x) * 111_000.0 * np.cos(np.deg2rad(y))
        dy = (ys - y) * 111_000.0
        dist_m = float(np.hypot(dx, dy))
    else:
        dist_m = float(np.hypot(xs - x, ys - y))

    if dist_m > max_dist_m:
        raise ValueError(f"Snapped too far ({dist_m:.1f} m). Check outlet location / stream threshold.")

    return xs, ys, dist_m

# -----------------------------
# MAIN WORKFLOW
# -----------------------------
grid = Grid.from_raster(DEM_PATH)
dem = grid.read_raster(DEM_PATH).astype(float)

dem_crs, to_ll, to_dem = get_transformers(grid)
print("DEM CRS:", dem_crs)
print("DEM bbox:", grid.bbox)

# ---- Plot DEM (pretty) ----
fig, ax = plt.subplots(figsize=FIGSIZE)
im = ax.imshow(dem, extent=grid.extent, cmap="terrain", interpolation=INTERP)
plt.colorbar(im, ax=ax, label="Elevation (m)")
pretty_axes(ax, "Digital Elevation Model")
set_latlon_ticks(ax, grid)
plt.tight_layout()
plt.show()

# ---- Hydrologic conditioning ----
pit_filled = grid.fill_pits(dem)
flooded = grid.fill_depressions(pit_filled)
inflated = grid.resolve_flats(flooded)

# ---- Flow direction ----
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
fdir = grid.flowdir(inflated, dirmap=dirmap)

fig, ax = plt.subplots(figsize=FIGSIZE)
im = ax.imshow(fdir, extent=grid.extent, cmap="viridis", interpolation=INTERP)
bounds = ([0] + sorted(list(dirmap)))
plt.colorbar(im, ax=ax, boundaries=bounds, values=sorted(dirmap), label="D8 direction code")
pretty_axes(ax, "Flow Direction Grid")
set_latlon_ticks(ax, grid)
plt.tight_layout()
plt.show()

# ---- Accumulation ----
acc = grid.accumulation(fdir, dirmap=dirmap)
acc_arr = finite(acc)
pos = acc_arr[acc_arr > 0]
vmin = np.percentile(pos, ACC_VMIN_PCT)
vmax = np.percentile(pos, ACC_VMAX_PCT)

fig, ax = plt.subplots(figsize=FIGSIZE)
im = ax.imshow(
    np.where(acc_arr > 0, acc_arr, np.nan),
    extent=grid.extent,
    cmap="cubehelix",
    norm=colors.LogNorm(vmin, vmax),
    interpolation=INTERP,
)
plt.colorbar(im, ax=ax, label="Upstream cells (log scale)")
pretty_axes(ax, "Flow Accumulation")
set_latlon_ticks(ax, grid)
plt.tight_layout()
plt.show()

# ---- Outlet selection (lat/lon input OR click) ----
if OUTLET_MODE.lower() == "click":
    print("Click ONE point on the Accumulation map window, then press Enter...")
    pt = plt.ginput(1)  # requires interactive backend (qt/widget)
    x_dem, y_dem = pt[0]
    lon_pt, lat_pt = to_ll.transform(x_dem, y_dem)
    print(f"Clicked outlet lat/lon:  {lat_pt:.6f}, {lon_pt:.6f}")

    OUTLET_LAT, OUTLET_LON = lat_pt, lon_pt

# Convert lat/lon -> DEM CRS
x0, y0 = to_dem.transform(float(OUTLET_LON), float(OUTLET_LAT))
print("Outlet (lat,lon):", OUTLET_LAT, OUTLET_LON)
print("Outlet in DEM CRS:", x0, y0)

# Check inside bbox
inside = (grid.bbox[0] <= x0 <= grid.bbox[2]) and (grid.bbox[1] <= y0 <= grid.bbox[3])
print("Outlet inside DEM bbox:", inside)
if not inside:
    raise ValueError("Outlet is outside DEM extent. Use a larger DEM tile or correct coordinates.")

# ---- Stream mask + snapping ----
stream_mask, thr_used, thr_note = build_stream_mask(
    grid, acc,
    method=STREAM_MASK_METHOD,
    percentile=STREAM_PERCENTILE,
    area_km2=STREAM_AREA_KM2
)
print("Stream mask:", thr_note)

x_snap, y_snap, snap_dist = snap_with_guard(grid, stream_mask, x0, y0, MAX_SNAP_DISTANCE_M)
lon_snap, lat_snap = to_ll.transform(x_snap, y_snap)

print(f"Snapped outlet distance: {snap_dist:.1f} m")
print(f"Snapped outlet lat/lon: {lat_snap:.6f}, {lon_snap:.6f}")

# ---- Catchment ----
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype="coordinate")
catch_view = grid.view(catch)
catch_arr = np.asarray(catch_view)

# Extract rivers (vector)
branches = grid.extract_river_network(fdir, stream_mask, dirmap=dirmap)

ensure_outdir(OUT_DIR)

# 1) Create a finer stream mask for denser network
fine_stream_mask, fine_thr_used, fine_thr_note = build_stream_mask(
    grid, acc,
    method="percentile",
    percentile=FINE_STREAM_PERCENTILE,
    area_km2=STREAM_AREA_KM2
)
print("Fine stream mask:", fine_thr_note)

# 2) Extract finer river network
fine_branches = grid.extract_river_network(fdir, fine_stream_mask, dirmap=dirmap)

# 3) Convert basin raster to polygon and save as SHP
basin_poly = mask_to_polygon(grid, catch_view)

if basin_poly is None:
    raise ValueError("Basin polygon conversion failed (empty basin mask).")

basin_gdf = gpd.GeoDataFrame({"name": ["Basin_1"]}, geometry=[basin_poly], crs=str(grid.crs))
basin_gdf.to_file(BASIN_SHP)
print("✅ Basin shapefile saved:", BASIN_SHP)

# 4) Convert streams to GeoDataFrame and (optionally) clip to basin
streams_gdf = branches_to_gdf(fine_branches, crs_wkt=str(grid.crs))

if CLIP_STREAMS_TO_BASIN:
    streams_gdf = gpd.overlay(streams_gdf, basin_gdf, how="intersection")

streams_gdf.to_file(STREAMS_SHP)
print("✅ Streams shapefile saved:", STREAMS_SHP)


# =========================================================
# ========= OPTIONAL: BEAUTIFUL PLOT OF FINE NETWORK =======
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8))

# hillshade-like background
gy, gx = np.gradient(np.nan_to_num(inflated, nan=np.nanmean(inflated)))
slope = np.hypot(gx, gy)
shade = 1 - (slope / (np.nanmax(slope) if np.nanmax(slope) > 0 else 1))
ax.imshow(shade, extent=grid.extent, cmap="gray", alpha=0.85, interpolation=INTERP)

# basin outline + fill
basin_gdf.boundary.plot(ax=ax, linewidth=2)
basin_gdf.plot(ax=ax, alpha=0.15)

# fine streams
streams_gdf.plot(ax=ax, linewidth=1.0)

# outlet points
ax.plot(x0, y0, "r*", ms=12, label=f"Original ({OUTLET_LAT:.4f}, {OUTLET_LON:.4f})")
ax.plot(x_snap, y_snap, "ko", ms=6, label=f"Snapped ({lat_snap:.4f}, {lon_snap:.4f})")

pretty_axes(ax, "Basin + Fine Stream Network (Shapefile Exported)")
set_latlon_ticks(ax, grid)
ax.legend(loc="lower left", frameon=True)

plt.tight_layout()
plt.show()


# ---- Beautiful final plot: Catchment + Streams + Outlet ----
fig, ax = plt.subplots(figsize=(10, 8))

# Background: hillshade-like contrast from inflated DEM gradient (simple & fast)
gy, gx = np.gradient(np.nan_to_num(inflated, nan=np.nanmean(inflated)))
slope = np.hypot(gx, gy)
shade = 1 - (slope / (np.nanmax(slope) if np.nanmax(slope) > 0 else 1))
ax.imshow(shade, extent=grid.extent, cmap="gray", alpha=0.85, interpolation=INTERP)

# Catchment overlay
ax.imshow(np.where(catch_arr, 1, np.nan), extent=grid.extent,
          cmap="Greys_r", alpha=0.25, interpolation=INTERP)

# Streams as vectors
for branch in branches["features"]:
    coords = np.asarray(branch["geometry"]["coordinates"])
    if coords.shape[0] >= 2:
        ax.plot(coords[:, 0], coords[:, 1], linewidth=1.0)

# Original + snapped outlet
ax.plot(x0, y0, "r*", ms=12, label=f"Original ({OUTLET_LAT:.4f}, {OUTLET_LON:.4f})")
ax.plot(x_snap, y_snap, "ko", ms=6, label=f"Snapped ({lat_snap:.4f}, {lon_snap:.4f})")

pretty_axes(ax, "Catchment Delineation with Extracted Stream Network")
set_latlon_ticks(ax, grid)
ax.legend(loc="lower left", frameon=True)
plt.tight_layout()
plt.show()
