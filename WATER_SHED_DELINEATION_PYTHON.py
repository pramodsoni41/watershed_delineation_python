# ==============================================
# WATERSHED DELINEATION USING PYSHEDS
# Dr. Pramod Soni
# ==============================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pysheds.grid import Grid

# Optional but recommended (for CRS transform)
from pyproj import CRS, Transformer


# =====================================================
# ========= USER INPUT SECTION (EDIT HERE) ============
# =====================================================

DEM_PATH = Path("data/dem.tif")          # Path to DEM file
OUTPUT_FOLDER = Path("outputs")          # Output directory

# --- Pour point ---
# If your coordinates are Lat/Lon (WGS84), set mode="latlon"
# If your coordinates are already in DEM projected CRS, set mode="projected"
POUR_POINT_MODE = "latlon"               # "latlon" OR "projected"

POUR_LAT = 25.2                         # used if mode="latlon"
POUR_LON = 82.9                         # used if mode="latlon"

POUR_X = 326500                          # used if mode="projected"
POUR_Y = 1289000                         # used if mode="projected"

NODATA_VALUE = -9999                     # Change if different (or set None)
MAX_ELEVATION = None                     # Example: 200 (or None)

SNAP_THRESHOLD = 2000                    # Use a realistic threshold for your DEM
CHANNEL_THRESHOLD = 500                  # River extraction threshold

XYTYPE = "coordinate"                    # Keep "coordinate" for projected CRS

# =====================================================
# =====================================================


def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_plot(fig, out_path: Path, filename: str):
    fig.savefig(out_path / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_pour_point_in_dem_crs(grid: Grid):
    """
    Returns (x, y) in DEM CRS.
    - If POUR_POINT_MODE = "latlon": converts WGS84 lon/lat -> DEM CRS
    - If "projected": uses POUR_X, POUR_Y directly
    """
    if POUR_POINT_MODE.lower() == "projected":
        return float(POUR_X), float(POUR_Y), "projected input"

    # Lat/Lon input
    # PySheds stores CRS in grid.crs (often as rasterio CRS or WKT)
    # We'll robustly convert it using pyproj.CRS
    dem_crs = CRS.from_user_input(grid.crs)
    wgs84 = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(wgs84, dem_crs, always_xy=True)
    x, y = transformer.transform(float(POUR_LON), float(POUR_LAT))
    return float(x), float(y), "latlon->DEM CRS transform"


def finite_acc(acc):
    arr = np.array(acc, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def robust_snap_to_mask(grid: Grid, acc_raster, pour_xy, snap_threshold: float):
    """
    In some pysheds versions, snap_to_mask requires mask to be a Raster.
    So we build the mask as a Raster using grid.view().
    """
    x0, y0 = pour_xy

    acc_arr = finite_acc(acc_raster)
    vmax = float(np.nanmax(acc_arr))

    if vmax <= 0:
        raise ValueError("Flow accumulation max is <= 0. Check DEM/flowdir.")

    # If user threshold too high, reduce automatically
    thr = float(snap_threshold)
    if thr >= vmax:
        thr = max(1.0, 0.25 * vmax)
        print(f"⚠️ SNAP_THRESHOLD too high for this DEM. Using thr={thr:.2f} instead.")

    # Build mask as Raster (not numpy)
    # grid.view(acc_raster) returns a Raster-like view aligned to grid
    acc_view = grid.view(acc_raster)
    mask_raster = acc_view > thr  # this stays as Raster in pysheds

    # If still empty, relax progressively (percentiles)
    if not np.any(np.asarray(mask_raster)):
        pos = acc_arr[acc_arr > 0]
        for q in [99, 98, 97, 95, 90]:
            t = float(np.percentile(pos, q))
            mask_raster = acc_view >= t
            if np.any(np.asarray(mask_raster)):
                xs, ys = grid.snap_to_mask(mask_raster, (x0, y0))
                return xs, ys, f"percentile={q} (thr≈{t:.2f})"
        raise ValueError("Could not find any cells for snapping. Try lowering SNAP_THRESHOLD.")

    xs, ys = grid.snap_to_mask(mask_raster, (x0, y0))
    return xs, ys, f"threshold={thr:.2f}"


ensure_folder(OUTPUT_FOLDER)

if not DEM_PATH.exists():
    raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

grid = Grid.from_raster(str(DEM_PATH))
dem = grid.read_raster(str(DEM_PATH)).astype(float)

# Handle NoData
if NODATA_VALUE is not None:
    dem[dem == NODATA_VALUE] = np.nan
if MAX_ELEVATION is not None:
    dem[dem > MAX_ELEVATION] = np.nan

# Pour point conversion / verification
px, py, mode_note = get_pour_point_in_dem_crs(grid)

print("DEM bbox:", grid.bbox)
print("Pour point (DEM CRS):", (px, py), f"[{mode_note}]")
if not (grid.bbox[0] <= px <= grid.bbox[2] and grid.bbox[1] <= py <= grid.bbox[3]):
    print("⚠️ WARNING: Pour point is outside DEM bbox.")
    print("   - If using lat/lon, confirm DEM CRS and your coordinates (lat,lon order).")
    print("   - If using projected, confirm units/meters and EPSG/UTM zone.")

# Plot DEM
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(dem, extent=grid.extent, cmap="terrain")
fig.colorbar(im, ax=ax, label="Elevation")
ax.set_title("Digital Elevation Model")
ax.set_xlabel("X")
ax.set_ylabel("Y")
save_plot(fig, OUTPUT_FOLDER, "01_dem.png")

# Hydrological conditioning
pit_filled = grid.fill_pits(dem)
flooded = grid.fill_depressions(pit_filled)
inflated = grid.resolve_flats(flooded)

dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
fdir = grid.flowdir(inflated, dirmap=dirmap)

# Flow accumulation
acc = grid.accumulation(fdir, dirmap=dirmap)
acc_arr = finite_acc(acc)
print("acc min/max:", float(np.nanmin(acc_arr)), float(np.nanmax(acc_arr)))

fig, ax = plt.subplots(figsize=(8, 6))
vmax = float(np.nanmax(acc_arr)) if np.nanmax(acc_arr) > 1 else 2.0
im = ax.imshow(acc_arr, extent=grid.extent, cmap="cubehelix",
               norm=colors.LogNorm(1, vmax))
fig.colorbar(im, ax=ax, label="Upstream Cells")
ax.set_title("Flow Accumulation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
save_plot(fig, OUTPUT_FOLDER, "02_flow_accumulation.png")

# Snap pour point (Raster mask)
x_snap, y_snap, snap_note = robust_snap_to_mask(grid, acc, (px, py), SNAP_THRESHOLD)
print(f"✅ Snapped pour point: ({x_snap:.3f}, {y_snap:.3f}) using {snap_note}")

# Catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype=XYTYPE)
catch_view = grid.view(catch)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(np.where(catch_view, 1, np.nan), extent=grid.extent, cmap="Greys_r")
ax.plot(px, py, "r*", markersize=10, label="Original")
ax.plot(x_snap, y_snap, "bo", markersize=6, label="Snapped")
ax.set_title("Delineated Catchment")
ax.set_xlabel("X/Longitude")
ax.set_ylabel("Y/Latitude")
ax.legend(loc="lower left")
save_plot(fig, OUTPUT_FOLDER, "03_catchment.png")

# River network
branches = grid.extract_river_network(fdir, grid.view(acc) > CHANNEL_THRESHOLD, dirmap=dirmap)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect("equal")
for branch in branches["features"]:
    coords = np.asarray(branch["geometry"]["coordinates"])
    ax.plot(coords[:, 0], coords[:, 1])
ax.plot(x_snap, y_snap, "ro", markersize=5)
ax.set_title("Extracted River Network")
ax.set_xlabel("X/Longitude")
ax.set_ylabel("Y/Latitude")
save_plot(fig, OUTPUT_FOLDER, "04_river_network.png")

# Flow distance
dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype=XYTYPE)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(dist, extent=grid.extent, cmap="cubehelix_r")
fig.colorbar(im, ax=ax, label="Distance (cells)")
ax.set_title("Flow Distance to Outlet")
ax.set_xlabel("X")
ax.set_ylabel("Y")
save_plot(fig, OUTPUT_FOLDER, "05_flow_distance.png")

print("✅ Watershed analysis complete.")
print("Outputs saved in:", OUTPUT_FOLDER.resolve())

