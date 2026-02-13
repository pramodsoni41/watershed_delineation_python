# ==============================================
# WATERSHED DELINEATION USING PYSHEDS
# Dr. Pramod Soni
# ==============================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pysheds.grid import Grid


# =====================================================
# ========= USER INPUT SECTION (EDIT HERE) ===========
# =====================================================

DEM_PATH = Path("data/dem.tif")     # Path to DEM file
OUTPUT_FOLDER = Path("outputs")     # Output directory

POUR_POINT_X = 78.9                 # Longitude or X coordinate
POUR_POINT_Y = 29.9                 # Latitude or Y coordinate

NODATA_VALUE = -9999                # Change if different
MAX_ELEVATION = None                # Example: 200 (or None)

SNAP_THRESHOLD = 200000             # Accumulation threshold for snapping
CHANNEL_THRESHOLD = 500             # River extraction threshold

XYTYPE = "coordinate"               # "coordinate" or "index"

# =====================================================
# =====================================================


def ensure_folder(path):
    path.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    fig.savefig(OUTPUT_FOLDER / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():

    ensure_folder(OUTPUT_FOLDER)

    # Load DEM
    grid = Grid.from_raster(str(DEM_PATH))
    dem = grid.read_raster(str(DEM_PATH)).astype(float)

    # Handle NoData
    if NODATA_VALUE is not None:
        dem[dem == NODATA_VALUE] = np.nan

    if MAX_ELEVATION is not None:
        dem[dem > MAX_ELEVATION] = np.nan

    # ---------------------------------------------------
    # Plot DEM
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dem, extent=grid.extent, cmap="terrain")
    fig.colorbar(im, ax=ax, label="Elevation")
    ax.set_title("Digital Elevation Model")
    ax.set_xlabel("X / Longitude")
    ax.set_ylabel("Y / Latitude")
    save_plot(fig, "01_dem.png")

    # ---------------------------------------------------
    # Hydrological Conditioning
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)

    # ---------------------------------------------------
    # Flow Accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(acc, extent=grid.extent,
                   cmap="cubehelix",
                   norm=colors.LogNorm(1, np.nanmax(acc)))
    fig.colorbar(im, ax=ax, label="Upstream Cells")
    ax.set_title("Flow Accumulation")
    save_plot(fig, "02_flow_accumulation.png")

    # ---------------------------------------------------
    # Snap Pour Point
    mask = acc > SNAP_THRESHOLD
    x_snap, y_snap = grid.snap_to_mask(mask, (POUR_POINT_X, POUR_POINT_Y))

    # Delineate Catchment
    catch = grid.catchment(x=x_snap,
                           y=y_snap,
                           fdir=fdir,
                           dirmap=dirmap,
                           xytype=XYTYPE)

    grid.clip_to(catch)
    clipped = grid.view(catch)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.where(clipped, 1, np.nan),
              extent=grid.extent,
              cmap="Greys_r")
    ax.plot(POUR_POINT_X, POUR_POINT_Y, "r*", markersize=12)
    ax.set_title("Delineated Catchment")
    save_plot(fig, "03_catchment.png")

    # ---------------------------------------------------
    # River Network
    branches = grid.extract_river_network(fdir,
                                          acc > CHANNEL_THRESHOLD,
                                          dirmap=dirmap)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(grid.bbox[0], grid.bbox[2])
    ax.set_ylim(grid.bbox[1], grid.bbox[3])
    ax.set_aspect("equal")

    for branch in branches["features"]:
        coords = np.asarray(branch["geometry"]["coordinates"])
        ax.plot(coords[:, 0], coords[:, 1])

    ax.set_title("Extracted River Network")
    save_plot(fig, "04_river_network.png")

    # ---------------------------------------------------
    # Flow Distance
    dist = grid.distance_to_outlet(x=x_snap,
                                   y=y_snap,
                                   fdir=fdir,
                                   dirmap=dirmap,
                                   xytype=XYTYPE)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist, extent=grid.extent, cmap="cubehelix_r")
    fig.colorbar(im, ax=ax, label="Distance (cells)")
    ax.set_title("Flow Distance to Outlet")
    save_plot(fig, "05_flow_distance.png")

    print("✅ Watershed analysis complete.")
    print("Outputs saved in:", OUTPUT_FOLDER.resolve())


# Run directly
main()
