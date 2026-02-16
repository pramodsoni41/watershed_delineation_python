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

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pyproj import CRS, Transformer
# --- ADD THESE IMPORTS AT TOP (after existing imports) ---
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon
from shapely.ops import unary_union
import rasterio.features

# --- ADD THESE USER SETTINGS (near USER SETTINGS) ---
OUT_DIR = r"E:/QUALITY_1/outputs_pysheds"
BASIN_SHP = f"{OUT_DIR}/basin_f4.shp"
STREAMS_SHP = f"{OUT_DIR}/streams_f4.shp"

# "Fine" streams: lower percentile => denser network (try 90–97)
FINE_STREAM_PERCENTILE = 93

# Optional: keep only streams inside basin in output shapefile
CLIP_STREAMS_TO_BASIN = True
# -----------------------------
# USER SETTINGS
# -----------------------------
DEM_PATH = r"E:/QUALITY_1/Terrain/Terrain.ASTGTMV003_N26E083_dem.tif"
def ensure_outdir(path):
    import os
    os.makedirs(path, exist_ok=True)
import numpy as np
from collections import deque

# pysheds D8 convention: [N, NE, E, SE, S, SW, W, NW]
# default dirmap often is: (64, 128, 1, 2, 4, 8, 16, 32)
DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def strahler_order_d8(fdir, stream_mask, dirmap):
    """
    Compute Strahler stream order on a D8 stream network raster.
    fdir: 2D array of flow directions (D8 codes)
    stream_mask: 2D boolean array where True = stream cell
    dirmap: tuple/list of 8 D8 codes in order [N,NE,E,SE,S,SW,W,NW]
    returns: order raster (uint16), 0 where not stream
    """
    nrows, ncols = fdir.shape
    stream = stream_mask.astype(bool)

    # Maps direction code -> (dr, dc)
    code2delta = {int(code): DIRS[i] for i, code in enumerate(dirmap)}

    # We'll do a topological pass from upstream -> downstream using indegree (upstream count)
    indeg = np.zeros_like(fdir, dtype=np.int32)      # number of upstream stream-neighbors
    down_r = np.full_like(fdir, -1, dtype=np.int32)  # downstream row for each stream cell
    down_c = np.full_like(fdir, -1, dtype=np.int32)  # downstream col for each stream cell

    # Precompute downstream link and indegree
    rr, cc = np.where(stream)
    for r, c in zip(rr, cc):
        code = int(fdir[r, c])
        if code not in code2delta:
            continue
        dr, dc = code2delta[code]
        r2, c2 = r + dr, c + dc
        if 0 <= r2 < nrows and 0 <= c2 < ncols and stream[r2, c2]:
            down_r[r, c] = r2
            down_c[r, c] = c2
            indeg[r2, c2] += 1

    # Strahler DP state on each cell:
    # max_up = maximum order among upstream tributaries seen so far
    # nmax   = how many upstream tributaries achieved that max
    max_up = np.zeros_like(fdir, dtype=np.uint16)
    nmax   = np.zeros_like(fdir, dtype=np.uint16)
    order  = np.zeros_like(fdir, dtype=np.uint16)

    # Initialize queue with sources (no upstream stream-neighbors)
    q = deque()
    for r, c in zip(rr, cc):
        if indeg[r, c] == 0:
            order[r, c] = 1
            q.append((r, c))

    # Process upstream->downstream
    while q:
        r, c = q.popleft()
        r2, c2 = down_r[r, c], down_c[r, c]
        if r2 < 0:
            continue  # outlet or leaving stream mask

        o = order[r, c]

        # update downstream DP state
        if o > max_up[r2, c2]:
            max_up[r2, c2] = o
            nmax[r2, c2] = 1
        elif o == max_up[r2, c2]:
            nmax[r2, c2] += 1

        # reduce indegree; when all upstream processed, finalize downstream order
        indeg[r2, c2] -= 1
        if indeg[r2, c2] == 0:
            mo = max_up[r2, c2]
            order[r2, c2] = mo + 1 if nmax[r2, c2] >= 2 else mo
            q.append((r2, c2))

    # non-stream cells remain 0
    order[~stream] = 0
    return order

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

# Outlet input mode:
#   "latlon"  -> use OUTLET_LAT/OUTLET_LON below
#   "click"   -> click on accumulation plot; it prints lat/lon and uses that
OUTLET_MODE = "latlon"   # "latlon" or "click"

OUTLET_LAT = 26.374909
OUTLET_LON = 83.617662

# Stream mask method:
#   "percentile" -> uses STREAM_PERCENTILE of accumulation
#   "area_km2"   -> uses STREAM_AREA_KM2 converted to cells (works best for projected CRS)
STREAM_MASK_METHOD = "percentile"   # "percentile" or "area_km2"
STREAM_PERCENTILE = 98              # 95 (more streams) ... 99 (fewer streams)
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
dem2 = dem.astype(float)
dem2[dem2 < 0] = np.nan
# ---- Plot DEM (pretty) ----
fig, ax = plt.subplots(figsize=FIGSIZE)
im = ax.imshow(dem2, extent=grid.extent, cmap="terrain", interpolation=INTERP)
plt.colorbar(im, ax=ax, label="Elevation (m)")
pretty_axes(ax, "Digital Elevation Model")
set_latlon_ticks(ax, grid)
plt.tight_layout()
plt.show()
plt.savefig(f"{OUT_DIR}/DEM.png", dpi=150)

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
plt.savefig(f"{OUT_DIR}/FDIR.png", dpi=150)

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
plt.savefig(f"{OUT_DIR}/FACC.png", dpi=150)
#%%
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
#%%
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
#%%

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



# add a representative accumulation value to each line by sampling points along it
import numpy as np
strahler = strahler_order_d8(fdir, fine_stream_mask, dirmap)

streams_list = []

for k in range(1, int(strahler.max()) + 1):
    mask_k = (strahler == k)
    if mask_k.sum() == 0:
        continue

    branches_k = grid.extract_river_network(fdir, mask_k, dirmap=dirmap)
    gdf_k = branches_to_gdf(branches_k, crs_wkt=str(grid.crs))
    gdf_k["order"] = k
    streams_list.append(gdf_k)

streams_gdf = gpd.GeoDataFrame(pd.concat(streams_list, ignore_index=True), crs=str(grid.crs))

print("Unique Strahler:", np.unique(strahler[strahler > 0]))

def densify_xy(geom, n=25):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiLineString":
        geom = max(list(geom.geoms), key=lambda g: g.length)
    if geom.length == 0:
        x, y = list(geom.coords)[0]
        return [(x, y)]
    pts = [geom.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
    return [(p.x, p.y) for p in pts]


def sample_order_max(line_geom, grid, order_raster, n=25):
    pts = densify_xy(line_geom, n=n)
    nrows, ncols = order_raster.shape
    best = 0

    for x, y in pts:
        try:
            r, c = grid.nearest_cell(x, y)
        except TypeError:
            r, c = grid.nearest_cell((x, y))

        r = int(np.clip(r, 0, nrows - 1))
        c = int(np.clip(c, 0, ncols - 1))

        v = int(order_raster[r, c])
        if v > best:
            best = v

    return best




if CLIP_STREAMS_TO_BASIN:
    # Ensure CRS matches
    streams_gdf = streams_gdf.to_crs(basin_gdf.crs)
    
    # Clip
    streams_gdf = gpd.clip(streams_gdf, basin_gdf)
    
    # Clean geometry
    streams_gdf = streams_gdf[~streams_gdf.is_empty]
    streams_gdf.reset_index(drop=True, inplace=True)
    
streams_gdf = streams_gdf.copy()
streams_gdf["order"] = [sample_order_max(g, grid, strahler, n=25) for g in streams_gdf.geometry]

print(streams_gdf["order"].value_counts().sort_index())
streams_gdf.to_file(STREAMS_SHP)
print("✅ Streams shapefile saved:", STREAMS_SHP)



#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def plot_basin_streams_pretty(
    ax,
    basin_gdf,
    streams_gdf,
    grid,
    inflated,
    INTERP,
    x0=None, y0=None, x_snap=None, y_snap=None,
    min_plot_order=4,
    title="Basin + Stream Network (Strahler order)",
    pad_frac=0.08,
    hillshade_alpha=0.55,
    basin_fill_alpha=0.18,
    basin_fill_color="lightblue",
    basin_edge_color="black",
    basin_edge_lw=2.2,
    cmap_name="tab10",      # distinct colors
    simplify_tol=None       # e.g., 5 or 10 (meters) to speed up
):
    # ---- Center/zoom to basin ----
    basin1 = basin_gdf.dissolve().geometry.iloc[0]
    minx, miny, maxx, maxy = basin1.bounds
    dx, dy = (maxx - minx), (maxy - miny)
    ax.set_xlim(minx - dx*pad_frac, maxx + dx*pad_frac)
    ax.set_ylim(miny - dy*pad_frac, maxy + dy*pad_frac)

    # ---- Hillshade ----
    gy, gx = np.gradient(np.nan_to_num(inflated, nan=np.nanmean(inflated)))
    slope = np.hypot(gx, gy)
    shade = 1 - (slope / (np.nanmax(slope) if np.nanmax(slope) > 0 else 1))
    ax.imshow(shade, extent=grid.extent, cmap="gray",
              alpha=hillshade_alpha, interpolation=INTERP, zorder=0)

    # ---- Basin ----
    basin_gdf.plot(ax=ax, alpha=basin_fill_alpha, color=basin_fill_color, zorder=2)
    basin_gdf.boundary.plot(ax=ax, linewidth=basin_edge_lw, color=basin_edge_color, zorder=8)

    # ---- Streams (filter + optional simplify) ----
    s = streams_gdf.copy()
    s = s[s["order"] >= min_plot_order].copy()

    # Fast bbox prefilter before drawing
    s = s.cx[minx:maxx, miny:maxy].copy()

    if simplify_tol is not None:
        s["geometry"] = s.geometry.simplify(simplify_tol, preserve_topology=True)

    if s.empty:
        ax.set_title(f"{title}\n(No streams with order ≥ {min_plot_order})")
        return

    orders = sorted(s["order"].unique())
    cmap = plt.cm.get_cmap(cmap_name, len(orders))

    def geom_to_segments(geom):
        if geom is None or geom.is_empty:
            return []
        gt = geom.geom_type
        if gt == "LineString":
            arr = np.asarray(geom.coords)
            return [arr] if arr.shape[0] >= 2 else []
        if gt == "MultiLineString":
            out = []
            for g in geom.geoms:
                if g is None or g.is_empty:
                    continue
                arr = np.asarray(g.coords)
                if arr.shape[0] >= 2:
                    out.append(arr)
            return out
        return []

    legend_handles = []

    # Draw low orders first, high orders last (on top)
    for i, o in enumerate(orders):
        sub = s[s["order"] == o]

        segs = []
        for geom in sub.geometry:
            segs.extend(geom_to_segments(geom))

        if not segs:
            continue

        # thickness scaling (higher order thicker)
        width = 0.9 + 0.9 * (o - orders[0])  # tweak if you want more/less contrast
        color = cmap(i)

        lc = LineCollection(
            segs,
            linewidths=width,
            colors=[color] * len(segs),
            capstyle="round",
            joinstyle="round",
            zorder=5 + (o - orders[0])
        )
        ax.add_collection(lc)

        legend_handles.append(Line2D([0], [0], color=color, lw=width, label=f"Order {o}"))

    # ---- Outlets ----
    if x0 is not None and y0 is not None:
        ax.plot(x0, y0, "r*", ms=12, zorder=10, label="Original outlet")
    if x_snap is not None and y_snap is not None:
        ax.plot(x_snap, y_snap, "ko", ms=6, zorder=10, label="Snapped outlet")

    # ---- Cosmetics ----
    try:
        pretty_axes(ax, title)
        set_latlon_ticks(ax, grid)
    except Exception:
        ax.set_title(title)

    # Legend: orders first, then outlets
    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles + h2,
        [h.get_label() for h in legend_handles] + l2,
        loc="lower left",
        frameon=True,
        fontsize=9
    )

# ----------------- USE IT -----------------
fig, ax = plt.subplots(figsize=(10, 8))

plot_basin_streams_pretty(
    ax=ax,
    basin_gdf=basin_gdf,
    streams_gdf=streams_gdf,   # must have "order"
    grid=grid,
    inflated=inflated,
    INTERP=INTERP,
    x0=x0, y0=y0, x_snap=x_snap, y_snap=y_snap,
    min_plot_order=4,          # try 5 for very clean map
    title="Basin + Stream Network (Strahler order)",
    cmap_name="tab10",         # distinct colors; try "Set2" or "viridis"
    simplify_tol=10            # meters (UTM). Set None to disable.
)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/BASIN_DEL_ORDERED.png", dpi=300, bbox_inches="tight")
plt.show()


#%%
import geopandas as gpd
from shapely.geometry import Point
import os

os.makedirs(OUT_DIR, exist_ok=True)

# Create shapely point (in DEM CRS)
outlet_point = Point(x_snap, y_snap)


orig_point = Point(x0, y0)

outlet_gdf = gpd.GeoDataFrame(
    {
        "type": ["original", "snapped"],
        "lat": [OUTLET_LAT, lat_snap],
        "lon": [OUTLET_LON, lon_snap]
    },
    geometry=[orig_point, outlet_point],
    crs=str(grid.crs)
)


OUTLET_SHP = f"{OUT_DIR}/outlet_point.shp"
outlet_gdf.to_file(OUTLET_SHP)

outlet_gdf_ll = outlet_gdf.to_crs("EPSG:4326")
outlet_gdf_ll.to_file(f"{OUT_DIR}/outlet_point_wgs84.shp")
