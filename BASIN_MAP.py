import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping, LineString, MultiLineString
from pyproj import Geod
from shapely.geometry import LineString, MultiLineString
from pyproj import Geod
from matplotlib.patches import Polygon

def plot_streams_cased(ax, gdf, color="#1f78b4", lw=1.4, casing_color="white", casing_lw=2.6, zorder=8):
    """
    Draw streams with a white casing underneath so they remain visible on DEM/hillshade.
    gdf must be in lon/lat (EPSG:4326) same as ax.
    """
    # casing first
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            x, y = geom.xy
            ax.plot(x, y, color=casing_color, linewidth=casing_lw, zorder=zorder-1, solid_capstyle="round")
            ax.plot(x, y, color=color, linewidth=lw, zorder=zorder, solid_capstyle="round")
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if part is None or part.is_empty:
                    continue
                x, y = part.xy
                ax.plot(x, y, color=casing_color, linewidth=casing_lw, zorder=zorder-1, solid_capstyle="round")
                ax.plot(x, y, color=color, linewidth=lw, zorder=zorder, solid_capstyle="round")


def add_scalebar_segments_lonlat(
    ax,
    length_km=10,
    n_segments=4,
    location=(0.08, 0.06),  # axes fraction
    height_frac=0.012,
    text_offset_frac=0.018,
    font_size=10
):
    """
    Beautiful alternating black/white scalebar for a lon/lat map (geodesic length).
    Drawn in axis data coordinates using WGS84 geodesics.
    """
    geod = Geod(ellps="WGS84")

    # current extent
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # anchor in data coords using axes fraction
    x0 = xmin + location[0] * (xmax - xmin)
    y0 = ymin + location[1] * (ymax - ymin)

    # choose a latitude for geodesic calculation
    lat = y0
    lon0 = x0

    # compute end point for full length
    lon1, lat1, _ = geod.fwd(lon0, lat, 90, length_km * 1000)

    # if scalebar too long for map, auto-shrink
    if lon1 > xmax:
        length_km = max(2, int(length_km * 0.5))
        lon1, lat1, _ = geod.fwd(lon0, lat, 90, length_km * 1000)

    # segment length in meters
    seg_m = (length_km * 1000) / n_segments

    # bar height in lat degrees approx from axis range
    h = height_frac * (ymax - ymin)

    # draw segments
    cur_lon, cur_lat = lon0, lat
    for i in range(n_segments):
        next_lon, next_lat, _ = geod.fwd(cur_lon, cur_lat, 90, seg_m)
        face = "black" if i % 2 == 0 else "white"

        rect = plt.Rectangle(
            (cur_lon, cur_lat),
            next_lon - cur_lon,
            h,
            facecolor=face,
            edgecolor="black",
            linewidth=1.0,
            zorder=20
        )
        ax.add_patch(rect)
        cur_lon, cur_lat = next_lon, next_lat

    # outline box (crisp border)
    outline = plt.Rectangle(
        (lon0, lat),
        lon1 - lon0,
        h,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        zorder=21
    )
    ax.add_patch(outline)

    # labels: 0, mid, end
    ax.text(lon0, lat + h + text_offset_frac*(ymax-ymin), "0",
            ha="center", va="bottom", fontsize=font_size,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
            zorder=22)

    mid_lon, mid_lat, _ = geod.fwd(lon0, lat, 90, (length_km*1000)/2)
    ax.text(mid_lon, lat + h + text_offset_frac*(ymax-ymin), f"{length_km/2:g}",
            ha="center", va="bottom", fontsize=font_size,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
            zorder=22)

    ax.text(lon1, lat + h + text_offset_frac*(ymax-ymin), f"{length_km:g} km",
            ha="center", va="bottom", fontsize=font_size, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
            zorder=22)

import matplotlib.pyplot as plt
from pyproj import Geod
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def add_north_arrow_png(ax, path, zoom=0.10, xy=(0.90, 0.88)):
    img = mpimg.imread(path)

    # If JPG -> convert to RGBA with transparency by removing near-white
    if img.ndim == 3 and img.shape[2] == 3:
        rgb = img.astype(np.float32) / 255.0 if img.max() > 1.0 else img
        alpha = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        # treat near-white as transparent
        mask = (rgb[..., 0] > 0.92) & (rgb[..., 1] > 0.92) & (rgb[..., 2] > 0.92)
        alpha[mask] = 0.0

        img = np.dstack([rgb, alpha])

    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, xycoords='axes fraction', frameon=False)
    ax.add_artist(ab)

from matplotlib.patches import Polygon

def add_north_arrow_vector(ax, location=(0.90, 0.88), size=0.07, text_offset=0.02):
    x, y = location

    tip   = (x, y)
    left  = (x - size/2, y - size)
    right = (x + size/2, y - size)
    inner = (x, y - size*0.62)

    tri_black = Polygon([tip, right, inner], closed=True,
                        transform=ax.transAxes,
                        facecolor="black", edgecolor="black",
                        linewidth=1.2, zorder=300)

    tri_white = Polygon([tip, left, inner], closed=True,
                        transform=ax.transAxes,
                        facecolor="white", edgecolor="black",
                        linewidth=1.2, zorder=300)

    ax.add_patch(tri_black)
    ax.add_patch(tri_white)

    ax.text(x, y + text_offset, "N",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            zorder=301)
def plot_streams_by_order_cased(ax, streams_ll, order_field,
                               lw=1.8, casing_lw=3.2,
                               zorder=200):
    """
    Plot streams colored by order with white casing beneath.
    Works in EPSG:4326.
    """
    orders = np.sort(streams_ll[order_field].dropna().unique())
    cmap = plt.get_cmap("tab10", len(orders))

    for i, o in enumerate(orders):
        subset = streams_ll.loc[streams_ll[order_field] == o]
        if subset.empty:
            continue

        # casing first
        plot_streams_cased(
            ax, subset,
            color=cmap(i),
            lw=lw,
            casing_color="white",
            casing_lw=casing_lw,
            zorder=zorder
        )

        # dummy handle for legend (so legend shows correct color)
        ax.plot([], [], color=cmap(i), lw=2.5, label=f"Order {int(o)}")

def add_scalebar_compact(ax,
                         length_km=5,
                         segments=2,
                         location=(0.10, 0.06),
                         height_frac=0.008,
                         font_size=9):

    geod = Geod(ellps="WGS84")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    lon0 = xmin + location[0] * (xmax - xmin)
    lat0 = ymin + location[1] * (ymax - ymin)

    lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000)

    h = height_frac * (ymax - ymin)
    seg_len = (length_km * 1000) / segments

    cur_lon, cur_lat = lon0, lat0

    for i in range(segments):
        next_lon, next_lat, _ = geod.fwd(cur_lon, cur_lat, 90, seg_len)

        color = "black" if i % 2 == 0 else "white"

        rect = plt.Rectangle(
            (cur_lon, cur_lat),
            next_lon - cur_lon,
            h,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=50
        )

        ax.add_patch(rect)

        cur_lon, cur_lat = next_lon, next_lat

    # outline
    outline = plt.Rectangle(
        (lon0, lat0),
        lon1 - lon0,
        h,
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        zorder=51
    )

    ax.add_patch(outline)

    # labels
    ax.text(lon0, lat0 + 2*h, "0",
            fontsize=font_size,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.7, pad=0.2))

    ax.text(lon1, lat0 + 2*h, f"{length_km} km",
            fontsize=font_size,
            ha="center",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, pad=0.2))


# -----------------------------
# PATHS (edit)
# -----------------------------
OUT_DIR = r"E:/QUALITY_1/outputs_pysheds"
os.makedirs(OUT_DIR, exist_ok=True)

BASIN_SHP   = f"{OUT_DIR}/basin_f15.shp"
STREAMS_SHP = f"{OUT_DIR}/streams_f15.shp"
INDIA_DEM_TIF = r"E:/Ashwin/DEM/AST14DEM_00403102025044241_20251211132210.tif"

OUT_PNG = os.path.join(OUT_DIR, "Basin_DEM_Streams.png")

# -----------------------------
# SETTINGS
# -----------------------------
DEM_CMAP = "terrain"
STREAM_LW = 1.0
BASIN_LW  = 2.4

# If your streams have an order field, list candidates here:
STREAM_ORDER_FIELD_CANDIDATES = ["order", "Order", "strahler", "Strahler", "streamord", "StreamOrd"]

# Grid density (lon/lat ticks)
N_TICKS = 5

# Scale bar length (km)
SCALEBAR_KM = 5

# -----------------------------
# HELPERS
# -----------------------------
def hillshade(arr, azdeg=315, altdeg=45):
    a = np.array(arr, dtype=float)
    nanmask = ~np.isfinite(a)
    if np.any(nanmask):
        finite = a[np.isfinite(a)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        a[nanmask] = fill

    x, y = np.gradient(a)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    az = np.deg2rad(azdeg)
    alt = np.deg2rad(altdeg)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min() + 1e-9)
    shaded[nanmask] = np.nan
    return shaded

def add_north_arrow(ax, x=0.92, y=0.88, size=0.10):
    ax.annotate(
        'N', xy=(x, y), xytext=(x, y - size),
        xycoords='axes fraction', textcoords='axes fraction',
        ha='center', va='center', fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='-|>', lw=1.4)
    )

def plot_lines_matplotlib(ax, gdf, color, lw, zorder, label=None):
    first = True
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            x, y = geom.xy
            ax.plot(x, y, color=color, linewidth=lw, zorder=zorder,
                    label=(label if first else None))
            first = False
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if part is None or part.is_empty:
                    continue
                x, y = part.xy
                ax.plot(x, y, color=color, linewidth=lw, zorder=zorder,
                        label=(label if first else None))
                first = False

def add_scalebar_lonlat(ax, extent_ll, length_km=20):
    """Draw geodesic scalebar on a lon/lat axis."""
    minx, maxx, miny, maxy = extent_ll
    geod = Geod(ellps="WGS84")

    lat = miny + 0.06 * (maxy - miny)
    lon0 = minx + 0.08 * (maxx - minx)

    lon1, lat1, _ = geod.fwd(lon0, lat, 90, length_km * 1000)

    # if too long, reduce
    if lon1 > maxx:
        length_km = max(5, int(length_km * 0.5))
        lon1, lat1, _ = geod.fwd(lon0, lat, 90, length_km * 1000)

    ax.plot([lon0, lon1], [lat, lat1], color="k", lw=4, zorder=20)
    tick_h = 0.01 * (maxy - miny)
    ax.plot([lon0, lon0], [lat - tick_h, lat + tick_h], color="k", lw=2, zorder=20)
    ax.plot([lon1, lon1], [lat1 - tick_h, lat1 + tick_h], color="k", lw=2, zorder=20)
    ax.text((lon0 + lon1)/2, lat + 2.2*tick_h, f"{length_km} km",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.6),
            zorder=21)

def clip_dem_to_basin_reproject_4326(dem_path, basin_gdf):
    """Clip DEM to basin and reproject result to EPSG:4326 (for clean lon/lat plotting)."""
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        basin_dem = basin_gdf.to_crs(dem_crs)
        basin_union = basin_dem.geometry.union_all()

        out, out_transform = mask(src, [mapping(basin_union)], crop=True, filled=True)
        arr = out[0].astype("float32")

        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        arr[arr <= -9999] = np.nan

        # Reproject clipped raster to EPSG:4326
        dst_crs = "EPSG:4326"
        bounds = rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], out_transform)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            dem_crs, dst_crs, arr.shape[1], arr.shape[0], *bounds
        )

        dst = np.full((dst_h, dst_w), np.nan, dtype="float32")

        reproject(
            source=arr,
            destination=dst,
            src_transform=out_transform,
            src_crs=dem_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

        xmin = dst_transform.c
        ymax = dst_transform.f
        xmax = xmin + dst_w * dst_transform.a
        ymin = ymax + dst_h * dst_transform.e
        extent_ll = (xmin, xmax, ymin, ymax)

    return dst, extent_ll

# -----------------------------
# READ DATA
# -----------------------------
basin = gpd.read_file(BASIN_SHP)
streams = gpd.read_file(STREAMS_SHP)

basin["geometry"] = basin.geometry.buffer(0)
# Fix only invalid lines safely (do NOT buffer(0) for lines)
streams = streams[~streams.geometry.is_empty].copy()  # keep non-empty if any
streams = streams[streams.geometry.notna()].copy()

# Convert vectors to EPSG:4326 for plotting
basin_ll = basin.to_crs(4326)
streams_ll = streams.to_crs(4326)

# Clip DEM to basin and reproject to EPSG:4326
dem_ll, dem_extent_ll = clip_dem_to_basin_reproject_4326(INDIA_DEM_TIF, basin_ll)
hs = hillshade(dem_ll)

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 9))

# DEM + hillshade
im = ax.imshow(dem_ll, extent=dem_extent_ll, origin="upper", cmap=DEM_CMAP, alpha=0.95, zorder=1)
ax.imshow(hs, extent=dem_extent_ll, origin="upper", cmap="gray", alpha=0.22, zorder=2)

# Basin boundary
basin_ll.boundary.plot(ax=ax, color="black", linewidth=BASIN_LW, zorder=10)

# Streams (ONLY ONCE, clearly visible)
# Detect stream order field
order_field = None
for c in STREAM_ORDER_FIELD_CANDIDATES:
    if c in streams_ll.columns:
        order_field = c
        break

if order_field is None:
    # single-color streams
    plot_streams_cased(
        ax, streams_ll,
        color="#0072B2",
        lw=2.0,
        casing_color="white",
        casing_lw=3.6,
        zorder=200
    )
else:
    plot_streams_by_order_cased(
        ax, streams_ll, order_field,
        lw=1.8, casing_lw=3.2, zorder=200
    )
    ax.legend(title="Stream Order", loc="upper left",
              fontsize=9, title_fontsize=10, frameon=True)


# Extent padded
bminx, bminy, bmaxx, bmaxy = basin_ll.total_bounds
padx = 0.01 * (bmaxx - bminx)
pady = 0.01 * (bmaxy - bminy)
ax.set_xlim(bminx - padx, bmaxx + padx)
ax.set_ylim(bminy - pady, bmaxy + pady)

# Title
ax.set_title("Basin DEM with Stream Network", fontsize=16, fontweight="bold", pad=10)

# North arrow (PNG only)

# Grid + lon/lat ticks
xt = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], N_TICKS)
yt = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], N_TICKS)
ax.set_xticks(xt)
ax.set_yticks(yt)
ax.set_xticklabels([f"{x:.2f}°E" for x in xt], fontsize=10)
ax.set_yticklabels([f"{y:.2f}°N" for y in yt], fontsize=10)
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

# Beautiful compact scalebar
add_scalebar_compact(
    ax,
    length_km=5,
    segments=2,
    location=(0.08, 0.045),
    height_frac=0.006,
    font_size=9
)

# Colorbar
cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Elevation (m)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", OUT_PNG)
