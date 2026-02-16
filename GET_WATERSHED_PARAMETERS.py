# -*- coding: utf-8 -*-
"""
Morphometric + Relief parameters from basin.shp + streams.shp (+ optional DEM)
- Loads basin polygon and stream lines
- Reprojects to a metric CRS automatically (UTM zone from basin centroid)
- Computes watershed geometric/morphometric indices including:
  A, P, Lb (geometric + Schumm), Rf, Bs, Re,
  Rc (Miller), Compactness (Gravelius), Compactness (Horton), Lemniscate ratio,
  Total stream length, Drainage density (Dd)
- Adds RELIEF metrics (requires DEM):
  Zmax, Zmin, Zmean, Zmedian, Zstddev,
  H = Zmax - Zmin (watershed relief),
  Rr = H/Lb, Rhp = H*100/P, Rn = H*Dd, Dis = H/Zmax

Inputs:
  basin.shp   (polygon, single basin)
  streams.shp (polyline network, preferably clipped to basin)
  dem.tif     (optional but needed for relief stats)

Outputs:
  Prints summary table and optionally saves CSV.

Dr. Pramod Soni (extended)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# USER INPUTS
# -------------------------
BASIN_SHP   = r"E:/QUALITY_1/outputs_pysheds/basin.shp"
STREAMS_SHP = r"E:/QUALITY_1/outputs_pysheds/streams_f7.shp"

# DEM is REQUIRED for relief metrics (Zmax, Zmin, etc.)
DEM_TIF     = r"E:/QUALITY_1/Terrain/Terrain.ASTGTMV003_N26E083_dem.tif"  # set None to skip relief

SAVE_CSV    = True
CSV_OUT     = r"E:/QUALITY_1/outputs_pysheds/morphometry_with_relief.csv"

COMPUTE_DRAINAGE_DENSITY = True   # Dd = total_stream_length / basin_area
ROUND_DIGITS = 4

# -------------------------
# HELPERS
# -------------------------
def pick_utm_epsg_from_lonlat(lon, lat):
    zone = int(np.floor((lon + 180) / 6) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)

def to_metric_crs(gdf_basin, gdf_streams):
    basin_ll = gdf_basin.to_crs("EPSG:4326")
    c = basin_ll.geometry.iloc[0].centroid
    lon, lat = float(c.x), float(c.y)

    utm_epsg = pick_utm_epsg_from_lonlat(lon, lat)
    basin_m   = gdf_basin.to_crs(f"EPSG:{utm_epsg}")
    streams_m = gdf_streams.to_crs(f"EPSG:{utm_epsg}")
    return basin_m, streams_m, utm_epsg
def drainage_network_analysis(streams_m, order_field="order"):
    """
    Drainage network analysis (Horton/Schumm) from stream GeoDataFrame with Strahler order.

    Inputs:
      streams_m: GeoDataFrame in metric CRS (meters)
      order_field: column name containing Strahler order (int)

    Returns:
      dict with:
        - per-order table (DataFrame): order, Nu, Lu_km
        - summary scalars: Rbm_mean, Rbm_list, Lur_mean, Lur_list, rho
        - Horton regression: b, RbmH (optional; NaN if not enough orders)
        - weighted metrics: Lur_r, total_weighted_stream_length, Luwm, Rbwm
    Notes:
      - Nu = number of stream segments per order (depends on how your network is segmented).
      - Lu = total length (km) per order.
    """
    import numpy as np
    import pandas as pd

    if order_field not in streams_m.columns:
        raise ValueError(f"'{order_field}' not found in streams GeoDataFrame. Compute Strahler order first.")

    g = streams_m.dropna(subset=[order_field]).copy()
    g = g[g.geometry.notnull() & (~g.geometry.is_empty)].copy()
    g[order_field] = g[order_field].astype(int)

    # length in km
    g["_len_km"] = g.length / 1000.0

    # --- Per order Nu and Lu ---
    tab = (
        g.groupby(order_field)
         .agg(Nu=("geometry", "count"), Lu_km=("_len_km", "sum"))
         .reset_index()
         .rename(columns={order_field: "order"})
         .sort_values("order")
         .reset_index(drop=True)
    )

    orders = tab["order"].values
    Nu = tab["Nu"].values.astype(float)
    Lu = tab["Lu_km"].values.astype(float)

    # --- Mean bifurcation ratio (Schumm): average of Nu_i / Nu_{i+1} ---
    Rbm_list = []
    for i in range(len(tab) - 1):
        if Nu[i+1] > 0:
            Rbm_list.append(Nu[i] / Nu[i+1])
    Rbm_mean = float(np.nanmean(Rbm_list)) if len(Rbm_list) > 0 else np.nan

    # --- Horton "b" from log10(Nu) vs order; RbmH = antilog(b) ---
    # log10(Nu) = a - b*order  (commonly negative slope if using +order)
    # We'll fit log10(Nu) = m*order + c, then RbmH = 10^(-m) if m is negative.
    b = np.nan
    RbmH = np.nan
    if len(tab) >= 3 and np.all(Nu > 0):
        x = orders.astype(float)
        y = np.log10(Nu)
        m, c = np.polyfit(x, y, 1)  # y = m x + c
        # In many texts, b is the absolute slope magnitude.
        b = float(-m)               # positive
        RbmH = float(10 ** b)

    # --- Stream length ratio (Horton): average of (Lu_{i+1}/Nu_{i+1}) / (Lu_i/Nu_i) ---
    # Mean stream length per order: Lbar_u = Lu/Nu
    Lbar = np.where(Nu > 0, Lu / Nu, np.nan)

    Lur_list = []
    for i in range(len(tab) - 1):
        if np.isfinite(Lbar[i]) and np.isfinite(Lbar[i+1]) and Lbar[i] > 0:
            Lur_list.append(Lbar[i+1] / Lbar[i])
    Lur_mean = float(np.nanmean(Lur_list)) if len(Lur_list) > 0 else np.nan

    # --- Stream length used in ratio (Lur-r) = Lur_{u+1} + Lur_u (table notation) ---
    # We'll define Lur-r per step i as Lbar_{i+1} + Lbar_i
    Lur_r = []
    for i in range(len(tab) - 1):
        if np.isfinite(Lbar[i]) and np.isfinite(Lbar[i+1]):
            Lur_r.append(Lbar[i+1] + Lbar[i])
    Lur_r = np.array(Lur_r, dtype=float)
    sum_Lur_r = float(np.nansum(Lur_r)) if len(Lur_r) else np.nan

    # --- Total Weighted Stream Length = (Lur_mean * sum(Lur-r)) (as in your table line) ---
    total_weighted_stream_length = (Lur_mean * sum_Lur_r) if (np.isfinite(Lur_mean) and np.isfinite(sum_Lur_r)) else np.nan

    # --- Weighted Mean Stream Length Ratio (Luwm) = (Lur_mean * sum(Lur-r)) / sum(Lur-r) = Lur_mean
    # But some notes interpret differently; we’ll keep it explicit:
    Luwm = safe_div(total_weighted_stream_length, sum_Lur_r)

    # --- Weighted Mean Bifurcation Ratio (Rbwm) = Σ(Rbm_i * Lur-r_i) / Σ(Lur-r_i) ---
    # Need Rbm per step and matching Lur-r per step
    Rbwm = np.nan
    if len(Rbm_list) == len(Lur_r) and len(Lur_r) > 0:
        num = np.nansum(np.array(Rbm_list) * Lur_r)
        den = np.nansum(Lur_r)
        Rbwm = safe_div(num, den)

    # --- Rho coefficient (Horton): Rho = Lur / Rbm ---
    rho = safe_div(Lur_mean, Rbm_mean)

    # Add intermediate columns to tab for convenience
    tab["Lbar_km"] = Lbar

    return {
        "per_order_table": tab,
        "Rbm_list": Rbm_list,
        "Rbm_mean": Rbm_mean,
        "b_horton": b,
        "RbmH": RbmH,
        "Lur_list": Lur_list,
        "Lur_mean": Lur_mean,
        "Lur_r": Lur_r,
        "sum_Lur_r": sum_Lur_r,
        "total_weighted_stream_length": total_weighted_stream_length,
        "Luwm": Luwm,
        "Rbwm": Rbwm,
        "rho": rho,
    }

def ensure_single_polygon(geom):
    if geom is None or geom.is_empty:
        raise ValueError("Basin geometry is empty.")
    g = geom
    if not g.is_valid:
        g = g.buffer(0)
    if isinstance(g, (MultiPolygon,)):
        g = g.unary_union
    if not g.is_valid:
        g = g.buffer(0)
    return g

def watershed_length_km(basin_poly):
    mrr = basin_poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]
    d01 = np.hypot(coords[1][0]-coords[0][0], coords[1][1]-coords[0][1])
    d12 = np.hypot(coords[2][0]-coords[1][0], coords[2][1]-coords[1][1])
    return max(d01, d12) / 1000.0

def safe_div(a, b):
    return (a / b) if (b is not None and b != 0 and np.isfinite(b)) else np.nan

def relief_stats_from_dem(dem_path, basin_geom_ll):
    """
    Compute Z statistics inside basin using raster mask.
    basin_geom_ll must be in EPSG:4326 or same CRS as DEM (we will reproject).
    """
    import rasterio
    from rasterio.mask import mask as rio_mask

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

        # Reproject basin to DEM CRS
        basin_in_dem = basin_geom_ll.to_crs(dem_crs)

        geoms = [basin_in_dem.geometry.iloc[0]]
        out, out_transform = rio_mask(src, geoms, crop=True, filled=True)

        arr = out[0].astype(float)

        # Handle NoData
        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan

        # Extra safety for ASTER/odd tiles
        arr[arr <= -9999] = np.nan
        arr[arr > 1e5] = np.nan

        if np.all(~np.isfinite(arr)):
            raise ValueError("All DEM values became NaN after masking. Check DEM nodata and basin extent.")

        Zmax = float(np.nanmax(arr))
        Zmin = float(np.nanmin(arr))
        Zmean = float(np.nanmean(arr))
        Zmedian = float(np.nanmedian(arr))
        Zstd = float(np.nanstd(arr))

        return Zmax, Zmin, Zmean, Zmedian, Zstd
def drainage_texture_metrics(streams_m, A_km2, P_km, total_stream_length_km=None):
    """
    Horton (1945) drainage texture metrics.
    Inputs:
      streams_m: streams GeoDataFrame in metric CRS
      A_km2: basin area (km^2)
      P_km: basin perimeter (km)
      total_stream_length_km: optional; if None computed from streams_m
    Returns dict with: ΣLu, ΣNu, Fs, Dt, Dd, Di, If, Lo, C
    """
    # total length (ΣLu)
    if total_stream_length_km is None:
        total_stream_length_km = float(streams_m.length.sum()) / 1000.0

    # total number of streams (ΣNu) = number of polyline features
    # (If your streams are split into many tiny segments, this will be large.)
    Nu = int(len(streams_m))

    # Drainage density (Dd = ΣLu / A)
    Dd = safe_div(total_stream_length_km, A_km2)

    # Stream frequency (Fs = ΣNu / A)
    Fs = safe_div(Nu, A_km2)

    # Drainage texture (Dt = ΣNu / P)
    Dt = safe_div(Nu, P_km)

    # Drainage intensity (Di = Fs / Dd)
    Di = safe_div(Fs, Dd)

    # Infiltration number (If = Fs * Dd)
    If = (Fs * Dd) if (np.isfinite(Fs) and np.isfinite(Dd)) else np.nan

    # Length of overland flow (Lo = 1 / (2 * Dd))
    Lo = safe_div(1.0, (2.0 * Dd))

    # Constant of channel maintenance (C = 1 / Dd)
    C = safe_div(1.0, Dd)

    return {
        "Total length of all streams (ΣLu)": total_stream_length_km,
        "Total number of streams (ΣNu)": Nu,
        "Stream frequency (Fs = ΣNu/A)": Fs,
        "Drainage texture (Dt = ΣNu/P)": Dt,
        "Drainage density (Dd = ΣLu/A)": Dd,
        "Drainage intensity (Di = Fs/Dd)": Di,
        "Infiltration number (If = Fs*Dd)": If,
        "Length of overland flow (Lo = 1/(2Dd))": Lo,
        "Constant of channel maintenance (C = 1/Dd)": C,
    }

# -------------------------
# LOAD DATA
# -------------------------
basin = gpd.read_file(BASIN_SHP).dissolve().reset_index(drop=True)
streams = gpd.read_file(STREAMS_SHP)
streams = streams[streams.geometry.notnull() & (~streams.geometry.is_empty)].copy()

# -------------------------
# PROJECT TO METRIC CRS
# -------------------------
basin_m, streams_m, utm_epsg = to_metric_crs(basin, streams)
basin_poly = ensure_single_polygon(basin_m.geometry.iloc[0])
net = drainage_network_analysis(streams_m, order_field="order")

print("\nPer-order drainage table:")
print(net["per_order_table"].to_string(index=False))

print("\nNetwork summary:")
print("Mean bifurcation ratio (Rbm):", net["Rbm_mean"])
print("Horton slope b:", net["b_horton"])
print("Mean bifurcation ratio from Horton diagram (RbmH):", net["RbmH"])
print("Mean stream length ratio (Lur):", net["Lur_mean"])
print("Weighted mean bifurcation ratio (Rbwm):", net["Rbwm"])
print("Rho coefficient:", net["rho"])
# -------------------------
# BASIC GEOMETRY
# -------------------------
A_km2 = basin_poly.area / 1e6
P_km  = basin_poly.length / 1000.0
total_stream_length_km = float(streams_m.length.sum()) / 1000.0  # ΣLu

dr_tex = drainage_texture_metrics(
    streams_m=streams_m,
    A_km2=A_km2,
    P_km=P_km,
    total_stream_length_km=total_stream_length_km
)

# keep Dd for later relief metrics
Dd = dr_tex["Drainage density (Dd = ΣLu/A)"]

Lb_km = watershed_length_km(basin_poly)

Lb_schumm_km = 1.312 * (A_km2 ** 0.568) if A_km2 > 0 else np.nan

# -------------------------
# MORPHOMETRIC INDICES
# -------------------------
Rf = safe_div(A_km2, (Lb_km ** 2))
Bs = safe_div((Lb_km ** 2), A_km2)
Re = safe_div((2.0 * np.sqrt(A_km2 / np.pi)), Lb_km)

Rc = safe_div((4.0 * np.pi * A_km2), (P_km ** 2))
Cc_gravelius = safe_div(P_km, (2.0 * np.sqrt(np.pi * A_km2)))
Cc_horton = safe_div((0.2821 * P_km), np.sqrt(A_km2))
k_lemniscate = safe_div((np.pi * (Lb_km ** 2)), (4.0 * A_km2))

# -------------------------
# STREAM METRICS
# -------------------------
total_stream_length_km = float(streams_m.length.sum()) / 1000.0
Dd = safe_div(total_stream_length_km, A_km2) if COMPUTE_DRAINAGE_DENSITY else np.nan

# -------------------------
# RELIEF METRICS (needs DEM)
# -------------------------
Zmax = Zmin = Zmean = Zmedian = Zstd = np.nan
H_m = Rr = Rhp = Rn = Dis = np.nan

if DEM_TIF:
    # Basin geometry to GeoDataFrame for reprojection convenience (start from basin layer)
    basin_ll = basin.to_crs("EPSG:4326")

    Zmax, Zmin, Zmean, Zmedian, Zstd = relief_stats_from_dem(DEM_TIF, basin_ll)
    H_m = Zmax - Zmin                                  # watershed relief (m)
    Rr  = safe_div(H_m, (Lb_km * 1000.0))              # relief ratio (H/Lb in m/m)
    Rhp = safe_div(H_m * 100.0, (P_km * 1000.0))       # relative relief (H*100/P), P in m
    Rn  = H_m * Dd if np.isfinite(Dd) else np.nan      # ruggedness number (H * Dd)
    Dis = safe_div(H_m, Zmax)                          # dissection index (H/Zmax)

# -------------------------
# OUTPUT TABLE
# -------------------------
results = [
    ("Area of Watershed (A)", A_km2, "km²"),
    ("Perimeter of Watershed (P)", P_km, "km"),
    ("Watershed Length (Lb) – geometric", Lb_km, "km"),
    ("Watershed Length (Lb) – Schumm (1956)", Lb_schumm_km, "km"),
    ("Form Factor (Rf = A/Lb²)", Rf, "-"),
    ("Shape Factor / Basin Shape (Bs = Lb²/A)", Bs, "-"),
    ("Elongation Ratio (Re)", Re, "-"),
    ("Circularity Ratio (Rc = 4πA/P²)", Rc, "-"),
    ("Compactness Coefficient (Gravelius)", Cc_gravelius, "-"),
    ("Compactness Constant (Horton)", Cc_horton, "-"),
    ("Lemniscate Ratio (k = πLb²/4A)", k_lemniscate, "-"),
    ("Total Stream Length", total_stream_length_km, "km"),
]

if COMPUTE_DRAINAGE_DENSITY:
    results.append(("Drainage Density (Dd = ΣL/A)", Dd, "km/km²"))

# Relief block
if DEM_TIF:
    results += [
        ("Maximum Elevation (Zmax)", Zmax, "m"),
        ("Minimum Elevation (Zmin)", Zmin, "m"),
        ("Mean Elevation (Zmean)", Zmean, "m"),
        ("Median Elevation (Zmedian)", Zmedian, "m"),
        ("Std Dev of Elevation (Zstddev)", Zstd, "m"),
        ("Watershed Relief (H = Zmax - Zmin)", H_m, "m"),
        ("Relief Ratio (Rr = H/Lb)", Rr, "-"),
        ("Relative Relief (Rhp = H*100/P)", Rhp, "-"),
        ("Ruggedness Number (Rn = H * Dd)", Rn, "-"),
        ("Dissection Index (Dis = H/Zmax)", Dis, "-"),
    ]

results.append(("Projected CRS used", utm_epsg, f"EPSG:{utm_epsg}"))
# add stream/drainage texture parameters
results += [
    ("Total Length of All Streams (ΣLu)", dr_tex["Total length of all streams (ΣLu)"], "km"),
    ("Total Number of Streams (ΣNu)", dr_tex["Total number of streams (ΣNu)"], "count"),
    ("Stream Frequency (Fs = ΣNu/A)", dr_tex["Stream frequency (Fs = ΣNu/A)"], "1/km²"),
    ("Drainage Texture (Dt = ΣNu/P)", dr_tex["Drainage texture (Dt = ΣNu/P)"], "1/km"),
    ("Drainage Density (Dd = ΣLu/A)", dr_tex["Drainage density (Dd = ΣLu/A)"], "km/km²"),
    ("Drainage Intensity (Di = Fs/Dd)", dr_tex["Drainage intensity (Di = Fs/Dd)"], "-"),
    ("Infiltration Number (If = Fs*Dd)", dr_tex["Infiltration number (If = Fs*Dd)"], "-"),
    ("Length of Overland Flow (Lo = 1/(2Dd))", dr_tex["Length of overland flow (Lo = 1/(2Dd))"], "km"),
    ("Constant of Channel Maintenance (C = 1/Dd)", dr_tex["Constant of channel maintenance (C = 1/Dd)"], "km²/km"),
]
results += [
    ("Mean Bifurcation Ratio (Rbm)", net["Rbm_mean"], "-"),
    ("Horton Diagram Slope (b)", net["b_horton"], "-"),
    ("Mean Bifurcation Ratio (RbmH = antilog b)", net["RbmH"], "-"),
    ("Mean Stream Length Ratio (Lur)", net["Lur_mean"], "-"),
    ("Weighted Mean Bifurcation Ratio (Rbwm)", net["Rbwm"], "-"),
    ("Rho Coefficient (Rho = Lur/Rbm)", net["rho"], "-"),
]
df = pd.DataFrame(results, columns=["Parameter", "Value", "Unit"])
df["Value"] = pd.to_numeric(df["Value"], errors="ignore").round(ROUND_DIGITS)

pd.set_option("display.max_colwidth", 95)
print(df.to_string(index=False))

if SAVE_CSV:
    df.to_csv(CSV_OUT, index=False)
    print("\n✅ Saved:", CSV_OUT)
