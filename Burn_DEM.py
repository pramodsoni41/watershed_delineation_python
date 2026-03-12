import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

dem_path = r"E:/GANGA_MODEL/DATA/GANGA_DEM.tif"
rivers_path = r"E:/GANGA_MODEL/DATA/RIVERS_INDIA.shp"
districts_path = r"E:/GANGA_MODEL/DATA/Ganga_Disricts.shp"

out_burned_dem = r"E:/GANGA_MODEL/DATA/ganga_dem_burned7.tif"
out_river_mask = r"E:/GANGA_MODEL/DATA/river_mask7.tif"
out_dist_mask  = r"E:/GANGA_MODEL/DATA/district_mask7.tif"

RIVER_BUFFER_M = 8000.0
DIST_BUFFER_M  = 1000.0
RIVER_BURN_M   = 200.0
DIST_BURN_M    = 100.0

# --- DEM
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float32")
    meta = src.meta.copy()
    crs = src.crs
    transform = src.transform
    nodata = src.nodata
    out_shape = (src.height, src.width)

dem_nan = dem.copy()
if nodata is not None:
    dem_nan[dem_nan == nodata] = np.nan

# --- vectors
rivers = gpd.read_file(rivers_path)
districts = gpd.read_file(districts_path)

rivers = rivers[rivers.geometry.notnull() & ~rivers.geometry.is_empty].to_crs(crs)
districts = districts[districts.geometry.notnull() & ~districts.geometry.is_empty].to_crs(crs)

# --- buffers (IMPORTANT CHANGE)
rivers_buf = rivers.geometry.buffer(RIVER_BUFFER_M)

# Districts are lines => buffer lines directly (NOT boundary)
dist_buf = districts.geometry.buffer(DIST_BUFFER_M)

# drop empties
rivers_buf = [g for g in rivers_buf if g and (not g.is_empty)]
dist_buf   = [g for g in dist_buf   if g and (not g.is_empty)]

print("Buffered rivers polygons:", len(rivers_buf))
print("Buffered district polygons:", len(dist_buf))

# --- rasterize masks
river_mask = rasterize(
    ((geom, 1) for geom in rivers_buf),
    out_shape=out_shape, transform=transform,
    fill=0, dtype="uint8", all_touched=True
)

dist_mask = rasterize(
    ((geom, 1) for geom in dist_buf),
    out_shape=out_shape, transform=transform,
    fill=0, dtype="uint8", all_touched=True
)

print("River mask pixels:", int(river_mask.sum()))
print("District mask pixels:", int(dist_mask.sum()))

# --- burn
burned = dem_nan.copy()
burned[river_mask == 1] -= RIVER_BURN_M
burned[dist_mask == 1]  -= DIST_BURN_M

# restore nodata
if nodata is not None:
    burned_out = burned.copy()
    burned_out[np.isnan(burned_out)] = nodata
else:
    burned_out = burned

# --- write outputs
meta.update(dtype="float32", compress="lzw")
with rasterio.open(out_burned_dem, "w", **meta) as dst:
    dst.write(burned_out.astype("float32"), 1)

mask_meta = meta.copy()
mask_meta.update(dtype="uint8", nodata=0)

with rasterio.open(out_river_mask, "w", **mask_meta) as dst:
    dst.write(river_mask, 1)

with rasterio.open(out_dist_mask, "w", **mask_meta) as dst:
    dst.write(dist_mask, 1)

print("Saved:", out_burned_dem)
print("Saved masks:", out_river_mask, out_dist_mask)

#%%

import numpy as np
import rasterio

dem_path = r"E:/GANGA_MODEL/DATA/ganga_dem_burned5.tif"
out_print_dem = r"E:/GANGA_MODEL/DATA/ganga_dem_print_ready6.tif"

# --- Controls (tune these) ---
LOW_MAX = 800.0        # plains + foothills range where you want more detail
CAP_PCT = 99.0         # cap Himalaya using percentile (try 97–99)
EXAG_LOW = 10.0         # exaggeration for plains/foothills
EXAG_HIGH = 2.5        # exaggeration above LOW_MAX (keeps Himalaya under control)

with rasterio.open(dem_path) as src:
    z = src.read(1).astype("float32")
    meta = src.meta.copy()
    nodata = src.nodata

z2 = z.copy()
if nodata is not None:
    z2[z2 == nodata] = np.nan

# Reference base (so output starts near 0)
zmin = np.nanpercentile(z2, 2)   # robust minimum (avoids pits)
z0 = z2 - zmin

# Cap extreme mountains
cap_val = np.nanpercentile(z0, CAP_PCT)
z0 = np.minimum(z0, cap_val)

# Piecewise exaggeration
out = np.where(z0 <= LOW_MAX,
               z0 * EXAG_LOW,
               LOW_MAX * EXAG_LOW + (z0 - LOW_MAX) * EXAG_HIGH)

# Restore nodata
if nodata is not None:
    out[np.isnan(z2)] = nodata

meta.update(dtype="float32", compress="lzw")
with rasterio.open(out_print_dem, "w", **meta) as dst:
    dst.write(out.astype("float32"), 1)

print("Saved:", out_print_dem)
print("Cap value (after base shift):", cap_val)
print("Base (zmin):", zmin)
#%%
import numpy as np
import rasterio

inp = r"E:/GANGA_MODEL/DATA/ganga_dem_burned7.tif"
out = r"E:/GANGA_MODEL/DATA/ganga_dem_print_ready12.tif"

LOW_MAX = 500.0        # plains + foothills threshold
CAP = 20000.0           # cap Himalaya (use 4500 or 5000)
EXAG_LOW = 15.0         # exaggerate plains strongly
EXAG_HIGH = 3.0        # keep mountains under control

with rasterio.open(inp) as src:
    z = src.read(1).astype("float32")
    meta = src.meta.copy()
    nodata = src.nodata
    

z2 = z.copy()
if nodata is not None:
    z2[z2 == nodata] = np.nan

# shift so base starts at 0 (use robust min)
zmin = np.nanpercentile(z2, 1)   # ~ -174 in your case
z0 = z2 - zmin
z0 = np.maximum(z0, 0)

# cap extreme Himalaya
z0 = np.minimum(z0, CAP)

# piecewise exaggeration
z_print = np.where(
    z0 <= LOW_MAX,
    z0 * EXAG_LOW,
    LOW_MAX * EXAG_LOW + (z0 - LOW_MAX) * EXAG_HIGH
)

# restore nodata
if nodata is not None:
    z_print[np.isnan(z2)] = nodata

meta.update(dtype="float32", compress="lzw")
with rasterio.open(out, "w", **meta) as dst:
    dst.write(z_print.astype("float32"), 1)

print("Saved:", out)
print("Used zmin:", zmin, "cap:", CAP)
print("Output range (approx):",
      np.nanmin(z_print[z_print!=nodata]) if nodata is not None else np.nanmin(z_print),
      np.nanmax(z_print[z_print!=nodata]) if nodata is not None else np.nanmax(z_print))

#%%

import numpy as np
import rasterio

inp = r"E:/GANGA_MODEL/DATA/ganga_dem_burned7.tif"
out = r"E:/GANGA_MODEL/DATA/ganga_dem_print_ready_log2.tif"

CAP = 20000.0          # cap extreme Himalaya before transform (keep as you wish)
TARGET_MAX = 6000.0    # <-- final max after exaggeration (choose to fit print thickness)

# Controls curvature:
# bigger LOG_K => stronger compression of high elevations
LOG_K = 100.0          # try 100 to 800 (meters-ish if your DEM is meters)

with rasterio.open(inp) as src:
    z = src.read(1).astype("float32")
    meta = src.meta.copy()
    nodata = src.nodata

z2 = z.copy()
if nodata is not None:
    z2[z2 == nodata] = np.nan

# shift base to 0 using robust min
zmin = np.nanpercentile(z2, 1)
z0 = z2 - zmin
z0 = np.maximum(z0, 0)

# cap extremes
z0 = np.minimum(z0, CAP)

# log transform (compress highs, expand lows relatively)
t = np.log1p(z0 / LOG_K)  # monotonic

# scale so max becomes TARGET_MAX
tmax = np.nanmax(t)
z_print = (t / tmax) * TARGET_MAX if tmax > 0 else t

# restore nodata
if nodata is not None:
    z_print[np.isnan(z2)] = nodata

meta.update(dtype="float32", compress="lzw", nodata=nodata)
with rasterio.open(out, "w", **meta) as dst:
    dst.write(z_print.astype("float32"), 1)

valid = z_print[z_print != nodata] if nodata is not None else z_print
print("Saved:", out)
print("Used zmin:", zmin, "cap:", CAP)
print("Output range:", np.nanmin(valid), np.nanmax(valid))
print("LOG_K:", LOG_K, "TARGET_MAX:", TARGET_MAX)

#%%

import numpy as np
import rasterio

inp = r"E:/GANGA_MODEL/DATA/ganga_dem_burned7.tif"
out = r"E:/GANGA_MODEL/DATA/ganga_dem_print_ready_logfactor5.tif"

CAP = 6000.0
EXAG_MAX = 25.0   # exaggeration near sea/plains
EXAG_MIN = 1    # exaggeration in high mountains
SCALE = 300.0     # transition scale: bigger => slower drop of exaggeration

with rasterio.open(inp) as src:
    z = src.read(1).astype("float32")
    meta = src.meta.copy()
    nodata = src.nodata

z2 = z.copy()
if nodata is not None:
    z2[z2 == nodata] = np.nan

zmin = np.nanpercentile(z2, 1)
z0 = np.maximum(z2 - zmin, 0)
z0 = np.minimum(z0, CAP)

# smooth decreasing exaggeration with elevation
# exag(z) drops from EXAG_MAX to EXAG_MIN as z increases
w = np.log1p(z0 / SCALE) / np.log1p(CAP / SCALE)  # normalized 0..1
exag = EXAG_MAX - (EXAG_MAX - EXAG_MIN) * w

z_print = z0 * exag

# restore nodata
if nodata is not None:
    z_print[np.isnan(z2)] = nodata

meta.update(dtype="float32", compress="lzw", nodata=nodata)
with rasterio.open(out, "w", **meta) as dst:
    dst.write(z_print.astype("float32"), 1)

valid = z_print[z_print != nodata] if nodata is not None else z_print
print("Saved:", out)
print("Used zmin:", zmin, "cap:", CAP)
print("Output range:", np.nanmin(valid), np.nanmax(valid))
print("EXAG_MAX:", EXAG_MAX, "EXAG_MIN:", EXAG_MIN, "SCALE:", SCALE)