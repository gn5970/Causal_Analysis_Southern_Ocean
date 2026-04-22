import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
q=-25
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.mpl.ticker as cticker
from matplotlib import ticker
import xesmf as xe
from eofs.xarray import Eof
from scipy.stats import linregress

def linear_trend(y, x):
    """Return slope and p-value of y vs x"""
    mask = np.isfinite(y)
    y_valid = y[mask]
    x_valid = x[mask]

    # Not enough valid points for regression
    if len(y_valid) < 3:
        return np.nan, np.nan

    # All x are identical (no temporal span)
    if np.all(x_valid == x_valid[0]):
        return np.nan, np.nan

    try:
        res = linregress(x_valid, y_valid)
        return res.slope, res.pvalue
    except Exception:
        # Catch any residual numerical corner cases
        return np.nan, np.nan

#def deseasonalize(ds):
#    # Calculate monthly climatology (mean over years)
#    clim = ds.groupby("time.month").mean("time")
    # Subtract climatology from original data
#    return (ds.groupby("time.month") - clim)/ds.std()
def deseasonalize(ds):
    clim = ds.groupby("time.month").mean("time")
    #std_month = (ds.groupby("time.month") - clim).std("time")
    ds1=ds.groupby("time.month") - clim
    ds1=ds1.coarsen(YC=12, XC=12, boundary='trim').mean()
    return ds1 / ds1.std("time")

def pdens(S, theta):
    a0, a1, a2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
    b0, b1, b2, b3, b4, b5 = 5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3
    c0, c1, c2, c3, c4, c5 = 1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464
    p = 100000.
    alpha0 = a0 + a1 * theta + a2 * S
    p0 = b0 + b1 * theta + b2 * theta**2 + b3 * theta**3 + b4 * S + b5 * theta * S
    lambd = c0 + c1 * theta + c2 * theta**2 + c3 * theta**3 + c4 * S + c5 * theta * S
    return (p + p0) / (lambd + alpha0 * (p + p0))

lat_bin_edges = np.arange(-90, -29, 2)  # latitude bins


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Chl_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()
data_Chl=ds
seconds_per_year = 24 * 60 * 60  # 31,536,000
# Select first 132 months
data_Chl = data_Chl.isel(time=slice(0, 132))

seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000

min_depth=-0

max_depth=-25
# Multiply all data variables by seconds per yea
#data_Chl = data_Chl.where(data_Chl.Z >= -100, drop=True).mean('Z')
data_Chl=data_Chl.isel(time=slice(0,132))
data_Chl.sel(Z=slice(min_depth,max_depth))
chl = data_Chl.BLGCHL  # Assuming this is the variable name

# Step 2: Mask top 100 m
Z_top = data_Chl.Z.where(data_Chl.Z >=q, drop=True)
data = data_Chl.sel(Z=Z_top)


drF = data.drF  # (Z), vertical cell thickness in m
hFacC = data.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_Chl = (chl * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]
data_Chl  = (
    (chl * drF_top * hFacC_top).sum(dim="Z") /
    (drF_top * hFacC_top).sum(dim="Z")
)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-1,1, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(data_Chl.XC), np.array(data_Chl.YC), np.array(data_Chl.mean('time')).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())

cb.set_label('Chl (mg/m^$3$)', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("CHL_"+str(q)+".pdf")

chl_spatial_ts = data_Chl.mean(dim=['YC','XC'])   # 1D over time

# Plot time series
fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(chl_spatial_ts.time, chl_spatial_ts, color='black', linewidth=2)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Chl (mg/m^$3$)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.grid(True)
plt.tight_layout()
plt.savefig("CHL_spatial_mean_timeseries_"+str(q)+".pdf")
plt.show()





time_num = data_Chl.time.dt.year.values

data_Chl=deseasonalize(data_Chl)

data_Chl_trend, pval = xr.apply_ufunc(
    linear_trend,
    data_Chl,
    time_num,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[[], []],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float, float]
)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.3,0.3, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(data_Chl_trend.XC), np.array(data_Chl_trend.YC), np.array(data_Chl_trend).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
cb.set_label('Chl (mg/m^$3$/yr)', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Chltrend.pdf")


data_Chl_zonal = data_Chl.mean(dim="XC", skipna=True)
data_Chl_latbin = data_Chl_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------
slopes = []
for m in range(1, 13):
    data_Chl_m = data_Chl_latbin.sel(time= data_Chl_latbin["time"].dt.month == m)
    time_num_m = data_Chl_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        data_Chl_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

data_Chl_trend = xr.concat(slopes, dim="month")


data_Chl_trend = data_Chl_trend.assign_coords(month=np.arange(1, 13))

data_Chl_trend_clim = data_Chl_zonal.groupby("time.month").mean(dim="time")

# --- Coordinates for plotting ---
lat_mid = data_Chl_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field (Hovmöller: month × latitude)
im = ax.pcolormesh(
    data_Chl_trend["month"].values,
    lat_mid,
    data_Chl_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid = data_Chl_trend_clim["YC"].values
#cs = ax.contour(
#months, lat_mid,
#data_Chl_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Tick label size (numbers on axes) ---
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="Chl trend ESA")
cb.set_label("Chl trend", fontsize=18)
cb.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig("data_Chltrend_Hovmoller_trend+mean.pdf", dpi=300)

#valid = np.isfinite(lon_flat) & np.isfinite(lat_flat)

#lon_out   = lon_flat[valid]
#lat_out   = lat_flat[valid]
#trend_out = trend_flat[valid]
#pval_out  = pval_flat[valid]


import scipy.io as io

#io.savemat(
#    "Chl_trend_significant.mat",
#    {
#        "lon": lon_out,
#        "lat": lat_out,
#        "chl_trend": trend_out,
#        "p_value": pval_out
#    }
#)

import scipy.io as io
import numpy as np

# 1. Create the significance mask
alpha = 0.05
sig_mask = pval < alpha

# 2. Mask the p-values (Keep the 2D shape!)
# Points that are NOT significant will become NaN
pval_2d = pval.where(sig_mask, np.nan)

# 3. Save the unique 1D axis vectors AND the 2D grid
# Note: We save XC and YC directly as the axes
io.savemat(
    "Chl_trend_significant.mat",
    {
        "lon_axis": data_Chl.XC.values,  # 1D array of longitudes
        "lat_axis": data_Chl.YC.values,  # 1D array of latitudes
        "p_value_2d": pval_2d.values           # 2D array (lat, lon)
    }
)


### same from satellite

ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/ESA_Chl_south_surface_for_BSOSE.nc",
    engine="h5netcdf"
)
ds=ds.sel(time=slice("2013-01-01", "2023-12-31"))
#ds=ds.coarsen(lat=4, lon=4, boundary='trim').mean()
print('ds',ds)
max_val = ds.chlor_a.max().values
print(f"Max Chlorophyll value after regridding: {max_val}")

if np.isnan(max_val):
    print("WARNING: Data is still all NaNs. Check latitude ranges.")

# --- 4. PLOTTING ---
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Calculate mean, ensuring we handle NaNs in the time dimension
data_to_plot = ds.chlor_a.mean(dim='time', skipna=True)

levels = np.linspace(-1, 1, 40)
fill = ax.contourf(
    ds.lon,
    ds.lat,
    data_to_plot,
    cmap='RdBu_r',
    levels=levels,
    transform=ccrs.PlateCarree(),
    extend='both' # This handles values outside your -0.9 to 0.9 range
)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.7, pad=0.05)
cb.set_label('Chlorophyll-a Concentration', size=12)

plt.savefig("Chlspatialmean_satellite.pdf", dpi=300)


def deseasonalize_sat(ds):
    clim = ds.groupby("time.month").mean("time")
    #std_month = (ds.groupby("time.month") - clim).std("time")
    ds1=ds.groupby("time.month") - clim
    return ds1 / ds1.std("time")

time_num = ds.time.dt.year.values

ds1=deseasonalize_sat(ds.chlor_a)

data_Chl_trend, pval = xr.apply_ufunc(
    linear_trend,
    ds1,
    time_num,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[[], []],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float, float]
)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.15,0.15, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(data_Chl_trend.lon), np.array(data_Chl_trend.lat), np.array(data_Chl_trend).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
cb.set_label('Chl (mg/m^$3$/yr)', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("Chltrend_satellite.pdf")


data_Chl_zonal = ds1.mean(dim="lon", skipna=True)
data_Chl_latbin = data_Chl_zonal.groupby_bins("lat", bins=lat_bin_edges).mean()

# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------
slopes = []
for m in range(1, 13):
    data_Chl_m = data_Chl_latbin.sel(time= data_Chl_latbin["time"].dt.month == m)
    time_num_m = data_Chl_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        data_Chl_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

data_Chl_trend = xr.concat(slopes, dim="month")


data_Chl_trend = data_Chl_trend.assign_coords(month=np.arange(1, 13))

data_Chl_trend_clim = data_Chl_zonal.groupby("time.month").mean(dim="time")

# --- Coordinates for plotting ---
lat_mid = data_Chl_trend["lat_bins"].to_index().mid.values
months = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field (Hovmöller: month × latitude)
im = ax.pcolormesh(
    data_Chl_trend["month"].values,
    lat_mid,
    data_Chl_trend.transpose("lat_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.15, vmax=0.15,
    shading="auto"
)
lat_mid = data_Chl_trend_clim["lat"].values
#cs = ax.contour(
#months, lat_mid,
#data_Chl_trend_clim.transpose("lat", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Tick label size (numbers on axes) ---
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="Chl trend ESA")
cb.set_label("Chl trend ", fontsize=18)
cb.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig("data_Chltrendsatellite_Hovmoller_trend+mean.pdf", dpi=300)






data_NPP = xr.open_dataset("/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_NPP = data_NPP.coarsen(YC=8, XC=8, boundary="trim").mean()
seconds_per_year=12.011*1000*86400
ds=data_NPP

# Coarsen grid spatially (21x21)
# Select chl-a
chl = ds.BLGNPP*seconds_per_year

# Restrict to top q m
Z_top = ds.Z.where(ds.Z >= q, drop=True)
chl = chl.sel(Z=Z_top)
drF = ds.drF.sel(Z=Z_top)
hFacC = ds.hFacC.sel(Z=Z_top)

# Volume-weighted vertical integration
chl_weighted = (chl * drF * hFacC).sum(dim="Z") / (drF * hFacC).sum(dim="Z")


from matplotlib import ticker
mean_chl = chl_weighted.mean(dim=['YC', 'XC'])
# 3. Plot the time series (e.g. at a given depth level, or integrated)
#    Here we’ll pick the surface depth (Z=0) as an example:
fig = plt.figure(figsize=(12, 4), dpi=300)
plt.plot(mean_chl.time, mean_chl, color='black', linewidth=2)
plt.xlabel("Time", fontsize=18)
plt.ylabel("NPP (mgC/m2/day)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.grid(True)
plt.tight_layout()
plt.grid(True)
plt.savefig("NPPspatialmean.pdf", dpi=300)



def linear_trend(y, x):
    """Return slope and p-value of y vs x"""
    mask = np.isfinite(y)
    y_valid = y[mask]
    x_valid = x[mask]

    # Not enough valid points for regression
    if len(y_valid) < 3:
        return np.nan, np.nan

    # All x are identical (no temporal span)
    if np.all(x_valid == x_valid[0]):
        return np.nan, np.nan

    try:
        res = linregress(x_valid, y_valid)
        return res.slope, res.pvalue
    except Exception:
        # Catch any residual numerical corner cases
        return np.nan, np.nan


time_num = chl_weighted.time.dt.year.values
npp_trend, pval = xr.apply_ufunc(
    linear_trend,
    chl_weighted,
    time_num,
    input_core_dims=[["time"], ["time"]],
    output_core_dims=[[], []],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float, float]
)
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-11,11, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(chl_weighted.XC), np.array(chl_weighted.YC), np.array(npp_trend).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
cb.set_label('NPP Trend (mgC/m²/day²)', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
plt.savefig('NPP_trend.pdf')






chl_weighted=deseasonalize(chl_weighted)
# -------------------------------
# 3. Zonal mean & latitude binning
# -------------------------------
chl_zonal = chl_weighted.mean(dim="XC", skipna=True)
chl_latbin = chl_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------
slopes = []
for m in range(1, 13):
    chl_m = chl_latbin.sel(time=chl_latbin["time"].dt.month == m)
    time_num_m = chl_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        chl_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    ) 
    slopes.append(slope)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- Combine monthly slopes into single array ---
chl_trend = xr.concat(slopes, dim="month")
chl_trend = chl_trend.assign_coords(month=np.arange(1, 13))

# Keep full seasonal climatology (lat × month)
chl_trend_clim =  chl_zonal.groupby("time.month").mean(dim="time")
print('chl_trend_clim',chl_trend_clim)
# --- Coordinates for plotting ---
lat_mid = chl_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

# --- Base figure ---
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field (Hovmöller: month × latitude)
im = ax.pcolormesh(
    chl_trend["month"].values,
    lat_mid,
    chl_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid = chl_trend_clim["YC"].values
#cs = ax.contour(
#months, lat_mid,
#chl_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="NPP trend BSOSE")
cb.set_label("NPP trend BSOSE", fontsize=18)
cb.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig("data_NPPtrend_Hovmoller_trend+mean.pdf", dpi=300)
plt.close()





ds_thetao = xr.open_dataset("/projects/CDEUTSCH/DATA/Theta_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
ds = ds_thetao.isel(time=slice(0, 132))

# Coarsen spatially
# Extract flux variable
THETA = ds["THETA"]   # (time, Zl, YC, XC)

# -------------------------------------------------
# Compute vertical cell thickness from rLowC - rSurfC
# -------------------------------------------------
layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": THETA.time, "Z": THETA.Z}
).transpose("time", "Z", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": THETA.time, "Z": THETA.Z}).transpose("time","Z","YC","XC")

THETA = THETA.where(mask_4d)
max_depth = -25
THETA = THETA.sel(Z=slice(0, max_depth))
layer_thickness_4d = layer_thickness_4d.sel(Z=slice(0, max_depth))

# -------------------------------------------------
# Volume-weighted mean vertical flux
## -------------------------------------------------
data_THETA_mean = (THETA * layer_thickness_4d).sum(dim="Z") / layer_thickness_4d.sum(dim="Z")

data_THETA_mean=deseasonalize(data_THETA_mean)
# -------------------------------
# 3. Zonal mean & latitude binning
# -------------------------------
data_THETA_mean_zonal = data_THETA_mean.mean(dim="XC", skipna=True)
data_THETA_mean_latbin = data_THETA_mean_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()
# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------:wq
slopes = []
for m in range(1, 13):
    data_THETA_mean_m = data_THETA_mean_latbin.sel(time=data_THETA_mean_latbin["time"].dt.month == m)
    time_num_m = data_THETA_mean_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        data_THETA_mean_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

data_THETA_mean_trend = xr.concat(slopes, dim="month")
data_THETA_mean_trend = data_THETA_mean_trend.assign_coords(month=np.arange(1, 13))

# Keep full seasonal climatology (lat × month)
#data_THETA_mean_trend_clim =  data_THETA_mean_zonal.groupby("time.month").mean(dim="time")
data_THETA_mean_clim_latbin = (
    data_THETA_mean_latbin
    .groupby("time.month")
    .mean(dim="time")
)
#print('chl_trend_clim',data_THETA_mean_trend_clim)
# --- Coordinates for plotting ---
lat_mid = data_THETA_mean_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

# --- Base figure ---
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field (Hovmöller: month × latitude)
im = ax.pcolormesh(
    data_THETA_mean_trend["month"].values,
    lat_mid,
    data_THETA_mean_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid = data_THETA_mean_trend["YC_bins"].to_index().mid.values
#clevels = np.linspace(
#    np.nanmin(data_THETA_mean_clim_latbin),
#    np.nanmax(data_THETA_mean_clim_latbin),
#    6
#)

#cs = ax.contour(
#    months,
#    lat_mid,
#    data_THETA_mean_clim_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)
#ax.clabel(cs, fmt="%.1f", fontsize=10)

#cs = ax.contour(
#months, lat_mid,
#data_THETA_mean_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="Theta trend BSOSE")
cb.set_label("Theta trend", fontsize=18)
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("data_Thetatrend_Hovmoller_trend+mean.pdf", dpi=300)
plt.close()



ds1 = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/surfTflx_bsoseI156_2013to2024_monthly.nc",
    engine="netcdf4"
)
#ds1=ds1.coarsen(YC=8, XC=8, boundary='trim').mean()
ds1 = ds1.isel(time=slice(0, 132))
Fz = ds1["TFLUX"]
Tflux = Fz
#Tflux=Tflux.coarsen(YC=8, XC=8, boundary='trim').mean()
surfTflx=deseasonalize(Tflux)

surfTflx_zonal = surfTflx.mean(dim="XC", skipna=True)
surfTflx_latbin = surfTflx_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()
# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------:wq
slopes = []
for m in range(1, 13):
    surfTflx_m = surfTflx_latbin.sel(time=surfTflx_latbin["time"].dt.month == m)
    time_num_m = surfTflx_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        surfTflx_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

surfTflx_trend = xr.concat(slopes, dim="month")
surfTflx_trend = surfTflx_trend.assign_coords(month=np.arange(1, 13))

# Keep full seasonal climatology (lat × month)
#data_THETA_mean_trend_clim =  data_THETA_mean_zonal.groupby("time.month").mean(dim="time")
surfTflx_clim_latbin = (
    surfTflx_latbin
    .groupby("time.month")
    .mean(dim="time")
)
#print('chl_trend_clim',data_THETA_mean_trend_clim)
# --- Coordinates for plotting ---
lat_mid = surfTflx_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)


fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field (Hovmöller: month × latitude)
im = ax.pcolormesh(
    surfTflx_trend["month"].values,
    lat_mid,
    surfTflx_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid = surfTflx_trend["YC_bins"].to_index().mid.values
clevels = np.linspace(
    np.nanmin(surfTflx_clim_latbin),
    np.nanmax(surfTflx_clim_latbin),
    6
)

#cs = ax.contour(
#    months,
#    lat_mid,
#    surfTflx_clim_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)
#ax.clabel(cs, fmt="%.1f", fontsize=10)

#cs = ax.contour(
#months, lat_mid,
#data_THETA_mean_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="surfTflx trend BSOSE")
cb.set_label("surfTflx trend", fontsize=18)
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("data_surfTflx_Hovmoller_trend+mean.pdf", dpi=300)
plt.close()







ds = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
SALT = ds["SALT"]
layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": SALT.time, "Z": SALT.Z}
).transpose("time", "Z", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": SALT.time, "Z": SALT.Z}).transpose("time","Z","YC","XC")

SALT = SALT.where(mask_4d)
max_depth = -25
SALT = SALT.sel(Z=slice(0, max_depth))
layer_thickness_4d = layer_thickness_4d.sel(Z=slice(0, max_depth))

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
data_SALT_mean = (SALT * layer_thickness_4d).sum(dim="Z") / layer_thickness_4d.sum(dim="Z")

data_SALT_mean=deseasonalize(data_SALT_mean)

data_SALT_mean_zonal = data_SALT_mean.mean(dim="XC", skipna=True)
data_SALT_mean_latbin = data_SALT_mean_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------:wq
slopes = []
for m in range(1, 13):
    data_SALT_mean_m = data_SALT_mean_latbin.sel(time=data_SALT_mean_latbin["time"].dt.month == m)
    time_num_m = data_SALT_mean_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        data_SALT_mean_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

data_SALT_mean_trend = xr.concat(slopes, dim="month")
data_SALT_mean_trend = data_SALT_mean_trend.assign_coords(month=np.arange(1, 13))

data_SALT_latbin = (
    data_SALT_mean_latbin
    .groupby("time.month")
    .mean(dim="time")
)

# Keep full seasonal climatology (lat × month)
#data_SALT_mean_trend_clim =  data_SALT_mean_zonal.groupby("time.month").mean(dim="time")

#print('chl_trend_clim',data_SALT_mean_trend_clim)
# --- Coordinates for plotting ---
lat_mid = data_SALT_mean_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

# --- Base figure ---
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field (Hovmöller: month × latitude)
im = ax.pcolormesh(
    data_SALT_mean_trend["month"].values,
    lat_mid,
    data_SALT_mean_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)

#cs = ax.contour(
#months, lat_mid,
#data_THETA_mean_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#clevels = np.linspace(
#    np.nanmin(data_SALT_latbin),
#    np.nanmax(data_SALT_latbin),
#    6
#)

#cs = ax.contour(
#    months,
#    lat_mid,
#    data_SALT_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)
#ax.clabel(cs, fmt="%.1f", fontsize=10)

# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="SALT trend BSOSE")
cb.set_label("Salt trend ", fontsize=18)
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("data_Salttrend_Hovmoller_trend+mean.pdf", dpi=300)
plt.close()


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/BioMass_bsoseI155_2013to2023_monthly.nc",
    engine="netcdf4"
)
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()

# Select first 132 months
ds = ds.isel(time=slice(0, 132))
biomass = ds["TRAC09"]   # (time, Zl, YC, XC)i
C_to_N = 106.0 / 16.0
biomass = biomass * C_to_N

layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": biomass.time, "Z": biomass.Z}
).transpose("time", "Z", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": biomass.time, "Z": biomass.Z}).transpose("time","Z","YC","XC")
biomass = biomass.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
biomass = biomass.sel(Z=slice(0, max_depth))


data_Llim  = xr.open_dataset("/projects/CDEUTSCH/DATA/Llim_bsoseI156_2013to2024_monthly.nc", engine="netcdf4")
#data_Llim=data_Llim.coarsen(YC=8, XC=8, boundary='trim').mean()
data_Llim=data_Llim.isel(time=slice(0,132))
#data_Llim.sel(Z=slice(min_depth,max_depth))
Llim = data_Llim.BLGLLIM  # Assuming this is the variable name

# Step 2: Mask top 100 m
Z_top = data_Llim.Z.where(data_Llim.Z >= q, drop=True)
data = data_Llim.sel(Z=Z_top)


drF = data.drF  # (Z), vertical cell thickness in m
hFacC = data.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_Llim = (Llim * biomass* drF_top * hFacC_top).sum(dim="Z")/(biomass* drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]
data_Llim = deseasonalize(data_Llim)
Llim_zonal = data_Llim.mean(dim="XC", skipna=True)
Llim_latbin = Llim_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------
slopes = []
for m in range(1, 13):
    Llim_m = Llim_latbin.sel(time= Llim_latbin["time"].dt.month == m)
    time_num_m = Llim_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        Llim_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    ) 
    slopes.append(slope)

Llim_trend = xr.concat(slopes, dim="month")
Llim_trend = Llim_trend.assign_coords(month=np.arange(1, 13))

Llim_trend_clim = Llim_zonal.groupby("time.month").mean(dim="time")

# --- Coordinates for plotting ---
lat_mid = Llim_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

# --- Base figure ---
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    Llim_trend["month"].values,
    lat_mid,
    Llim_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid =Llim_trend_clim["YC"].values
# 2. Contours of long-term mean
#cs = ax.contour(
#months, lat_mid,
#Llim_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)
# --- Colorbar ---
cb = plt.colorbar(im, ax=ax, label="Llim trend BSOSE")
cb.set_label("Llim trend BSOSE", fontsize=18)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("data_Llim_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()



data_Felim = xr.open_dataset("/projects/CDEUTSCH/DATA/FElim_bsoseI156_2013to2024_monthly.nc", engine="netcdf4")
data_Felim=data_Felim.isel(time=slice(0,132))
#Felim = data_Felim.BLGFELIM
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000

min_depth=-0

max_depth=q
# Multiply all data variables by seconds per yea
#data_Chl = data_Chl.where(data_Chl.Z >= -100, drop=True).mean('Z')
#data_FElim=data_FElim.sel(Z=slice(min_depth,max_depth))
data_Felim = data_Felim.BLGFELIM  # Assuming this is the variable name
#data_Felim = deseasonalize(data_Felim)
# Step 2: Mask top 100 m
Z_top = data_Felim.Z.where(data_Felim.Z >= q, drop=True)
data = data_Felim.sel(Z=Z_top)


drF = data.drF  # (Z), vertical cell thickness in m
hFacC = data.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_Felim = (data * chl* drF_top * hFacC_top).sum(dim="Z")/(chl* drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]
data_Felim = deseasonalize(data_Felim)
Felim_zonal = data_Felim.mean(dim="XC", skipna=True)
Felim_latbin = Felim_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

# -------------------------------
# 4. Compute linear trends per month × latitude bin
# -------------------------------
slopes = []
for m in range(1, 13):
    Felim_m = Felim_latbin.sel(time=Felim_latbin["time"].dt.month == m)
    time_num_m = Felim_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        Felim_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    ) 
    slopes.append(slope)


Felim_trend = xr.concat(slopes, dim="month")
Felim_trend = Felim_trend.assign_coords(month=np.arange(1, 13))

Felim_trend_clim = Felim_zonal.groupby("time.month").mean(dim="time")

# --- Coordinates for plotting ---
lat_mid = Felim_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    Felim_trend["month"].values,
    lat_mid,
    Felim_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid =Felim_trend_clim["YC"].values
# 2. Contours of long-term mean
#cs = ax.contour(
#months, lat_mid,
#Felim_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Colorbar ---
ax.tick_params(axis="both", which="major", labelsize=20)
cb = plt.colorbar(im, ax=ax, label="Fe Lim trend BSOSE")
cb.set_label("Fe lim trend BSOSE", fontsize=18)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("data_Felim_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()




ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Chl_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()
data_Chl=ds
seconds_per_year = 24 * 60 * 60  # 31,536,000
# Select first 132 months
data_Chl = data_Chl.isel(time=slice(0, 132))

seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000

min_depth=-0

max_depth=-25
# Multiply all data variables by seconds per yea
#data_Chl = data_Chl.where(data_Chl.Z >= -100, drop=True).mean('Z')
data_Chl=data_Chl.isel(time=slice(0,132))
data_Chl.sel(Z=slice(min_depth,max_depth))
chl = data_Chl.BLGCHL  # Assuming this is the variable name

# Step 2: Mask top 100 m
Z_top = data_Chl.Z.where(data_Chl.Z >=q, drop=True)
data = data_Chl.sel(Z=Z_top)


drF = data.drF  # (Z), vertical cell thickness in m
hFacC = data.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = data.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
data_Chl = (chl * drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]
data_Chl  = (
    (chl * drF_top * hFacC_top).sum(dim="Z") /
    (drF_top * hFacC_top).sum(dim="Z")
)
data_Chl = deseasonalize(data_Chl)
Chl_zonal = data_Chl.mean(dim="XC", skipna=True)
Chl_latbin =Chl_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

slopes = []
for m in range(1, 13):
    Chl_m =Chl_latbin.sel(time= Llim_latbin["time"].dt.month == m)
    time_num_m = Chl_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        Chl_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    ) 
    slopes.append(slope)

Chl_trend = xr.concat(slopes, dim="month")
Chl_trend = Chl_trend.assign_coords(month=np.arange(1, 13))

Chl_trend_clim = Chl_zonal.groupby("time.month").mean(dim="time")

# --- Coordinates for plotting ---
lat_mid = Chl_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    Chl_trend["month"].values,
    lat_mid,
    Chl_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.5, vmax=0.5,
    shading="auto"
)
lat_mid =Chl_trend_clim["YC"].values
# 2. Contours of long-term mean
#cs = ax.contour(
#months, lat_mid,
#Chl_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Colorbar ---
ax.tick_params(axis="both", which="major", labelsize=20)
cb = plt.colorbar(im, ax=ax, label="Chl trend BSOSE")
cb.set_label("Chl trend BSOSE", fontsize=18)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("data_Chl_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()




ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/SIArea_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
ds=ds.isel(time=slice(0,132))
#ds=ds.coarsen(YC=8, XC=8, boundary='trim').mean()
C = ds["SIarea"]

#C = C.Z.where(C.Z >= q, drop=True)
Siarea=deseasonalize(C)

Siarea_zonal = Siarea.mean(dim="XC", skipna=True)
Siarea_latbin =Siarea_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

slopes = []
for m in range(1, 13):
    Siarea_m =Siarea_latbin.sel(time= Siarea_latbin["time"].dt.month == m)
    time_num_m = Siarea_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        Siarea_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    ) 
    slopes.append(slope)

Siarea_trend = xr.concat(slopes, dim="month")
Siarea_trend = Siarea_trend.assign_coords(month=np.arange(1, 13))

Siarea_trend_clim = Siarea_zonal.groupby("time.month").mean(dim="time")

# --- Coordinates for plotting ---
lat_mid = Siarea_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    Siarea_trend["month"].values,
    lat_mid,
    Siarea_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
lat_mid =Siarea_trend_clim["YC"].values
# 2. Contours of long-term mean
#cs = ax.contour(
#months, lat_mid,
#Siarea_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Colorbar ---
ax.tick_params(axis="both", which="major", labelsize=20)
cb = plt.colorbar(im, ax=ax, label="Si Area trend BSOSE")
cb.set_label("Si Area (m^-2/m^-2)", fontsize=18)
#cb.set_label("Si Area trend BSOSE", fontsize=20)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("data_Siarea_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()

ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/BioMass_bsoseI155_2013to2023_monthly.nc",
    engine="netcdf4"
)
#ds=ds.coarsen(YC=8, XC=8, boundary='trim').mean()
# Select first 132 months
ds = ds.isel(time=slice(0, 132))
biomass = ds["TRAC09"]   # (time, Zl, YC, XC)i
C_to_N = 106.0 / 16.0
biomass = biomass * C_to_N


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
seconds_per_year = 365 * 24 * 3600
#ds=ds.coarsen(YC=8, XC=8, boundary='trim').mean()
# Select first 132 months
ds = ds.isel(time=slice(0, 132))
NPP = ds["BLGNPP"]*seconds_per_year   # (time, Zl, YC, XC)

phytoplankotn_growth_rate=NPP/biomass
Z_top = phytoplankotn_growth_rate.Z.where(phytoplankotn_growth_rate.Z >= q, drop=True)
phytoplankotn_growth_rate = phytoplankotn_growth_rate.sel(Z=Z_top)
drF = ds.drF  # (Z), vertical cell thickness in m
hFacC = ds.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = ds.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
phytoplankotn_growth_rate  = (
    (phytoplankotn_growth_rate * drF_top * hFacC_top).sum(dim="Z") /
    (drF_top * hFacC_top).sum(dim="Z")
)
phytoplankotn_growth_rate=deseasonalize(phytoplankotn_growth_rate)

phytoplankotn_growth_rate_zonal = phytoplankotn_growth_rate.mean(dim="XC", skipna=True)
phytoplankotn_growth_rate_latbin =phytoplankotn_growth_rate_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

slopes = []
for m in range(1, 13):
    phytoplankotn_growth_rate_m =phytoplankotn_growth_rate_latbin.sel(time= phytoplankotn_growth_rate_latbin["time"].dt.month == m)
    time_num_m = phytoplankotn_growth_rate_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        phytoplankotn_growth_rate_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

phytoplankotn_growth_rate_trend = xr.concat(slopes, dim="month")
phytoplankotn_growth_rate_trend = phytoplankotn_growth_rate_trend.assign_coords(month=np.arange(1, 13))

phytoplankotn_growth_rate_clim_latbin = (
    phytoplankotn_growth_rate_latbin
    .groupby("time.month")
    .mean(dim="time")
)

# --- Coordinates for plotting ---
lat_mid = phytoplankotn_growth_rate_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    phytoplankotn_growth_rate_trend["month"].values,
    lat_mid,
    phytoplankotn_growth_rate_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.5, vmax=0.5,
    shading="auto"
)

clevels = np.linspace(
    np.nanmin(phytoplankotn_growth_rate_clim_latbin),
    np.nanmax(phytoplankotn_growth_rate_clim_latbin),
    6
)

#cs = ax.contour(
#    months,
#    lat_mid,
#    phytoplankotn_growth_rate_clim_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)

#ax.clabel(cs, fmt="%.1f", fontsize=10)

# 2. Contours of long-term mean
#cs = ax.contour(
#months, lat_mid,
#phytoplankotn_growth_rate_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Colorbar ---
ax.tick_params(axis="both", which="major", labelsize=20)
cb = plt.colorbar(im, ax=ax, label="Phyotplankton Growth Rate trend BSOSE")
cb.set_label("Phyotplankton Growth Rate trend BSOSE", fontsize=18)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("data_pythonplankton_growth_rate_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()



import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load MLD
# -------------------------------------------------
data_MLD = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/MLD_bsoseI156_2013to2024_monthly.nc",
    engine="netcdf4"
)

data_MLD = data_MLD.isel(time=slice(0, 132)).BLGMLD
data_MLD =deseasonalize(data_MLD)
# -------------------------------------------------
# Zonal mean + latitude binning
# -------------------------------------------------
data_MLD_zonal = data_MLD.mean(dim="XC", skipna=True)

data_MLD_latbin = (
    data_MLD_zonal
    .groupby_bins("YC", bins=lat_bin_edges)
    .mean()
)
# dims: (time, YC_bins)

# -------------------------------------------------
# Monthly trends
# -------------------------------------------------
slopes = []

for m in range(1, 13):
    data_MLD_m = data_MLD_latbin.sel(
        time=data_MLD_latbin["time"].dt.month == m
    )

    time_num_m = data_MLD_m["time"].dt.year.values

    slope, pval = xr.apply_ufunc(
        linear_trend,
        data_MLD_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )

    slopes.append(slope)

data_MLD_trend = xr.concat(slopes, dim="month")
data_MLD_trend = data_MLD_trend.assign_coords(month=np.arange(1, 13))
# dims: (month, YC_bins)

# -------------------------------------------------
# Monthly climatology (FOR CONTOURS)
# -------------------------------------------------
data_MLD_clim_latbin = (
    data_MLD_latbin
    .groupby("time.month")
    .mean(dim="time")
)
# dims: (month, YC_bins)

# -------------------------------------------------
# Coordinates
# -------------------------------------------------
lat_mid = data_MLD_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend (color)
im = ax.pcolormesh(
    months,
    lat_mid,
    data_MLD_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3,
    vmax=0.3,
    shading="auto"
)

# 2. Climatological mean (contours)
#clevels = np.linspace(
#    np.nanmin(data_MLD_clim_latbin),
#    np.nanmax(data_MLD_clim_latbin),
#    6
#)

#cs = ax.contour(
#    months,
#    lat_mid,
#    data_MLD_clim_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)

ax.clabel(cs, fmt="%.1f", fontsize=10)

# -------------------------------------------------
# Formatting
# -------------------------------------------------
ax.set_xticks(months)
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)

cb = plt.colorbar(im, ax=ax)
cb.set_label("MLD trend (m)", fontsize=20)
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("data_MLD_Hovmoller_trend+mean.pdf", dpi=300)
plt.close()



data_oceQsw = xr.open_dataset("/projects/CDEUTSCH/DATA/oceQsw_bsoseI156_2013to2024_monthly.nc", engine="netcdf4")
print('data_oceQsw',data_oceQsw)
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000
#data_oceQsw=data_oceQsw.coarsen(YC=8, XC=8, boundary='trim').mean()
data_oceQsw=data_oceQsw.isel(time=slice(0,132))
data_oceQsw=data_oceQsw.oceQsw
data_oceQsw = deseasonalize(data_oceQsw)

data_oceQsw_zonal = data_oceQsw.mean(dim="XC", skipna=True)
data_oceQsw_latbin =data_oceQsw_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

slopes = []
for m in range(1, 13):
    data_oceQsw_m =data_oceQsw_latbin.sel(time= data_oceQsw_latbin["time"].dt.month == m)
    time_num_m = data_oceQsw_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        data_oceQsw_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    ) 
    slopes.append(slope)

data_oceQsw_trend = xr.concat(slopes, dim="month")
data_oceQsw_trend = data_oceQsw_trend.assign_coords(month=np.arange(1, 13))

data_oceQsw_latbin = (
    data_oceQsw_latbin
    .groupby("time.month")
    .mean(dim="time")
)


lat_mid = data_oceQsw_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    months,
    lat_mid,
    data_oceQsw_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
#clevels = np.linspace(
#    np.nanmin(data_oceQsw_latbin),
#    np.nanmax(data_oceQsw_latbin),
#    6
#)

#cs = ax.contour(
#    months,
#    lat_mid,
#    data_oceQsw_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)
#ax.clabel(cs, fmt="%.1f", fontsize=10)

# -------------------------------------------------
# Formatting
# -------------------------------------------------
ax.set_xticks(months)
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)

cb = plt.colorbar(im, ax=ax)
cb.set_label("oceQsw trend ", fontsize=20)
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("data_oceQsw_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()



###### 
data_TAUX = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUX_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")

def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XG'].attrs
    if (_ds['XG'].min() >= 0) and (_ds['XG'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XG'] = xr.where(_ds['XG'] > 180, _ds['XG'] - 360, _ds['XG'])
        _ds = _ds.sortby('XG')
    return _ds

# Apply the conversion to your dataset
#data_TAUX = convert_0_360_to_neg180_180(data_TAUX)
#data_TAUX  = data_TAUX.coarsen(XG=coarsen_factor,YC=coarsen_factor, boundary='trim').mean()

data_TAUY = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUY_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
#data_TAUY = data_TAUY.coarsen(XC=coarsen_factor,YG=coarsen_factor, boundary='trim').mean()


target_grid = {"lon": data_TAUY["XC"], "lat": data_TAUX["YC"]}

# --- Regrid UVEL: (YG, XC) → (YC, XC)
regridder_TAUX = xe.Regridder(
    data_TAUX.oceTAUX, target_grid, method="bilinear", periodic=True, reuse_weights=False
)
TAUX_on_tracer = regridder_TAUX(data_TAUX.oceTAUX)

# --- Regrid VVEL: (YC, XG) → (YC, XC)
regridder_TAUY = xe.Regridder(
    data_TAUY.oceTAUY, target_grid, method="bilinear", periodic=True, reuse_weights=False
)
TAUY_on_tracer = regridder_TAUY(data_TAUY.oceTAUY)
wind_amp = np.sqrt(TAUX_on_tracer**2 + TAUY_on_tracer**2)

time_mean_wind_amp_anom2=deseasonalize(wind_amp)

time_mean_wind_amp_anom2_zonal = time_mean_wind_amp_anom2.mean(dim="XC", skipna=True)
time_mean_wind_amp_anom2_latbin =time_mean_wind_amp_anom2_zonal.groupby_bins("YC", bins=lat_bin_edges).mean()

slopes = []
for m in range(1, 13):
    time_mean_wind_amp_anom2_m =time_mean_wind_amp_anom2_latbin.sel(time= time_mean_wind_amp_anom2_latbin["time"].dt.month == m)
    time_num_m = time_mean_wind_amp_anom2_m["time"].dt.year.values   # aligned with chl_m
    slope, pval = xr.apply_ufunc(
        linear_trend,
        time_mean_wind_amp_anom2_m,
        time_num_m,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],      # <<< TWO outputs
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]    # <<< TWO dtypes
    )
    slopes.append(slope)

time_mean_wind_amp_anom2_trend = xr.concat(slopes, dim="month")
time_mean_wind_amp_anom2_trend = time_mean_wind_amp_anom2_trend.assign_coords(month=np.arange(1, 13))

time_mean_wind_amp_anom2_clim_latbin = (
    time_mean_wind_amp_anom2_latbin
    .groupby("time.month")
    .mean(dim="time")
)


lat_mid = time_mean_wind_amp_anom2_trend["YC_bins"].to_index().mid.values
months = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(6, 6))

# 1. Trend as color field
im = ax.pcolormesh(
    time_mean_wind_amp_anom2_trend["month"].values,
    lat_mid,
    time_mean_wind_amp_anom2_trend.transpose("YC_bins", "month").values,
    cmap="RdBu_r",
    vmin=-0.3, vmax=0.3,
    shading="auto"
)
#clevels = np.linspace(
#    np.nanmin(time_mean_wind_amp_anom2_clim_latbin),
#    np.nanmax(time_mean_wind_amp_anom2_clim_latbin),
#    6
#)

#cs = ax.contour(
#    months,
#    lat_mid,
#    time_mean_wind_amp_anom2_clim_latbin.transpose("YC_bins", "month").values,
#    levels=clevels,
#    colors="black",
#    linewidths=1.0
#)

#ax.clabel(cs, fmt="%.2f", fontsize=10)

# 2. Contours of long-term mean
#cs = ax.contour(
#months, lat_mid,
#Fe_trend_clim.transpose("YC", "month").values,
#levels=6, colors="black", linewidths=1
#)
#ax.clabel(cs, fmt="%.3f", fontsize=9)
# --- Axis formatting ---
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Latitude (°)", fontsize=16)
# --- Colorbar ---
ax.tick_params(axis="both", which="major", labelsize=20)
cb = plt.colorbar(im, ax=ax, label="MLD trend BSOSE")
cb.set_label("Wind Amplitude ", fontsize=20)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig("data_windamplitude_Hovmoller_trend+mean.pdf", dpi=300)
#plt.savefig("Llim_Hovmoller.pdf", dpi=300)
plt.close()


