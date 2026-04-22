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
import xesmf as xe
from eofs.xarray import Eof
rho_ref=1027.5
g=9.83
def linear_trend(y, x):
    """Return slope of y vs x"""
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return slope

#def deseasonalize(ds):
#    # Calculate monthly climatology (mean over years)
#    clim = ds.groupby("time.month").mean("time")
#    # Subtract climatology from original data
#    return (ds.groupby("time.month") - clim)/ds.std()

def deseasonalize(ds):
    clim = ds.groupby("time.month").mean("time")
    #clim=clim.coarsen(YC=2, XC=2, boundary='trim').mean()
    ds1=ds.groupby("time.month") - clim
    ds1=ds1.coarsen(YC=12, XC=12, boundary='trim').mean()
    return ds1/ds1.std('time')

def pdens(S, theta):
    a0, a1, a2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
    b0, b1, b2, b3, b4, b5 = 5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3
    c0, c1, c2, c3, c4, c5 = 1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464
    p = 100000.
    alpha0 = a0 + a1 * theta + a2 * S
    p0 = b0 + b1 * theta + b2 * theta**2 + b3 * theta**3 + b4 * S + b5 * theta * S
    lambd = c0 + c1 * theta + c2 * theta**2 + c3 * theta**3 + c4 * S + c5 * theta * S
    return (p + p0) / (lambd + alpha0 * (p + p0))



ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc",
    engine="netcdf4"
)
#ds=ds.coarsen(YC=8, XC=8, boundary='trim').mean()
# Select first 132 months
ds = ds.isel(time=slice(0, 132))

# Coarsen spatially
# Extract flux variable
flux = ds["SALT"]   # (time, Zl, YC, XC)

# -------------------------------------------------
# Compute vertical cell thickness from rLowC - rSurfC
# -------------------------------------------------
layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": flux.time, "Z": flux.Z}
).transpose("time", "Z", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": flux.time, "Z": flux.Z}).transpose("time","Z","YC","XC")

flux = flux.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
flux = flux.sel(Z=slice(0, max_depth))
layer_thickness_4d = layer_thickness_4d.sel(Z=slice(0, max_depth))

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
data_salt_mean = (flux * layer_thickness_4d).sum(dim="Z") / layer_thickness_4d.sum(dim="Z")


# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
data_salt_mean = deseasonalize(data_salt_mean)

data_salt_mean1=data_salt_mean.isel(time=slice(0,36)).mean('time')
data_salt_mean2=data_salt_mean.isel(time=slice(132-36,132)).mean('time')

print('data_salt_mean',data_salt_mean)
print('data_salt_mean',np.array(data_salt_mean2-data_salt_mean1).shape)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-2.5,2.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(data_salt_mean.XC), np.array(data_salt_mean.YC), np.array(data_salt_mean2-data_salt_mean1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("3_Saltdifference.pdf")



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
layer_thickness_4d = layer_thickness_4d.sel(Z=slice(0, max_depth))
biomass_mean = (biomass * layer_thickness_4d).sum(dim="Z") / layer_thickness_4d.sum(dim="Z")
biomass_mean=deseasonalize(biomass_mean)

biomass_mean1=biomass_mean.isel(time=slice(0,36)).mean('time')
biomass_mean2=biomass_mean.isel(time=slice(132-36,132)).mean('time')

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-1.5,1.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(biomass_mean.XC), np.array(biomass_mean.YC), np.array(biomass_mean2-biomass_mean1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("3_Biomassdifference.pdf")

ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/NPP_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()

seconds_per_year = 24 * 60 * 60  # 31,536,000
# Select first 132 months
ds = ds.isel(time=slice(0, 132))
NPP = ds["BLGNPP"]*seconds_per_year   # (time, Zl, YC, XC)

phytoplankotn_growth_rate=NPP/biomass

layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": phytoplankotn_growth_rate.time, "Z": phytoplankotn_growth_rate.Z}
).transpose("time", "Z", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": phytoplankotn_growth_rate.time, "Z": phytoplankotn_growth_rate.Z}).transpose("time","Z","YC","XC")
phytoplankotn_growth_rate = phytoplankotn_growth_rate.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
phytoplankotn_growth_rate = phytoplankotn_growth_rate.sel(Z=slice(0, max_depth))
layer_thickness_4d = layer_thickness_4d.sel(Z=slice(0, max_depth))
phytoplankotn_growth_rate_mean = (phytoplankotn_growth_rate * layer_thickness_4d).sum(dim="Z") / layer_thickness_4d.sum(dim="Z")

phytoplankotn_growth_rate_mean=deseasonalize(phytoplankotn_growth_rate_mean)

phytoplankotn_growth_rate_mean1=phytoplankotn_growth_rate_mean.isel(time=slice(0,36)).mean('time')
phytoplankotn_growth_rate_mean2=phytoplankotn_growth_rate_mean.isel(time=slice(132-36,132)).mean('time')

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-2,2, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(phytoplankotn_growth_rate_mean.XC), np.array(phytoplankotn_growth_rate_mean.YC), np.array(phytoplankotn_growth_rate_mean2-phytoplankotn_growth_rate_mean1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("3_phytoplankotn_growth_ratedifference.pdf")


data_MLD = xr.open_dataset("/projects/CDEUTSCH/DATA/MLD_bsoseI156_2013to2024_monthly.nc", engine="netcdf4")
print('data_MLD',data_MLD)
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000
#data_MLD=data_MLD.coarsen(YC=8, XC=8, boundary='trim').mean()
data_MLD=data_MLD.isel(time=slice(0,132))
data_MLD=data_MLD.BLGMLD
data_MLD = deseasonalize(data_MLD)


data_MLD_mean1=data_MLD.isel(time=slice(0,36)).mean('time')
data_MLD_mean2=data_MLD.isel(time=slice(132-36,132)).mean('time')


fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
#ax.add_feature(
#    cfeature.LAND,
#    facecolor="white",
#    edgecolor="black",
#    zorder=10
#)
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-1.6,1.6, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(data_MLD.XC), np.array(data_MLD.YC), np.array(data_MLD_mean2-data_MLD_mean1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("3_MLDdifference.pdf")

ds_thetao = xr.open_dataset("/projects/CDEUTSCH/DATA/Theta_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
ds_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
# Compute N from potential density
pt = xr.apply_ufunc(pdens, ds_salt.SALT, ds_thetao.THETA, dask='parallelized', output_dtypes=[ds_salt.SALT.dtype])
anom_density = pt - rho_ref
dUdz = anom_density.roll(Z=-1) - anom_density.roll(Z=1)
N = (-(g / rho_ref) * dUdz)
#N=N.coarsen(YC=8, XC=8, boundary='trim').mean()
N=N.isel(time=slice(0,132))
#Felim = data_Felim.BLGFELIM
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000

min_depth=-0

max_depth=q
# Multiply all data variables by seconds per yea
#data_Chl = data_Chl.where(data_Chl.Z >= -100, drop=True).mean('Z')
#data_FElim=data_FElim.sel(Z=slice(min_depth,max_depth))
#data_Felim = data_Felim.BLGFELIM  # Assuming this is the variable name
#data_Llim = deseasonalize(data_Felim)
# Step 2: Mask top 100 m
Z_top = N.Z.where(N.Z >= q, drop=True)
N = N.sel(Z=Z_top)


drF = N.drF  # (Z), vertical cell thickness in m
hFacC = N.hFacC  # (Z, YC, XC), vertical fraction of wet cell
rA = N.rA

# Step 3: Align shapes for multiplication
drF_top = drF.sel(Z=Z_top)
hFacC_top = hFacC.sel(Z=Z_top)

# Step 4: Broadcast to match NPP shape
#drF_exp = drF_top.broadcast_like(npp)
#hFacC_exp = hFacC_top.broadcast_like(npp)

# Step 5: Compute the volume-weighted NPP (mol C / m² / year)
N = (N * drF_top * hFacC_top).sum(dim="Z")/( drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]

ds = xr.open_dataset("/projects/CDEUTSCH/DATA/Strat_bsoseI155_2013to2023_monthly.nc", engine="h5netcdf")
print('depth',ds)
ds = ds.isel(time=slice(0, 132))

# Coarsen spatially
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()

# Extract flux variable
flux = ds["DRHODR"]   # (time, Zl, YC, XC)

# -------------------------------------------------
# Compute vertical cell thickness from rLowC - rSurfC
# -------------------------------------------------
layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": flux.time, "Zl": flux.Zl}
).transpose("time", "Zl", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": flux.time, "Zl": flux.Zl}).transpose("time","Zl","YC","XC")

flux = flux.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
flux = flux.sel(Zl=-25, method="nearest")
#layer_thickness_4d = layer_thickness_4d.sel(Zl=slice(0, max_depth))

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
#data_DRHODR_mean = (flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")
data_DRHODR_mean = flux#(flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")
#data_DRHODR_mean = deseasonalize(data_DRHODR_mean)
Richarson=N/(data_DRHODR_mean**2)

Richarson_mean = deseasonalize(Richarson)

Richarson_mean1=Richarson_mean.isel(time=slice(0,36)).mean('time')
Richarson_mean2=Richarson_mean.isel(time=slice(132-36,132)).mean('time')

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
fill = ax.contourf(np.array(Richarson_mean.XC), np.array(Richarson_mean.YC), np.array(Richarson_mean2-Richarson_mean1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("3_Richardondifference.pdf")

### W changes




ds_thetao = xr.open_dataset("/projects/CDEUTSCH/DATA/Theta_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
ds_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
# Compute N from potential density
pt = xr.apply_ufunc(pdens, ds_salt.SALT, ds_thetao.THETA, dask='parallelized', output_dtypes=[ds_salt.SALT.dtype])
anom_density = pt - rho_ref
dUdz = anom_density.roll(Z=-1) - anom_density.roll(Z=1)
N = (-(g / rho_ref) * dUdz)
#N=N.coarsen(YC=8, XC=8, boundary='trim').mean()
N=N.isel(time=slice(0,132))
#Felim = data_Felim.BLGFELIM
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000

min_depth=-0

max_depth=q
# Multiply all data variables by seconds per yea
#data_Chl = data_Chl.where(data_Chl.Z >= -100, drop=True).mean('Z')
#data_FElim=data_FElim.sel(Z=slice(min_depth,max_depth))
#data_Felim = data_Felim.BLGFELIM  # Assuming this is the variable name
#data_Llim = deseasonalize(data_Felim)
# Step 2: Mask top 100 m
Z_top = N.Z.where(N.Z >= q, drop=True)
N = N.sel(Z=Z_top)

N = N.mean('Z')#(N * drF_top * hFacC_top).sum(dim="Z")/( drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]

N=deseasonalize(N)

N1=N.isel(time=slice(0,36)).mean('time')
N2=N.isel(time=slice(132-36,132)).mean('time')

N1=N.isel(time=slice(0,36)).mean('time')
N2=N.isel(time=slice(132-36,132)).mean('time')

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-2.5,2.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(N.XC), np.array(N.YC), np.array(N2-N1).squeeze(), levels=levels,
cmap=plt.cm.RdBu_r,
transform=ccrs.PlateCarree())
# Make the aspect ratio equal to get a circular plot
ax.set_aspect('equal')
# Colorbar
ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
#ax.set_boundary(plt.Circle((0, 0), 0.5, transform=ax.transAxes), transform=ax.transAxes)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle_path = mpath.Path(verts * radius + center)
ax.set_boundary(circle_path, transform=ax.transAxes)

cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
cb.ax.tick_params(labelsize=20)
#cb.set_label('', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig('3_Ndifference.pdf')


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/SIArea_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
ds=ds.isel(time=slice(0,132))
#ds=ds.coarsen(YC=8, XC=8, boundary='trim').mean()
C = ds["SIarea"]

#C = C.Z.where(C.Z >= q, drop=True)
Siarea=deseasonalize(C)

Siarea1=Siarea.isel(time=slice(0,36)).mean('time')
Siarea2=Siarea.isel(time=slice(132-36,132)).mean('time')

time_num =Siarea.time.dt.year.values

Siarea_trend = xr.apply_ufunc(
    linear_trend,
    Siarea,
    time_num,
    input_core_dims=[["time"], ["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float])

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-1.5,1.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array(Siarea.XC), np.array(Siarea.YC), np.array(Siarea2-Siarea1).squeeze(), levels=levels,
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
cb.set_label('Siarea', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("3_siareatrend.pdf")


coarsen_factor=1
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
data_TAUX  = data_TAUX.coarsen(XG=coarsen_factor,YC=coarsen_factor, boundary='trim').mean()




data_TAUY = xr.open_dataset("/projects/CDEUTSCH/DATA/oceTAUY_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
data_TAUY = data_TAUY.coarsen(XC=coarsen_factor,YG=coarsen_factor, boundary='trim').mean()

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


def convert_0_360_to_neg180_180(_ds):
    """
    Convert longitude from 0-360 degrees to -180 to 180 degrees.
    """
    attrs = _ds['XC'].attrs
    if (_ds['XC'].min() >= 0) and (_ds['XC'].max() <= 360):
        with xr.set_options(keep_attrs=True):
            _ds.coords['XC'] = xr.where(_ds['XC'] > 180, _ds['XC'] - 360, _ds['XC'])
        _ds = _ds.sortby('XC')
    return _ds

# Apply the conversion to your dataset
data_TAUY = convert_0_360_to_neg180_180(data_TAUY)

print('data_TAUY',data_TAUY)

print('data_TAUX',data_TAUX)

dx=(2*np.pi)/360
dy=(2*np.pi)/360

def div_4pt_xr(U, V):
    """
    POP stencil operator for divergence
    using xarray
    """
    #U_at_lat_t = U + U.roll(lat=-1, roll_coords=False)  # avg U in y
    dUdx = U.roll(XC=-1, roll_coords=False) - U.roll(XC=1, roll_coords=False)  # dU/dx
    #V_at_lon_t = V + V.roll(lon=-1, roll_coords=False)  # avg V in x
    dVdy = V.roll(YC=-1, roll_coords=False) - V.roll(YC=1, roll_coords=False)  # dV/dy
    return dUdx,dVdy

# Get the 1D lat/lon arrays
lat = TAUY_on_tracer['YC']
lon = TAUX_on_tracer['XC']

lon2d, lat2d = xr.broadcast(lon, lat)  # Each will now have shape (YG, XG)

#print('lat_wsc',lat_wsc.shape)
#zcurl, Udy, Vdx,dcos = z_curl_xr(data_TAUX.oceTAUX, data_TAUY.oceTAUY, dx, dy, lat_wsc)


dx = dy = (2 * np.pi) / 360  # degrees to radians
lat_wsc = data_TAUX.YC.values

# Step 1: Interpolate TAUY from (YG, XC) to (YC, XG)
def z_curl_xr(U, V, dx, dy, lat_wsc):
    """
    Compute z-curl of wind stress using a POP-like stencil and cosine latitude correction.

    Parameters:
    - U, V: DataArrays with dims (time, YC, XC)
    - dx, dy: grid spacing in radians
    - lat_wsc: 1D array of latitudes (deg) matching YC

    Returns:
    - zcurl: DataArray with dims (time, YC, XC)
    """
    R = 6413e3  # Earth radius [m]
    dcos = np.cos(np.deg2rad(lat_wsc))
    const2 = 1 / (R * dcos**2)

    # Wrap as DataArray
    dcos_da = xr.DataArray(dcos, coords={"YC": U.coords["YC"]}, dims=["YC"])
    const2_da = xr.DataArray(const2, coords={"YC": U.coords["YC"]}, dims=["YC"])

    # Broadcast over full (time, YC, XC)
    dcos_3d = dcos_da.broadcast_like(U)
    const2_3d = const2_da.broadcast_like(U)

    u = 0.5 * U * dx * dcos_3d
    v = 0.5 * V * dy * dcos_3d

    dVdx, dUdy = div_4pt_xr(v, u)

    zcurl = (const2_3d * (dVdx - dcos_3d * dUdy)) / (dx * dy)
    zcurl.name = "wind_stress_curl"

    return zcurl

def compute_zcurl_with_time(U, V, dx, dy, lat_wsc):
    """
    Compute z-curl across all time steps and return (time, YC, XC).
    """
    zcurl_list = []

    for t in range(U.sizes["time"]):
        zcurl_t = z_curl_xr(U.isel(time=t), V.isel(time=t), dx, dy, lat_wsc)
        zcurl_t = zcurl_t.expand_dims(time=[U.time[t].values])
        zcurl_list.append(zcurl_t)

    zcurl_all = xr.concat(zcurl_list, dim="time")
    return zcurl_all

wind_amp = np.sqrt(TAUX_on_tracer**2 + TAUY_on_tracer**2)


#zcurl = compute_zcurl_with_time(
#    data_TAUX.oceTAUX,
#    data_TAUY.oceTAUY.interp(YG=data_TAUX.YC, XC=data_TAUX.XG),
#    dx, dy,
#    data_TAUX.YC.values
#)


#time_mean_wind_amp_clim = wind_amp.groupby('time.month').mean(dim='time',skipna=True)
#time_mean_wind_amp_anom = wind_amp.groupby('time.month')-time_mean_wind_amp_clim
#time_mean_wind_amp_anom2=time_mean_wind_amp_anom/time_mean_wind_amp_anom.std()
time_mean_wind_amp_anom2=deseasonalize(wind_amp)

ds1=time_mean_wind_amp_anom2.isel(time=slice(0,36)).mean('time')
ds2=time_mean_wind_amp_anom2.isel(time=slice(132-36,132)).mean('time')

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
fill = ax.contourf(np.array((ds2-ds1).XC), np.array((ds2-ds1).YC), np.array(ds2-ds1).squeeze(), levels=levels,
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
cb.set_label('Wind Amplitude Change', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.savefig("3_windtrend.pdf")



ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/DFrIFE_bsoseI155_2013to2024_monthly.nc",
    engine="netcdf4"
)

ds = ds.isel(time=slice(0, 132))

# Coarsen spatially
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()

# Extract flux variable
flux = ds["DFrITr06"]   # (time, Zl, YC, XC)


layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": flux.time, "Zl": flux.Zl}
).transpose("time", "Zl", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": flux.time, "Zl": flux.Zl}).transpose("time","Zl","YC","XC")

flux = flux.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
flux = flux.sel(Zl=max_depth,method='nearest')
layer_thickness_4d = layer_thickness_4d.sel(Zl=max_depth, method='nearest')

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
data_diffusive_mean = flux#(flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")

data_diffusive_mean=deseasonalize(data_diffusive_mean)

ds1=data_diffusive_mean.isel(time=slice(0,36)).mean('time')
ds2=data_diffusive_mean.isel(time=slice(132-36,132)).mean('time')

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-1.6,1.6, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array((ds2-ds1).XC), np.array((ds2-ds1).YC), np.array(ds2-ds1).squeeze(), levels=levels,
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
#cb.set_label('Wind Amplitude Change', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("3_Fediffusiontrend.pdf")










ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/ADVrFE_bsoseI155_2013to2024_monthly.nc",
    engine="netcdf4"
)
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()
# Select first 132 months
ds = ds.isel(time=slice(0, 132))

# Coarsen spatially
#ds = ds.coarsen(YC=21, XC=21, boundary="trim").mean()

# Extract flux variable
flux = ds["ADVrTr06"]   # (time, Zl, YC, XC)

# -------------------------------------------------
# Compute vertical cell thickness from rLowC - rSurfC
# -------------------------------------------------
layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": flux.time, "Zl": flux.Zl}
).transpose("time", "Zl", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": flux.time, "Zl": flux.Zl}).transpose("time","Zl","YC","XC")

flux = flux.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
flux = flux.sel(Zl=max_depth, method='nearest')
layer_thickness_4d = layer_thickness_4d.sel(Zl= max_depth, method='nearest')

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
data_advection_mean = flux#(flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")


data_advection_mean=deseasonalize(data_advection_mean)

ds1=data_advection_mean.isel(time=slice(0,36)).mean('time')
ds2=data_advection_mean.isel(time=slice(132-36,132)).mean('time')

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
fill = ax.contourf(np.array((ds2-ds1).XC), np.array((ds2-ds1).YC), np.array(ds2-ds1).squeeze(), levels=levels,
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
#cb.set_label('Wind Amplitude Change', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("3_Feadvectiontrend.pdf")



ds = xr.open_dataset("/projects/CDEUTSCH/DATA/Strat_bsoseI155_2013to2023_monthly.nc", engine="h5netcdf")
print('depth',ds)
ds = ds.isel(time=slice(0, 132))

# Coarsen spatially
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()

# Extract flux variable
flux = ds["DRHODR"]   # (time, Zl, YC, XC)

# -------------------------------------------------
# Compute vertical cell thickness from rLowC - rSurfC
# -------------------------------------------------
layer_thickness = ds["rLowC"] - ds["rSurfC"]   # (YC, XC)

# Expand to 4D so it matches (time,Zl,YC,XC)
layer_thickness_4d = layer_thickness.expand_dims(
    {"time": flux.time, "Zl": flux.Zl}
).transpose("time", "Zl", "YC", "XC")

# Apply interior mask
mask = ds["maskInC"].astype(bool)
mask_4d = mask.expand_dims({"time": flux.time, "Zl": flux.Zl}).transpose("time","Zl","YC","XC")

flux = flux.where(mask_4d)

# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
# -------------------------------------------------
max_depth = -25
flux = flux.sel(Zl=slice(0, max_depth))
layer_thickness_4d = layer_thickness_4d.sel(Zl=slice(0, max_depth))

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
#data_DRHODR_mean = (flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")
data_DRHODR_mean = (flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")

data_DRHODR_mean=deseasonalize(data_DRHODR_mean)

ds1=data_DRHODR_mean.isel(time=slice(0,36)).mean('time')
ds2=data_DRHODR_mean.isel(time=slice(132-36,132)).mean('time')

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
fill = ax.contourf(np.array((ds2-ds1).XC), np.array((ds2-ds1).YC), np.array(ds2-ds1).squeeze(), levels=levels,
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
#cb.set_label('DRHODR  Change', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("3_DRHODRtrend.pdf")



ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Chl_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)

# Coarsen grid spatially (21x21)
#ds = ds.coarsen(YC=8, XC=8, boundary="trim").mean()

# Select chl-a
chl = ds.BLGCHL

# Restrict to top q m
Z_top = ds.Z.where(ds.Z >= q, drop=True)
chl = chl.sel(Z=Z_top)


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
data_Llim = (Llim * chl* drF_top * hFacC_top).sum(dim="Z")/(chl* drF_top * hFacC_top).sum(dim="Z")  # [mol C / m² / yr]

data_Llim=deseasonalize(data_Llim)

ds1=data_Llim.isel(time=slice(0,36)).mean('time')
ds2=data_Llim.isel(time=slice(132-36,132)).mean('time')

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-1.5,1.5, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array((ds2-ds1).XC), np.array((ds2-ds1).YC), np.array(ds2-ds1).squeeze(), levels=levels,
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
#cb.set_label('DRHODR  Change', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("3_Lighttrend.pdf")


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Fe_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
#ds=ds.coarsen(YC=8, XC=8, boundary='trim').mean()
C = ds["TRAC06"]
dCdt=C.roll(time=-1, roll_coords=False) - C.roll(time=1, roll_coords=False)

Z_top = dCdt.Z.where(dCdt.Z >= q, drop=True)
dCdt = dCdt.sel(Z=Z_top)

dCdt = dCdt.mean('Z')

# -------------------------------------------------
# Volume-weighted mean vertical flux
# -------------------------------------------------
#data_DRHODR_mean = (flux * layer_thickness_4d).sum(dim="Zl") / layer_thickness_4d.sum(dim="Zl")

dCdt=deseasonalize(dCdt)

ds1=dCdt.isel(time=slice(0,36)).mean('time')
ds2=dCdt.isel(time=slice(132-36,132)).mean('time')

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

ax.coastlines(resolution='110m')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#lat_formatter = cticker.LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)
levels = np.linspace(-0.8,0.8, 40)
# Contour plot with PlateCarree projection
fill = ax.contourf(np.array((ds2-ds1).XC), np.array((ds2-ds1).YC), np.array(ds2-ds1).squeeze(), levels=levels,
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
#cb.set_label('Fe tendency  Change', fontsize=20)
#emphasize = ax.contour(lon, lat, np.abs(no3_mean-no3_mean_atlas)  < 1, levels=[0.5], colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.savefig("3_Fetendencytrend.pdf")


