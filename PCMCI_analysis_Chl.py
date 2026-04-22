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
from joblib import Parallel, delayed
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from scipy.stats import t as tdist
import pandas as pd
rho_ref=1027.5
g=9.81

def partial_corr_maps(chl, feats, lags=[0], alpha=0.05):
    # chl, feats[f]: (T,W,Y)
    T, W, Y = chl.shape
    feat_names = list(feats.keys())
    maps_r = {f: np.full((len(lags), W, Y), np.nan) for f in feat_names}
    maps_p = {f: np.full((len(lags), W, Y), np.nan) for f in feat_names}
    for li, tau in enumerate(lags):
        # Align time so X at t maps to Y at t+tau
        if tau == 0:
            Yt   = chl[1:]         # (T-1, W, Y)
            Z    = chl[:-1]        # control: Chl(t)
            for f in feat_names:
                Xt = feats[f][1:]
                # residualize w.r.t Z
                ZZ = Z.reshape(T-1, -1)
                YY = Yt.reshape(T-1, -1)
                XX = Xt.reshape(T-1, -1)
                # simple scalar OLS per column (Z is 1 regressor)
                z = ZZ
                denom = np.nansum(z*z, axis=0)
                betaY = np.nansum(z*YY, axis=0) / denom
                betaX = np.nansum(z*XX, axis=0) / denom
                Ry = YY - z*betaY
                Rx = XX - z*betaX
                # correlation
                Ry -= np.nanmean(Ry, axis=0)
                Rx -= np.nanmean(Rx, axis=0)
                num = np.nansum(Rx*Ry, axis=0)
                den = np.sqrt(np.nansum(Rx*Rx, axis=0) * np.nansum(Ry*Ry, axis=0))
                r = num/den
                n = np.sum(np.isfinite(Rx) & np.isfinite(Ry), axis=0)
                df = n - 2
                tstat = r*np.sqrt(np.maximum(df,1)/(1 - r*r))
                p = 2*(1 - tdist.cdf(np.abs(tstat), np.maximum(df,1)))
                maps_r[f][li] = r.reshape(W, Y)
                maps_p[f][li] = p.reshape(W, Y)
        else:
            Yt   = chl[tau:]        # (T-tau, W, Y)
            Xt_l = {f: feats[f][:-tau] for f in feat_names}
            Z    = chl[tau-1:-1]    # control: Chl(t+tau-1)
            ZZ = Z.reshape(T-tau, -1)
            YY = Yt.reshape(T-tau, -1)
            denomZ = np.nansum(ZZ*ZZ, axis=0)
            betaY  = np.nansum(ZZ*YY,  axis=0)/denomZ
            Ry     = YY - ZZ*betaY
            for f in feat_names:
                XX = Xt_l[f].reshape(T-tau, -1)
                betaX = np.nansum(ZZ*XX, axis=0)/denomZ
                Rx    = XX - ZZ*betaX
                Rx -= np.nanmean(Rx, axis=0)
                Ry2 = Ry - np.nanmean(Ry, axis=0)
                num = np.nansum(Rx*Ry2, axis=0)
                den = np.sqrt(np.nansum(Rx*Rx, axis=0) * np.nansum(Ry2*Ry2, axis=0))
                r = num/den
                n = np.sum(np.isfinite(Rx) & np.isfinite(Ry2), axis=0)
                df = n - 2
                tstat = r*np.sqrt(np.maximum(df,1)/(1 - r*r))
                p = 2*(1 - tdist.cdf(np.abs(tstat), np.maximum(df,1)))
                maps_r[f][li] = r.reshape(W, Y)
                maps_p[f][li] = p.reshape(W, Y)
    return maps_r, maps_p



from tigramite import plotting as tp


from tigramite import plotting as tp
import matplotlib.pyplot as plt
import numpy as np

def plot_causality_graph_from_maps(
    val_maps, p_maps, feat_names,
    alpha=0.05, title="", outfile=None
):
    """
    Circular causality graph (drivers → Chl)
    Compatible with older Tigramite versions.
    """

    var_names = feat_names + ["Chl"]
    N = len(var_names)
    chl_idx = var_names.index("Chl")

    # -------------------------------------------------
    # Build val_matrix and graph mask
    # -------------------------------------------------
    val_matrix = np.zeros((N, N, 1))
    graph      = np.zeros((N, N, 1), dtype=bool)

    for i, f in enumerate(feat_names):
        vals = val_maps[f][0]
        pval = p_maps[f][0]

        sig = (pval <= alpha) & np.isfinite(vals)
        if np.any(sig):
            mean_val = np.nanmean(vals[sig])
            val_matrix[i, chl_idx, 0] = mean_val
            graph[i, chl_idx, 0] = True

    # -------------------------------------------------
    # Node positions (REQUIRED)
    # -------------------------------------------------
    #node_pos = _circle_node_pos(N)
    
    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    out = tp.plot_graph(
    graph,
    val_matrix=val_matrix,
    var_names=var_names,
    figsize=(7, 7),
    node_size=0.4,
    )

    # --- Handle Tigramite return types safely ---
    if isinstance(out, tuple):
       fig, ax = out[0], out[1]
    else:
       fig = out
       ax = fig.axes[0]

    if title:
       ax.set_title(title, fontsize=14)

    if outfile:
       fig.savefig(outfile, bbox_inches="tight", dpi=300)
       plt.close(fig)
    



def pdens(S, theta):
    a0, a1, a2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
    b0, b1, b2, b3, b4, b5 = 5.790749e8, 3.516535e6, -4.002714e4, 2.084372e2, 5.944068e5, -9.643486e3
    c0, c1, c2, c3, c4, c5 = 1.704853e5, 7.904722e2, -7.984422, 5.140652e-2, -2.302158e2, -3.079464
    p = 100000.
    alpha0 = a0 + a1 * theta + a2 * S
    p0 = b0 + b1 * theta + b2 * theta**2 + b3 * theta**3 + b4 * S + b5 * theta * S
    lambd = c0 + c1 * theta + c2 * theta**2 + c3 * theta**3 + c4 * S + c5 * theta * S
    return (p + p0) / (lambd + alpha0 * (p + p0))


def linear_trend(y, x):
    """Return slope of y vs x"""
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return slope

def deseasonalize(ds):
    clim = ds.groupby("time.month").mean("time")
    #clim=clim.coarsen(YC=2, XC=2, boundary='trim').mean()
    ds1=ds.groupby("time.month") - clim
    ds1=ds1.coarsen(YC=12, XC=12, boundary='trim').mean()
    return ds1/ds1.std('time')


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Chl_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
#ds = ds.coarsen(YC=16, XC=16, boundary="trim").mean()

# Coarsen grid spatially (21x21)
#ds = ds.coarsen(YC=21, XC=21, boundary="trim").mean()

# Select chl-a
chl = ds.BLGCHL

# Restrict to top q m
Z_top = ds.Z.where(ds.Z >= q, drop=True)
chl = chl.sel(Z=Z_top)
drF = ds.drF.sel(Z=Z_top)
hFacC = ds.hFacC.sel(Z=Z_top)

# Volume-weighted vertical integration
chl_weighted = (chl * drF * hFacC).sum(dim="Z") / (drF * hFacC).sum(dim="Z")
chl_weighted=deseasonalize(chl_weighted)
#chl_weighted=chl_weighted.coarsen(YC=1, XC=1, boundary='trim').mean()

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

lat = TAUY_on_tracer['YC']
lon = TAUX_on_tracer['XC']

# Create 2D lat-lon meshgrid using xarray broadcasting
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

wind_amp = np.sqrt(TAUX_on_tracer**2 + TAUY_on_tracer**2)
#wind_amp =wind_amp.coarsen(XC=coarsen_factor,YC=coarsen_factor, boundary='trim').mean()

#zcurl = compute_zcurl_with_time(
#    data_TAUX.oceTAUX,
#    data_TAUY.oceTAUY.interp(YG=data_TAUX.YC, XC=data_TAUX.XG),
#    dx, dy,
#    data_TAUX.YC.values
#)
time_mean_wind_amp_anom2=deseasonalize(wind_amp)

lon_grid = time_mean_wind_amp_anom2.XC.values
lat_grid = time_mean_wind_amp_anom2.YC.values


ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/SIArea_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
ds=ds.isel(time=slice(0,132))
#ds=ds.coarsen(YC=16, XC=16, boundary='trim').mean()
C = ds["SIarea"]

#C = C.Z.where(C.Z >= q, drop=True)
Siarea=deseasonalize(C)
#Siarea=Siarea.coarsen(YC=1, XC=1, boundary='trim').mean()


ds_thetao = xr.open_dataset("/projects/CDEUTSCH/DATA/Theta_bsoseI155_2013to2023_monthly.h5", engine="h5netcdf")
ds_salt = xr.open_dataset("/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc", engine="netcdf4")
# Compute N from potential density
pt = xr.apply_ufunc(pdens, ds_salt.SALT, ds_thetao.THETA, dask='parallelized', output_dtypes=[ds_salt.SALT.dtype])
anom_density = pt - rho_ref
dUdz = anom_density.roll(Z=-1) - anom_density.roll(Z=1)
N = (-(g / rho_ref) * dUdz)
#N=N.coarsen(YC=16, XC=16, boundary='trim').mean()
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
#N=N.coarsen(YC=1, XC=1, boundary='trim').mean()

ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Salt_bsoseI155_2013to2023_monthly.nc",
    engine="netcdf4"
)
#ds=ds.coarsen(YC=16, XC=16, boundary='trim').mean()
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
#data_salt_mean =data_salt_mean.coarsen(YC=1, XC=1, boundary='trim').mean()

ds1 = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/DFrIFE_bsoseI155_2013to2024_monthly.nc",
    engine="netcdf4"
)
#ds1=ds1.coarsen(YC=16, XC=16, boundary='trim').mean()
ds1 = ds1.isel(time=slice(0, 132))
Fz = ds1["DFrITr06"]
flux = Fz.sel(Zl=-25, method="nearest")
flux=deseasonalize(flux)
flux=flux.coarsen(YC=1, XC=1, boundary='trim').mean()

data_MLD = xr.open_dataset("/projects/CDEUTSCH/DATA/MLD_bsoseI156_2013to2024_monthly.nc", engine="netcdf4")
print('data_MLD',data_MLD)
seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000
#data_MLD=data_MLD.coarsen(YC=16, XC=16, boundary='trim').mean()
data_MLD=data_MLD.isel(time=slice(0,132))
data_MLD=data_MLD.BLGMLD
data_MLD = deseasonalize(data_MLD)
data_MLD=data_MLD.coarsen(YC=1, XC=1, boundary='trim').mean()

ds = xr.open_dataset(
    "/projects/CDEUTSCH/DATA/Chl_bsoseI155_2013to2023_monthly.h5",
    engine="h5netcdf"
)
#ds = ds.coarsen(YC=16, XC=16, boundary="trim").mean()

# Coarsen grid spatially (21x21)
#ds = ds.coarsen(YC=21, XC=21, boundary="trim").mean()

# Select chl-a
chl = ds.BLGCHL

# Restrict to top q m
Z_top = ds.Z.where(ds.Z >= q, drop=True)
chl = chl.sel(Z=Z_top)
drF = ds.drF.sel(Z=Z_top)
hFacC = ds.hFacC.sel(Z=Z_top)

# Volume-weighted vertical integration
chl = (chl * drF * hFacC).sum(dim="Z") / (drF * hFacC).sum(dim="Z")


data_Llim  = xr.open_dataset("/projects/CDEUTSCH/DATA/Llim_bsoseI156_2013to2024_monthly.nc", engine="netcdf4")
#data_Llim=data_Llim.coarsen(YC=16, XC=16, boundary='trim').mean()
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
#data_Llim=data_Llim.coarsen(YC=1, XC=1, boundary='trim').mean()

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
# -------------------------------------------------
data_THETA_mean = (THETA * layer_thickness_4d).sum(dim="Z") / layer_thickness_4d.sum(dim="Z")


# -------------------------------------------------
# Restrict depth range (e.g. top 100 m)
data_THETA_mean = deseasonalize(data_THETA_mean)
#data_THETA_mean =data_THETA_mean.coarsen(YC=1, XC=1, boundary='trim').mean()

feats = {
    "wind":  time_mean_wind_amp_anom2.values,
    "salt":data_salt_mean.values,
    "N":N.values,
    "MLD":data_MLD.values,
    "Theta":data_THETA_mean.values
}
chl = chl_weighted.values

print("Chl shape:", chl.shape)
for k, v in feats.items():
    print(k, v.shape)

maps_r, maps_p = partial_corr_maps(chl, feats, lags=[0], alpha=0.05)

lon_grid = data_MLD.XC.values
lat_grid = data_MLD.YC.values
alpha_level = 0.05


import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import matplotlib.ticker as mticker

alpha_level = 0.05
lags_to_plot = [0]
features_to_plot = ["wind","salt","N","MLD","Theta"]
for li, tau in enumerate(lags_to_plot):
    for f in features_to_plot:
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

        # Add coastlines and gridlines
        ax.coastlines(resolution='110m')
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        # Extract significant partial correlations
        vals = maps_r[f][li]
        pval = maps_p[f][li]
        plot_data = np.where(pval <= alpha_level, vals, np.nan)

        # Contour levels
        levels = np.linspace(-0.7, 0.7, 40)

        # Filled contour plot in polar projection
        fill = ax.contourf(
            lon_grid, lat_grid, plot_data,
            levels=levels,
            cmap=plt.cm.RdBu_r,
            transform=ccrs.PlateCarree(),
            extend='both'
        )

        ax.set_aspect('equal')

        # Colorbar formatting
        cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
        cb.ax.tick_params(labelsize=16)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        cb.set_label(f'{f} → Chl partial corr (lag {tau})', fontsize=16, labelpad=15)

        # Save figure
        plt.savefig(f"pcorr_map_{f}_lag{tau}_polar_"+str(q)+".pdf", dpi=300, bbox_inches='tight')
        plt.close()


# sensitivity analysis
from joblib import Parallel, delayed
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc import GPDC
import numpy as np

def save_maps(val_maps, p_maps, outdir, tag="", lon2d=None, lat2d=None):
    os.makedirs(outdir, exist_ok=True)
    for f in val_maps.keys():
        field = val_maps[f][0]
        pval  = p_maps[f][0]
        sig   = np.where(pval <= 0.05, field, np.nan)

        fname = os.path.join(outdir, f"{f}{tag}.pdf")
        plot_map(
            sig,
            title=f"{f}{tag}",
            fname=fname,
            lon2d=lon2d,
            lat2d=lat2d
        )


from tigramite.independence_tests.gpdc import GPDC
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from joblib import Parallel, delayed
import numpy as np


def bootstrap_ci(data, n_boot=2000, ci=95):
    """
    Spatial bootstrap (longitude resampling)
    Returns mean, lower, upper CI
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if len(data) < 5:
        return np.nan, np.nan, np.nan

    means = np.empty(n_boot)

    for i in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)

    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, 100-(100-ci)/2)

    return np.mean(data), lower, upper

def collect_regional_values(val_maps, p_maps, lat2d, alpha=0.05):

    regions = {
        "Subtropics": (lat2d > -45) & (lat2d <= -30),
        "ACC":        (lat2d > -60) & (lat2d <= -45),
        "HighLat":    (lat2d <= -60),
    }

    regional_values = {r:{} for r in regions}

    for region, mask in regions.items():
        for drv in val_maps.keys():

            vals = val_maps[drv][0]
            pval = p_maps[drv][0]

            good = (pval <= alpha) & np.isfinite(vals) & mask
            regional_values[region][drv] = vals[good]

    return regional_values

def plot_bootstrap_significance(seasonal_maps, lat2d, outfile):

    seasons = list(seasonal_maps.keys())
    drivers = list(next(iter(seasonal_maps.values())).keys())

    width = 0.12
    x = np.arange(len(seasons))

    plt.figure(figsize=(12,6), dpi=200)

    for i, drv in enumerate(drivers):

        means, lo_err, hi_err = [], [], []

        for s in seasons:
            data = seasonal_maps[s][drv]
            m, lo, hi = bootstrap_ci(data)

            means.append(m)
            lo_err.append(m-lo)
            hi_err.append(hi-m)

        yerr = np.vstack([lo_err, hi_err])

        plt.bar(
            x + i*width,
            means,
            width,
            yerr=yerr,
            capsize=4,
            label=drv
        )
    
    #plt.xticks(x + width*(len(drivers)-1)/2, seasons)
    center_positions = x + width*(len(drivers)-1)/2
    plt.xticks(center_positions, seasons, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Causal influence (PCMCI)", fontsize=18)
    plt.xlabel("Season", fontsize=18) 
    plt.title("Bootstrap spatial confidence intervals (ACC)")
    plt.grid(alpha=0.3)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),   # below plot
        ncol=len(drivers),              # horizontal
        fontsize=14,
        frameon=False
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def run_pcmci_driver(
    features,
    chl,
    alpha=0.05,
    min_samples=12,
    tau_max=2,
    verbose=True
):
    """
    Run PCMCI with explicit lag dependence (tau = 0..tau_max).

    Returns
    -------
    val_maps, p_maps : dict
        val_maps[driver][tau, YC, XC]
        p_maps[driver][tau, YC, XC]
    """

    feat_names = list(features.keys())
    T, W, Y = chl.shape

    # --------------------------------------------------
    # Allocate output maps
    # --------------------------------------------------
    val_maps = {
        f: np.full((tau_max + 1, W, Y), np.nan)
        for f in feat_names
    }
    p_maps = {
        f: np.full((tau_max + 1, W, Y), np.nan)
        for f in feat_names
    }

    # --------------------------------------------------
    # Per-gridcell PCMCI
    # --------------------------------------------------
    def run_cell(i, j):
        try:
            A = np.column_stack(
                [features[f][:, i, j] for f in feat_names] +
                [chl[:, i, j]]
            )

            mask = np.all(np.isfinite(A), axis=1)
            A = A[mask]

            if A.shape[0] < min_samples:
                return None, None, i, j

            df = DataFrame(A, var_names=feat_names + ["Chl"])
            chl_idx = len(feat_names)

            pcmci = PCMCI(
                dataframe=df,
                cond_ind_test=GPDC(significance="analytic")
            )

            res = pcmci.run_pcmci(
                tau_min=0,
                tau_max=tau_max,
                pc_alpha=alpha
            )

            vals = res["val_matrix"]
            pval = res["p_matrix"]

            out_vals = {}
            out_pval = {}

            for fi, f in enumerate(feat_names):
                out_vals[f] = vals[fi, chl_idx, :]   # (tau,)
                out_pval[f] = pval[fi, chl_idx, :]   # (tau,)

            return out_vals, out_pval, i, j

        except Exception as e:
            if verbose:
                print(f"⚠️ PCMCI failed at ({i},{j}): {e}")
            return None, None, i, j

    # --------------------------------------------------
    # Parallel execution
    # --------------------------------------------------
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_cell)(i, j)
        for i in range(W)
        for j in range(Y)
    )

    # --------------------------------------------------
    # Fill maps
    # --------------------------------------------------
    for out_vals, out_pval, i, j in results:
        if out_vals is None:
            continue

        for f in feat_names:
            val_maps[f][:, i, j] = out_vals[f]
            p_maps[f][:, i, j]   = out_pval[f]

    return val_maps, p_maps


from statsmodels.stats.multitest import multipletests

drivers = {
    "PCMCI": run_pcmci_driver,
    #"Granger": run_granger_driver,
    #"VAR": run_var_driver,
}


seasons = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
#seasons = {"DJF":[12,1,2]}

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker


def get_region_masks(lat2d):
    """
    Define Southern Ocean dynamical regimes based on latitude.
    lat2d: 2D latitude array (YC, XC)
    """
    masks = {
        "Subtropics": (lat2d > -40),
        "ACC":        (lat2d <= -40) & (lat2d > -55),
        "HighLat":    (lat2d <= -55),
    }
    return masks

def compute_regional_causal_means(
    val_maps, p_maps, lat2d, alpha=0.05
):
    """
    Compute mean causal influence per driver and per region.
    Returns dict: region → driver → mean value
    """
    region_masks = get_region_masks(lat2d)
    out = {}

    for region, rmask in region_masks.items():
        out[region] = {}

        for f in val_maps.keys():
            vals = val_maps[f][0]
            pval = p_maps[f][0]

            good = (pval <= alpha) & np.isfinite(vals) & rmask

            if np.any(good):
                out[region][f] = np.nanmean(vals[good])
            else:
                out[region][f] = np.nan

    return out

def plot_regional_causal_importance(regional_stats, outfile):
    """
    Bar plot comparing regional mean causal influence.
    """
    regions = list(regional_stats.keys())
    drivers = list(next(iter(regional_stats.values())).keys())

    x = np.arange(len(regions))
    width = 0.12

    plt.figure(figsize=(10, 5), dpi=200)

    for i, drv in enumerate(drivers):
        vals = [regional_stats[r][drv] for r in regions]
        plt.bar(x + i * width, vals, width, label=drv)

    plt.xticks(x + width * (len(drivers) - 1) / 2, regions)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("Mean causal influence (PCMCI)", fontsize=14)
    plt.title("Regional causal influence on Chl", fontsize=16)
    plt.legend(ncol=3, fontsize=11)
    plt.grid(axis="y", alpha=0.3)

    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close()



def plot_map(field, title, fname, vmin=-0.6, vmax=0.6, cmap=plt.cm.RdBu_r,
             lon2d=None, lat2d=None):
    """Polar projection plot with Cartopy for lag 0 only."""
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax  = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m')
    ax.gridlines(draw_labels=True)
    fill = ax.contourf(lon2d, lat2d, field,
                       levels=np.linspace(vmin, vmax, 40),
                       cmap=cmap, transform=ccrs.PlateCarree(), extend='both')

    cb = plt.colorbar(fill, orientation='vertical', shrink=0.8)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    cb.ax.tick_params(labelsize=20)
    cb.set_label(title, fontsize=16)
    #plt.title(title, fontsize=16)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def compute_seasonal_mean_importance(val_maps, p_maps, alpha=0.05):
    """
    Compute spatial mean PCMCI influence for each driver in a season.
    Returns {driver: mean influence}.
    """
    mean_vals = {}
    for f in val_maps.keys():
        vals = val_maps[f][0]     # (W, Y)
        pval = p_maps[f][0]
        sig = np.where(pval <= alpha, vals, np.nan)
        mean_vals[f] = np.nanmean(sig)
    return mean_vals

def compute_seasonal_stats(val_maps, p_maps, alpha=0.05):
    """
    Robust seasonal stats:
      - mean/std of significant values only
      - n_sig: number of significant finite gridpoints
      - frac_sig: fraction significant among finite p-values
    """
    out = {}
    for f in val_maps.keys():
        vals = val_maps[f][0]   # (W, Y)
        pval = p_maps[f][0]

        good = np.isfinite(vals) & np.isfinite(pval)
        sig  = good & (pval <= alpha)

        n_good = int(good.sum())
        n_sig  = int(sig.sum())

        if n_sig == 0:
            out[f] = {"mean": np.nan, "std": np.nan, "n_sig": 0, "frac_sig": 0.0}
        else:
            vv = vals[sig]
            out[f] = {
                "mean": float(np.nanmean(vv)),
                "std":  float(np.nanstd(vv)),
                "n_sig": n_sig,
                "frac_sig": (n_sig / n_good) if n_good > 0 else np.nan
            }
    return out

def compute_seasonal_regional_stats(
    val_maps,
    p_maps,
    lat2d,
    alpha=0.05
):
    """
    Compute regional PCMCI statistics for a given season.

    Regions (Southern Ocean):
      - Subtropics:  -40 < lat <= -30
      - ACC:         -55 < lat <= -40
      - HighLat:     lat <= -55

    Parameters
    ----------
    val_maps : dict
        PCMCI value maps, val_maps[driver][0, YC, XC]
    p_maps : dict
        PCMCI p-value maps, p_maps[driver][0, YC, XC]
    lat2d : ndarray
        2D latitude array (YC, XC)
    alpha : float
        Significance threshold

    Returns
    -------
    regional_stats : dict
        regional_stats[region][driver] = {
            "mean": float,
            "std": float,
            "n_sig": int,
            "frac_sig": float
        }
    """

    sample_driver = next(iter(val_maps))
    ny, nx = val_maps[sample_driver][0].shape

    lat = np.asarray(lat2d)
    if lat.ndim == 1:
        # lat shape (YC,) → expand to (YC, XC)
        lat = lat[:, None] * np.ones((1, nx))
    elif lat.ndim != 2:
        raise ValueError("lat2d must be 1D or 2D")

    # ------------------------------------------------------------
    # Define regions
    # ------------------------------------------------------------
    regions = {
        "Subtropics": (lat > -45) & (lat <= -30),
        "ACC":        (lat > -60) & (lat <= -45),
        "HighLat":    (lat <= -60),
    }

    regional_stats = {r: {} for r in regions}

    # ------------------------------------------------------------
    # Loop over regions and drivers
    # ------------------------------------------------------------
    for region, rmask in regions.items():

        for driver in val_maps.keys():

            vals = val_maps[driver][0]   # (YC, XC)
            pval = p_maps[driver][0]

            # Valid data inside region
            good = (
                np.isfinite(vals) &
                np.isfinite(pval) &
                rmask
            )

            sig = good & (pval <= alpha)

            n_good = int(good.sum())
            n_sig  = int(sig.sum())

            if n_sig == 0:
                regional_stats[region][driver] = {
                    "mean": np.nan,
                    "std":  np.nan,
                    "n_sig": 0,
                    "frac_sig": 0.0
                }
            else:
                vv = vals[sig]
                regional_stats[region][driver] = {
                    "mean": float(np.nanmean(vv)),
                    "std":  float(np.nanstd(vv)),
                    "n_sig": n_sig,
                    "frac_sig": (n_sig / n_good) if n_good > 0 else np.nan
                }

    return regional_stats


import matplotlib.pyplot as plt


def plot_seasonal_mean_importance(seasonal_stats, outfile):
    seasons = list(seasonal_stats.keys())
    drivers = list(next(iter(seasonal_stats.values())).keys())

    colors = plt.cm.Set1(np.linspace(0, 1, len(drivers)))
    
    driver_colors = {
    "MLD":   "#1f77b4",  # blue
    "salt":  "#9467bd",  # purple
    "N":     "#2ca02c",  # green
    "wind":  "#ff7f0e",  # orange  ✅
    "Theta": "#b15928",  # brown
    }
    colors = [driver_colors[d] for d in drivers]

    plt.figure(figsize=(12, 6), dpi=200)

    for ci, drv in enumerate(drivers):
        means = [seasonal_stats[s][drv]["mean"] for s in seasons]
        stds  = [seasonal_stats[s][drv]["std"]  for s in seasons]

        #plt.errorbar(
        #    seasons, means, yerr=stds,
        #    fmt="o", lw=2, ms=9, capsize=6,
        #    color=colors[ci], label=drv
        #)
        plt.plot(
            seasons, means,
            marker="o", lw=2, ms=9,
            color=colors[ci], label=drv
        )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Mean causal influence (PCMCI)", fontsize=18)
    plt.xlabel("Season", fontsize=18)
    plt.title("Seasonal causal influence of drivers on Chl", fontsize=18)
    plt.grid(alpha=0.3)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),   # below plot
        ncol=len(drivers),              # horizontal
        fontsize=14,
        frameon=False
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()



def plot_driver_lag_comparison(seasonal_stats_by_lag, outdir):
    """
    Creates one figure per driver.
    Each figure contains one line per lag (tau),
    showing seasonal mean causal influence.
    """

    os.makedirs(outdir, exist_ok=True)

    # Get lags
    taus = sorted(seasonal_stats_by_lag.keys())

    # Get seasons from first lag
    seasons = list(next(iter(seasonal_stats_by_lag.values())).keys())

    # Get drivers
    drivers = list(
        next(iter(next(iter(seasonal_stats_by_lag.values())).values())).keys()
    )

    # Colors for lags
    lag_colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))

    for drv in drivers:

        plt.figure(figsize=(8, 6), dpi=200)

        for i, tau in enumerate(taus):

            means = [
                seasonal_stats_by_lag[tau][s][drv]["mean"]
                for s in seasons
            ]

            plt.plot(
                seasons,
                means,
                marker="o",
                lw=2.5,
                ms=8,
                color=lag_colors[i],
                label=f"Lag {tau}"
            )
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("Mean causal influence (PCMCI)", fontsize=16)
        plt.xlabel("Season", fontsize=16)
        plt.title(f"{drv} – Seasonal influence by lag", fontsize=18)
        plt.grid(alpha=0.3)
        plt.legend(title="Lag", fontsize=12)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        outfile = os.path.join(
            outdir,
            f"{drv}_lag_comparison.pdf"
        )

        plt.savefig(outfile, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"[Saved] {outfile}")



def bootstrap_ci(data, n_boot=2000, ci=95):
    """
    Bootstrap CI across spatial samples (e.g. longitudes).
    data : 1d array-like of spatial samples (may contain NaNs)
    returns: mean, lower, upper
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if data.size < 5:
        return np.nan, np.nan, np.nan

    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(data, size=data.size, replace=True)
        means[i] = np.mean(sample)

    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper


def collect_regional_values(val_maps, p_maps, lat2d, alpha=0.05, region="ACC"):
    """
    Extract significant spatial samples for each driver inside `region`.
    val_maps : dict driver -> array (tau, YC, XC)  (we use tau=0 slice)
    p_maps   : dict driver -> array (tau, YC, XC)
    lat2d    : 1D or 2D lat array (YC or YCxXC)
    region   : "Subtropics" | "ACC" | "HighLat"
    returns dict: driver -> 1d array of values (samples)
    """
    lat = np.asarray(lat2d)
    # expand 1D lat to 2D (YC, XC)
    if lat.ndim == 1:
        # need NX; pick shape from one val_map
        sample_driver = next(iter(val_maps))
        ny, nx = val_maps[sample_driver][0].shape
        lat = lat[:, None] * np.ones((1, nx))
    if lat.ndim != 2:
        raise ValueError("lat2d must be 1D or 2D")

    regions = {
        "Subtropics": (lat > -45) & (lat <= -30),
        "ACC":        (lat > -60) & (lat <= -45),
        "HighLat":    (lat <= -60),
    }

    mask = regions[region]

    samples = {}
    for drv in val_maps.keys():
        vals = val_maps[drv][0]   # shape (YC, XC)
        pval = p_maps[drv][0]
        good = (pval <= alpha) & np.isfinite(vals) & mask
        samples[drv] = vals[good]  # might be empty array
    return samples


def run_sensitivity(
    method,
    driver,
    features,
    chl,
    seasons=None,
    alpha=0.05,
    outdir="results",
    lon2d=None,
    lat2d=None,
    tau_max=2
):
    """
    Lag-aware PCMCI sensitivity analysis with PANEL B bootstrap CI for all regions.
    Replaces previous run_sensitivity; requires helpers:
      - save_maps(val_maps, p_maps, outdir, tag, lon2d, lat2d)
      - compute_seasonal_stats(val_maps, p_maps, alpha)
      - compute_seasonal_regional_stats(val_maps, p_maps, lat2d, alpha)
      - plot_seasonal_mean_importance(seasonal_stats, outfile)
      - plot_driver_lag_comparison(seasonal_stats_by_lag, outdir)
      - plot_bootstrap_significance(seasonal_bootstrap, outfile)
    """
    feat_names = list(features.keys())
    os.makedirs(outdir, exist_ok=True)

    print(f"\n--- Running {method} sensitivity analysis (tau=0..{tau_max}) ---")

    # 1) FULL-YEAR PCMCI
    print("→ Running full-year PCMCI")
    val_full, p_full = driver(
        features=features,
        chl=chl,
        alpha=alpha,
        tau_max=tau_max
    )

    for tau in range(tau_max + 1):
        lag_dir = os.path.join(outdir, f"full_lag{tau}")
        os.makedirs(lag_dir, exist_ok=True)

        val_tau = {f: val_full[f][tau:tau+1] for f in feat_names}
        p_tau   = {f: p_full[f][tau:tau+1]   for f in feat_names}

        save_maps(val_tau, p_tau, lag_dir, tag=f"_LAG{tau}", lon2d=lon2d, lat2d=lat2d)

    # 2) SEASONAL PCMCI
    if seasons is None:
        print("[INFO] No seasons provided → skipping seasonal analysis")
        return

    ntime = chl.shape[0]
    months = (np.arange(ntime) % 12) + 1

    season_dir = os.path.join(outdir, "seasons")
    os.makedirs(season_dir, exist_ok=True)

    seasonal_stats_global   = {tau: {} for tau in range(tau_max + 1)}
    seasonal_stats_regional = {tau: {"Subtropics": {}, "ACC": {}, "HighLat": {}} for tau in range(tau_max + 1)}

    # NEW: store raw maps for panel-B bootstrap (per tau → per season)
    # structure: bootstrap_by_lag[tau][sname] = {"val": val_tau, "p": p_tau}
    bootstrap_by_lag = {tau: {} for tau in range(tau_max + 1)}

    for sname, smonths in seasons.items():

        idx = np.where(np.isin(months, smonths))[0]
        if idx.size < 12:
            print(f"[WARNING] Season {sname} has only {idx.size} months → skipping.")
            continue

        print(f"→ Season {sname}: {idx.size} months")

        feat_sub = {k: v[idx] for k, v in features.items()}
        chl_sub  = chl[idx]

        val_s, p_s = driver(
            features=feat_sub,
            chl=chl_sub,
            alpha=alpha,
            tau_max=tau_max
        )

        for tau in range(tau_max + 1):

            tau_dir = os.path.join(season_dir, sname, f"lag{tau}")
            os.makedirs(tau_dir, exist_ok=True)

            val_tau = {f: val_s[f][tau:tau+1] for f in feat_names}
            p_tau   = {f: p_s[f][tau:tau+1]   for f in feat_names}

            # Save spatial maps for this season+lag
            save_maps(val_tau, p_tau, tau_dir, tag=f"_{sname}_LAG{tau}", lon2d=lon2d, lat2d=lat2d)

            # GLOBAL seasonal stats
            seasonal_stats_global[tau][sname] = compute_seasonal_stats(val_tau, p_tau, alpha)

            # REGIONAL seasonal stats
            regional_stats = compute_seasonal_regional_stats(val_tau, p_tau, lat2d=lat2d, alpha=alpha)
            for region in seasonal_stats_regional[tau]:
                seasonal_stats_regional[tau][region][sname] = regional_stats[region]

            # STORE raw val/p for Panel B (useful later to extract any region)
            bootstrap_by_lag[tau][sname] = {"val": val_tau, "p": p_tau}

    # 3) SEASONAL MEAN IMPORTANCE FIGURES
    print("→ Plotting seasonal mean importance")
    for tau in range(tau_max + 1):
        outfile = os.path.join(outdir, f"seasonal_mean_importance_GLOBAL_LAG{tau}.pdf")
        plot_seasonal_mean_importance(seasonal_stats_global[tau], outfile)
        for region, stats in seasonal_stats_regional[tau].items():
            outfile = os.path.join(outdir, f"seasonal_mean_importance_{region}_LAG{tau}.pdf")
            plot_seasonal_mean_importance(stats, outfile)

    # 4) LAG COMPARISON
    print("→ Plotting lag comparison per driver")
    plot_driver_lag_comparison(seasonal_stats_global, outdir=outdir)
    for region in seasonal_stats_regional[0].keys():
        regional_lag_stats = {tau: seasonal_stats_regional[tau][region] for tau in seasonal_stats_regional}
        plot_driver_lag_comparison(regional_lag_stats, outdir=outdir)

    # 5) PANEL B — BOOTSTRAP SIGNIFICANCE BARS FOR ALL REGIONS
    print("→ Plotting bootstrap significance bars (Subtropics, ACC, HighLat)")

    regions = ["Subtropics", "ACC", "HighLat"]
    for tau in range(tau_max + 1):
        for region in regions:
            # build seasonal dict: season -> driver->samples
            seasonal_bootstrap = {}
            for sname in bootstrap_by_lag[tau].keys():
                stored = bootstrap_by_lag[tau][sname]
                seasonal_bootstrap[sname] = collect_regional_values(
                    stored["val"], stored["p"], lat2d, alpha, region=region
                )

            outfile = os.path.join(outdir, f"{region}_bootstrap_significance_LAG{tau}.pdf")
            plot_bootstrap_significance(seasonal_bootstrap, lat2d, outfile)
            print(f"[SAVED] {outfile}")

    print("→ run_sensitivity finished.")


import scipy.io as io

#data = io.loadmat("Chl_trend_significant.mat")
#lon        = data["lon"].squeeze()
#lat        = data["lat"].squeeze()
#chl_trend = data["chl_trend"].squeeze()
#p_value    = data["p_value"].squeeze()



#lon_grid = data_MLD.XC.values
#lat_grid = data_MLD.YC.values

for method, driver in drivers.items():
    run_sensitivity(
    method=method,
    driver=driver,
    features=feats,
    chl=chl_weighted.values,
    seasons=seasons,
    alpha=0.05,
    outdir=f"results_noflux_{method}",
    lon2d=lon_grid,
    lat2d=lat_grid
    )



