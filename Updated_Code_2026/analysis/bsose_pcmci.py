"""Causal discovery (PCMCI+, PCMCI or LPCMCI) on B-SOSE.

    python bsose_pcmci.py                 B-SOSE, complete, PCMCI+
    METHOD=PCMCI python bsose_pcmci.py    the earlier PCMCI
    METHOD=LPCMCI python bsose_pcmci.py   allows for hidden confounders
    ANOM_ONLY=1 python bsose_pcmci.py     only save the block anomalies
                                          (anomalies.npz) and stop
    TEST=parcorr python bsose_pcmci.py    linear test (fast) in place of
                                          GPDC (nonlinear, slow)
    SAMPLED=1 python bsose_pcmci.py       B-SOSE chlorophyll kept only
                                          where and when the satellite
                                          has a value
    SATELLITE=1 python bsose_pcmci.py     the satellite's chlorophyll
                                          itself, on the B-SOSE grid,
                                          with the B-SOSE drivers
    DETREND=1 python bsose_pcmci.py       linear trends removed from
                                          every series
    COARSEN=6 python bsose_pcmci.py       6 x 6 blocks in place of 4 x 4
    TARGET=MLD python bsose_pcmci.py      the mixed-layer depth as the
                                          target, drivers wind, surfTflx,
                                          N, surfSflx, Theta, oceQsw

THE FIELDS are exactly those of fig1_chl_trend.py and
plot_driver_trends.py, so the causal maps and the trend figures rest on
the same numbers:

  chlorophyll  BLGCHL over the top 25 m, weighted by drF x hFacC
  Theta, salt  THETA, SALT over the top 25 m, weighted by drF x hFacC
  N            -(g / rho0) (sigma0(0-25 m) - sigma0(25-50 m)) / 25 m,
               the two layers weighted as above
  MLD          BLGMLD
  wind         wind-stress amplitude, oceTAUX and oceTAUY each moved to
               the cell centre by the mean of the two faces around it
               (WIND_DRIVER=wek: Ekman pumping from the same stresses)
  surfTflx, surfSflx, oceQsw   B-SOSE surface heat flux, salt flux and
               shortwave radiation (TARGET=MLD only)

THE ANOMALIES are the earlier deseasonalize, in its order, with xarray:

  clim = ds.groupby("time.month").mean("time")
  ds1  = ds.groupby("time.month") - clim
  ds1  = ds1.coarsen(YC=COARSEN, XC=COARSEN, boundary="trim").mean()
  (DETREND=1: linear trend removed here)
  ds1 / ds1.std("time")

with COARSEN = 4 (about 0.7 deg), the blocks of the trend figures.

THE CAUSAL STEP, per block, tau = 0..2, alpha = 0.05:

  * METHOD=PCMCIplus (default): PCMCI+ (Runge 2020). Like PCMCI for the
    lagged links, and it also tests and orients the CONTEMPORANEOUS
    (lag-0) links, which PCMCI only scores. A driver counts as a cause
    of chlorophyll at a lag when the graph has driver --> Chl there;
    p is set to 1 elsewhere, so every figure below reads the graph.
    CONTEMP=1 also counts unoriented lag-0 links (o-o).
  * METHOD=PCMCI: the earlier analysis (lag-0 values are not oriented).
  * METHOD=LPCMCI: Gerhardus & Runge (2020); allows for unobserved
    common drivers. Counted as for PCMCI+ (only -->).
  * TEST: gpdc (default, as before), parcorr (linear partial
    correlation), robustparcorr (rank-transformed, robust to outliers
    and non-Gaussian anomalies), cmiknn (nonparametric, slowest).

Gaps are handed to tigramite as missing values (missing_flag), so a lag
never reaches across a gap: before, gap months were dropped and the
months either side became neighbours.

SEASONS use tigramite's sample mask: the full series is kept and only
the months of the season are tested as the TARGET, while the lagged
drivers come from the true preceding months (DJF with November and
October at lags 1 and 2). Before, the DJF months of all years were
glued into one series, so lags 1-2 jumped from one February to the next
December. SEASON_MASK=0 restores that.

Every finished latitude row of every pass is saved, so a job that runs
out of time continues where it stopped when submitted again; a saved
file from another method, test or setup is not reused.
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from scipy import stats
from scipy.stats import t as tdist
from joblib import Parallel, delayed
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI

q = -25
rho_ref = 1027.5
g = 9.81

# ---------------------------------------------------------------------
# Where the data are, and the two choices this script adds
# ---------------------------------------------------------------------
BSOSE_DIR = os.environ.get(
    "BSOSE_DIR", "/work/bb1596/b384036/b384036/causality/bsose")
BSOSE_NAME = "{}_bsoseI156_2013to2024_monthly.nc"
# Months used from January 2013: 144 is 2013-2024, the whole record.
# (132, the earlier analysis's 2013-2023, keeps the earlier folder names.)
NT = int(os.environ.get("NT", 144))
LAST_YEAR = 2013 + (NT - 1) // 12
WINDOW_TAG = "" if NT == 132 else f"_2013to{LAST_YEAR}"
# "wind": wind-stress amplitude, as in the earlier script.
# "wek":  Ekman pumping, curl(tau / (rho f)), positive upward.
WIND_DRIVER = os.environ.get("WIND_DRIVER", "wind")
# SAMPLED=1: model chlorophyll only where and when the satellite saw the
# ocean. The drivers are untouched.
SAMPLED = os.environ.get("SAMPLED", "0").lower() in ("1", "true", "yes")
ESA_FILE = os.environ.get(
    "ESA_FILE", os.path.join(os.path.dirname(BSOSE_DIR.rstrip("/")),
                             "ESA_Chl_south_1998_2025_0p25deg.nc"))
# DETREND=1: the linear trend of every series (chlorophyll and drivers)
# removed after the climatology, so a drift shared by two fields does not
# count as dependence. Off by default, as in the earlier analysis.
DETREND = os.environ.get("DETREND", "0").lower() in ("1", "true", "yes")
# Side of the block the fields are averaged over, in model cells: 4 by
# default (about 0.7 deg), the blocks of the trend figures. The earlier
# script used 12. The size is always in the folder name.
COARSEN = int(os.environ.get("COARSEN", 4))
# SATELLITE=1: the satellite's chlorophyll on the model grid in place of
# the model's. It takes precedence over SAMPLED.
SATELLITE = os.environ.get("SATELLITE", "0").lower() in ("1", "true", "yes")
if SATELLITE:
    SAMPLED = False
# Causal method and independence test (see the top of the file).
METHOD = {"pcmci": "PCMCI", "pcmciplus": "PCMCIplus", "pcmci+": "PCMCIplus",
          "lpcmci": "LPCMCI"}[os.environ.get("METHOD", "PCMCIplus").lower()]
TEST = os.environ.get("TEST", "gpdc").lower()
CONTEMP = os.environ.get("CONTEMP", "0") == "1"
SEASON_MASK = os.environ.get("SEASON_MASK", "1") == "1"
MISSING = -999999.0
# TARGET=MLD: the same causal analysis with the mixed-layer depth as the
# target instead of chlorophyll, with the drivers of the earlier MLD
# analysis: the wind, the surface heat flux (surfTflx), the stratification
# N, the surface salt flux (surfSflx), the temperature of the top 25 m
# (Theta) and the shortwave radiation (oceQsw).
TARGET = os.environ.get("TARGET", "Chl")
if TARGET not in ("Chl", "MLD"):
    sys.exit(f"TARGET={TARGET}: Chl or MLD")
INTERP_GAP = int(os.environ.get("INTERP_GAP", 0))
SETUP = (f"{METHOD}-{TEST}-contemp{int(CONTEMP)}-mask{int(SEASON_MASK)}")
TAG = (("" if WIND_DRIVER == "wind" else "_" + WIND_DRIVER)
       + ("_sampled" if SAMPLED else "")
       + ("_satellite" if SATELLITE else "")
       + f"_c{COARSEN}"
       + WINDOW_TAG
       + ("_detrended" if DETREND else "")
       + (f"_interp{INTERP_GAP}" if INTERP_GAP > 0 else "")
       + ("" if TEST == "gpdc" else f"_{TEST}"))
OUTDIR = os.environ.get("OUTDIR", ("results_noflux_{method}" if TARGET == "Chl"
                                   else "results_mld_{method}") + TAG)


_COAST = None


def coastlines(ax):
    """Coastlines if cartopy has them on disk; a compute node cannot
    download them, and a missing coastline should not cost the run. The
    check is made once, up front: the download failure would otherwise
    come at savefig and take the figure with it."""
    global _COAST
    if _COAST is None:
        try:
            from cartopy.io import shapereader
            shapereader.natural_earth(resolution="110m", category="physical",
                                      name="coastline")
            _COAST = True
        except Exception as exc:
            print(f"  no coastlines ({type(exc).__name__}); maps without them")
            _COAST = False
    if _COAST:
        ax.coastlines(resolution='110m')


def open_bsose(name):
    """One B-SOSE file, first NT months, on a clean monthly time axis.

    The months are counted from January 2013, the first record, and not
    read from the time stamps, so every file carries the same axis.
    """
    path = os.path.join(BSOSE_DIR, BSOSE_NAME.format(name))
    ds = xr.open_dataset(path, decode_times=False)
    ds = ds.isel(time=slice(0, NT))
    ds = ds.assign_coords(time=pd.date_range("2013-01-01",
                                             periods=ds.sizes["time"],
                                             freq="MS"))
    print(f"{name}: {os.path.basename(path)}, {ds.sizes['time']} months",
          flush=True)
    return ds


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

ANOMALIES = {}     # the coarsened anomalies before the division by the
                   # standard deviation, in physical units, for the trend maps


def fill_short_gaps(a, max_gap):
    """(T, Y, X) -> (filled copy, number filled). Interior gaps of at most
    max_gap months, with a valid month on both sides, are filled linearly
    along time; longer gaps and the ends of the record stay missing.
    Vectorized, a few latitude rows at a time to keep memory small."""
    a = np.asarray(a)
    out = np.array(a, dtype="float32", copy=True)
    T = out.shape[0]
    t = np.arange(T, dtype=np.int32)[:, None, None]
    n_filled = 0
    step = max(1, 20_000_000 // max(1, T * out.shape[2]))
    for j0 in range(0, out.shape[1], step):
        x = out[:, j0:j0 + step]
        ok = np.isfinite(x)
        last = np.maximum.accumulate(np.where(ok, t, -1), axis=0)
        nxt = np.minimum.accumulate(np.where(ok, t, T)[::-1], axis=0)[::-1]
        fill = (~ok) & (last >= 0) & (nxt < T) & (nxt - last - 1 <= max_gap)
        if fill.any():
            lo = np.take_along_axis(x, np.clip(last, 0, T - 1), axis=0)
            hi = np.take_along_axis(x, np.clip(nxt, 0, T - 1), axis=0)
            w = (t - last) / np.maximum(nxt - last, 1)
            x[fill] = (lo + (hi - lo) * w)[fill]
            n_filled += int(fill.sum())
    return out, n_filled


def deseasonalize(ds, keep=None, fill_gap=0):
    clim = ds.groupby("time.month").mean("time")
    #clim=clim.coarsen(YC=2, XC=2, boundary='trim').mean()
    ds1=ds.groupby("time.month") - clim
    if fill_gap > 0:
        # Missing months filled in time in the anomalies, on the native
        # grid, before the block average (INTERP_GAP); the climatology
        # above is from the observed months only.
        ds1 = ds1.transpose("time", "YC", "XC")
        n_obs = int(np.isfinite(ds1.values).sum())
        filled, n_f = fill_short_gaps(ds1.values, fill_gap)
        ds1 = ds1.copy(data=filled)
        print(f"INTERP_GAP={fill_gap}: {n_f:,} cell-months filled in time "
              f"({100 * n_f / max(n_obs, 1):.1f}% more than the {n_obs:,} "
              "observed), before the block average", flush=True)
    ds1=ds1.coarsen(YC=COARSEN, XC=COARSEN, boundary='trim').mean()
    if keep:
        ANOMALIES[keep] = ds1          # with its trend: the trend maps
    if DETREND:
        ds1 = remove_trend(ds1)
    return ds1/ds1.std('time')


def remove_trend(da):
    """Least-squares linear trend removed per cell, gaps left as gaps."""
    a = da.transpose("time", "YC", "XC").values.astype("float64")
    t = np.arange(a.shape[0], dtype="float64")[:, None, None]
    ok = np.isfinite(a)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(ok, t, 0).sum(axis=0) / n
        am = np.where(ok, a, 0).sum(axis=0) / n
        dt = np.where(ok, t - tm, 0)
        slope = (dt * np.where(ok, a - am, 0)).sum(axis=0) / (dt * dt).sum(axis=0)
        out = a - am - slope * (t - tm)
    out[:, n < 24] = np.nan
    return da.copy(data=out.astype(da.dtype))



# =====================================================================
# Chlorophyll: top 25 m, volume weighted
# =====================================================================
ds = open_bsose("Chl")

# Select chl-a
chl = ds.BLGCHL

# Restrict to top q m
Z_top = ds.Z.where(ds.Z >= q, drop=True)
chl = chl.sel(Z=Z_top)
drF = ds.drF.sel(Z=Z_top)
hFacC = ds.hFacC.sel(Z=Z_top)

# Volume-weighted vertical integration
chl_weighted = (chl * drF * hFacC).sum(dim="Z") / (drF * hFacC).sum(dim="Z")


def satellite_on_model_grid(yc, xc, times):
    """Satellite chlorophyll on the model grid, (time, YC, XC), mg m-3.

    Each model cell takes the value of the satellite pixel it lies in
    (the nearest one). The model grid is finer than the satellite's
    0.25 deg, so nothing is interpolated or smoothed and a gap stays
    exactly the gap it was: a model cell is missing in a month precisely
    when its pixel is. Averaging these cells over a block afterwards is
    then the average of the pixels the block covers. A model cell
    outside the satellite's latitude range, or a month the file does not
    have, is missing.
    """
    with xr.open_dataset(ESA_FILE) as sat:
        name = "chlor_a" if "chlor_a" in sat else list(sat.data_vars)[0]
        v = sat[name].transpose("time", "lat", "lon")
        sat_month = pd.PeriodIndex(pd.to_datetime(v["time"].values), freq="M")
        want = pd.PeriodIndex(times, freq="M")
        pos = sat_month.get_indexer(want)
        missing = int((pos < 0).sum())
        vals = v.isel(time=np.where(pos >= 0, pos, 0)).values.astype("float32")
        slat = v["lat"].values.astype("float64")
        slon = v["lon"].values.astype("float64")
    vals[~(np.isfinite(vals) & (vals > 0))] = np.nan
    vals[pos < 0] = np.nan
    iy = np.abs(slat[None, :] - np.asarray(yc)[:, None]).argmin(axis=1)
    dlon = ((slon[None, :] - np.asarray(xc)[:, None] + 180.0) % 360.0) - 180.0
    ix = np.abs(dlon).argmin(axis=1)
    half = 0.5 * float(np.abs(np.diff(slat)).mean())
    inside = np.abs(slat[iy] - np.asarray(yc)) <= half + 1e-6
    out = vals[:, iy][:, :, ix]
    out[:, ~inside, :] = np.nan
    print(f"satellite: {os.path.basename(ESA_FILE)}, {name}, "
          f"{len(slat)} x {len(slon)} pixels, "
          f"{len(want) - missing} of {len(want)} months found", flush=True)
    if missing:
        print(f"  WARNING: {missing} month(s) of 2013-{LAST_YEAR} are not "
              "in the satellite file and are missing here too")
    return out


if SAMPLED or SATELLITE:
    sat_chl = satellite_on_model_grid(chl_weighted.YC.values,
                                      chl_weighted.XC.values,
                                      chl_weighted.time.values)
    seen = np.isfinite(sat_chl)
    chl_weighted = chl_weighted.load()
    ocean = np.isfinite(chl_weighted.values)
    month = pd.DatetimeIndex(chl_weighted.time.values).month
    what = ("satellite chlorophyll on the model grid" if SATELLITE
            else "sampled as the satellite")
    print(f"{what}: {int((ocean & seen).sum()):,} of "
          f"{int(ocean.sum()):,} ocean cell-months "
          f"({100 * (ocean & seen).sum() / max(ocean.sum(), 1):.0f}%)")
    lat_nat = chl_weighted.YC.values
    for sname, smon in (("DJF", (12, 1, 2)), ("MAM", (3, 4, 5)),
                        ("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
        tsel = np.isin(month, smon)
        row = []
        for lab, lo, hi in (("30-45S", -45, -30), ("45-60S", -60, -45),
                            ("south of 60S", -90, -60)):
            ysel = (lat_nat > lo) & (lat_nat <= hi)
            o = ocean[tsel][:, ysel]
            k = (ocean & seen)[tsel][:, ysel]
            row.append(f"{lab} {100 * k.sum() / max(o.sum(), 1):.0f}%")
        print(f"  kept in {sname}: " + ", ".join(row))
    if SATELLITE:
        # The satellite's own values, on the model's ocean cells only,
        # so this run and the sampled one cover exactly the same
        # cell-months and differ in whose chlorophyll it is.
        both = ocean & seen
        print(f"  median over those cell-months: satellite "
              f"{np.nanmedian(sat_chl[both]):.3f}, model "
              f"{np.nanmedian(chl_weighted.values[both]):.3f} mg m-3")
        chl_weighted = chl_weighted.copy(
            data=np.where(both, sat_chl, np.nan).astype(
                chl_weighted.dtype))
    else:
        chl_weighted = chl_weighted.where(seen)
    del seen, ocean, sat_chl

chl_weighted = deseasonalize(chl_weighted, keep="Chl", fill_gap=INTERP_GAP)

# =====================================================================
# The drivers, defined as in plot_driver_trends.py
# =====================================================================
DEPTH_MEAN, DEPTH_STRAT = 25.0, 50.0       # upper layer, lower layer
YC_T, XC_T = ds["YC"], ds["XC"]            # the tracer grid of the Chl file
WET = (ds["hFacC"].isel(Z=0, drop=True) > 0)


def sigma0(S, T):
    """Potential density anomaly, EOS-80 polynomial (gsw if available);
    as plot_driver_trends.py."""
    try:
        import gsw
        return gsw.sigma0(S, T)
    except ImportError:
        t, s = T, S
        rho = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2
               + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4
               + 6.536332e-9 * t ** 5
               + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t ** 2
                      - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4)
               + np.abs(s) ** 1.5 * (-5.72466e-3 + 1.0227e-4 * t
                                     - 1.6546e-6 * t ** 2)
               + 4.8314e-4 * s ** 2)
        return rho - 1000.0


def layer_mean(dsx, name, zmin, zmax):
    """Mean of dsx[name] over the levels with zmin < depth <= zmax,
    weighted by drF x hFacC (zero on land, partial at the sea floor), as
    the chlorophyll above. (time, YC, XC); land is NaN."""
    depth = np.abs(dsx["Z"].values.astype("float64"))
    keep = np.flatnonzero((depth > zmin) & (depth <= zmax))
    w = (dsx["drF"].isel(Z=keep) * dsx["hFacC"].isel(Z=keep)).astype("float64")
    den = w.sum("Z")
    v = dsx[name].isel(Z=keep)
    out = []
    for t0 in range(0, v.sizes["time"], 12):     # a year at a time
        vv = v.isel(time=slice(t0, t0 + 12)).astype("float64")
        out.append(((vv * w).sum("Z") / den.where(den > 0)).astype("float32"))
    return xr.concat(out, dim="time").transpose("time", "YC", "XC")


# ---- wind stress, on the cell centre: the mean of the two faces
data_TAUX = open_bsose("oceTAUX")          # (time, YC, XG): west faces
data_TAUY = open_bsose("oceTAUY")          # (time, YG, XC): south faces
tx = data_TAUX.oceTAUX.transpose("time", "YC", "XG").values.astype("float32")
tx = 0.5 * (tx + np.roll(tx, -1, axis=2))
ty = data_TAUY.oceTAUY.transpose("time", "YG", "XC").values.astype("float32")
tyc = np.full_like(ty, np.nan)
tyc[:, :-1] = 0.5 * (ty[:, :-1] + ty[:, 1:])
coords_t = {"time": chl_weighted["time"].values, "YC": YC_T.values,
            "XC": XC_T.values}
TAUX_on_tracer = xr.DataArray(tx, dims=("time", "YC", "XC"), coords=coords_t)
TAUY_on_tracer = xr.DataArray(tyc, dims=("time", "YC", "XC"), coords=coords_t)
del tx, ty, tyc, data_TAUX, data_TAUY


def ekman_pumping(TAUX, TAUY):
    """w = curl_z(tau / (rho f)) on the tracer grid, positive upward.

        w = 1 / (R cos(lat)) * [ d(Ay)/d(lon) - d(Ax cos(lat))/d(lat) ]

    with A = tau / (rho f). Centred differences, periodic in longitude.
    """
    R, Omega = 6.371e6, 7.2921e-5
    lat = np.deg2rad(TAUX.YC.values.astype("float64"))
    lon = np.deg2rad(TAUX.XC.values.astype("float64"))
    cos = np.cos(lat)[None, :, None]
    rf = (rho_ref * 2.0 * Omega * np.sin(lat))[None, :, None]
    out = np.empty(TAUX.shape, dtype="float32")
    for i0 in range(0, TAUX.shape[0], 24):
        sl = slice(i0, i0 + 24)
        ax = TAUX.values[sl] / rf
        ay = TAUY.values[sl] / rf
        day = (np.roll(ay, -1, axis=2) - np.roll(ay, 1, axis=2)) / (
            2.0 * (lon[1] - lon[0]))
        dax = np.gradient(ax * cos, lat, axis=1)
        out[sl] = (day - dax) / (R * cos)
    return TAUX.copy(data=out)


if WIND_DRIVER == "wek":
    wind_field = ekman_pumping(TAUX_on_tracer, TAUY_on_tracer)
else:
    wind_field = np.sqrt(TAUX_on_tracer ** 2 + TAUY_on_tracer ** 2)
wind_field = wind_field.where(WET)
time_mean_wind_amp_anom2 = deseasonalize(wind_field, keep=WIND_DRIVER)
del TAUX_on_tracer, TAUY_on_tracer, wind_field

# ---- temperature, salinity (top 25 m) and stratification (0-25 vs 25-50 m)
ds_thetao = open_bsose("Theta")
ds_salt = open_bsose("Salt")
theta_up = layer_mean(ds_thetao, "THETA", -1.0, DEPTH_MEAN)
salt_up = layer_mean(ds_salt, "SALT", -1.0, DEPTH_MEAN)
theta_dn = layer_mean(ds_thetao, "THETA", DEPTH_MEAN, DEPTH_STRAT)
salt_dn = layer_mean(ds_salt, "SALT", DEPTH_MEAN, DEPTH_STRAT)
dz = 0.5 * DEPTH_STRAT                     # 12.5 m to 37.5 m
N = (-(g / rho_ref) * (sigma0(salt_up, theta_up)
                       - sigma0(salt_dn, theta_dn)) / dz).astype("float32")
del theta_dn, salt_dn, ds_salt, ds_thetao
N = deseasonalize(N, keep="N")
data_salt_mean = deseasonalize(salt_up, keep="salt")
data_THETA_mean = deseasonalize(theta_up, keep="Theta")
del salt_up, theta_up

# ---- mixed-layer depth
data_MLD = open_bsose("MLD")
data_MLD = data_MLD.BLGMLD
data_MLD = data_MLD.where(WET & (data_MLD != 0))
data_MLD = deseasonalize(data_MLD, keep="MLD")

def surface_field(name):
    """A 2D B-SOSE diagnostic (time, YC, XC), ocean cells only, as
    anomalies like the other drivers."""
    dsx = open_bsose(name)
    var = name if name in dsx else next(
        v for v in dsx.data_vars if dsx[v].ndim == 3 and "time" in dsx[v].dims)
    da = dsx[var].squeeze(drop=True)
    yd = next(d for d in da.dims if d.startswith("Y"))
    xd = next(d for d in da.dims if d.startswith("X"))
    da = da.rename({yd: "YC", xd: "XC"}).transpose("time", "YC", "XC")
    da = da.assign_coords(YC=YC_T.values, XC=XC_T.values)
    return deseasonalize(da.where(WET), keep=name)


if TARGET == "MLD":
    feats = {
        WIND_DRIVER: time_mean_wind_amp_anom2.values,
        "surfTflx": surface_field("surfTflx").values,
        "N": N.values,
        "surfSflx": surface_field("surfSflx").values,
        "Theta": data_THETA_mean.values,
        "oceQsw": surface_field("oceQsw").values,
    }
    # the target takes the place of chlorophyll everywhere below
    chl_weighted = data_MLD
    print("TARGET=MLD: drivers " + ", ".join(feats), flush=True)
else:
    feats = {
        WIND_DRIVER: time_mean_wind_amp_anom2.values,
        "salt": data_salt_mean.values,
        "N": N.values,
        "MLD": data_MLD.values,
        "Theta": data_THETA_mean.values
    }

chl = chl_weighted.values

# ---------------------------------------------------------------------
# The block anomalies PCMCI sees, saved for summarize_links.py (the sign
# of each link). ANOM_ONLY=1 saves them and stops: for a finished run.
# ---------------------------------------------------------------------
_anom_dir = OUTDIR.format(method=METHOD)
os.makedirs(_anom_dir, exist_ok=True)
np.savez(os.path.join(_anom_dir, "anomalies.npz"),
         names=np.array(list(feats.keys())),
         feats=np.stack([feats[k] for k in feats]).astype("float32"),
         chl=chl.astype("float32"),
         lat=data_MLD.YC.values, lon=data_MLD.XC.values,
         month0=np.array(1))
print(f"anomalies saved: {_anom_dir}/anomalies.npz", flush=True)
if os.environ.get("ANOM_ONLY", "0") == "1":
    raise SystemExit(0)
print(f"blocks of {COARSEN} x {COARSEN} model cells -> grid "
      f"{chl.shape[1]} x {chl.shape[2]} = {chl.shape[1] * chl.shape[2]} "
      f"cells; results in {OUTDIR.format(method=METHOD)}", flush=True)
if SAMPLED or SATELLITE:
    n_months = np.isfinite(chl).sum(axis=0)
    print(f"coarse cells with chlorophyll: {int((n_months > 0).sum())} of "
          f"{n_months.size}; months per cell, median "
          f"{int(np.median(n_months[n_months > 0]))} of {chl.shape[0]}")


print("Chl shape:", chl.shape)
for k, v in feats.items():
    print(k, v.shape)

maps_r, maps_p = partial_corr_maps(chl, feats, lags=[0], alpha=0.05)

lon_grid = data_MLD.XC.values
lat_grid = data_MLD.YC.values
alpha_level = 0.05


# ---------------------------------------------------------------------
# Regions of the regional summaries: the three SSH zones of
# make_acc_zones.py (north of the SAF, SAF to Southern Boundary, south of
# the SB), each 4x4 / 8x8 block taking the zone of most of its model
# cells. Without the zone file: the old latitude bands.
#   ZONES_FILE=/work/bb1596/b384036/b384036/causality/acc_zones3_bsose.nc
#   ZONES_FILE=none   -> latitude bands
# ---------------------------------------------------------------------
ZONES_FILE = os.environ.get(
    "ZONES_FILE",
    "/work/bb1596/b384036/b384036/causality/acc_zones3_bsose.nc")
LAT_BANDS = {"Subtropics": (-45, -30), "ACC": (-60, -45), "HighLat": (-90, -60)}
_REGION_CACHE = {}


def _block_start(native, coarse, k):
    """Index of the native cell where the first block starts: the offset
    whose k-cell mean is the first coarse coordinate."""
    n = native.size - k + 1
    means = np.convolve(native, np.ones(k) / k, mode="valid")[:n]
    i0 = int(np.argmin(np.abs(means - coarse[0])))
    step = np.abs(np.diff(native)).mean()
    if abs(means[i0] - coarse[0]) > 0.25 * step:
        raise ValueError("block grid does not line up with the zone file")
    return i0


def block_regions(lat2d=None, nx=None):
    """{region name: (ny, nx) bool} on the block grid of this run."""
    lat_b = np.asarray(lat_grid, float)
    lon_b = np.asarray(lon_grid, float)
    ny, nxb = lat_b.size, lon_b.size
    key = (ny, nxb)
    if key in _REGION_CACHE:
        return _REGION_CACHE[key]
    masks = None
    if ZONES_FILE.lower() != "none" and os.path.exists(ZONES_FILE):
        import xarray as xr
        with xr.open_dataset(ZONES_FILE) as z:
            zone = z["zone"].transpose("YC", "XC")
            yz, xz = zone.YC.values, zone.XC.values
            zone = zone.values
            fv = str(z["zone"].attrs.get("flag_values", "")).split()
            fm = str(z["zone"].attrs.get("flag_meanings", "")).split()
        names = ({int(v): m for v, m in zip(fv, fm) if int(v) != 0}
                 if fv and len(fv) == len(fm) else
                 {1: "N_SAF", 2: "ACC", 3: "S_SB"})
        k = COARSEN
        try:
            i0 = _block_start(yz, lat_b, k)
            j0 = _block_start(xz, lon_b, k)
            b = zone[i0:i0 + ny * k, j0:j0 + nxb * k]
            if b.shape != (ny * k, nxb * k):
                raise ValueError("zone file smaller than the block grid")
            b = b.reshape(ny, k, nxb, k)
            counts = np.stack([(b == c).sum(axis=(1, 3)) for c in names])
            best = np.array(list(names))[counts.argmax(axis=0)]
            zb = np.where(counts.max(axis=0) >= k * k / 2, best, 0)
            masks = {{"ACC": "SAF_SB"}.get(n, n): zb == c for c, n in names.items()}
            print("regions from " + ZONES_FILE + ": " + ", ".join(
                f"{n} {int(m.sum())} blocks" for n, m in masks.items()),
                flush=True)
        except ValueError as exc:
            print(f"[WARNING] {ZONES_FILE}: {exc}; latitude bands used",
                  flush=True)
            masks = None
    elif ZONES_FILE.lower() != "none":
        print(f"[WARNING] {ZONES_FILE} not found; latitude bands used",
              flush=True)
    if masks is None:
        masks = {r: np.broadcast_to(((lat_b > lo) & (lat_b <= hi))[:, None],
                                    (ny, nxb)).copy()
                 for r, (lo, hi) in LAT_BANDS.items()}
    _REGION_CACHE[key] = masks
    return masks


REGION_NAMES = list(block_regions())


import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import matplotlib.ticker as mticker

alpha_level = 0.05
lags_to_plot = [0]
features_to_plot = list(feats.keys())
for li, tau in enumerate(lags_to_plot):
    for f in features_to_plot:
        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

        # Add coastlines and gridlines
        coastlines(ax)
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
        cb.set_label(f'{f} → {TARGET} partial corr (lag {tau})', fontsize=16, labelpad=15)

        # Save figure
        plt.savefig(f"pcorr_map_{f}_lag{tau}_polar_"+str(q)+TAG+("" if TARGET == "Chl" else "_" + TARGET)+".pdf", dpi=300, bbox_inches='tight')
        plt.close()


# sensitivity analysis
from joblib import Parallel, delayed
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
import numpy as np

def save_maps(val_maps, p_maps, outdir, tag="", lon2d=None, lat2d=None):
    os.makedirs(outdir, exist_ok=True)
    for f in val_maps.keys():
        field = val_maps[f][0]
        pval  = p_maps[f][0]
        sig   = np.where(pval <= 0.05, field, np.nan)

        fname = os.path.join(outdir, f"{f}{tag}.pdf")
        try:
            plot_map(
                sig,
                title=f"{f}{tag}",
                fname=fname,
                lon2d=lon2d,
                lat2d=lat2d
            )
        except Exception as exc:
            # A map with no significant cell, or a plotting problem, must
            # not cost the passes that follow it.
            plt.close("all")
            print(f"  map {os.path.basename(fname)} not drawn "
                  f"({type(exc).__name__}: {exc})", flush=True)


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

    regions = block_regions()

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
    plt.title("Bootstrap spatial confidence intervals")
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


def make_test(masked=False):
    """The conditional-independence test named by TEST. With a season
    mask, mask_type='y': the mask restricts the TARGET months only."""
    mt = {"mask_type": "y"} if masked else {}
    if TEST == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        return ParCorr(significance="analytic", **mt)
    if TEST == "robustparcorr":
        from tigramite.independence_tests.robust_parcorr import RobustParCorr
        return RobustParCorr(significance="analytic", **mt)
    if TEST == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        return CMIknn(significance="shuffle_test", **mt)
    from tigramite.independence_tests.gpdc import GPDC
    return GPDC(significance="analytic", **mt)


# Link marks counted as "driver causes chlorophyll" in the graph
LINKS = {"-->"} | ({"o-o", "x-x"} if CONTEMP else set())


def run_causal_driver(features, chl, alpha=0.05, min_samples=12, tau_max=2,
                      target_months=None, verbose=True):
    """METHOD with TEST per block, tau = 0..tau_max.

    Returns val_maps[driver][tau, YC, XC] and p_maps (same shape). For
    PCMCI+ and LPCMCI, p is set to 1 wherever the graph does not have
    driver --> Chl, so p <= alpha means "in the causal graph".
    target_months: calendar months tested as the target (a season);
    the other months only supply the lagged values.
    """
    feat_names = list(features.keys())
    T, W, Y = chl.shape
    val_maps = {f: np.full((tau_max + 1, W, Y), np.nan) for f in feat_names}
    p_maps = {f: np.full((tau_max + 1, W, Y), np.nan) for f in feat_names}
    months = (np.arange(T) % 12) + 1
    in_season = (np.ones(T, bool) if target_months is None
                 else np.isin(months, target_months))

    def run_cell(i, j):
        try:
            A = np.column_stack([features[f][:, i, j] for f in feat_names]
                                + [chl[:, i, j]]).astype("float64")
            ok = np.all(np.isfinite(A), axis=1)
            if (ok & in_season).sum() < min_samples:
                return None, None, i, j
            # Trim leading/trailing gaps; inner gaps stay as missing
            # values, which tigramite keeps out of every lagged sample.
            idx = np.flatnonzero(ok)
            sl = slice(idx[0], idx[-1] + 1)
            A = np.where(np.isfinite(A), A, MISSING)[sl]
            kw = dict(var_names=feat_names + [TARGET], missing_flag=MISSING)
            masked = target_months is not None
            if masked:
                kw["mask"] = np.repeat((~in_season[sl])[:, None],
                                       A.shape[1], axis=1)
            df = DataFrame(A, **kw)
            c = len(feat_names)
            if METHOD == "LPCMCI":
                from tigramite.lpcmci import LPCMCI
                res = LPCMCI(dataframe=df, cond_ind_test=make_test(masked),
                             verbosity=0).run_lpcmci(
                    tau_min=0, tau_max=tau_max, pc_alpha=alpha)
            else:
                pc = PCMCI(dataframe=df, cond_ind_test=make_test(masked),
                           verbosity=0)
                if METHOD == "PCMCI":
                    res = pc.run_pcmci(tau_min=0, tau_max=tau_max,
                                       pc_alpha=alpha)
                else:
                    res = pc.run_pcmciplus(tau_min=0, tau_max=tau_max,
                                           pc_alpha=alpha)
            vals, pval = res["val_matrix"], res["p_matrix"]
            out_v, out_p = {}, {}
            for fi, f in enumerate(feat_names):
                out_v[f] = np.asarray(vals[fi, c, :], float)
                pv = np.asarray(pval[fi, c, :], float)
                if METHOD != "PCMCI":
                    linked = np.isin(res["graph"][fi, c, :], list(LINKS))
                    pv = np.where(linked, np.minimum(pv, alpha), 1.0)
                out_p[f] = pv
            return out_v, out_p, i, j
        except Exception as e:
            if verbose:
                print(f"⚠️ {METHOD} failed at ({i},{j}): {e}")
            return None, None, i, j

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_cell)(i, j) for i in range(W) for j in range(Y))
    for out_v, out_p, i, j in results:
        if out_v is None:
            continue
        for f in feat_names:
            val_maps[f][:, i, j] = out_v[f]
            p_maps[f][:, i, j] = out_p[f]
    return val_maps, p_maps


drivers = {
    METHOD: run_causal_driver,
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
    masks = block_regions()
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
    plt.title(f"Regional causal influence on {TARGET}", fontsize=16)
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
    coastlines(ax)
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

    Regions: block_regions() -- the SSH zones N_SAF, SAF_SB, S_SB of
    ZONES_FILE (latitude bands if the file is missing).

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
    regions = block_regions()

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
    "wind":  "#ff7f0e",  # orange
    "wek":   "#ff7f0e",  # same slot as the wind
    "Theta": "#b15928",  # brown
    "surfTflx": "#d62728",  # red
    "surfSflx": "#17becf",  # cyan
    "oceQsw":   "#4d4d4d",  # dark grey
    }
    colors = [driver_colors.get(d, "0.5") for d in drivers]

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
    plt.title(f"Seasonal causal influence of drivers on {TARGET}", fontsize=18)
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



def plot_driver_lag_comparison(seasonal_stats_by_lag, outdir, region="GLOBAL"):
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
            f"{drv}_lag_comparison_{region}.pdf"
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


def collect_regional_values(val_maps, p_maps, lat2d, alpha=0.05, region="SAF_SB"):
    """
    Extract significant spatial samples for each driver inside `region`.
    val_maps : dict driver -> array (tau, YC, XC)  (we use tau=0 slice)
    p_maps   : dict driver -> array (tau, YC, XC)
    lat2d    : 1D or 2D lat array (YC or YCxXC)
    region   : a name in REGION_NAMES (N_SAF | SAF_SB | S_SB)
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

    regions = block_regions()

    mask = regions[region]

    samples = {}
    for drv in val_maps.keys():
        vals = val_maps[drv][0]   # shape (YC, XC)
        pval = p_maps[drv][0]
        good = (pval <= alpha) & np.isfinite(vals) & mask
        samples[drv] = vals[good]  # might be empty array
    return samples


def run_rows(driver, tag, outdir, features, chl, alpha, tau_max,
             target_months=None):
    """`driver` over the whole grid, one latitude row at a time, saved
    after every row.

    The cells are independent, so the maps are the ones a single call
    gives. The only purpose is that a job which stops early loses one
    row and not the whole pass: the next run picks up from the file.
    """
    import time
    feat_names = list(features.keys())
    T, W, Y = chl.shape
    path = os.path.join(outdir, f"pcmci_{tag}.npz")
    val = {f: np.full((tau_max + 1, W, Y), np.nan) for f in feat_names}
    pv = {f: np.full((tau_max + 1, W, Y), np.nan) for f in feat_names}
    done = np.zeros(W, dtype=bool)
    if os.path.exists(path):
        with np.load(path) as z:
            same = (list(z["names"]) == feat_names
                    and str(z["setup"]) == SETUP if "setup" in z else False)
            same = (same and z["done"].shape == (W,)
                    and z["val"].shape == (len(feat_names), tau_max + 1, W, Y)
                    and int(z["T"]) == T)
            if same:
                done = z["done"].copy()
                for k, f in enumerate(feat_names):
                    val[f], pv[f] = z["val"][k].copy(), z["p"][k].copy()
        if same:
            print(f"  {tag}: {int(done.sum())} of {W} rows already done "
                  f"({os.path.basename(path)})", flush=True)
        else:
            print(f"  {tag}: {os.path.basename(path)} is from a different "
                  "setup; starting again", flush=True)
    # Cells that have enough complete months to be fitted at all (the
    # driver's own threshold, 12). A sampled run has whole rows with
    # none -- the far south outside summer -- and that is not a failure.
    complete = np.isfinite(chl)
    for f in feat_names:
        complete &= np.isfinite(features[f])
    if target_months is not None:
        complete &= np.isin((np.arange(T) % 12) + 1,
                            target_months)[:, None, None]
    eligible = complete.sum(axis=0) >= 12                  # (W, Y)
    del complete
    print(f"  {tag}: {int(eligible.sum())} of {W * Y} cells have 12 or "
          "more complete months", flush=True)
    t0, n_new, tried = time.time(), 0, 0
    for i in range(W):
        if done[i]:
            continue
        v, p = driver(features={f: features[f][:, i:i + 1] for f in feat_names},
                      chl=chl[:, i:i + 1], alpha=alpha, tau_max=tau_max,
                      target_months=target_months)
        for f in feat_names:
            val[f][:, i], pv[f][:, i] = v[f][:, 0], p[f][:, 0]
        done[i] = True
        n_new += 1
        tmp = path + ".part.npz"
        np.savez(tmp, names=np.array(feat_names), done=done, T=T,
                 setup=np.array(SETUP),
                 val=np.stack([val[f] for f in feat_names]),
                 p=np.stack([pv[f] for f in feat_names]))
        os.replace(tmp, path)
        left = int((~done).sum())
        rate = (time.time() - t0) / n_new
        tried += int(eligible[i].sum())
        print(f"  {tag}: row {i + 1} of {W}, "
              f"{int(np.isfinite(v[feat_names[0]][0]).sum())} of "
              f"{int(eligible[i].sum())} cells with enough months fitted, "
              f"~{rate * left / 3600:.1f} h left in this pass",
              flush=True)
        fitted = sum(int(np.isfinite(val[f]).sum()) for f in feat_names)
        if fitted == 0 and tried >= 300 and tag == "full":
            # Only the first pass can stop the run: if nothing at all
            # comes back there, the test itself is not working. A season
            # with few usable cells is a result, not an error.
            raise RuntimeError(
                f"none of the first {tried} cells with enough months has "
                f"given a result; see the '{METHOD} failed' messages above")
    return val, pv


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
    print(f"→ Running full-year {METHOD} ({TEST})")
    val_full, p_full = run_rows(
        driver, "full", outdir,
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
    seasonal_stats_regional = {tau: {r: {} for r in REGION_NAMES} for tau in range(tau_max + 1)}

    # NEW: store raw maps for panel-B bootstrap (per tau → per season)
    # structure: bootstrap_by_lag[tau][sname] = {"val": val_tau, "p": p_tau}
    bootstrap_by_lag = {tau: {} for tau in range(tau_max + 1)}

    for sname, smonths in seasons.items():

        idx = np.where(np.isin(months, smonths))[0]
        if idx.size < 12:
            print(f"[WARNING] Season {sname} has only {idx.size} months → skipping.")
            continue

        if SEASON_MASK:
            # Full series; only the season's months are the target.
            print(f"→ Season {sname}: {idx.size} target months, lags "
                  "from the months before them")
            val_s, p_s = run_rows(
                driver, sname, outdir, features=features, chl=chl,
                alpha=alpha, tau_max=tau_max, target_months=smonths)
        else:
            print(f"→ Season {sname}: {idx.size} months glued together")
            feat_sub = {k: v[idx] for k, v in features.items()}
            val_s, p_s = run_rows(
                driver, sname, outdir, features=feat_sub, chl=chl[idx],
                alpha=alpha, tau_max=tau_max)

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
        plot_driver_lag_comparison(regional_lag_stats, outdir=outdir, region=region)

    # 5) PANEL B — BOOTSTRAP SIGNIFICANCE BARS FOR ALL REGIONS
    print("→ Plotting bootstrap significance bars (" + ", ".join(REGION_NAMES) + ")")

    regions = REGION_NAMES
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



#data = io.loadmat("Chl_trend_significant.mat")
#lon        = data["lon"].squeeze()
#lat        = data["lat"].squeeze()
#chl_trend = data["chl_trend"].squeeze()
#p_value    = data["p_value"].squeeze()



#lon_grid = data_MLD.XC.values
#lat_grid = data_MLD.YC.values

# =====================================================================
# Trend of the anomalies at every grid cell
# =====================================================================
UNITS = {"wind": "N m$^{-2}$", "wek": "m s$^{-1}$", "salt": "psu",
         "N": "m s$^{-2}$", "MLD": "m", "Theta": "$^\\circ$C",
         "Chl": "mg m$^{-3}$", "surfTflx": "W m$^{-2}$",
         "surfSflx": "g m$^{-2}$ s$^{-1}$", "oceQsw": "W m$^{-2}$"}


def cell_trends(a):
    """Least-squares slope per decade and its p-value at every cell of a
    (time, Y, X) array with gaps; a cell needs 24 months."""
    T = a.shape[0]
    t = (np.arange(T) / 120.0)[:, None, None]          # decades
    ok = np.isfinite(a)
    n = ok.sum(axis=0)
    a0 = np.where(ok, a, 0.0)
    t0 = np.where(ok, t, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = t0.sum(axis=0) / n
        am = a0.sum(axis=0) / n
        dt = np.where(ok, t - tm, 0.0)
        da = np.where(ok, a - am, 0.0)
        sxx = (dt * dt).sum(axis=0)
        slope = (dt * da).sum(axis=0) / sxx
        resid = np.where(ok, da - slope * dt, 0.0)
        se = np.sqrt((resid ** 2).sum(axis=0) / (n - 2) / sxx)
        tstat = slope / se
        pval = 2 * (1 - tdist.cdf(np.abs(tstat), np.maximum(n - 2, 1)))
    bad = n < 24
    slope[bad] = np.nan
    pval[bad] = np.nan
    return slope, pval


def plot_trend_maps(anoms, lon, lat, outdir):
    """One polar map per field of the trend of its anomalies, cells with
    p <= 0.05 stippled; and all of them on one page."""
    os.makedirs(outdir, exist_ok=True)
    names = [k for k in anoms if k != "Chl"] + (["Chl"] if "Chl" in anoms else [])
    results = {}
    for k in names:
        results[k] = cell_trends(anoms[k].transpose("time", "YC", "XC").values)
    # one column repeated at lon + 360, so the map has no seam
    lon_c = np.concatenate([lon, [lon[0] + 360.0]])
    wrap = lambda f: np.concatenate([f, f[:, :1]], axis=1)

    def draw(ax, k, cb_label=True):
        slope, pval = (wrap(f) for f in results[k])
        lim = np.nanpercentile(np.abs(slope), 95)
        if not np.isfinite(lim) or lim == 0:
            lim = 1.0
        ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
        coastlines(ax)
        fill = ax.contourf(lon_c, lat, slope, levels=np.linspace(-lim, lim, 21),
                           cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree(),
                           extend="both")
        sig = np.where(pval <= 0.05, 1.0, 0.0)
        if np.nansum(sig) > 0:
            ax.contourf(lon_c, lat, sig, levels=[0.5, 1.5], colors="none",
                        hatches=["..."], transform=ccrs.PlateCarree())
        cb = plt.colorbar(fill, ax=ax, orientation="vertical", shrink=0.7)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
        if cb_label:
            cb.set_label(f"{k} trend, {UNITS.get(k, '')} per decade",
                         fontsize=11)
        share = 100 * np.nanmean(pval <= 0.05)
        ax.set_title(f"{k}: {share:.0f}% of cells significant", fontsize=11)
        return fill

    for k in names:
        fig = plt.figure(figsize=(8, 8), dpi=200)
        ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        draw(ax, k)
        fig.savefig(os.path.join(outdir, f"trend_{k}{TAG}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 5.2 * nrow), dpi=200)
    for i, k in enumerate(names):
        ax = fig.add_subplot(nrow, ncol, i + 1,
                             projection=ccrs.SouthPolarStereo())
        draw(ax, k)
    fig.suptitle(f"Linear trend of the anomalies, 2013-{LAST_YEAR}, "
                 f"{COARSEN} x {COARSEN} blocks; stippled: p <= 0.05",
                 fontsize=13)
    out = os.path.join(outdir, f"trend_maps{TAG}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}", flush=True)
    np.savez(os.path.join(outdir, f"trend_maps{TAG}.npz"), lon=lon, lat=lat,
             names=np.array(names),
             slope=np.stack([results[k][0] for k in names]),
             pval=np.stack([results[k][1] for k in names]))


plot_trend_maps(ANOMALIES, lon_grid, lat_grid, OUTDIR.format(method=METHOD))

for method, driver in drivers.items():
    run_sensitivity(
    method=method,
    driver=driver,
    features=feats,
    chl=chl_weighted.values,
    seasons=seasons,
    alpha=0.05,
    outdir=OUTDIR.format(method=method),
    lon2d=lon_grid,
    lat2d=lat_grid
    )
