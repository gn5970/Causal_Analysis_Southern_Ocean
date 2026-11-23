"""Causal discovery (PCMCI by default; PCMCI+ or LPCMCI) on the
observations: satellite chlorophyll with ORAS5 physics, 1998-2025, on the
ORAS5 (ORCA025) grid.

    python oras5_pcmci.py                 PCMCI (default)
    COARSEN=4 python oras5_pcmci.py       4 x 4 ORCA025 blocks (default 2)
    TEST=robustparcorr python oras5_pcmci.py
    MIN_SAMPLES=36                        complete target months a block needs

The same analysis as bsose_pcmci.py, on other data:

  grid         the ORAS5 ORCA025 T grid itself, south of 30S, where it is
               a regular Mercator grid (latitude constant along a row);
               columns sorted by longitude, the duplicated wrap columns
               dropped. COARSEN x COARSEN blocks of it (2: ~0.5 deg in
               longitude, 0.15-0.45 deg in latitude)
  chlorophyll  ESA CCI 0.25 deg; each ORCA025 cell takes the satellite
               pixel it lies in (nearest); gaps stay gaps
  drivers      ORAS5 on its own grid, no regridding,
               defined exactly as in plot_driver_trends.py:
                 Theta, salt  thickness-weighted means over the top 25 m
                 N            -(g / rho0) (sigma0(0-25 m) - sigma0(25-50 m)) / 25 m
                 MLD          somxl010
                 wind         amplitude of sozotaux and sometauy
  anomalies    the earlier deseasonalize, in its order: monthly
               climatology removed, COARSEN x COARSEN blocks, (DETREND=1:
               linear trend removed), divided by the standard deviation
  causal step  as bsose_pcmci.py: METHOD (PCMCI default, PCMCIplus,
               LPCMCI), TEST (gpdc default, parcorr, robustparcorr,
               cmiknn), tau = 0..2, alpha = 0.05; gaps as missing values,
               seasons by a target mask

YEAR0/YEAR1 set the window (1998-2025). Results in
results_obs_<METHOD>_c<COARSEN>_1998to2025[_<test>]/.
"""
import os
import re
import sys
import glob
import warnings
from pathlib import Path
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
# Where the data are, and the choices
# ---------------------------------------------------------------------
WORK = os.environ.get("WORK", "/work/bb1596/b384036/b384036/causality")
BSOSE_DIR = os.environ.get("BSOSE_DIR", os.path.join(WORK, "bsose"))
BSOSE_NAME = "{}_bsoseI156_2013to2024_monthly.nc"    # for the grid only
ESA_FILE = os.environ.get(
    "ESA_FILE", os.path.join(WORK, "ESA_Chl_south_1998_2025_0p25deg.nc"))
YEAR0 = int(os.environ.get("YEAR0", 1998))
YEAR1 = int(os.environ.get("YEAR1", 2025))
YEAR_MIN, YEAR_MAX = YEAR0, YEAR1          # names used by the ORAS5 readers
MONTHS = pd.period_range(f"{YEAR0}-01", f"{YEAR1}-12", freq="M").to_timestamp()
N_MONTHS = NT = len(MONTHS)
LAST_YEAR = YEAR1
WIND_DRIVER = "wind"
SAMPLED = SATELLITE = False
DETREND = os.environ.get("DETREND", "0").lower() in ("1", "true", "yes")
COARSEN = int(os.environ.get("COARSEN", 2))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", 36))
METHOD = {"pcmci": "PCMCI", "pcmciplus": "PCMCIplus", "pcmci+": "PCMCIplus",
          "lpcmci": "LPCMCI"}[os.environ.get("METHOD", "PCMCI").lower()]
TEST = os.environ.get("TEST", "gpdc").lower()
CONTEMP = os.environ.get("CONTEMP", "0") == "1"
SEASON_MASK = os.environ.get("SEASON_MASK", "1") == "1"
MISSING = -999999.0
INTERP_GAP = int(os.environ.get("INTERP_GAP", 0))
# LOG_CHL=1: log10 of the satellite Chl before the anomalies (log-normal,
# multiplicative retrieval error)
LOG_CHL = os.environ.get("LOG_CHL", "0") == "1"
SETUP = (f"obs-{METHOD}-{TEST}-contemp{int(CONTEMP)}-mask{int(SEASON_MASK)}")
TAG = (f"_c{COARSEN}_{YEAR0}to{YEAR1}"
       + ("_detrended" if DETREND else "")
       + (f"_interp{INTERP_GAP}" if INTERP_GAP > 0 else "")
       + ("_log" if LOG_CHL else "")
       + ("" if TEST == "gpdc" else f"_{TEST}"))
OUTDIR = os.environ.get("OUTDIR", "results_obs_{method}" + TAG)
# ORAS5 drivers, as in plot_driver_trends.py
LAT_NORTH = -30.0
DEPTH_MEAN, DEPTH_STRAT = 25.0, 50.0
RHO_REF, G = 1027.5, 9.81
PATTERNS = {"theta": "*votemper*.nc", "salt": "*vosaline*.nc",
            "MLD": "*somxl010*.nc", "taux": "*sozotaux*.nc",
            "tauy": "*sometauy*.nc"}


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


def _unused_open_bsose(name):
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


def deseasonalize(ds, keep=None):
    clim = ds.groupby("time.month").mean("time")
    #clim=clim.coarsen(YC=2, XC=2, boundary='trim').mean()
    ds1=ds.groupby("time.month") - clim
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
# ORAS5 readers (from plot_driver_trends.py)
# =====================================================================
_SLOT_CACHE, _ROWS_CACHE = {}, {}
ORAS5_DIR = REF_PATTERN = REF_ROWS = NAV_LAT = NAV_LON = None
NY = NX = REF_ROWLAT = None


def main_var(ds):
    """The field in this file: the data variable with the most dims."""
    cands = [v for v in ds.data_vars
             if ds[v].ndim >= 2 and v not in ("nav_lat", "nav_lon",
                                              "time_counter_bnds",
                                              "deptht_bounds")]
    if not cands:
        return None
    return max(cands, key=lambda v: (ds[v].ndim, ds[v].size))


def month_slot(ds, path):
    """Position of this file's month on the analysis axis, or None."""
    stamp = None
    for nm in ("time_counter", "time", "t", "time_centered"):
        if nm in ds.coords or nm in ds.dims:
            try:
                stamp = pd.to_datetime(np.atleast_1d(ds[nm].values)[0])
            except (TypeError, ValueError):
                stamp = None
            break
    if stamp is None:
        m = re.search(r"((?:19|20)\d{2})(0[1-9]|1[0-2])", path.name)
        if not m:
            return None
        yy, mm = int(m.group(1)), int(m.group(2))
    else:
        yy, mm = stamp.year, stamp.month
    k = (yy - YEAR_MIN) * 12 + (mm - 1)
    return k if 0 <= k < N_MONTHS else None


def file_slots(pattern):
    """{month slot: path} for every file matching `pattern`.

    Cached: the scan opens every file to read its time stamp, and the
    same pattern is asked for up to four times (theta and salt are each
    read on two layers for the stratification).
    """
    if pattern in _SLOT_CACHE:
        return _SLOT_CACHE[pattern]
    out = {}
    for f in sorted(ORAS5_DIR.glob(pattern)):
        try:
            with xr.open_dataset(f, decode_times=True) as ds:
                k = month_slot(ds, f)
        except (OSError, ValueError):
            try:
                with xr.open_dataset(f, decode_times=False) as ds:
                    k = month_slot(ds, f)
            except (OSError, ValueError):
                continue
        if k is not None:
            out.setdefault(k, f)
    _SLOT_CACHE[pattern] = out
    return out


def depth_dim(ds, da):
    for d in ("deptht", "depthw", "depthu", "depthv", "depth", "olevel",
              "nav_lev", "lev", "z"):
        if d in da.dims:
            return d
    return None


def grid_of(path):
    """(nav_lat, nav_lon) for one file, 2-D, longitudes on -180..180."""
    with xr.open_dataset(path, decode_times=False) as ds:
        if "nav_lat" in ds:
            la = np.asarray(np.squeeze(ds["nav_lat"].values), float)
            lo = np.asarray(np.squeeze(ds["nav_lon"].values), float)
        elif "lat" in ds:
            la = np.asarray(ds["lat"].values, float)
            lo = np.asarray(ds["lon"].values, float)
        else:
            sys.exit(f"\n{path.name} carries no nav_lat/nav_lon or lat/lon")
    if la.ndim == 1:
        la, lo = np.meshgrid(la, lo, indexing="ij")
    return la, ((lo + 180.0) % 360.0) - 180.0


def rows_for(pattern):
    """Which rows of `pattern`'s own grid line up with the reference.

    ORCA025's T, U and V grids are staggered by half a cell, so a
    latitude crop applied at download time keeps a DIFFERENT NUMBER OF
    ROWS for each of them -- 373 for one wind component and 372 for the
    other is what prompted this. Rows are therefore matched by their
    mean latitude rather than by index, which pairs each reference row
    with the model row physically on top of it. No interpolation
    happens: the values are the model's own, read from the matched row.
    The residual offset is the half-cell stagger (~6 km at ORCA025,
    reported below), far under one plotted quad.
    """
    if pattern in _ROWS_CACHE:
        return _ROWS_CACHE[pattern]
    slots = file_slots(pattern)
    if not slots:
        _ROWS_CACHE[pattern] = None
        return None
    la, _ = grid_of(slots[min(slots)])
    if la.shape[1] != NX:
        sys.exit(f"\n{pattern} has {la.shape[1]} columns, the reference "
                 f"grid has {NX}. Different zonal subsets cannot be "
                 f"paired cell by cell -- re-download them on one window.")
    src = la.mean(axis=1)
    idx = np.abs(src[None, :] - REF_ROWLAT[:, None]).argmin(axis=1)
    off = float(np.abs(src[idx] - REF_ROWLAT).max())
    if la.shape[0] != len(REF_ROWS) or off > 1e-6:
        print(f"  {pattern:18s} {la.shape[0]} rows -> matched to the "
              f"reference by latitude, worst offset {off:.4f} deg")
    if off > 0.25:
        print(f"  WARNING: {pattern} rows sit up to {off:.2f} deg from "
              f"the reference. That is more than a staggered grid; check "
              f"the download window.")
    _ROWS_CACHE[pattern] = idx
    return idx


def layer_mean(vals, z, zmin, zmax):
    """Thickness-weighted mean over (zmin, zmax] metres.

    `vals` is (nz, ny, nx), `z` the level depths. Levels are weighted by
    their own thickness, so a 25 m mean is not biased by ORAS5's levels
    being ~1 m thick at the surface and ~10 m by 50 m depth.
    """
    sel = (z > zmin) & (z <= zmax)
    if not sel.any():
        sel = np.zeros(z.size, bool)
        sel[int(np.argmin(np.abs(z - 0.5 * (zmin + zmax))))] = True
    edges = np.r_[0.0, 0.5 * (z[:-1] + z[1:]), z[-1]]
    thick = np.diff(edges)[sel].astype("float64")
    sub = vals[sel]
    w = thick.reshape(-1, 1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = np.isfinite(sub)
        num = np.where(ok, sub * w, 0.0).sum(axis=0)
        den = np.where(ok, np.broadcast_to(w, sub.shape), 0.0).sum(axis=0)
    return np.where(den > 0, num / np.maximum(den, 1e-12),
                    np.nan).astype("float32")


def read_one(path, collapse, rows):
    """One month on the native grid, reference rows only, as float32."""
    with xr.open_dataset(path, decode_times=False) as ds:
        name = main_var(ds)
        if name is None:
            return None
        da = ds[name].squeeze(drop=True)
        zname = depth_dim(ds, da)
        if collapse is not None and zname is not None:
            z = np.asarray(ds[zname].values, float) \
                if zname in ds else np.arange(da.sizes[zname], dtype=float)
            da = da.transpose(zname, ...)
            v = layer_mean(np.asarray(da.values, "float32"), z, *collapse)
        else:
            v = np.asarray(np.squeeze(da.values), "float32")
        v = np.squeeze(v)
    if v.ndim != 2:
        return None
    out = v[rows]
    # land is 0 or a huge fill value in ORAS5; both become gaps
    return np.where(np.isfinite(out) & (np.abs(out) < 1e15) & (out != 0.0),
                    out, np.nan)


def read_stack(items, combine=None, label=""):
    """(336, NY, NX) stack, streamed a month at a time.

    `items` is a list of (pattern, collapse); with more than one, only
    the months present in ALL of them are filled and `combine` is
    applied to that month's arrays. Streaming the combination is what
    keeps stratification -- four fields on two layers -- inside one
    stack's worth of memory instead of four.
    """
    maps = [file_slots(p) for p, _ in items]
    if any(not m for m in maps):
        return None, 0
    rows = [rows_for(p) for p, _ in items]
    if any(r is None for r in rows):
        return None, 0
    slots = sorted(set(maps[0]).intersection(*maps[1:])) \
        if len(maps) > 1 else sorted(maps[0])
    out = np.full((N_MONTHS, NY, NX), np.nan, "float32")
    seen = 0
    for k in slots:
        vals = [read_one(maps[j][k], items[j][1], rows[j])
                for j in range(len(items))]
        if any(v is None for v in vals):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[k] = vals[0] if combine is None else combine(*vals)
        seen += 1
    nfile = len(set().union(*[set(m.values()) for m in maps]))
    print(f"  {label:7s} {seen:4d} of {N_MONTHS} months from "
          f"{nfile} file(s), native {NY} x {NX}")
    return (out, seen) if seen else (None, 0)


def sigma0(S, T):
    """Potential density anomaly, EOS-80 polynomial (gsw if available)."""
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


def find_oras5():
    if os.environ.get("ORAS5_DIR"):
        return Path(os.environ["ORAS5_DIR"])
    root = Path(WORK)
    for d in [root / "reanalysis/oras5_so", root, root / "oras5_so",
              root / "reanalysis", root / "oras5"]:
        if d.is_dir() and any(d.glob("*votemper*.nc")):
            return d
    hits = sorted(root.rglob("*votemper*.nc"))
    if hits:
        return hits[0].parent
    sys.exit(f"no ORAS5 votemper files under {root}")


def init_oras5():
    """The ORCA025 rows south of LAT_NORTH (+1 deg margin)."""
    global ORAS5_DIR, REF_PATTERN, REF_ROWS, NAV_LAT, NAV_LON, NY, NX
    global REF_ROWLAT
    ORAS5_DIR = find_oras5()
    slots = file_slots(PATTERNS["theta"])
    if not slots:
        sys.exit(f"no ORAS5 votemper files for {YEAR_MIN}-{YEAR_MAX} "
                 f"in {ORAS5_DIR}")
    la, lo = grid_of(slots[min(slots)])
    keep = (la <= LAT_NORTH + 1.0).any(axis=1)
    REF_PATTERN, REF_ROWS = PATTERNS["theta"], np.where(keep)[0]
    NAV_LAT, NAV_LON = la[REF_ROWS], lo[REF_ROWS]
    NY, NX = NAV_LAT.shape
    REF_ROWLAT = NAV_LAT.mean(axis=1)
    _ROWS_CACHE[REF_PATTERN] = REF_ROWS
    print(f"ORAS5: {ORAS5_DIR}, {YEAR_MIN}-{YEAR_MAX}, ORCA025 rows "
          f"{NY} x {NX}", flush=True)


def oras5_stack(name):
    if name == "theta":
        return read_stack([(PATTERNS["theta"], (0.0, DEPTH_MEAN))],
                          label="theta")
    if name == "salt":
        return read_stack([(PATTERNS["salt"], (0.0, DEPTH_MEAN))],
                          label="salt")
    if name == "MLD":
        return read_stack([(PATTERNS["MLD"], None)], label="MLD")
    if name == "wind":
        return read_stack([(PATTERNS["taux"], None),
                           (PATTERNS["tauy"], None)],
                          combine=lambda x, y: np.sqrt(x ** 2 + y ** 2),
                          label="wind")
    if name == "N":
        dz = 0.5 * DEPTH_STRAT

        def strat(tu, su, td, sd):
            return (-(G / RHO_REF)
                    * (sigma0(su, tu) - sigma0(sd, td)) / dz).astype("float32")

        return read_stack([(PATTERNS["theta"], (0.0, DEPTH_MEAN)),
                           (PATTERNS["salt"], (0.0, DEPTH_MEAN)),
                           (PATTERNS["theta"], (DEPTH_MEAN, DEPTH_STRAT)),
                           (PATTERNS["salt"], (DEPTH_MEAN, DEPTH_STRAT))],
                          combine=strat, label="N")


# =====================================================================
# The ORAS5 (ORCA025) grid, south of LAT_NORTH, and the satellite on it
# =====================================================================
init_oras5()
_ROWSEL = np.flatnonzero(REF_ROWLAT <= LAT_NORTH)
_lon_mid = NAV_LON[_ROWSEL[len(_ROWSEL) // 2]]
_u, _first = np.unique(np.round(_lon_mid, 4), return_index=True)
_COLS = _first[np.argsort(_u)]                 # sorted, wrap duplicates out
YC_O = REF_ROWLAT[_ROWSEL].astype("float64")
XC_O = _lon_mid[_COLS].astype("float64")       # -180..180, increasing
_dev = float(np.nanmax(np.abs(NAV_LAT[_ROWSEL][:, _COLS] - YC_O[:, None])))
_dlo = float(np.nanmax(np.abs(((NAV_LON[_ROWSEL][:, _COLS] - XC_O[None, :]
                                + 180.0) % 360.0) - 180.0)))
COORDS = {"time": MONTHS, "YC": YC_O, "XC": XC_O}
print(f"ORAS5 grid: {YC_O.size} rows ({YC_O.min():.2f} to {YC_O.max():.2f}) x "
      f"{XC_O.size} columns ({NX - XC_O.size} wrap column(s) dropped); "
      f"largest departure from a regular grid: {_dev:.4f} deg lat, "
      f"{_dlo:.4f} deg lon; blocks of {COARSEN} x {COARSEN}; "
      f"{YEAR0}-{YEAR1}, {NT} months", flush=True)
if _dev > 0.01 or _dlo > 0.01:
    print("  WARNING: the ORCA025 rows are not regular here; the 1-D "
          "coordinates are approximate", flush=True)


def on_oras5(name):
    """One ORAS5 driver on its own grid, south of LAT_NORTH, columns
    sorted by longitude. No regridding."""
    stack, seen = oras5_stack(name)
    if stack is None:
        raise SystemExit(f"no ORAS5 files for {name} in {ORAS5_DIR}")
    a = stack[:, _ROWSEL][:, :, _COLS]
    del stack
    return xr.DataArray(a, dims=("time", "YC", "XC"), coords=COORDS)


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
        print(f"  WARNING: {missing} month(s) of {YEAR0}-{LAST_YEAR} are not "
              "in the satellite file and are missing here too")
    return out


# ---- chlorophyll: the satellite on the ORAS5 grid, ocean cells only
theta_raw = on_oras5("theta")
OCEAN = np.isfinite(theta_raw.values).any(axis=0)
sat = satellite_on_model_grid(YC_O, XC_O, MONTHS)
sat[:, ~OCEAN] = np.nan
if LOG_CHL:
    with np.errstate(invalid="ignore", divide="ignore"):
        sat = np.where(sat > 0, np.log10(sat), np.nan).astype("float32")
    print("LOG_CHL=1: log10 of the satellite chlorophyll before the anomalies", flush=True)
print(f"satellite on the ORAS5 grid: "
      f"{100 * np.isfinite(sat).sum() / max(OCEAN.sum() * NT, 1):.0f}% of "
      "ocean cell-months with data", flush=True)
chl_weighted = deseasonalize(xr.DataArray(sat, dims=("time", "YC", "XC"),
                                          coords=COORDS), keep="Chl")
del sat

# ---- drivers, one at a time, on the ORAS5 grid
data_THETA_mean = deseasonalize(theta_raw, keep="Theta")
del theta_raw
data_salt_mean = deseasonalize(on_oras5("salt"), keep="salt")
N = deseasonalize(on_oras5("N"), keep="N")
data_MLD = deseasonalize(on_oras5("MLD"), keep="MLD")
time_mean_wind_amp_anom2 = deseasonalize(on_oras5("wind"), keep="wind")

feats = {
    WIND_DRIVER: time_mean_wind_amp_anom2.values,
    "salt": data_salt_mean.values,
    "N": N.values,
    "MLD": data_MLD.values,
    "Theta": data_THETA_mean.values
}

# ---------------------------------------------------------------------
# INTERP_GAP=N: missing chlorophyll months filled by linear interpolation
# in time, in the block anomalies, but only for gaps of at most N months
# with a valid month on both sides. Longer gaps (the polar winter) and the
# ends of the record stay missing: nothing is extrapolated. Default 0 (off).
# ---------------------------------------------------------------------
def fill_short_gaps(a, max_gap):
    """(T, Y, X) -> copy with interior gaps of <= max_gap months filled
    linearly along time; returns (filled array, number of values filled)."""
    a = np.array(a, dtype="float64", copy=True)
    T = a.shape[0]
    flat = a.reshape(T, -1)
    t = np.arange(T)
    n_filled = 0
    for j in np.flatnonzero(np.isfinite(flat).any(axis=0)
                            & ~np.isfinite(flat).all(axis=0)):
        y = flat[:, j]
        ok = np.isfinite(y)
        idx = np.flatnonzero(ok)
        if idx.size < 2:
            continue
        gaps = np.diff(idx) - 1                 # missing months between
        for k in np.flatnonzero((gaps >= 1) & (gaps <= max_gap)):
            i0, i1 = idx[k], idx[k + 1]
            tt = t[i0 + 1:i1]
            y[i0 + 1:i1] = y[i0] + (y[i1] - y[i0]) * (tt - i0) / (i1 - i0)
            n_filled += tt.size
    return flat.reshape(a.shape), n_filled


chl = chl_weighted.values

if INTERP_GAP > 0:
    _before = int(np.isfinite(chl).sum())
    chl, _nf = fill_short_gaps(chl, INTERP_GAP)
    chl = chl.astype("float32")
    chl_weighted = chl_weighted.copy(data=chl)
    print(f"INTERP_GAP={INTERP_GAP}: {_nf:,} block-months of chlorophyll "
          f"filled in time ({100 * _nf / max(_before, 1):.1f}% more than "
          f"the {_before:,} observed); longer gaps left missing", flush=True)
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


def block_regions(lat2d=None, nx=None):
    """{region name: (ny, nx) bool} on the block grid of this run.

    The SSH zones of ZONES_FILE are on the B-SOSE grid: every ORAS5 cell
    takes the zone of the nearest B-SOSE cell, and every COARSEN x COARSEN
    block the zone of most of its cells (at least half), the blocks being
    those of xarray's coarsen(boundary="trim") -- from the first row and
    column, the remainder dropped. Without the zone file: latitude bands."""
    lat_b = np.asarray(lat_grid, float)
    lon_b = np.asarray(lon_grid, float)
    ny, nxb = lat_b.size, lon_b.size
    key = (ny, nxb)
    if key in _REGION_CACHE:
        return _REGION_CACHE[key]
    masks = None
    if ZONES_FILE.lower() != "none" and os.path.exists(ZONES_FILE):
        with xr.open_dataset(ZONES_FILE) as z:
            zone = z["zone"].transpose("YC", "XC")
            yz, xz = zone.YC.values.astype(float), zone.XC.values.astype(float)
            zone = zone.values
            fv = str(z["zone"].attrs.get("flag_values", "")).split()
            fm = str(z["zone"].attrs.get("flag_meanings", "")).split()
        names = ({int(v): m for v, m in zip(fv, fm) if int(v) != 0}
                 if fv and len(fv) == len(fm) else
                 {1: "N_SAF", 2: "ACC", 3: "S_SB"})
        k = COARSEN
        iy = np.abs(yz[None, :] - YC_O[:, None]).argmin(axis=1)
        far = np.abs(yz[iy] - YC_O) > 0.5          # outside the zone file
        dl = ((xz[None, :] - XC_O[:, None] + 180.0) % 360.0) - 180.0
        ix = np.abs(dl).argmin(axis=1)
        zn = zone[iy][:, ix]
        zn[far] = 0
        zn = np.where(OCEAN, zn, 0)
        b = zn[:ny * k, :nxb * k].reshape(ny, k, nxb, k)
        counts = np.stack([(b == c).sum(axis=(1, 3)) for c in names])
        best = np.array(list(names))[counts.argmax(axis=0)]
        zb = np.where(counts.max(axis=0) >= k * k / 2, best, 0)
        masks = {n: zb == c for c, n in names.items()}
        print("regions from " + ZONES_FILE + " (nearest B-SOSE cell): " +
              ", ".join(f"{n} {int(m.sum())} blocks" for n, m in masks.items()),
              flush=True)
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
        cb.set_label(f'{f} → Chl partial corr (lag {tau})', fontsize=16, labelpad=15)

        # Save figure
        plt.savefig(f"pcorr_map_{f}_lag{tau}_polar_"+str(q)+TAG+".pdf", dpi=300, bbox_inches='tight')
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


def run_causal_driver(features, chl, alpha=0.05, min_samples=MIN_SAMPLES, tau_max=2,
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
            kw = dict(var_names=feat_names + ["Chl"], missing_flag=MISSING)
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
import re
import sys
import glob
import warnings
from pathlib import Path
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

    Regions: block_regions() -- the SSH zones N_SAF, ACC, S_SB of
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
    "wind":  "#ff7f0e",  # orange  ✅
    "wek":   "#ff7f0e",  # same slot as the wind
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


def collect_regional_values(val_maps, p_maps, lat2d, alpha=0.05, region="ACC"):
    """
    Extract significant spatial samples for each driver inside `region`.
    val_maps : dict driver -> array (tau, YC, XC)  (we use tau=0 slice)
    p_maps   : dict driver -> array (tau, YC, XC)
    lat2d    : 1D or 2D lat array (YC or YCxXC)
    region   : a name in REGION_NAMES (N_SAF | ACC | S_SB)
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
    eligible = complete.sum(axis=0) >= MIN_SAMPLES                  # (W, Y)
    del complete
    print(f"  {tag}: {int(eligible.sum())} of {W * Y} cells have {MIN_SAMPLES} or "
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
         "Chl": "mg m$^{-3}$"}


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
    names = [k for k in anoms if k != "Chl"] + ["Chl"]
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
    fig.suptitle(f"Linear trend of the anomalies, {YEAR0}-{LAST_YEAR}, "
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
