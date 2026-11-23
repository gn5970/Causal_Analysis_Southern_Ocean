"""Trends of the physical drivers, ORAS5 and B-SOSE: two figures.

    python plot_driver_trends.py                 both figures
    FIGS=oras5 python plot_driver_trends.py      one of them
    FIGS=bsose python plot_driver_trends.py

    driver_trends_oras5.pdf    ORAS5, 1998-2025
    driver_trends_bsose.pdf    B-SOSE, 2013-2024

One column per driver (temperature and salinity over the top 25 m,
stratification between 0-25 and 25-50 m, mixed-layer depth, wind-stress
amplitude), two rows:

    top     trend, block by block, stippled where FDR q <= 0.05
    bottom  trend of each calendar month, zonal mean in LAT_STEP bands
            (only with SEASON=1; by default the maps alone)

THE SAME STEPS AS fig1_chl_trend.py, so the drivers can be read against
the chlorophyll figure block for block:

  * ORAS5 is put on the B-SOSE grid: each 1/6 deg B-SOSE cell takes the
    value of the ORCA025 cell nearest to it (ORCA025 rows and columns
    are lines of constant latitude and longitude south of 30S), so
    nothing is smoothed here and land stays land;
  * the anomalies are the earlier script's deseasonalize, in its order,
    with xarray:
        clim = ds.groupby("time.month").mean("time")
        ds1  = ds.groupby("time.month") - clim
        ds1  = ds1.coarsen(YC=k, XC=k, boundary="trim").mean()
        ds1 / ds1.std("time")                 (STANDARDIZE=1, default)
    with k = COARSEN (4, ~0.7 deg), the same blocks for both products;
  * trend per block, least squares against time in years; p-value with
    an effective sample size reduced for the lag-1 autocorrelation of the
    residuals, Benjamini-Hochberg FDR over the map;
  * bottom row: per block, each calendar month fitted on its own (one
    value a year), then the area-weighted mean over the latitude band.

Units: standard deviations per year by default; STANDARDIZE=0 gives
physical units per year (degC, PSU, s-2, m, N m-2).

Settings (environment):
    WORK                  folder with the data
    ORAS5_DIR             ORAS5 monthly files (searched below WORK if unset)
    BSOSE_DIR             B-SOSE files (WORK/bsose)
    ORAS5_YEAR0/1         ORAS5 window (1998, 2025)
    BSOSE_NT              B-SOSE months from Jan 2013 (144)
    COARSEN               k x k blocks of the B-SOSE grid (4)
    STANDARDIZE           1 (default) or 0
    SAME_SCALE=1          one colour scale per driver across both figures
    SEASON=1              add the row of trends by calendar month
    FIGS                  oras5,bsose (default both)

Needs ~30 GB for ORAS5 on the B-SOSE grid: run it on a compute node.
"""
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

FMT = mticker.FuncFormatter(lambda v, _: f"{v:.3g}")

WORK = os.environ.get("WORK", "/work/bb1596/b384036/b384036/causality")
BSOSE_DIR = os.environ.get("BSOSE_DIR", os.path.join(WORK, "bsose"))
BSOSE_NT = int(os.environ.get("BSOSE_NT", 144))
YEAR_MIN = int(os.environ.get("ORAS5_YEAR0", 1998))
YEAR_MAX = int(os.environ.get("ORAS5_YEAR1", 2025))
MONTHS = pd.period_range(f"{YEAR_MIN}-01", f"{YEAR_MAX}-12",
                         freq="M").to_timestamp()
N_MONTHS = len(MONTHS)
COARSEN = int(os.environ.get("COARSEN", 4))
STANDARDIZE = os.environ.get("STANDARDIZE", "1") == "1"
SAME_SCALE = os.environ.get("SAME_SCALE", "0") == "1"
# SEASON=1 adds the second row (trend by calendar month and latitude)
SEASON = os.environ.get("SEASON", "0") == "1"
FIGS = [f.strip().lower() for f in
        os.environ.get("FIGS", "oras5,bsose").split(",") if f.strip()]

LAT_NORTH, LAT_SOUTH_PLOT = -30.0, -80.0
DEPTH_MEAN, DEPTH_STRAT = 25.0, 50.0       # upper layer, lower layer
RHO_REF, G = 1027.5, 9.81
LAT_STEP = 2.5
MIN_MONTHS_TREND = 120
MIN_YEARS_TREND = 12
MIN_CELLS_ZONAL = 20
TREND_ALPHA = 0.05
BANDS = {"Subtropics": (-45, -30), "ACC": (-60, -45), "HighLat": (-90, -60)}

ORDER = ["theta", "salt", "N", "MLD", "wind"]
TITLES = {"theta": "Temperature, 0-25 m", "salt": "Salinity, 0-25 m",
          "N": "Stratification N$^2$", "MLD": "Mixed-layer depth",
          "wind": "Wind-stress amplitude"}
PHYS = {"theta": "$^\\circ$C", "salt": "PSU", "N": "s$^{-2}$", "MLD": "m",
        "wind": "N m$^{-2}$"}
COLORS = {"theta": "#b15928", "salt": "#9467bd", "N": "#2ca02c",
          "MLD": "#1f77b4", "wind": "#ff7f0e"}
PATTERNS = {"theta": "*votemper*.nc", "salt": "*vosaline*.nc",
            "MLD": "*somxl010*.nc", "taux": "*sozotaux*.nc",
            "tauy": "*sometauy*.nc"}


def unit(name):
    return ("$\\sigma$ yr$^{-1}$" if STANDARDIZE
            else f"{PHYS[name]} yr$^{{-1}}$")


def to_unit(slope):
    return slope


def to_pm180(lon, *fields):
    """Longitudes to -180..180, sorted, the fields' last axis with them."""
    lon = ((np.asarray(lon, "float64") + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    return (lon[order],) + tuple(f[..., order] for f in fields)


# =====================================================================
# The method of fig1_chl_trend.py
# =====================================================================
def deseasonalise_xr(a, t, k):
    """The earlier script's deseasonalize, in its order (see the top).
    a is (time, y, x) on the B-SOSE grid; returns the anomalies on the
    k x k block grid."""
    da = xr.DataArray(a, dims=("time", "YC", "XC"), coords={"time": t})
    clim = da.groupby("time.month").mean("time")
    ds1 = da.groupby("time.month") - clim
    if k > 1:
        ds1 = ds1.coarsen(YC=k, XC=k, boundary="trim").mean()
    if STANDARDIZE:
        sd = ds1.std("time")
        ds1 = ds1 / sd.where(sd > 0)
    return ds1.transpose("time", "YC", "XC").values.astype("float32")


def month_band_trends(field, months, years, lat, south):
    """Trend of each calendar month, (band, month): per block, the slope
    of that month's one-value-a-year series, then the area-weighted mean
    of the block slopes over the latitude band; and the p-value of a
    one-sample t-test of the block slopes against zero (sign
    consistency around the circle)."""
    from scipy import stats
    step = max(LAT_STEP, float(np.abs(np.diff(lat)).max()) * 1.01)
    edges = np.arange(south, LAT_NORTH + 1e-6, step)
    nb = edges.size - 1
    band = np.digitize(lat, edges) - 1
    coslat = np.cos(np.deg2rad(lat))
    tr = np.full((nb, 12), np.nan)
    pv = np.full((nb, 12), np.nan)
    for m in range(1, 13):
        sel = months == m
        if sel.sum() < MIN_YEARS_TREND:
            continue
        block = field[sel].astype("float64")
        yy = years[sel]
        ok = np.isfinite(block)
        n = ok.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = np.where(ok, yy[:, None, None], np.nan)
            dx = x - np.nanmean(x, axis=0)
            dy = block - np.nanmean(block, axis=0)
            sxx = np.nansum(dx ** 2, axis=0)
            slope = np.where((n >= MIN_YEARS_TREND) & (sxx > 0),
                             np.nansum(dx * dy, axis=0)
                             / np.maximum(sxx, 1e-12), np.nan)
        for b in range(nb):
            rows = band == b
            if not rows.any():
                continue
            v = slope[rows]
            w = np.broadcast_to(coslat[rows][:, None], v.shape)
            f = np.isfinite(v)
            if f.sum() >= MIN_CELLS_ZONAL:
                tr[b, m - 1] = np.average(v[f], weights=w[f])
                pv[b, m - 1] = stats.ttest_1samp(v[f], 0.0).pvalue
    return tr, pv, edges


def trend_per_cell(anom, years):
    """OLS slope per cell, in log10 units per year, and its p-value.

    The p-value uses an effective sample size reduced for the lag-1
    autocorrelation of the residuals. Monthly anomalies are not
    independent month to month, and a naive t-test on 336 of them calls
    almost everything significant.
    """
    from scipy import stats
    ok = np.isfinite(anom)
    n = ok.sum(axis=0)
    x = np.where(ok, years[:, None, None], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.nanmean(x, axis=0)
        ym = np.nanmean(anom, axis=0)
        dx, dy = x - xm, anom - ym
        sxx = np.nansum(dx ** 2, axis=0)
        slope = np.where((n >= MIN_MONTHS_TREND) & (sxx > 0),
                         np.nansum(dx * dy, axis=0) / np.maximum(sxx, 1e-12),
                         np.nan)
        resid = anom - (ym + slope * dx)
        # lag-1 autocorrelation of the residuals
        a, b = resid[:-1], resid[1:]
        m = np.isfinite(a) & np.isfinite(b)
        r1 = np.clip(np.where(m, a * b, 0.0).sum(axis=0)
                     / np.maximum(np.nansum(resid ** 2, axis=0), 1e-12),
                     -0.95, 0.95)
        neff = np.clip(n * (1 - r1) / np.maximum(1 + r1, 1e-6), 5, n)
        se = np.sqrt(np.nansum(resid ** 2, axis=0)
                     / np.maximum(neff - 2, 1) / np.maximum(sxx, 1e-12))
        tval = slope / np.maximum(se, 1e-20)
        pval = 2 * stats.t.sf(np.abs(tval), np.maximum(neff - 2, 1))
    return slope, np.where(np.isfinite(slope), pval, np.nan), n


def fdr(pv, alpha):
    """Benjamini-Hochberg q-values over every finite cell."""
    flat = pv.ravel()
    ok = np.isfinite(flat)
    n = int(ok.sum())
    q = np.full(flat.shape, np.nan)
    if n:
        p = flat[ok]
        order = np.argsort(p)
        ranked = p[order] * n / np.arange(1, n + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        qq = np.empty(n)
        qq[order] = np.clip(ranked, 0, 1)
        q[ok] = qq
    return q.reshape(pv.shape)


def nice_lim(*fields):
    """95th percentile of |trend| over the given fields, rounded up to
    a round number (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8 x 10^k)."""
    v = np.concatenate([np.abs(f[np.isfinite(f)]).ravel() for f in fields])
    x = float(np.percentile(v, 95)) if v.size else 1.0
    if not x > 0:
        return 1.0
    e = 10.0 ** np.floor(np.log10(x))
    m = next(c for c in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10) if c * e >= x)
    return m * e


def sym_ticks(lim):
    """Five ticks: -lim, -lim/2, 0, lim/2, lim (narrow colour bars)."""
    return np.array([-lim, -lim / 2, 0.0, lim / 2, lim])


_COAST = None


def coast(ax):
    """Coastlines only if cartopy already has them on disk. A compute
    node cannot download them, and the failure would come at savefig,
    taking the whole figure with it."""
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
        ax.coastlines(resolution="110m", lw=0.4)


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


def analyse(a, lat, lon, t, label, name):
    """One driver: anomalies, blocks, trend map, seasonal trends."""
    months = t.month.to_numpy()
    years = (t.year.to_numpy() + (months - 0.5) / 12.0).astype(float)
    k = max(1, COARSEN)
    ny, nx = (lat.size // k) * k, (lon.size // k) * k
    a = a[:, :ny, :nx]
    anoms = np.empty((t.size, ny // k, nx // k), "float32")
    rows = 24 * k
    for j0 in range(0, ny, rows):
        j1 = min(j0 + rows, ny)
        anoms[:, j0 // k:j1 // k] = deseasonalise_xr(a[:, j0:j1], t, k)
    lat = lat[:ny].reshape(-1, k).mean(axis=1)
    lon = lon[:nx].reshape(-1, k).mean(axis=1)
    slope, pval, _ = trend_per_cell(anoms.astype("float64"), years)
    q = fdr(pval, TREND_ALPHA)
    sig = np.isfinite(q) & (q <= TREND_ALPHA)
    tr, pv, edges = month_band_trends(anoms, months, years, lat,
                                      LAT_SOUTH_PLOT)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    fin = np.isfinite(slope)
    line = [f"  {label} {name:6s}"]
    for nm, (lo, hi) in BANDS.items():
        m = fin & ((lat > lo) & (lat <= hi))[:, None]
        if m.any():
            line.append(f"{nm} {np.average(slope[m], weights=w[m]):+.4g}"
                        f" ({100 * (sig & m).sum() / m.sum():.0f}% sig)")
    print(",  ".join(line), flush=True)
    return dict(val=slope, sig=sig, tr=tr, pv=pv, edges=edges, lat=lat,
                lon=lon, period=f"{t[0]:%Y}-{t[-1]:%Y}")


# =====================================================================
# B-SOSE
# =====================================================================
BSOSE_WORDS = {"theta": ("Theta", ["THETA"]), "salt": ("Salt", ["SALT"]),
               "MLD": ("MLD", ["BLGMLD", "MLD"]),
               "taux": ("oceTAUX", ["oceTAUX"]),
               "tauy": ("oceTAUY", ["oceTAUY"])}


def bsose_path(key):
    word = BSOSE_WORDS[key][0]
    hits = sorted(p for p in glob.glob(os.path.join(BSOSE_DIR, f"{word}_*.nc"))
                  if "snap" not in os.path.basename(p).lower())
    monthly = [h for h in hits if "monthly" in h.lower()]
    path = (monthly or hits or [None])[0]
    if path is None:
        sys.exit(f"no B-SOSE file {word}_*.nc in {BSOSE_DIR}")
    return path


def bsose_var(ds, key):
    for n in BSOSE_WORDS[key][1]:
        if n in ds:
            return ds[n]
    return ds[main_var(ds)]


_BGRID = {}


def bsose_grid():
    """B-SOSE tracer grid south of LAT_NORTH: latitudes, longitudes on
    -180..180 (sorted), the column order that sorts them, the rows kept,
    and the ocean mask of the top cell."""
    if _BGRID:
        return _BGRID
    with xr.open_dataset(bsose_path("theta"), decode_times=False) as ds:
        lat = ds["YC"].values.astype("float64")
        lon = ds["XC"].values.astype("float64")
        h = ds["hFacC"]
        zd = h.dims[0]
        wet = h.isel({zd: 0}).transpose("YC", "XC").values > 0
    rows = np.flatnonzero(lat <= LAT_NORTH)
    lonp = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lonp)
    _BGRID.update(lat=lat[rows], lon=lonp[order], rows=rows, order=order,
                  wet=wet[rows][:, order])
    return _BGRID


def bsose_time(nt):
    return pd.date_range("2013-01-01", periods=nt, freq="MS")


def bsose_layers(key, layers):
    """Thickness-weighted (drF x hFacC) means of a 3D field over each
    (zmin, zmax] layer: a list of (time, y, x) arrays on the grid."""
    gr = bsose_grid()
    with xr.open_dataset(bsose_path(key), decode_times=False) as ds:
        v = bsose_var(ds, key)
        zd = next(d for d in v.dims if d.upper().startswith("Z"))
        v = v.transpose("time", zd, "YC", "XC")
        depth = np.abs(ds[zd].values.astype("float64"))
        nt = min(BSOSE_NT, v.sizes["time"])
        hf = ds["hFacC"].transpose(zd, "YC", "XC").values[:, gr["rows"]]
        drf = ds["drF"].values.astype("float64")
        out = []
        for zmin, zmax in layers:
            keep = np.flatnonzero((depth > zmin) & (depth <= zmax))
            w = drf[keep][:, None, None] * hf[keep]
            den = w.sum(axis=0)
            arr = np.empty((nt,) + den.shape, "float32")
            for t0 in range(0, nt, 12):
                t1 = min(t0 + 12, nt)
                a = v.isel(time=slice(t0, t1), **{zd: keep}).values[
                    :, :, gr["rows"]].astype("float64")
                with np.errstate(invalid="ignore", divide="ignore"):
                    arr[t0:t1] = np.where(den > 0,
                                          (a * w[None]).sum(axis=1) / den,
                                          np.nan)
            out.append(arr[..., gr["order"]])
    return out, bsose_time(nt)


def bsose_2d(key):
    gr = bsose_grid()
    with xr.open_dataset(bsose_path(key), decode_times=False) as ds:
        v = bsose_var(ds, key)
        nt = min(BSOSE_NT, v.sizes["time"])
        ydim = next(d for d in v.dims if d.startswith("Y"))
        xdim = next(d for d in v.dims if d.startswith("X"))
        a = v.isel(time=slice(0, nt)).transpose("time", ydim, xdim).values
    return a.astype("float32"), ydim, xdim, bsose_time(nt)


def bsose_driver(name):
    gr = bsose_grid()
    if name in ("theta", "salt"):
        (a,), t = bsose_layers(name, [(-1.0, DEPTH_MEAN)])
    elif name == "N":
        (tu, td), t = bsose_layers("theta", [(-1.0, DEPTH_MEAN),
                                             (DEPTH_MEAN, DEPTH_STRAT)])
        (su, sd), _ = bsose_layers("salt", [(-1.0, DEPTH_MEAN),
                                            (DEPTH_MEAN, DEPTH_STRAT)])
        dz = 0.5 * DEPTH_STRAT
        with np.errstate(invalid="ignore"):
            a = (-(G / RHO_REF) * (sigma0(su, tu) - sigma0(sd, td)) / dz
                 ).astype("float32")
        del tu, td, su, sd
    elif name == "MLD":
        a, _, _, t = bsose_2d("MLD")
        a = a[:, gr["rows"]][..., gr["order"]]
    elif name == "wind":
        # C grid: taux on the west face of each cell (XG), tauy on the
        # south face (YG); each moved to the cell centre by the mean of
        # the two faces around it.
        tx, _, _, t = bsose_2d("taux")
        tx = 0.5 * (tx + np.roll(tx, -1, axis=2))
        ty, _, _, _ = bsose_2d("tauy")
        tyc = np.full_like(ty, np.nan)
        tyc[:, :-1] = 0.5 * (ty[:, :-1] + ty[:, 1:])
        a = np.sqrt(tx ** 2 + tyc ** 2)[:, gr["rows"]][..., gr["order"]]
        del tx, ty, tyc
    a = np.where(gr["wet"][None], a, np.nan).astype("float32")
    if name == "MLD":
        a[a == 0] = np.nan
    print(f"  B-SOSE {name:6s} {a.shape[0]} months, grid "
          f"{a.shape[1]} x {a.shape[2]}", flush=True)
    return a, t


# =====================================================================
# ORAS5, on the B-SOSE grid
# =====================================================================
_SLOT_CACHE, _ROWS_CACHE = {}, {}
ORAS5_DIR = REF_PATTERN = REF_ROWS = NAV_LAT = NAV_LON = None
NY = NX = REF_ROWLAT = None


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


def oras5_to_bsose_index():
    """For each B-SOSE row and column, the nearest ORCA025 row and
    column (rows by their mean latitude, columns by their longitude on
    the middle row); rows further than 0.3 deg are left empty."""
    gr = bsose_grid()
    iy = np.abs(REF_ROWLAT[None, :] - gr["lat"][:, None]).argmin(axis=1)
    far = np.abs(REF_ROWLAT[iy] - gr["lat"]) > 0.3
    colon = NAV_LON[NY // 2]
    d = np.abs(((colon[None, :] - gr["lon"][:, None] + 180.0) % 360.0) - 180.0)
    ix = d.argmin(axis=1)
    return iy, ix, far


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


def oras5_driver(name, index):
    stack, seen = oras5_stack(name)
    if stack is None:
        return None, None
    iy, ix, far = index
    a = stack[:, iy][:, :, ix]
    a[:, far] = np.nan
    del stack
    return a, MONTHS


# =====================================================================
# Figures
# =====================================================================
def figure(results, product, lims, out):
    names = [n for n in ORDER if n in results]
    ncol = len(names)
    if SEASON:
        fig = plt.figure(figsize=(4.4 * ncol, 10.5), dpi=200)
        gs = fig.add_gridspec(2, ncol, height_ratios=[1.15, 1], hspace=0.32,
                              wspace=0.35)
    else:
        fig = plt.figure(figsize=(4.4 * ncol, 5.6), dpi=200)
        gs = fig.add_gridspec(1, ncol, wspace=0.35)
    proj = ccrs.SouthPolarStereo(central_longitude=0)
    for i, name in enumerate(names):
        r, lim = results[name], lims[name]
        # top: the trend, block by block
        ax = fig.add_subplot(gs[0, i], projection=proj)
        m = ax.pcolormesh(r["lon"], r["lat"], np.ma.masked_invalid(r["val"]),
                          cmap=plt.cm.RdBu_r, vmin=-lim, vmax=lim,
                          shading="nearest", rasterized=True,
                          transform=ccrs.PlateCarree())
        kk = max(1, int(round(1.0 / abs(float(np.diff(r["lat"]).mean())))))
        s = np.zeros_like(r["sig"])
        s[::kk, ::kk] = r["sig"][::kk, ::kk]
        if s.any():
            lon2d, lat2d = np.meshgrid(r["lon"], r["lat"])
            ax.scatter(lon2d[s], lat2d[s], s=0.8, c="k", alpha=0.6,
                       linewidths=0, transform=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -90, LAT_NORTH], crs=ccrs.PlateCarree())
        coast(ax)
        ax.gridlines(lw=0.3, alpha=0.5)
        ax.set_title(f"{'abcde'[i]}   {TITLES[name]}", loc="left",
                     fontsize=13, color=COLORS[name])
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", shrink=0.9,
                          pad=0.05, extend="both")
        cb.set_ticks(sym_ticks(lim))
        cb.ax.xaxis.set_major_formatter(FMT)
        cb.set_label(f"trend ({unit(name)})", fontsize=13)
        cb.ax.tick_params(labelsize=11)
        if not SEASON:
            continue
        # bottom: by calendar month and latitude
        ax = fig.add_subplot(gs[1, i])
        m = ax.pcolormesh(np.arange(0.5, 13.5), r["edges"], r["tr"],
                          cmap=plt.cm.RdBu_r, vmin=-lim, vmax=lim,
                          shading="flat", rasterized=True)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=10)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(LAT_SOUTH_PLOT, LAT_NORTH)
        ax.set_xlabel("Month", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        if i == 0:
            ax.set_ylabel("Latitude ($^\\circ$)", fontsize=12)
        ax.set_title(f"{'fghij'[i]}   by calendar month, zonal mean",
                     loc="left", fontsize=12)
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", shrink=0.9,
                          pad=0.18, extend="both")
        cb.set_ticks(sym_ticks(lim))
        cb.ax.xaxis.set_major_formatter(FMT)
        cb.set_label(f"trend ({unit(name)})", fontsize=13)
        cb.ax.tick_params(labelsize=11)
    period = next(iter(results.values()))["period"].replace("-", "–")
    fig.suptitle(f"{product}, {period}: trend of the drivers "
                 f"(stippled: FDR q $\\leq$ {TREND_ALPHA:g})",
                 fontsize=15, y=0.995 if SEASON else 0.93)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    np.savez(f"{out}.npz", **{f"{n}_{k}": v for n, r in results.items()
                              for k, v in r.items()})
    print(f"wrote {out}.pdf, {out}.png, {out}.npz", flush=True)


def main():
    print(f"settings: STANDARDIZE={int(STANDARDIZE)} COARSEN={COARSEN} "
          f"SAME_SCALE={int(SAME_SCALE)} FIGS={','.join(FIGS)}", flush=True)
    gr = bsose_grid()
    print(f"B-SOSE grid: {gr['lat'].size} x {gr['lon'].size}, blocks of "
          f"{COARSEN} x {COARSEN}", flush=True)
    res = {}
    if "oras5" in FIGS:
        init_oras5()
        index = oras5_to_bsose_index()
        res["oras5"] = {}
        for name in ORDER:
            a, t = oras5_driver(name, index)
            if a is None:
                print(f"  ORAS5  {name:6s} skipped, no files")
                continue
            res["oras5"][name] = analyse(a, gr["lat"], gr["lon"], t,
                                         "ORAS5 ", name)
            del a
    if "bsose" in FIGS:
        res["bsose"] = {}
        for name in ORDER:
            a, t = bsose_driver(name)
            res["bsose"][name] = analyse(a, gr["lat"], gr["lon"], t,
                                         "B-SOSE", name)
            del a
    lims = {p: {n: nice_lim(r["val"], r["tr"]) for n, r in rr.items()}
            for p, rr in res.items()}
    if SAME_SCALE and len(res) == 2:
        for n in ORDER:
            if n in lims["oras5"] and n in lims["bsose"]:
                lims["oras5"][n] = lims["bsose"][n] = max(
                    lims["oras5"][n], lims["bsose"][n])
    tag = "" if STANDARDIZE else "_physical"
    if "oras5" in res and res["oras5"]:
        figure(res["oras5"], "ORAS5", lims["oras5"],
               f"driver_trends_oras5{tag}")
    if "bsose" in res and res["bsose"]:
        figure(res["bsose"], "B-SOSE", lims["bsose"],
               f"driver_trends_bsose{tag}")


if __name__ == "__main__":
    main()
