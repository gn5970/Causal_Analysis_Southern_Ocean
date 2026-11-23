"""Figure 1: the chlorophyll trend, cell by cell and by season, in B-SOSE
and in the satellite record.

    python fig1_chl_trend.py

    a  B-SOSE trend, cell by cell, 2013-2024
    b  satellite (ESA CCI) trend, cell by cell, 1998-2025
    c  B-SOSE trend of each calendar month, zonal mean in LAT_STEP bands
    d  the same for the satellite

The anomalies are made with xarray, monthly climatology removed per
cell (groupby("time.month")), as in the earlier script.

Units: standard deviations per year by default (STANDARDIZE=1, as the
earlier script). STANDARDIZE=0 gives mg m-3 per year; with RELATIVE=1,
% per decade
instead (log10 chlorophyll, (10**(10*slope) - 1) * 100), which removes
the difference in mean chlorophyll between model and satellite.

  * a, b: the monthly climatology removed per cell, least-squares slope
    against time; p-value with an effective sample size reduced for the
    lag-1 autocorrelation of the residuals, Benjamini-Hochberg FDR at
    0.05 across the cells of each map: stippled where q <= 0.05;
  * c, d: each calendar month fitted per cell as its own series of one
    value a year (no deseasonalisation needed), then the area-weighted
    mean of the cell slopes in each latitude band. SEASON_DOTS=1 dots
    the bands where the sign is not consistent around the circle.

The satellite is put on the B-SOSE grid (each B-SOSE cell takes the
satellite pixel its centre lies in), and from there both products go
through the same steps: anomalies, k x k blocks, standardisation,
trend. Each keeps its own period. Each product has its own colour scale (the 95th percentile of
its |trend|), the same for its map and its seasonal panel; SAME_SCALE=1
puts both on one.

Settings (environment):
    BSOSE_DIR, BSOSE_FILE   B-SOSE Chl file (BLGCHL, iteration 156)
    BSOSE_NT                months of B-SOSE used from Jan 2013 (144)
    ESA_FILE                satellite file
    SAT_YEAR0, SAT_YEAR1    satellite window (1998, 2025)
    SAT_LAT_SOUTH           satellite used north of this latitude (-70)
    RELATIVE=1              % per decade instead of mg m-3 per year
    STANDARDIZE             1 (default): anomalies / their std in time,
                            trend in sigma per year, the earlier
                            script's units; 0: mg m-3 per year
    BSOSE_COARSEN           k x k blocks of the B-SOSE grid, for both
                            products (4, ~0.7 deg; earlier script: 12)
    LAT_STEP                band width of c, d in degrees (2.5)
    LIM_BSOSE, LIM_SAT      colour limits (automatic)
    SAME_SCALE=1            one scale for both products
    TREND_LIM               one fixed limit for all four panels
    INTERP_MAX_GAP          short-gap filling of the satellite anomalies
                            before the trend, as plot_chl_mean.py (2);
                            0 for observations only

Writes fig1_chl_trend.pdf, .png and .npz in the working directory.
B-SOSE at 1/6 deg needs ~20 GB: run it on a compute node.
"""
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

WORK = "/work/bb1596/b384036/b384036/causality"
BSOSE_DIR = os.environ.get("BSOSE_DIR", os.path.join(WORK, "bsose"))
BSOSE_FILE = os.environ.get(
    "BSOSE_FILE", os.path.join(BSOSE_DIR, "Chl_bsoseI156_2013to2024_monthly.nc"))
BSOSE_NT = int(os.environ.get("BSOSE_NT", 144))
ESA_FILE = os.environ.get(
    "ESA_FILE", os.path.join(WORK, "ESA_Chl_south_1998_2025_0p25deg.nc"))
SAT_YEAR0 = int(os.environ.get("SAT_YEAR0", 1998))
SAT_YEAR1 = int(os.environ.get("SAT_YEAR1", 2025))
DEPTH = 25.0                         # top layer for B-SOSE, as bsose_pcmci.py
LAT_NORTH = -30.0
OUT = os.environ.get("OUT", "fig1_chl_trend")


# ---------------------------------------------------------------------
# Settings of the trend method (those of plot_chl_mean.py)
# ---------------------------------------------------------------------
BANDS = {"Subtropics": (-45, -30), "ACC": (-60, -45),
         "HighLat": (-90, -60)}
MIN_MONTHS_TREND = int(os.environ.get("MIN_MONTHS_TREND", 120))
TREND_ALPHA = 0.05
# Colour limits, % per decade. By default each product gets its own,
# from the 95th percentile of its |trend| (B-SOSE's trend is several times
# the satellite's, and one scale leaves either the model saturated or
# the satellite white). LIM_BSOSE / LIM_SAT fix them; SAME_SCALE=1 puts
# both on the larger one; TREND_LIM fixes one limit for everything.
TREND_LIM = os.environ.get("TREND_LIM")
LIM_BSOSE = os.environ.get("LIM_BSOSE")
LIM_SAT = os.environ.get("LIM_SAT")
SAME_SCALE = os.environ.get("SAME_SCALE", "0") == "1"
MIN_YEARS_TREND = int(os.environ.get("MIN_YEARS_TREND", 12))
MIN_CELLS_ZONAL = 20       # cells along a latitude before a zonal mean
# Linear filling in time across runs of at most this many missing
# months, satellite only (B-SOSE has no gaps). 0: observations only.
# STANDARDIZE=1 (default): the anomalies divided by each cell's standard
# deviation in time before the trend, exactly as the earlier script did
# (ds1 / ds1.std("time")); the trend is then in standard deviations per
# year. That script filled no gaps, so the default filling is off here.
# STANDARDIZE=0: mg m-3 per year (or % per decade with RELATIVE=1).
STANDARDIZE = os.environ.get("STANDARDIZE", "1") == "1"
INTERP_MAX_GAP = int(os.environ.get("INTERP_MAX_GAP",
                                    0 if STANDARDIZE else 2))
RELATIVE = os.environ.get("RELATIVE", "0") == "1" and not STANDARDIZE
if STANDARDIZE:
    UNIT, UNIT_TXT = "$\\sigma$ yr$^{-1}$", "standard deviations per year"
elif RELATIVE:
    UNIT, UNIT_TXT = "% per decade", "% per decade"
else:
    UNIT, UNIT_TXT = "mg m$^{-3}$ yr$^{-1}$", "mg m-3 per year"
# B-SOSE block average before the analysis, 4 x 4 cells (~0.7 deg; the
# earlier script used 12); 1 keeps the native 1/6 deg grid.
BSOSE_COARSEN = int(os.environ.get("BSOSE_COARSEN", 4))
LAT_STEP = float(os.environ.get("LAT_STEP", 2.5))
LAT_SOUTH_PLOT = -80.0
# The satellite is used only north of SAT_LAT_SOUTH (70S): further south
# the record is too sparse (sea ice, polar night) for a trend.
SAT_LAT_SOUTH = float(os.environ.get("SAT_LAT_SOUTH", -70.0))
SEASON_DOTS = os.environ.get("SEASON_DOTS", "0") == "1"


def to_unit(slope):
    """Slope per year of the analysed field -> the plotted unit."""
    if RELATIVE:
        return (10 ** (slope * 10.0) - 1.0) * 100.0
    return slope


def deseasonalise_xr(a, t, k):
    """Anomalies as the earlier script's deseasonalize, in its order:

        clim = ds.groupby("time.month").mean("time")
        ds1  = ds.groupby("time.month") - clim
        ds1  = ds1.coarsen(YC=k, XC=k, boundary="trim").mean()
        ds1 / ds1.std("time")              (STANDARDIZE=1 only)

    a is (time, y, x) on the native grid; returns the anomalies on the
    k x k block grid, and the block-mean field itself."""
    da = xr.DataArray(a, dims=("time", "YC", "XC"), coords={"time": t})
    clim = da.groupby("time.month").mean("time")
    ds1 = da.groupby("time.month") - clim
    if k > 1:
        ds1 = ds1.coarsen(YC=k, XC=k, boundary="trim").mean()
        da = da.coarsen(YC=k, XC=k, boundary="trim").mean()
    if STANDARDIZE:
        ds1 = ds1 / ds1.std("time")
    return (ds1.transpose("time", "YC", "XC").values.astype("float32"),
            da.values.astype("float32"))


def deseasonalise(logchl, months):
    """Subtract the monthly climatology, per cell. NaN-aware.

    This has to happen before any trend is fitted. Satellite coverage is
    seasonal -- summer months are observed far more often than winter
    ones -- so a straight line through the raw series partly measures
    which months happened to be visible in which year, not how
    chlorophyll changed. Removing each calendar month's own mean makes
    every value a departure from zero and the sampling largely
    irrelevant.
    """
    out = np.full_like(logchl, np.nan)
    for m in range(1, 13):
        sel = months == m
        if not sel.any():
            continue
        block = logchl[sel]
        with np.errstate(invalid="ignore"):
            clim = np.nanmean(block, axis=0)
        # No minimum-years screen, matching the pipeline. A calendar
        # month seen once gets a climatology equal to that observation,
        # so its anomaly is exactly zero -- reported by clim_depth
        # below rather than silently dropped.
        out[sel] = block - clim
    return out


def interpolate_time(a, max_gap=None):
    """Linear fill along time across SHORT gaps only. Same rule as the
    pipeline's interpolate_time, so the two agree on what counts as
    fillable.

    Returns (filled, mask). A run of missing months is filled only if it
    is at most max_gap long AND bounded by an observation on both sides,
    so nothing is extrapolated past the ends and a four-month polar
    winter is never invented at max_gap = 2.
    """
    max_gap = INTERP_MAX_GAP if max_gap is None else max_gap
    a = np.asarray(a, dtype="float64")
    nt = a.shape[0]
    if not max_gap or nt < 3:
        return a, np.zeros(a.shape, dtype=bool)
    valid = np.isfinite(a)
    idx = np.arange(nt)[:, None, None] * np.ones((1,) + a.shape[1:])
    prev_i = np.maximum.accumulate(np.where(valid, idx, -1.0), axis=0)
    nxt_i = np.minimum.accumulate(
        np.where(valid, idx, float(nt))[::-1], axis=0)[::-1]
    pi = np.clip(prev_i, 0, nt - 1).astype(np.intp)
    ni = np.clip(nxt_i, 0, nt - 1).astype(np.intp)
    pv = np.take_along_axis(a, pi, axis=0)
    nv = np.take_along_axis(a, ni, axis=0)
    run = nxt_i - prev_i - 1.0
    fill = ((~valid) & (prev_i >= 0) & (nxt_i <= nt - 1)
            & (run <= max_gap) & (run >= 1))
    lin = pv + (idx - prev_i) * (nv - pv) / np.maximum(nxt_i - prev_i, 1e-9)
    return np.where(fill, lin, a), fill


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
    from matplotlib.ticker import MaxNLocator
    t = MaxNLocator(nbins=6, symmetric=True).tick_values(-lim, lim)
    return t[(t >= -lim * 1.0001) & (t <= lim * 1.0001)]


def to_pm180(lon, *fields):
    """Longitudes to -180..180, sorted, the fields' last axis with them."""
    lon = ((np.asarray(lon, "float64") + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    return (lon[order],) + tuple(f[..., order] for f in fields)


def read_bsose():
    """BLGCHL over the top 25 m, weighted by drF x hFacC, as in
    bsose_pcmci.py; months counted from January 2013. Native grid,
    longitudes -180..180."""
    with xr.open_dataset(BSOSE_FILE, decode_times=False) as ds:
        v = ds["BLGCHL"]
        zdim = next(d for d in v.dims if d.upper().startswith("Z"))
        v = v.transpose("time", zdim, "YC", "XC")
        keep = np.flatnonzero(np.abs(ds[zdim].values) <= DEPTH)
        w = (ds["drF"].values[keep][:, None, None]
             * ds["hFacC"].isel({zdim: keep}).transpose(zdim, "YC", "XC").values)
        den = w.sum(axis=0)
        nt = min(BSOSE_NT, v.sizes["time"])
        out = np.empty((nt,) + den.shape, "float32")
        for t0 in range(0, nt, 12):
            t1 = min(t0 + 12, nt)
            a = v.isel(time=slice(t0, t1), **{zdim: keep}).values.astype("float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                out[t0:t1] = np.where(den > 0, (a * w[None]).sum(axis=1) / den,
                                      np.nan)
        lat = ds["YC"].values.astype("float64")
        lon = ds["XC"].values.astype("float64")
    t = pd.date_range("2013-01-01", periods=nt, freq="MS")
    lon, out = to_pm180(lon, out)
    sel = lat <= LAT_NORTH
    print(f"B-SOSE: {os.path.basename(BSOSE_FILE)}, {int(keep.size)} levels "
          f"above {DEPTH:g} m, {t[0]:%Y-%m} to {t[-1]:%Y-%m}, grid "
          f"{int(sel.sum())} x {lon.size}", flush=True)
    return out[:, sel], lat[sel], lon, t


def read_satellite(lat_t, lon_t):
    """The satellite on the B-SOSE grid (lat_t, lon_t): each B-SOSE cell
    takes the value of the satellite pixel its centre lies in. Both
    grids are finer than the 4 x 4 blocks that follow, so nothing is
    smoothed here and a gap stays a gap; the block average then treats
    both products identically."""
    with xr.open_dataset(ESA_FILE) as ds:
        name = ("chlor_a" if "chlor_a" in ds.data_vars
                else max(ds.data_vars, key=lambda v: ds[v].ndim))
        da = ds[name].transpose("time", "lat", "lon")
        t = pd.to_datetime(ds["time"].values)
        keep = (t.year >= SAT_YEAR0) & (t.year <= SAT_YEAR1)
        da = da.isel(time=np.flatnonzero(keep)).load()
    t = t[keep]
    slat = da["lat"].values.astype("float64")
    a = da.values.astype("float32")
    slon, a = to_pm180(da["lon"].values, a)
    o = np.argsort(slat)
    slat, a = slat[o], a[:, o]
    dlat = float(np.diff(slat).mean())
    dlon = float(np.diff(slon).mean())
    # first row at or north of SAT_LAT_SOUTH, moved to the start of a
    # B-SOSE block, so the satellite's k x k blocks are B-SOSE's blocks
    k = max(1, BSOSE_COARSEN)
    r0 = int(np.flatnonzero(lat_t >= SAT_LAT_SOUTH)[0])
    r0 = -(-r0 // k) * k
    lat_t = lat_t[r0:]
    iy = np.round((lat_t - slat[0]) / dlat).astype(int)
    inside = (iy >= 0) & (iy < slat.size)
    ix = np.round((lon_t - slon[0]) / dlon).astype(int) % slon.size
    out = a[:, np.clip(iy, 0, slat.size - 1)][:, :, ix]
    out[:, ~inside] = np.nan
    del a
    print(f"satellite: {os.path.basename(ESA_FILE)}, {name}, {t[0]:%Y-%m} "
          f"to {t[-1]:%Y-%m} ({t.size} months), {-SAT_LAT_SOUTH:g}-"
          f"{-LAT_NORTH:g}S, on the B-SOSE grid {out.shape[1]} x "
          f"{out.shape[2]}; {100 * np.isfinite(out).mean():.0f}% of "
          "cell-months with data", flush=True)
    return out, lat_t, lon_t, t


# =====================================================================
# The trends, in latitude chunks so B-SOSE at 1/6 deg fits in memory
# (cells are independent; FDR is over the whole map)
# =====================================================================
def month_band_trends(field, months, years, lat, south):
    """Trend of each calendar month, (band, month): per cell, the slope
    of that month's one-value-a-year series, then the area-weighted mean
    of the cell slopes over the latitude band. Also the p-value of a
    one-sample t-test of the cell slopes against zero (treats cells as
    independent, so read it as sign consistency, not significance)."""
    from scipy import stats
    # bands no narrower than the grid, or some would hold no cell
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
        val = to_unit(slope)
        del block, x, dx, dy
        for b in range(nb):
            rows = band == b
            if not rows.any():
                continue
            v = val[rows]
            w = np.broadcast_to(coslat[rows][:, None], v.shape)
            f = np.isfinite(v)
            if f.sum() >= MIN_CELLS_ZONAL:
                tr[b, m - 1] = np.average(v[f], weights=w[f])
                pv[b, m - 1] = stats.ttest_1samp(v[f], 0.0).pvalue
    return tr, pv, edges


def analyse(chl, lat, lon, t, label, fill, south=LAT_SOUTH_PLOT):
    """Both products the same way: anomalies on the B-SOSE grid, k x k
    blocks, (standardised), trend per block; seasonal panels from the
    same block anomalies."""
    months = t.month.to_numpy()
    years = (t.year.to_numpy() + (months - 0.5) / 12.0).astype(float)
    k = max(1, BSOSE_COARSEN)
    ny, nx = (lat.size // k) * k, (lon.size // k) * k
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(chl > 0, chl, np.nan).astype("float32")[:, :ny, :nx]
        if RELATIVE:
            field = np.log10(field)
    del chl
    anoms = np.empty((t.size, ny // k, nx // k), "float32")
    blocks = np.empty_like(anoms)
    rows = 24 * k
    for j0 in range(0, ny, rows):
        j1 = min(j0 + rows, ny)
        an, bl = deseasonalise_xr(field[:, j0:j1], t, k)
        anoms[:, j0 // k:j1 // k], blocks[:, j0 // k:j1 // k] = an, bl
    del field
    lat = lat[:ny].reshape(-1, k).mean(axis=1)
    lon = lon[:nx].reshape(-1, k).mean(axis=1)
    field = blocks                      # for the mean Chl in the log
    anom = anoms.astype("float64")
    if fill and INTERP_MAX_GAP:
        anom, _ = interpolate_time(anom)
    slope, pval, _ = trend_per_cell(anom, years)
    del anom
    val = to_unit(slope)
    q = fdr(pval, TREND_ALPHA)
    sig = np.isfinite(q) & (q <= TREND_ALPHA)
    tr, pv, edges = month_band_trends(anoms, months, years, lat, south)
    del anoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_chl = np.nanmean(10 ** field if RELATIVE else field, axis=0)

    fmt = (lambda v: f"{v:+8.1f}") if RELATIVE else (lambda v: f"{v:+8.4f}")
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    fin = np.isfinite(val)
    print(f"\n{label}, {t[0]:%Y}-{t[-1]:%Y}: trend in {int(fin.sum()):,} "
          f"cells, {100 * sig.sum() / max(fin.sum(), 1):.1f}% significant "
          f"(FDR q <= {TREND_ALPHA:g}); unit {UNIT_TXT}")
    print("    {:11s} {:>8s} {:>8s} {:>8s} {:>9s}".format(
        "band", "mean Chl", "trend", "sig", "positive"))
    for nm, (lo, hi) in BANDS.items():
        m = fin & ((lat > lo) & (lat <= hi))[:, None]
        if m.any():
            mc = np.average(mean_chl[m], weights=w[m])
            print(f"    {nm:11s} {mc:8.3f} {fmt(np.average(val[m], weights=w[m]))} "
                  f"{100 * (sig & m).sum() / m.sum():7.1f}% "
                  f"{100 * (val[m] > 0).mean():8.0f}%")
    print("    (area-weighted means; mean Chl in mg m-3)")
    print(f"  by calendar month, band mean ({UNIT_TXT})")
    print("    {:11s}".format("") + "".join(f"{c:>8s}" for c in "JFMAMJJASOND"))
    mid = 0.5 * (edges[:-1] + edges[1:])
    for nm, (lo, hi) in BANDS.items():
        s = (mid > lo) & (mid <= hi)
        if s.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row = np.nanmean(tr[s], axis=0)
            print(f"    {nm:11s}" + "".join(
                "       ." if not np.isfinite(v) else fmt(v) for v in row))
    return dict(val=val, sig=sig, tr=tr, pv=pv, edges=edges, lat=lat,
                lon=lon, south=south, period=f"{t[0]:%Y}-{t[-1]:%Y}")


# =====================================================================
# Figure
# =====================================================================
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


def colorbar(fig, m, ax, lim):
    cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.85,
                      pad=0.05, extend="both")
    cb.set_ticks(sym_ticks(lim))
    cb.set_label(f"Chl trend ({UNIT})", fontsize=15, labelpad=10)
    cb.ax.tick_params(labelsize=14)


def label_box(ax, text):
    ax.text(0.5, 0.05, text, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=12, zorder=10,
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="k",
                      lw=0.8))


def draw_map(fig, ax, r, letter, name, lim):
    # pcolormesh, not contourf: cartopy's filled contours turn
    # inside-out on a polar map when one level covers most of it, and
    # paint land and the area outside the data.
    lon, lat = r["lon"], r["lat"]
    m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(r["val"]),
                      cmap=plt.cm.RdBu_r, vmin=-lim, vmax=lim,
                      shading="nearest", rasterized=True,
                      transform=ccrs.PlateCarree())
    # stipple on a ~1 deg sub-grid, so the two resolutions ink alike
    k = max(1, int(round(1.0 / abs(float(np.diff(lat).mean())))))
    s = np.zeros_like(r["sig"])
    s[::k, ::k] = r["sig"][::k, ::k]
    if s.any():
        lon2d, lat2d = np.meshgrid(lon, lat)
        ax.scatter(lon2d[s], lat2d[s], s=0.6, c="k", alpha=0.6,
                   linewidths=0, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, LAT_NORTH], crs=ccrs.PlateCarree())
    coast(ax)
    ax.gridlines(lw=0.3, alpha=0.5)
    colorbar(fig, m, ax, lim)
    ax.set_title(f"{letter}   trend, cell by cell "
                 f"(stippled: FDR q $\\leq$ {TREND_ALPHA:g})",
                 loc="left", fontsize=12)
    label_box(ax, f"Chl-a ({name}, {r['period'].replace('-', '–')})")


def draw_season(fig, ax, r, letter, name, lim):
    tr, pv, edges = r["tr"], r["pv"], r["edges"]
    m = ax.pcolormesh(np.arange(0.5, 13.5), edges, tr,
                      cmap=plt.cm.RdBu_r, vmin=-lim, vmax=lim,
                      shading="flat", rasterized=True)
    if SEASON_DOTS:
        mid = 0.5 * (edges[:-1] + edges[1:])
        MM, LL = np.meshgrid(np.arange(1, 13), mid)
        weak = np.isfinite(tr) & ~(pv <= TREND_ALPHA)
        if weak.any():
            ax.scatter(MM[weak], LL[weak], s=3, c="0.25", linewidths=0)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(r["south"], LAT_NORTH)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Latitude ($^\\circ$)", fontsize=11)
    colorbar(fig, m, ax, lim)
    ax.set_title(f"{letter}   trend of each calendar month, zonal mean",
                 loc="left", fontsize=12)
    label_box(ax, f"Chl-a ({name}, {r['period'].replace('-', '–')})")


def main():
    print(f"settings: STANDARDIZE={int(STANDARDIZE)} RELATIVE={int(RELATIVE)} "
          f"BSOSE_COARSEN={BSOSE_COARSEN} INTERP_MAX_GAP={INTERP_MAX_GAP} "
          f"SAT_LAT_SOUTH={SAT_LAT_SOUTH:g} satellite {SAT_YEAR0}-{SAT_YEAR1}; "
          f"unit: {UNIT_TXT}", flush=True)
    b_chl, b_lat, b_lon, b_t = read_bsose()
    bs = analyse(b_chl, b_lat, b_lon, b_t, "B-SOSE", fill=False)
    del b_chl
    sat = analyse(*read_satellite(b_lat, b_lon), "Satellite (ESA CCI)",
                  fill=True, south=SAT_LAT_SOUTH)

    if TREND_LIM:
        lb = ls = float(TREND_LIM)
    else:
        lb = float(LIM_BSOSE) if LIM_BSOSE else nice_lim(bs["val"], bs["tr"])
        ls = float(LIM_SAT) if LIM_SAT else nice_lim(sat["val"], sat["tr"])
        if SAME_SCALE:
            lb = ls = max(lb, ls)
    print(f"\ncolour limits: B-SOSE +/-{lb:g}, satellite +/-{ls:g} {UNIT_TXT}")

    fig = plt.figure(figsize=(15.5, 12), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.2,
                          wspace=0.55)
    proj = ccrs.SouthPolarStereo(central_longitude=0)
    draw_map(fig, fig.add_subplot(gs[0, 0], projection=proj), bs,
             "a", "B-SOSE", lb)
    draw_map(fig, fig.add_subplot(gs[0, 1], projection=proj), sat,
             "b", "satellite", ls)
    draw_season(fig, fig.add_subplot(gs[1, 0]), bs, "c", "B-SOSE", lb)
    draw_season(fig, fig.add_subplot(gs[1, 1]), sat, "d", "satellite", ls)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    plt.close(fig)
    np.savez(f"{OUT}.npz",
             **{f"{p}_{k}": v for p, r in (("bsose", bs), ("sat", sat))
                for k, v in r.items()})
    print(f"\nwrote {OUT}.pdf, {OUT}.png, {OUT}.npz")


if __name__ == "__main__":
    main()
