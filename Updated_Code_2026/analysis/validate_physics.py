"""B-SOSE physics against the ORAS5 reanalysis, 2013-2024, block by block on
the B-SOSE grid: mixed-layer depth, temperature and salinity (top 25 m).

    python /work/bb1596/b384036/b384036/causality/validate_physics.py
    COARSEN=1 python /work/bb1596/b384036/b384036/causality/validate_physics.py
    REPLOT=1 COARSEN=1 python /work/bb1596/b384036/b384036/causality/validate_physics.py

THE FIELDS, exactly as the drivers of the PCMCI runs
  B-SOSE  THETA, SALT: mean over the top 25 m, weighted by drF x hFacC
          (bsose_pcmci.py); MLD: BLGMLD
  ORAS5   votemper, vosaline: thickness-weighted mean over 0-25 m;
          MLD: somxl010 (oras5_pcmci.py); each B-SOSE cell takes the
          nearest ORCA025 cell, as on_bsose() in oras5_pcmci.py
  The two MLDs are not the same definition (BLGMLD vs the 0.01 kg m-3
  criterion of somxl010), so the MLD bias is partly definition; the
  correlations are not affected by a constant offset.

PER BLOCK (COARSEN x COARSEN; COARSEN=1 is the native grid), common
cell-months only:
  r_raw    correlation of the fields themselves (seasonal cycle included)
  r_anom   correlation of the anomalies (monthly climatology of every
           native cell removed first, as the PCMCI pipeline does) -- the
           test that matters for the causal analysis
  p        t test (SIG=ar1: AR(1)-corrected), significant at p < ALPHA (0.01)
  bias     time mean of B-SOSE minus ORAS5
  sd_ratio standard deviation of the B-SOSE anomalies / ORAS5's

OUTPUT in /work/bb1596/b384036/b384036/causality/results_validate_physics_c<K>/
  validate_physics_c<K>.pdf      ONE ROW, THREE MAPS (MLD, Theta, salt): the
                                 correlation of B-SOSE with ORAS5.
                                 PLOT_METRIC=raw (default): monthly values,
                                 seasonal cycle included.
                                 PLOT_METRIC=anom: monthly anomalies; written
                                 to validate_physics_c<K>_anom.pdf
  validate_physics_c<K>.csv      per region (N_SAF, SAF_SB, S_SB, All):
                                 median r, share significant, bias
  physics_<var>_c<K>.npz         r, p, sig, n, bias, ... for each variable
  physics_mask_mld_c<K>.npz      mask for summarize_links.py --mask: MLD
  physics_mask_all_c<K>.npz      mask: MLD, Theta and salt all pass
  The masks use MASK_FROM=anom (default) or raw.

REPLOT=1 redraws the figure (and rewrites the csv and the masks) from the
saved physics_<var>_c<K>.npz files; nothing is recomputed.

CAVEAT: ORAS5 and B-SOSE both assimilate Argo T/S profiles, so they are
not independent; agreement means B-SOSE's state is consistent with a
second, independently built reanalysis, not with raw observations.
"""
import os
import sys
import re
import csv
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
from corr_bsose_sat import blocks, corr_ar1, anomalies, region_masks  # noqa: E402

WORK = os.environ.get("WORK", CAUS)
BSOSE_DIR = os.environ.get("BSOSE_DIR", os.path.join(CAUS, "bsose"))
BSOSE_NAME = "{}_bsoseI156_2013to2024_monthly.nc"
COARSEN = int(os.environ.get("COARSEN", 8))
YEAR_MIN, YEAR_MAX = 2013, 2024
N_MONTHS = NT = (YEAR_MAX - YEAR_MIN + 1) * 12
TIMES = pd.date_range(f"{YEAR_MIN}-01-01", periods=NT, freq="MS")
MIN_MONTHS = int(os.environ.get("MIN_MONTHS", 24))
ALPHA = float(os.environ.get("ALPHA", 0.01))
MASK_FROM = os.environ.get("MASK_FROM", "anom")
PLOT_METRIC = os.environ.get("PLOT_METRIC", "raw")
REPLOT = os.environ.get("REPLOT", "0") == "1"
OUTDIR = os.environ.get("OUTDIR", f"{CAUS}/results_validate_physics_c{COARSEN}")
LAT_NORTH = -30.0
DEPTH_MEAN = 25.0
PATTERNS = {"theta": "*votemper*.nc", "salt": "*vosaline*.nc",
            "MLD": "*somxl010*.nc"}
VARS = [("MLD", "MLD", "m"), ("Theta", "theta", "$^\\circ$C"),
        ("salt", "salt", "psu")]
TITLES = {"MLD": "Mixed-layer depth", "Theta": "Temperature, 0-25 m",
          "salt": "Salinity, 0-25 m"}

# =====================================================================
# ORAS5 readers, copied from oras5_pcmci.py (that script runs its whole
# analysis on import, so the functions cannot be imported from it)
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
    """{month slot: path} for every file matching `pattern` (cached)."""
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
    """Rows of `pattern`'s own grid matched to the reference by latitude."""
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
    """Thickness-weighted mean over (zmin, zmax] metres; vals (nz, ny, nx)."""
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
    """(N_MONTHS, NY, NX) stack, streamed a month at a time."""
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


def oras5_on_bsose(name, yc, xc):
    """(T, YC, XC): the ORAS5 field, nearest ORCA025 cell per B-SOSE cell."""
    collapse = None if name == "MLD" else (0.0, DEPTH_MEAN)
    stack, _ = read_stack([(PATTERNS[name], collapse)], label=name)
    if stack is None:
        sys.exit(f"no ORAS5 {PATTERNS[name]} files for {YEAR_MIN}-{YEAR_MAX} in {ORAS5_DIR}")
    iy = np.abs(REF_ROWLAT[None, :] - yc[:, None]).argmin(axis=1)
    far = np.abs(REF_ROWLAT[iy] - yc) > 0.3
    colon = NAV_LON[NY // 2]
    ix = np.abs(((colon[None, :] - xc[:, None] + 180.0) % 360.0) - 180.0).argmin(axis=1)
    a = stack[:, iy][:, :, ix]
    a[:, far] = np.nan
    return a


# =====================================================================
# B-SOSE
# =====================================================================
def bsose_field(name):
    """(T, YC, XC) float32; land NaN."""
    fname = {"MLD": "MLD", "theta": "Theta", "salt": "Salt"}[name]
    path = os.path.join(BSOSE_DIR, BSOSE_NAME.format(fname))
    ds = xr.open_dataset(path, decode_times=False).isel(time=slice(0, NT))
    wet = ds["hFacC"].isel(Z=0).values > 0 if "hFacC" in ds else None
    if name == "MLD":
        v = np.abs(ds["BLGMLD"].values.astype("float32"))
        v[v == 0] = np.nan
    else:
        var = {"theta": "THETA", "salt": "SALT"}[name]
        depth = np.abs(ds["Z"].values.astype("float64"))
        keep = np.flatnonzero((depth > -1.0) & (depth <= DEPTH_MEAN))
        w = (ds["drF"].isel(Z=keep) * ds["hFacC"].isel(Z=keep)).astype("float64")
        den = w.sum("Z")
        out = []
        for t0 in range(0, ds.sizes["time"], 12):
            vv = ds[var].isel(Z=keep, time=slice(t0, t0 + 12)).astype("float64")
            out.append(((vv * w).sum("Z") / den.where(den > 0)).astype("float32").values)
        v = np.concatenate(out, axis=0)
    if wet is not None:
        v[:, ~wet] = np.nan
    yc, xc = ds["YC"].values.astype("float64"), ds["XC"].values.astype("float64")
    ds.close()
    print(f"B-SOSE {fname}: {os.path.basename(path)}, {v.shape[0]} months", flush=True)
    return v, yc, xc


# =====================================================================
def compare(name):
    k = COARSEN
    mod, yc, xc = bsose_field(name)
    ref = oras5_on_bsose(name, yc, xc)
    both = np.isfinite(mod) & np.isfinite(ref)
    mod[~both] = np.nan
    ref[~both] = np.nan
    mb, rb = blocks(mod, k), blocks(ref, k)
    r_raw, p_raw, n = corr_ar1(mb, rb, MIN_MONTHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bias = np.nanmean(mb - rb, axis=0)
    del mb, rb
    ma, ra = blocks(anomalies(mod, TIMES), k), blocks(anomalies(ref, TIMES), k)
    del mod, ref
    r_anom, p_anom, _ = corr_ar1(ma, ra, MIN_MONTHS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd_ratio = np.nanstd(ma, axis=0) / np.nanstd(ra, axis=0)
    bad = ~np.isfinite(r_raw)
    bias[bad] = np.nan
    sd_ratio[bad | ~np.isfinite(sd_ratio)] = np.nan
    lat = yc[:yc.size // k * k].reshape(-1, k).mean(axis=1)
    lon = xc[:xc.size // k * k].reshape(-1, k).mean(axis=1)
    sig_raw = np.isfinite(p_raw) & (p_raw < ALPHA)
    sig_anom = np.isfinite(p_anom) & (p_anom < ALPHA)
    return dict(r_raw=r_raw, p_raw=p_raw, sig_raw=sig_raw, r_anom=r_anom,
                p_anom=p_anom, sig_anom=sig_anom, n=n, bias=bias,
                sd_ratio=sd_ratio, lat=lat, lon=lon)


def save_mask(path, res_list):
    """r, sig in the format summarize_links.py --mask reads."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmin(np.stack([d["r_" + MASK_FROM] for d in res_list]), axis=0)
    sig = np.ones_like(r, bool)
    untested = np.zeros_like(r, bool)
    for d in res_list:
        sig &= d["sig_" + MASK_FROM].astype(bool) & (d["r_" + MASK_FROM] > 0)
        untested |= ~np.isfinite(d["r_" + MASK_FROM])
    r = np.where(untested, np.nan, r)
    np.savez(path, r=r, sig=sig, lat=res_list[0]["lat"], lon=res_list[0]["lon"],
             mask_from=MASK_FROM, alpha=ALPHA)
    ok = np.isfinite(r)
    print(f"  {os.path.basename(path)}: kept {int((sig & ok).sum())}, rejected "
          f"{int((~sig & ok).sum())}, untested {int((~ok).sum())} blocks")


def plot_correlations(res, lat, lon, k):
    """One row, three maps: the correlation of B-SOSE with ORAS5."""
    import cartopy.crs as ccrs
    metric = PLOT_METRIC
    what = ("monthly values, seasonal cycle included" if metric == "raw"
            else "monthly anomalies, seasonal cycle removed")
    lonw = np.concatenate([lon, [lon[0] + 360.0]])

    def wrapf(f):
        return np.concatenate([f, f[:, :1]], axis=1)

    try:          # coastlines only if the Natural Earth files are there
        from cartopy.io import shapereader
        shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
        coast = True
    except Exception:
        coast = False

    fig = plt.figure(figsize=(15, 5.6), dpi=200)
    axes = []
    print(f"\nfigure ({metric}: {what}):")
    for i, (label, _, unit) in enumerate(VARS):
        d = res[label]
        r, sig = d["r_" + metric], d["sig_" + metric].astype(bool)
        ax = fig.add_subplot(1, 3, i + 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
        if coast:
            ax.coastlines(resolution="110m", lw=0.5)
        # grey where tested but not significant
        tested = np.where(np.isfinite(r), 0.0, np.nan)
        ax.pcolormesh(lonw, lat, wrapf(tested), cmap=plt.cm.Greys, vmin=-1, vmax=3,
                      transform=ccrs.PlateCarree(), shading="nearest", rasterized=True)
        m = ax.pcolormesh(lonw, lat, wrapf(np.where(sig, r, np.nan)), cmap=plt.cm.RdBu_r,
                          vmin=-1, vmax=1, transform=ccrs.PlateCarree(),
                          shading="nearest", rasterized=True)
        ok = np.isfinite(r)
        share = 100.0 * (sig & (r > 0) & ok).sum() / max(int(ok.sum()), 1)
        med = float(np.nanmedian(r))
        ax.set_title(f"{'abc'[i]}  {TITLES.get(label, label)}", loc="left",
                     fontsize=13, fontweight="bold")
        ax.text(0.5, -0.03, f"median r = {med:.2f}; {share:.0f}% of cells "
                f"significant (p < {ALPHA:g})", transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
        print(f"  {label:6s} median r = {med:.2f}, {share:.0f}% of cells "
              f"significant and positive")
        axes.append(ax)
    cb = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.045,
                      pad=0.1, shrink=0.5)
    cb.set_label(f"correlation B-SOSE vs ORAS5, {YEAR_MIN}-{YEAR_MAX} ({what}); "
                 f"grey: not significant", fontsize=10)
    name = f"validate_physics_c{k}.pdf" if metric == "raw" \
        else f"validate_physics_c{k}_{metric}.pdf"
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    k = COARSEN
    saved = [os.path.join(OUTDIR, f"physics_{label}_c{k}.npz") for label, _, _ in VARS]
    res = {}
    if REPLOT and all(os.path.exists(f) for f in saved):
        for (label, _, _), f in zip(VARS, saved):
            z = np.load(f)
            res[label] = {key: z[key] for key in z.files}
        print(f"REPLOT=1: read the saved results in {OUTDIR}", flush=True)
    else:
        if REPLOT:
            print(f"REPLOT=1 but no saved results in {OUTDIR}: computing", flush=True)
        init_oras5()
        for (label, name, _), f in zip(VARS, saved):
            res[label] = compare(name)
            np.savez(f, **res[label])
    lat, lon = res["MLD"]["lat"], res["MLD"]["lon"]

    rows = []
    regs = region_masks(lat.size, lon.size, k)
    for label, _, unit in VARS:
        d = res[label]
        for reg, m in regs.items():
            sel = m & np.isfinite(d["r_raw"])
            if not sel.any():
                continue
            sa = sel & np.isfinite(d["r_anom"])
            rows.append(dict(
                variable=label, region=reg, blocks=int(sel.sum()),
                median_r_raw=round(float(np.median(d["r_raw"][sel])), 3),
                share_sig_pos_raw=round(float((d["sig_raw"].astype(bool) & (d["r_raw"] > 0))[sel].mean()), 3),
                median_r_anom=round(float(np.median(d["r_anom"][sa])), 3) if sa.any() else np.nan,
                share_sig_pos_anom=round(float((d["sig_anom"].astype(bool) & (d["r_anom"] > 0))[sel].mean()), 3),
                median_bias=round(float(np.nanmedian(d["bias"][sel])), 4),
                median_sd_ratio=round(float(np.nanmedian(d["sd_ratio"][sel])), 3)))
    with open(os.path.join(OUTDIR, f"validate_physics_c{k}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'var':6s} {'region':7s} {'blocks':>7s} {'r_raw':>6s} {'sig>0':>6s} "
          f"{'r_anom':>6s} {'sig>0':>6s} {'bias':>9s} {'sd_ratio':>8s}   (p < {ALPHA:g})")
    for d in rows:
        print(f"{d['variable']:6s} {d['region']:7s} {d['blocks']:7d} {d['median_r_raw']:6.2f} "
              f"{100 * d['share_sig_pos_raw']:5.0f}% {d['median_r_anom']:6.2f} "
              f"{100 * d['share_sig_pos_anom']:5.0f}% "
              f"{d['median_bias']:9.3f} {d['median_sd_ratio']:8.2f}")

    print("\nmasks for summarize_links.py --mask (from r_" + MASK_FROM + "):")
    save_mask(os.path.join(OUTDIR, f"physics_mask_mld_c{k}.npz"), [res["MLD"]])
    save_mask(os.path.join(OUTDIR, f"physics_mask_all_c{k}.npz"), list(res.values()))

    plot_correlations(res, lat, lon, k)


if __name__ == "__main__":
    main()
