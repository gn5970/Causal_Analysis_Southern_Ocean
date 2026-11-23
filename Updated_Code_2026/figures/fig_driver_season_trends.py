"""Seasonally resolved trends (latitude x calendar month) of the physical drivers in
B-SOSE, 2013-2024. NORM=std: slope / SD of that month's yearly values (SD per year,
as Fig. 2); NORM=none: physical units per year; NORM=cell: original method (block-
standardized anomalies). Output: results_driver_season_trends/driver_season_trends_{NORM}.*"""
import os
import sys
import glob
import warnings
import numpy as np
import xarray as xr
from scipy.stats import t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUS = "/work/bb1596/b384036/b384036/causality"
BSOSE_DIR = os.environ.get("BSOSE_DIR", CAUS + "/bsose")
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_driver_season_trends")
FIELDS = os.environ.get("FIELDS", "Theta,MLD,salt,surfTflx,oceQsw,wind").split(",")
DEPTH = float(os.environ.get("DEPTH", 25.0))
DEPTH_STRAT = 50.0
NORM = os.environ.get("NORM", "std")
LAT_STEP = float(os.environ.get("LAT_STEP", 2.0 if NORM == "cell" else 2.5))
BLOCK = int(os.environ.get("BLOCK", 12))
VMAX = float(os.environ.get("VMAX", 0.3))
NT = int(os.environ.get("NT", 144))
YEAR0 = 2013
NAME = "{}_bsoseI156_2013to2024_monthly.nc"
SKIP = {"drF", "hFacC", "rA", "Depth", "drC", "dxG", "dyG", "rAw", "rAs",
        "rAz", "dxC", "dyC", "maskC", "maskW", "maskS", "PHrefC"}
TITLES = {"Theta": "Temperature, 0-25 m", "MLD": "Mixed-layer depth",
          "salt": "Salinity, 0-25 m", "surfTflx": "Net surface heat flux",
          "oceQsw": "Short-wave radiation", "wind": "Wind-stress amplitude",
          "N": "Stratification $N^2$"}
UNITS = {"Theta": "$^\\circ$C", "MLD": "m", "salt": "psu",
         "surfTflx": "W m$^{-2}$", "oceQsw": "W m$^{-2}$",
         "wind": "N m$^{-2}$", "N": "s$^{-2}$"}


def open_b(word):
    p = os.path.join(BSOSE_DIR, NAME.format(word))
    if not os.path.exists(p):
        hits = [f for f in glob.glob(os.path.join(BSOSE_DIR, "*.nc"))
                if word.lower() in os.path.basename(f).lower().replace(".", "_").split("_")]
        if not hits:
            sys.exit(f"no B-SOSE file for {word} in {BSOSE_DIR}")
        p = hits[0]
    return xr.open_dataset(p, decode_times=False).isel(time=slice(0, NT))


def main_var(ds):
    c = [v for v in ds.data_vars if v not in SKIP and ds[v].ndim >= 3]
    return max(c, key=lambda v: (ds[v].ndim, ds[v].size))


def layer(ds, var, zmin, zmax):
    z = np.abs(ds["Z"].values.astype("float64"))
    keep = np.flatnonzero((z > zmin) & (z <= zmax))
    w = (ds["drF"].isel(Z=keep) * ds["hFacC"].isel(Z=keep)).values.astype("float64")
    den = w.sum(axis=0)
    out = []
    for t0 in range(0, ds.sizes["time"], 12):
        v = ds[var].isel(Z=keep, time=slice(t0, t0 + 12)).values.astype("float64")
        ok = np.isfinite(v)
        num = np.where(ok, v * w[None], 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, num / den, np.nan).astype("float32"))
    return np.concatenate(out, axis=0)


def sigma0(S, T):
    try:
        import gsw
        return gsw.sigma0(S, T)
    except ImportError:
        t, s = T, S
        rho = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2
               + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4 + 6.536332e-9 * t ** 5
               + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t ** 2
                      - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4)
               + np.abs(s) ** 1.5 * (-5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2)
               + 4.8314e-4 * s ** 2)
        return rho - 1000.0


def field(name, wet):
    if name in ("Theta", "salt"):
        word, var = ("Theta", "THETA") if name == "Theta" else ("Salt", "SALT")
        with open_b(word) as ds:
            v = layer(ds, var, -1.0, DEPTH)
    elif name == "N":
        with open_b("Theta") as dt, open_b("Salt") as dsl:
            tu, td = layer(dt, "THETA", -1.0, DEPTH), layer(dt, "THETA", DEPTH, DEPTH_STRAT)
            su, sd = layer(dsl, "SALT", -1.0, DEPTH), layer(dsl, "SALT", DEPTH, DEPTH_STRAT)
        v = (-(9.81 / 1027.5) * (sigma0(su, tu) - sigma0(sd, td)) / (0.5 * DEPTH_STRAT)).astype("float32")
    elif name == "wind":
        with open_b("oceTAUX") as dx, open_b("oceTAUY") as dy:
            tx = dx["oceTAUX"].transpose("time", "YC", "XG").values.astype("float32")
            ty = dy["oceTAUY"].transpose("time", "YG", "XC").values.astype("float32")
        tx = 0.5 * (tx + np.roll(tx, -1, axis=2))
        tyc = np.full_like(ty, np.nan)
        tyc[:, :-1] = 0.5 * (ty[:, :-1] + ty[:, 1:])
        v = np.sqrt(tx ** 2 + tyc ** 2)
    else:
        with open_b(name) as ds:
            var = main_var(ds) if name != "MLD" else "BLGMLD"
            v = ds[var].values.astype("float32")
            if v.ndim == 4:
                v = v[:, 0]
        if name == "MLD":
            v = np.abs(v)
            v[v == 0] = np.nan
    v[:, ~wet] = np.nan
    return v


def cell_bands(v, yc, edges, nyear):
    T, Y, X = v.shape
    v = v[:nyear * 12].reshape(nyear, 12, Y, X)
    with np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        v -= np.nanmean(v, axis=0, keepdims=True)
        v = v.reshape(nyear * 12, Y, X)
        k = BLOCK
        yb, xb = Y // k, X // k
        v = np.nanmean(v[:, :yb * k, :xb * k].reshape(-1, yb, k, xb, k), axis=(2, 4))
        sd = np.nanstd(v, axis=0)
        sd[~(sd > 0)] = np.nan
        v = v / sd[None]
        zm = np.nanmean(v, axis=2)
    ycb = yc[:yb * k].reshape(yb, k).mean(axis=1)
    band = np.digitize(ycb, edges) - 1
    bm = np.full((edges.size - 1, zm.shape[0]), np.nan)
    for b in range(edges.size - 1):
        rows = band == b
        if rows.any():
            with np.errstate(invalid="ignore"):
                bm[b] = np.nanmean(zm[:, rows], axis=1)
    return bm


def trend(y, yrs):
    ok = np.isfinite(y)
    n = ok.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(ok, yrs, 0).sum(-1, keepdims=True) / n[..., None]
        ym = np.where(ok, y, 0).sum(-1, keepdims=True) / n[..., None]
        dt, dy = np.where(ok, yrs - tm, 0.0), np.where(ok, y - ym, 0.0)
        sxx = (dt * dt).sum(-1)
        b = (dt * dy).sum(-1) / sxx
        res = np.where(ok, dy - b[..., None] * dt, 0.0)
        se = np.sqrt((res ** 2).sum(-1) / np.maximum(n - 2, 1) / sxx)
        p = 2 * tdist.sf(np.abs(b / se), np.maximum(n - 2, 1))
        sd = np.sqrt((dy ** 2).sum(-1) / np.maximum(n - 1, 1))
    bad = n < 8
    b[bad], p[bad], sd[bad] = np.nan, np.nan, np.nan
    return b, p, sd


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"NORM={NORM}, LAT_STEP={LAT_STEP}, FIELDS={FIELDS}", flush=True)
    with open_b("Theta") as g:
        wet = g["hFacC"].isel(Z=0).values > 0
        yc = g["YC"].values.astype("float64")
    edges = np.arange(-90.0, -30.0 + LAT_STEP / 2, LAT_STEP)
    centres = 0.5 * (edges[1:] + edges[:-1])
    band = np.digitize(yc, edges) - 1
    nyear = NT // 12
    yrs = YEAR0 + np.arange(nyear, dtype=float)
    area = np.cos(np.deg2rad(yc))[:, None] * np.ones(wet.shape)
    res = {}
    for name in FIELDS:
        v = field(name, wet)
        print(f"{name}: {v.shape[0]} months", flush=True)
        if NORM == "cell":
            bm = cell_bands(v, yc, edges, nyear)
            del v
            vals = bm.reshape(centres.size, nyear, 12).transpose(0, 2, 1)
            s, p, sd = trend(vals, yrs)
            res[name] = (s, p, vals)
            continue
        ok = np.isfinite(v)
        num = np.where(ok, v * area[None], 0.0).sum(axis=2)
        den = np.where(ok, area[None], 0.0).sum(axis=2)
        del v, ok
        bm = np.full((centres.size, NT), np.nan)
        for b in range(centres.size):
            rows = band == b
            if rows.any():
                d = den[:, rows].sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bm[b] = np.where(d > 0, num[:, rows].sum(axis=1) / d, np.nan)
        vals = bm[:, :nyear * 12].reshape(centres.size, nyear, 12).transpose(0, 2, 1)
        s, p, sd = trend(vals, yrs)
        if NORM == "std":
            with np.errstate(invalid="ignore", divide="ignore"):
                s = s / sd
        res[name] = (s, p, vals)

    out = os.path.join(OUTDIR, f"driver_season_trends_{NORM}")
    np.savez(out + ".npz", lat=centres, edges=edges, years=yrs, norm=NORM,
             **{f"{k}_slope": r[0] for k, r in res.items()},
             **{f"{k}_p": r[1] for k, r in res.items()},
             **{f"{k}_vals": r[2] for k, r in res.items()})
    n = len(res)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow), squeeze=False)
    months = np.arange(12)
    for ax, (name, (s, p, _)), letter in zip(axes.flat, res.items(), "abcdefghij"):
        if NORM in ("std", "cell"):
            lim, lab = VMAX, "trend (SD yr$^{-1}$)"
        else:
            lim = np.nanpercentile(np.abs(s), 98) if np.isfinite(s).any() else 1.0
            lim = float(f"{lim:.1g}") or 1.0
            lab = f"trend ({UNITS[name]} yr$^{{-1}}$)"
        m = ax.pcolormesh(np.arange(13) - 0.5, edges, s, cmap=plt.cm.RdBu_r,
                          vmin=-lim, vmax=lim, shading="flat")
        yy, xx = np.nonzero(np.isfinite(p) & (p < 0.05))
        ax.scatter(months[xx], centres[yy], s=3, c="k", lw=0)
        ax.set_xticks(months)
        ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_xlabel("Month")
        ax.set_ylabel("Latitude (°)")
        ax.set_ylim(-80, -30)
        if NORM in ("std", "cell"):
            cb = fig.colorbar(m, ax=ax, extend="both", ticks=np.linspace(-lim, lim, 7))
        else:
            cb = fig.colorbar(m, ax=ax)
        cb.ax.tick_params(labelsize=14)
        cb.set_label(lab, fontsize=13)
        ax.set_title(TITLES[name], fontsize=13)
        ax.text(-0.15, 1.04, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {out}.pdf, .png, .npz")


if __name__ == "__main__":
    main()
