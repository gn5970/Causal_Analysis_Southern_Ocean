"""Correlation of B-SOSE chlorophyll with the satellite chlorophyll, block by
block, on the B-SOSE grid; figure as in the NPP comparison (Figure 2).

    python corr_bsose_sat.py                 chlorophyll
    ANOM=1 python corr_bsose_sat.py          anomalies (native climatology removed, then blocks)
    INTERP_GAP=1 python corr_bsose_sat.py    satellite gaps <= 1 month filled in time
    SIG=ar1 python corr_bsose_sat.py         AR(1)-corrected test instead of the plain t test

Fields: B-SOSE BLGCHL over the top 25 m (drF x hFacC); satellite
ESA_Chl_south_1998_2025_0p25deg.nc, nearest pixel on the B-SOSE grid; only
cell-months where both have a value; COARSEN x COARSEN block means (8).
r: Pearson over time per block. Significance at p < ALPHA (0.01):
SIG=ttest (default) t = r sqrt((n-2)/(1-r^2)), df = n-2; SIG=ar1 with
n_eff = n (1 - r1x r1y)/(1 + r1x r1y) (Bretherton et al. 1999), lag-1
autocorrelations over consecutive months only.
Bias: time mean of B-SOSE minus satellite chlorophyll, observed months.
"""
import os
import csv
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CAUS = "/work/bb1596/b384036/b384036/causality"
BSOSE_DIR = os.environ.get("BSOSE_DIR", CAUS + "/bsose")
BSOSE_NAME = "{}_bsoseI156_2013to2024_monthly.nc"
ESA_FILE = os.environ.get("ESA_FILE", CAUS + "/ESA_Chl_south_1998_2025_0p25deg.nc")
ZONES_FILE = os.environ.get("ZONES_FILE", CAUS + "/acc_zones3_bsose.nc")
COARSEN = int(os.environ.get("COARSEN", 8))
NT = int(os.environ.get("NT", 144))
MIN_MONTHS = int(os.environ.get("MIN_MONTHS", 24))
ALPHA = float(os.environ.get("ALPHA", 0.01))
SIG = os.environ.get("SIG", "ttest").lower()
ANOM = os.environ.get("ANOM", "0") == "1"
INTERP_GAP = int(os.environ.get("INTERP_GAP", 0))
OUTDIR = os.environ.get("OUTDIR", f"{CAUS}/results_corr_bsose_sat"
                        + ("_anom" if ANOM else "") + f"_c{COARSEN}"
                        + (f"_interp{INTERP_GAP}" if INTERP_GAP > 0 else ""))
TOP = 25.0


def bsose_chl():
    path = os.path.join(BSOSE_DIR, BSOSE_NAME.format("Chl"))
    ds = xr.open_dataset(path, decode_times=False).isel(time=slice(0, NT))
    keep = np.flatnonzero(np.abs(ds["Z"].values) <= TOP)
    w = (ds["drF"].isel(Z=keep) * ds["hFacC"].isel(Z=keep)).astype("float64")
    den = w.sum("Z")
    v = ds["BLGCHL"].isel(Z=keep)
    out = []
    for t0 in range(0, v.sizes["time"], 12):
        vv = v.isel(time=slice(t0, t0 + 12)).astype("float64")
        out.append(((vv * w).sum("Z") / den.where(den > 0)).astype("float32"))
    chl = xr.concat(out, dim="time").transpose("time", "YC", "XC")
    times = pd.date_range("2013-01-01", periods=chl.sizes["time"], freq="MS")
    print(f"B-SOSE Chl: {os.path.basename(path)}, {len(times)} months", flush=True)
    return chl.values, ds["YC"].values, ds["XC"].values, times


def satellite_on_grid(yc, xc, times):
    with xr.open_dataset(ESA_FILE) as sat:
        name = "chlor_a" if "chlor_a" in sat else list(sat.data_vars)[0]
        v = sat[name].transpose("time", "lat", "lon")
        sat_month = pd.PeriodIndex(pd.to_datetime(v["time"].values), freq="M")
        pos = sat_month.get_indexer(pd.PeriodIndex(times, freq="M"))
        vals = v.isel(time=np.where(pos >= 0, pos, 0)).values.astype("float32")
        slat = v["lat"].values.astype("float64")
        slon = v["lon"].values.astype("float64")
    vals[~(np.isfinite(vals) & (vals > 0))] = np.nan
    vals[pos < 0] = np.nan
    iy = np.abs(slat[None, :] - yc[:, None]).argmin(axis=1)
    dlon = ((slon[None, :] - xc[:, None] + 180.0) % 360.0) - 180.0
    ix = np.abs(dlon).argmin(axis=1)
    half = 0.5 * float(np.abs(np.diff(slat)).mean())
    inside = np.abs(slat[iy] - yc) <= half + 1e-6
    out = vals[:, iy][:, :, ix]
    out[:, ~inside, :] = np.nan
    print(f"satellite: {os.path.basename(ESA_FILE)}, {int((pos >= 0).sum())} "
          f"of {len(times)} months found", flush=True)
    return out


def anomalies(a, times):
    month = pd.DatetimeIndex(times).month.values
    out = np.full_like(a, np.nan)
    for m in range(1, 13):
        sel = month == m
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clim = np.nanmean(a[sel], axis=0)
        out[sel] = a[sel] - clim
    return out


def fill_short_gaps(a, max_gap):
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


def blocks(a, k):
    T, Y, X = a.shape
    a = a[:, :Y // k * k, :X // k * k].reshape(T, Y // k, k, X // k, k)
    with np.errstate(invalid="ignore"):
        n = np.isfinite(a).sum(axis=(2, 4))
        s = np.nansum(a, axis=(2, 4))
        return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype("float32")


def lag1(d, ok):
    pair = ok[1:] & ok[:-1]
    num = np.where(pair, d[1:] * d[:-1], 0).sum(axis=0)
    den = np.where(ok, d * d, 0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = num / den
    return np.where(pair.sum(axis=0) >= 3, r1, np.nan)


def corr_ar1(x, y, min_n):
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(ok, x, 0).sum(axis=0) / n
        ym = np.where(ok, y, 0).sum(axis=0) / n
        dx = np.where(ok, x - xm, 0)
        dy = np.where(ok, y - ym, 0)
        r = (dx * dy).sum(axis=0) / np.sqrt((dx * dx).sum(axis=0) * (dy * dy).sum(axis=0))
        rx, ry = lag1(dx, ok), lag1(dy, ok)
        rr = np.where(np.isfinite(rx * ry), rx * ry, 0.0)
        if SIG == "ar1":
            neff = np.maximum(n * (1 - rr) / (1 + rr), 3.0)
        else:
            neff = n.astype(float)
        tt = r * np.sqrt((neff - 2) / np.maximum(1 - r * r, 1e-12))
        p = 2 * tdist.sf(np.abs(tt), np.maximum(neff - 2, 1))
    bad = n < min_n
    r[bad], p[bad] = np.nan, np.nan
    return r, p, n


def region_masks(ny, nx, k):
    if not os.path.exists(ZONES_FILE):
        print(f"[WARNING] {ZONES_FILE} not found: All only")
        return {"All": np.ones((ny, nx), bool)}
    with xr.open_dataset(ZONES_FILE) as z:
        zone = z["zone"].transpose("YC", "XC").values
        fv = str(z["zone"].attrs.get("flag_values", "")).split()
        fm = str(z["zone"].attrs.get("flag_meanings", "")).split()
    names = ({int(v): m for v, m in zip(fv, fm) if int(v) != 0}
             if fv and len(fv) == len(fm) else {1: "N_SAF", 2: "ACC", 3: "S_SB"})
    b = zone[:ny * k, :nx * k].reshape(ny, k, nx, k)
    counts = np.stack([(b == c).sum(axis=(1, 3)) for c in names])
    best = np.array(list(names))[counts.argmax(axis=0)]
    zb = np.where(counts.max(axis=0) >= k * k / 2, best, 0)
    out = {"All": np.ones((ny, nx), bool)}
    out.update({{"ACC": "SAF_SB"}.get(n, n): zb == c for c, n in names.items()})
    return out


def setup_ax(ax, letter, title):
    import cartopy.crs as ccrs
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    try:
        from cartopy.io import shapereader
        shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
        ax.coastlines(resolution="110m")
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title, fontsize=18)
    ax.text(0.02, 0.96, letter, transform=ax.transAxes, fontsize=24,
            fontweight="bold", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))


def wrap(lon, f):
    return (np.concatenate([lon, [lon[0] + 360.0]]),
            np.concatenate([f, f[:, :1]], axis=1))


def main():
    import cartopy.crs as ccrs
    os.makedirs(OUTDIR, exist_ok=True)
    k = COARSEN
    mod, yc, xc, times = bsose_chl()
    sat = satellite_on_grid(yc, xc, times)
    sat = np.where(np.isfinite(mod), sat, np.nan)
    lat = yc[:yc.size // k * k].reshape(-1, k).mean(axis=1)
    lon = xc[:xc.size // k * k].reshape(-1, k).mean(axis=1)
    n_obs = int(np.isfinite(sat).sum())
    both0 = np.isfinite(sat)
    mb = blocks(np.where(both0, mod, np.nan), k)
    sb = blocks(sat, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bias = np.nanmean(mb - sb, axis=0)
    if ANOM:
        sa = anomalies(sat, times)
        if INTERP_GAP > 0:
            sa, nf = fill_short_gaps(sa, INTERP_GAP)
            print(f"INTERP_GAP={INTERP_GAP}: {nf:,} satellite cell-months "
                  f"filled ({100 * nf / max(n_obs, 1):.1f}% more than the "
                  f"{n_obs:,} observed)", flush=True)
        both = np.isfinite(sa)
        ma = anomalies(np.where(both, mod, np.nan), times)
        r, p, n = corr_ar1(blocks(ma, k), blocks(np.where(both, sa, np.nan), k),
                           MIN_MONTHS)
        del ma, sa
    else:
        if INTERP_GAP > 0:
            sat, nf = fill_short_gaps(sat, INTERP_GAP)
            print(f"INTERP_GAP={INTERP_GAP}: {nf:,} satellite cell-months "
                  f"filled ({100 * nf / max(n_obs, 1):.1f}% more than the "
                  f"{n_obs:,} observed)", flush=True)
            sb = blocks(sat, k)
            mb = blocks(np.where(np.isfinite(sat), mod, np.nan), k)
        r, p, n = corr_ar1(mb, sb, MIN_MONTHS)
    print(f"cell-months with both: {n_obs:,} observed"
          + (" (before filling)" if INTERP_GAP > 0 else ""), flush=True)
    del mod, sat
    bias = np.where(n >= MIN_MONTHS, bias, np.nan)
    sig = np.isfinite(p) & (p < ALPHA)

    tag = (f"corr_bsose_sat{'_anom' if ANOM else ''}_c{k}"
           + (f"_interp{INTERP_GAP}" if INTERP_GAP > 0 else "")
           + ("_ar1" if SIG == "ar1" else ""))
    np.savez(os.path.join(OUTDIR, tag + ".npz"), r=r, p=p, sig=sig, n=n,
             bias=bias, lat=lat, lon=lon, alpha=ALPHA)
    rows = []
    for reg, m in region_masks(lat.size, lon.size, k).items():
        sel = m & np.isfinite(r)
        if not sel.any():
            continue
        rows.append(dict(region=reg, blocks=int(sel.sum()),
                         median_r=round(float(np.median(r[sel])), 3),
                         q25=round(float(np.percentile(r[sel], 25)), 3),
                         q75=round(float(np.percentile(r[sel], 75)), 3),
                         share_sig=round(float(sig[sel].mean()), 3),
                         share_sig_pos=round(float((sig & (r > 0))[sel].mean()), 3),
                         median_bias=round(float(np.nanmedian(bias[sel])), 4),
                         median_months=int(np.median(n[sel]))))
    with open(os.path.join(OUTDIR, tag + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'region':7s} {'blocks':>6s} {'median r':>8s} {'IQR':>13s} "
          f"{'sig.':>5s} {'sig.>0':>6s} {'bias':>8s} {'months':>6s}   "
          f"(sig.: {'AR(1)-corrected' if SIG == 'ar1' else 't test,'} p < {ALPHA:g})")
    for d in rows:
        print(f"{d['region']:7s} {d['blocks']:6d} {d['median_r']:8.2f} "
              f"[{d['q25']:5.2f},{d['q75']:5.2f}] {100 * d['share_sig']:4.0f}% "
              f"{100 * d['share_sig_pos']:5.0f}% {d['median_bias']:8.3f} "
              f"{d['median_months']:6d}")

    fig = plt.figure(figsize=(24, 7.5), dpi=300)
    levels = np.linspace(-1, 1, 41)
    what = "anomalies" if ANOM else "chlorophyll"

    ax = fig.add_subplot(1, 3, 1, projection=ccrs.SouthPolarStereo(central_longitude=0))
    setup_ax(ax, "a", f"Correlation, {what}")
    lc, rc = wrap(lon, r)
    ax.contourf(lc, lat, rc, levels=levels, cmap="Greys", alpha=0.3,
                transform=ccrs.PlateCarree())
    _, rs = wrap(lon, np.where(sig, r, np.nan))
    fill = ax.contourf(lc, lat, rs, levels=levels, cmap=plt.cm.RdBu_r,
                       transform=ccrs.PlateCarree())
    _, sc = wrap(lon, sig.astype(float))
    ax.contour(lc, lat, sc, levels=[0.5], colors="black", linewidths=0.8,
               transform=ccrs.PlateCarree())
    ax.set_aspect("equal")
    cb = plt.colorbar(fill, ax=ax, orientation="vertical", shrink=0.75)
    cb.set_label(f"Correlation (r); coloured: p < {ALPHA:g}, "
                 + ("AR(1) t test" if SIG == "ar1" else "t test"), fontsize=14)
    cb.ax.tick_params(labelsize=13)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax = fig.add_subplot(1, 3, 2, projection=ccrs.SouthPolarStereo(central_longitude=0))
    setup_ax(ax, "b", "Bias (B-SOSE - satellite)")
    lim = float(np.nanpercentile(np.abs(bias), 98)) if np.isfinite(bias).any() else 1.0
    lim = float(f"{lim:.1g}") or 1.0
    lc, bc = wrap(lon, np.clip(bias, -lim, lim))
    fill = ax.contourf(lc, lat, bc, levels=np.linspace(-lim, lim, 31),
                       cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect("equal")
    cb = plt.colorbar(fill, ax=ax, orientation="vertical", shrink=0.75,
                      ticks=np.linspace(-lim, lim, 5))
    cb.set_label("Chl difference (mg m$^{-3}$)", fontsize=14)
    cb.ax.tick_params(labelsize=13)

    ax = fig.add_subplot(1, 3, 3, projection=ccrs.SouthPolarStereo(central_longitude=0))
    setup_ax(ax, "c", "Months with both")
    lc, nc = wrap(lon, np.where(n >= MIN_MONTHS, n, np.nan).astype(float))
    fill = ax.contourf(lc, lat, nc, levels=np.linspace(MIN_MONTHS, NT, 13),
                       cmap=plt.cm.viridis, transform=ccrs.PlateCarree())
    ax.set_aspect("equal")
    cb = plt.colorbar(fill, ax=ax, orientation="vertical", shrink=0.75)
    cb.set_label("months", fontsize=14)
    cb.ax.tick_params(labelsize=13)

    fig.suptitle(f"B-SOSE vs satellite chlorophyll"
                 + (" anomalies" if ANOM else "")
                 + f", {times[0]:%Y}-{times[-1]:%Y}, {k} x {k} blocks"
                 + (f", satellite gaps <= {INTERP_GAP} month(s) filled in time"
                    if INTERP_GAP > 0 else ""), fontsize=18)
    out = os.path.join(OUTDIR, tag + ".pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nwrote {out}\n      {os.path.join(OUTDIR, tag + '.npz')}\n"
          f"      {os.path.join(OUTDIR, tag + '.csv')}")


if __name__ == "__main__":
    main()
