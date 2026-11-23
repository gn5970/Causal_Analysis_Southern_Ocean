"""Vertical diffusive iron supply in B-SOSE and its trend: mixing or iron drift?
    python /work/bb1596/b384036/b384036/causality/iron_supply.py          stream + compute
    PLOT_ONLY=1 python /work/bb1596/b384036/b384036/causality/iron_supply.py
Run on the LOGIN node. Kz from DFrI_TH/DFrI_SLT (buoyancy), F = Kz * dFe/dz at the
first interface (50/100/150/200 m) below the MLD; trend split into mixing and
gradient parts; Fe 100-500 m drift test; Chl-a trend for comparison.
Output: /work/bb1596/b384036/b384036/causality/results_iron_supply/
"""
import os
import sys
import time
import numpy as np
import xarray as xr

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
BSOSE_URL = os.environ.get("BSOSE_URL", "https://sose.ucsd.edu/SO6/ITER156/MONTHLY/")
BSOSE_DIR = os.environ.get("BSOSE_DIR", CAUS + "/bsose")
NAME = "{}_bsoseI156_2013to2024_monthly.nc"
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_iron_supply")
ZONES_FILE = os.environ.get("ZONES_FILE", CAUS + "/acc_zones3_bsose.nc")
K = int(os.environ.get("COARSEN", 4))
NT = int(os.environ.get("NT", 144))
IFACES = [float(x) for x in os.environ.get("IFACES", "50,100,150,200").split(",")]
SUB = (100.0, 500.0)
N2MIN = float(os.environ.get("N2MIN", 1e-6))
G0 = 9.81
YEAR0 = 2013
SKIP = {"drF", "hFacC", "rA", "Depth", "drC", "dxG", "dyG", "rAw", "rAs", "rAz",
        "dxC", "dyC", "maskC", "maskW", "maskS", "PHrefC", "Zl", "Zu", "Zp1", "Z",
        "XC", "YC", "XG", "YG", "time", "iter"}


def open_remote(word):
    import fsspec
    url = BSOSE_URL + NAME.format(word)
    f = fsspec.open(url, "rb", block_size=8 << 20).open()
    return xr.open_dataset(f, engine="h5netcdf", decode_times=False)


def main_var(ds):
    c = [v for v in ds.data_vars if v not in SKIP and ds[v].ndim == 4]
    return max(c, key=lambda v: ds[v].size)


def zdim(da):
    return next(d for d in da.dims if d.upper().startswith("Z"))


def read_levels(word, levels, t0, tries=6):
    for a in range(1, tries + 1):
        try:
            with open_remote(word) as ds:
                da = ds[main_var(ds)]
                return da.isel({zdim(da): list(levels), "time": slice(t0, t0 + 12)}
                               ).values.astype("float32")
        except Exception as exc:
            print(f"    {word}: {type(exc).__name__}: {exc}; retry {a}/{tries}", flush=True)
            time.sleep(30 * a)
    raise RuntimeError(f"could not read {word}")


def grid():
    with open_remote("Theta") as ds:
        z = np.abs(ds["Z"].values.astype("float64"))
        drf = ds["drF"].values.astype("float64")
        hfac = ds["hFacC"].values.astype("float32")
        yc = ds["YC"].values.astype("float64")
        xc = ds["XC"].values.astype("float64")
        ra = ds["rA"].values.astype("float64") if "rA" in ds else None
        zl = np.abs(ds["Zl"].values.astype("float64")) if "Zl" in ds else None
    if ra is None:
        R = 6.371e6
        dlon = np.deg2rad(np.abs(np.diff(xc).mean()))
        dlat = np.abs(np.diff(yc).mean())
        s = np.sin(np.deg2rad(yc + dlat / 2)) - np.sin(np.deg2rad(yc - dlat / 2))
        ra = (R * R * dlon * np.abs(s))[:, None] * np.ones((1, xc.size))
    if zl is None:
        zl = np.concatenate([[0.0], np.cumsum(drf)[:-1]])
    return z, zl, drf, hfac, yc, xc, ra


def alpha_beta(S, T):
    try:
        import gsw
        return gsw.alpha(S, T, 0.0), gsw.beta(S, T, 0.0)
    except ImportError:
        def rho(s, t):
            return (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t ** 2
                    + 1.001685e-4 * t ** 3 - 1.120083e-6 * t ** 4 + 6.536332e-9 * t ** 5
                    + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t ** 2
                           - 8.2467e-7 * t ** 3 + 5.3875e-9 * t ** 4)
                    + np.abs(s) ** 1.5 * (-5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2)
                    + 4.8314e-4 * s ** 2)
        r = rho(S, T)
        return (-(rho(S, T + 0.05) - rho(S, T - 0.05)) / 0.1 / r,
                (rho(S + 0.05, T) - rho(S - 0.05, T)) / 0.1 / r)


def blocks(a):
    Y, X = a.shape[-2:]
    yb, xb = Y // K, X // K
    a = a[..., :yb * K, :xb * K].reshape(a.shape[:-2] + (yb, K, xb, K))
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as s:
        s.filter(RuntimeWarning)
        return np.nanmean(a, axis=(-3, -1)).astype("float32")


def one_year(y, g, iface_k, sub_k, mld):
    z, zl, drf, hfac, yc, xc, ra = g
    t0 = (y - YEAR0) * 12
    need = sorted({k for kk in iface_k for k in (kk - 1, kk)})
    lev = {k: i for i, k in enumerate(need)}
    T = read_levels("Theta", need, t0)
    S = read_levels("Salt", need, t0)
    fe_lev = sorted(set(need) | set(sub_k))
    fl = {k: i for i, k in enumerate(fe_lev)}
    FE = read_levels("Fe", fe_lev, t0)
    FT = read_levels("DFrI_TH", iface_k, t0)
    FS = read_levels("DFrI_SLT", iface_k, t0)
    print(f"  {y}: read", flush=True)
    nm = T.shape[0]
    Ks, Gs = [], []
    for i, k in enumerate(iface_k):
        wet = (hfac[k - 1] > 0) & (hfac[k] > 0)
        dz = z[k] - z[k - 1]
        tu, td = T[:, lev[k - 1]], T[:, lev[k]]
        su, sd = S[:, lev[k - 1]], S[:, lev[k]]
        a, b = alpha_beta(0.5 * (su + sd), 0.5 * (tu + td))
        gb = a * (tu - td) / dz - b * (su - sd) / dz
        fb = (a * FT[:, i] - b * FS[:, i]) / ra[None]
        with np.errstate(invalid="ignore", divide="ignore"):
            kz = -fb / gb
        kz[(G0 * gb < N2MIN) | ~(kz > 0) | ~wet[None]] = np.nan
        gfe = (FE[:, fl[k]] - FE[:, fl[k - 1]]) / dz
        gfe[:, ~wet] = np.nan
        Ks.append(kz)
        Gs.append(gfe)
    Ks, Gs = np.stack(Ks, 1), np.stack(Gs, 1)
    dI = np.array([zl[k] for k in iface_k])
    m = mld[t0:t0 + nm]
    idx = np.clip((dI[None, :, None, None] < m[:, None]).sum(axis=1), 0, len(iface_k) - 1)
    Kml = np.take_along_axis(Ks, idx[:, None], 1)[:, 0]
    Gml = np.take_along_axis(Gs, idx[:, None], 1)[:, 0]
    i100 = int(np.argmin(np.abs(dI - 100.0)))
    w = (drf[sub_k][:, None, None] * hfac[sub_k]).astype("float64")
    fs = FE[:, [fl[k] for k in sub_k]].astype("float64")
    ok = np.isfinite(fs) & (w[None] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fesub = np.where(ok, fs * w[None], 0).sum(1) / np.where(ok, w[None], 0).sum(1)
    return dict(F=blocks(Kml * Gml), K=blocks(Kml), G=blocks(Gml),
                F100=blocks(Ks[:, i100] * Gs[:, i100]),
                K100=blocks(Ks[:, i100]), G100=blocks(Gs[:, i100]),
                Fe_sub=blocks(fesub.astype("float32")))


def chl_blocks():
    p = os.path.join(BSOSE_DIR, NAME.format("Chl"))
    with xr.open_dataset(p, decode_times=False) as ds:
        v = [x for x in ds.data_vars if ds[x].ndim == 4][0]
        keep = np.flatnonzero(np.abs(ds["Z"].values) <= 25.0)
        w = (ds["drF"].isel(Z=keep) * ds["hFacC"].isel(Z=keep)).values.astype("float64")
        out = []
        for t0 in range(0, NT, 12):
            c = ds[v].isel(Z=keep, time=slice(t0, t0 + 12)).values.astype("float64")
            ok = np.isfinite(c) & (w[None] > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(ok, c * w[None], 0).sum(1) / np.where(ok, w[None], 0).sum(1)
            out.append(blocks(c.astype("float32")))
    return np.concatenate(out, 0)


def deseason(a):
    n = a.shape[0] // 12
    b = a[:n * 12].reshape(n, 12, *a.shape[1:])
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as s:
        s.filter(RuntimeWarning)
        clim = np.nanmean(b, 0, keepdims=True)
    return (b - clim).reshape(a.shape), np.broadcast_to(clim, b.shape).reshape(a.shape)


def trend(a):
    from scipy.stats import t as tdist
    t = np.arange(a.shape[0]) / 12.0
    ok = np.isfinite(a)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(ok, t[:, None, None], 0).sum(0) / n
        am = np.where(ok, a, 0).sum(0) / n
        dt = np.where(ok, t[:, None, None] - tm, 0)
        da = np.where(ok, a - am, 0)
        sxx = (dt * dt).sum(0)
        b = (dt * da).sum(0) / sxx
        res = np.where(ok, da - b * dt, 0)
        se = np.sqrt((res ** 2).sum(0) / np.maximum(n - 2, 1) / sxx)
        p = 2 * tdist.sf(np.abs(b / se), np.maximum(n - 2, 1))
    b[n < 36] = np.nan
    return b, p


def std_trend(a):
    an, _ = deseason(a)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as s:
        s.filter(RuntimeWarning)
        sd = np.nanstd(an, 0)
    b, p = trend(an)
    return b / np.where(sd > 0, sd, np.nan), p


def zones(yc, ny, nx):
    try:
        with xr.open_dataset(ZONES_FILE) as z:
            zone = z["zone"].transpose("YC", "XC").values
        zb = zone[:ny * K, :nx * K].reshape(ny, K, nx, K)
        names = {1: "N_SAF", 2: "SAF_SB", 3: "S_SB"}
        cnt = np.stack([(zb == c).sum((1, 3)) for c in names])
        best = np.array(list(names))[cnt.argmax(0)]
        best = np.where(cnt.max(0) >= K * K / 2, best, 0)
        return {n: best == c for c, n in names.items()}
    except Exception as exc:
        print(f"[zones] {exc}; latitude bands", flush=True)
        lat = yc[:ny * K].reshape(ny, K).mean(1)[:, None] * np.ones((1, nx))
        return {"N_SAF": lat > -45, "SAF_SB": (lat <= -45) & (lat > -60), "S_SB": lat <= -60}


def compute():
    os.makedirs(os.path.join(OUTDIR, "cache"), exist_ok=True)
    g = grid()
    z, zl = g[0], g[1]
    iface_k = sorted({int(np.argmin(np.abs(zl - d))) for d in IFACES})
    sub_k = [k for k in range(z.size) if SUB[0] < z[k] <= SUB[1]]
    print("interfaces (m): " + ", ".join(f"{zl[k]:.0f}" for k in iface_k)
          + f"; Fe {SUB[0]:g}-{SUB[1]:g} m: {len(sub_k)} levels", flush=True)
    with xr.open_dataset(os.path.join(BSOSE_DIR, NAME.format("MLD")), decode_times=False) as ds:
        mld = np.abs(ds["BLGMLD"].isel(time=slice(0, NT)).values.astype("float32"))
    for y in range(YEAR0, YEAR0 + NT // 12):
        c = os.path.join(OUTDIR, "cache", f"year_{y}.npz")
        if os.path.exists(c):
            continue
        r = one_year(y, g, iface_k, sub_k, mld)
        np.savez(c + ".tmp.npz", **r)
        os.replace(c + ".tmp.npz", c)
        print(f"  {y}: done", flush=True)
    parts = [dict(np.load(os.path.join(OUTDIR, "cache", f"year_{y}.npz")))
             for y in range(YEAR0, YEAR0 + NT // 12)]
    out = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    out["Chl"] = chl_blocks()
    yc, xc = g[4], g[5]
    ny, nx = out["F"].shape[1:]
    out["lat"] = yc[:ny * K].reshape(ny, K).mean(1)
    out["lon"] = xc[:nx * K].reshape(nx, K).mean(1)
    np.savez(os.path.join(OUTDIR, "iron_supply.npz"), **out)
    return out


def plot(d):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from corr_bsose_sat import setup_ax, wrap

    lat, lon = d["lat"], d["lon"]
    ny, nx = d["F"].shape[1:]
    zn = zones(np.repeat(lat, K), ny, nx)
    yrs = YEAR0 + np.arange(NT // 12)
    tF, pF = trend(deseason(d["F"])[0])
    Kc, Gc = deseason(d["K"])[1], deseason(d["G"])[1]
    tKpart, _ = trend(deseason(d["K"] * Gc)[0])
    tGpart, _ = trend(deseason(Kc * d["G"])[0])
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as s:
        s.filter(RuntimeWarning)
        Fm, Km, FeM = (np.nanmean(d[k], 0) for k in ("F", "K", "Fe_sub"))
    tK, _ = trend(deseason(d["K"])[0])
    tFe, pFe = trend(deseason(d["Fe_sub"])[0])
    sF, _ = std_trend(d["F"])
    sC, _ = std_trend(d["Chl"])
    ok = np.isfinite(sF) & np.isfinite(sC)
    r_pat = float(np.corrcoef(sF[ok], sC[ok])[0, 1])
    print(f"pattern correlation, trend of Fe supply vs trend of Chl-a (sigma/yr): r = {r_pat:.2f}")
    rows = []
    for zname, m in zn.items():
        def zm(a):
            return float(np.nanmean(np.where(m, a, np.nan)))
        F0 = zm(Fm)
        rows.append(dict(zone=zname, blocks=int(m.sum()),
                         mean_F_umol_m2_d=F0 * 1e6 * 86400,
                         trend_F_pct_yr=100 * zm(tF) / F0,
                         mixing_part_pct_yr=100 * zm(tKpart) / F0,
                         gradient_part_pct_yr=100 * zm(tGpart) / F0,
                         trend_Kz_pct_yr=100 * zm(tK) / zm(Km),
                         trend_Fe100_500_pct_yr=100 * zm(tFe) / zm(FeM),
                         r_trend_F_vs_Chl=float(np.corrcoef(sF[m & ok], sC[m & ok])[0, 1])))
    with open(os.path.join(OUTDIR, "iron_supply.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 11))
    col = {"N_SAF": "tab:orange", "SAF_SB": "tab:green", "S_SB": "tab:blue"}
    ax = fig.add_subplot(2, 3, 1)
    for zname, m in zn.items():
        s = np.nanmean(np.where(m[None], d["Fe_sub"], np.nan), (1, 2))
        ann = s[:yrs.size * 12].reshape(yrs.size, 12).mean(1)
        ax.plot(yrs, 100 * (ann / ann[0] - 1), "o-", c=col[zname], label=zname)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_ylabel("dissolved Fe, 100-500 m (% change from 2013)")
    ax.legend(frameon=False)
    ax.text(-0.12, 1.03, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    def pmap(i, letter, title, f, lim, lab, p=None):
        ax = fig.add_subplot(2, 3, i, projection=ccrs.SouthPolarStereo())
        setup_ax(ax, letter, title)
        lc, fc = wrap(lon, f)
        mm = ax.pcolormesh(lc, lat, fc, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="nearest", transform=pc)
        if p is not None:
            yy, xx = np.nonzero(np.isfinite(p) & (p < 0.05))
            ax.scatter(lon[xx][::4], lat[yy][::4], s=0.3, c="k", lw=0, transform=pc)
        fig.colorbar(mm, ax=ax, shrink=0.75).set_label(lab)

    with np.errstate(invalid="ignore", divide="ignore"):
        pmap(2, "b", "Fe 100-500 m", 100 * tFe / FeM, 3, "trend (% yr$^{-1}$)", pFe)
        pmap(3, "c", "Chl-a 0-25 m", sC, 0.16, "trend ($\\sigma$ yr$^{-1}$)")
        pmap(4, "d", f"Fe supply (r = {r_pat:.2f} with c)", sF, 0.16, "trend ($\\sigma$ yr$^{-1}$)")
        pmap(6, "f", "Kz below the mixed layer", 100 * tK / Km, 5, "trend (% yr$^{-1}$)")
    ax = fig.add_subplot(2, 3, 5)
    x = np.arange(len(rows))
    wdt = 0.27
    ax.bar(x - wdt, [r["trend_F_pct_yr"] for r in rows], wdt, color="k", label="Fe supply")
    ax.bar(x, [r["mixing_part_pct_yr"] for r in rows], wdt, color="tab:purple", label="mixing (Kz)")
    ax.bar(x + wdt, [r["gradient_part_pct_yr"] for r in rows], wdt, color="tab:brown", label="Fe gradient")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r["zone"] for r in rows])
    ax.set_ylabel("trend (% of mean supply yr$^{-1}$)")
    ax.legend(frameon=False)
    ax.text(-0.12, 1.03, "e", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "iron_supply")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {out}.pdf/.png/.csv")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    npz = os.path.join(OUTDIR, "iron_supply.npz")
    if os.environ.get("PLOT_ONLY", "0") == "1":
        plot(dict(np.load(npz)))
    else:
        plot(compute())
