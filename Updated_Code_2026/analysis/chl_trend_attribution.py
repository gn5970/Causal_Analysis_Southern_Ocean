"""How much of the B-SOSE Chl-a trend do the PCMCI drivers reproduce?
    RUN_DIR=<c4 PCMCI folder> python /work/bb1596/b384036/b384036/causality/chl_trend_attribution.py
Per block: sensitivities b_k fitted on DETRENDED standardized anomalies
(Chl' = sum b_k X_k', LAGS=0 or 0,1), then predicted trend = sum b_k trend(X_k),
compared with the actual Chl trend (sigma/yr, as Fig. 1a). Zone means over the
blocks kept by the satellite mask (untested sea-ice blocks kept); MASK=none for all.
Output: RUN_DIR/attribution/chl_trend_attribution.pdf/.png/.csv
"""
import os
import sys
import csv
import numpy as np

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
RUN_DIR = os.environ.get("RUN_DIR", CAUS + "/results_final_PCMCI_c4_bsose_2013to2024")
K = int(os.environ.get("COARSEN", 4))
LAGS = [int(x) for x in os.environ.get("LAGS", "0").split(",")]
ZONES_FILE = os.environ.get("ZONES_FILE", CAUS + "/acc_zones3_bsose.nc")
MASK = os.environ.get("MASK", f"{CAUS}/results_corr_bsose_sat_c{K}/corr_bsose_sat_c{K}_cal8.npz")
OUT = os.path.join(RUN_DIR, "attribution")
MIN_MONTHS = 60


def detrend(a, t):
    ok = np.isfinite(a)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(ok, t[:, None], 0).sum(0) / n
        am = np.where(ok, a, 0).sum(0) / n
        dt = np.where(ok, t[:, None] - tm, 0)
        da = np.where(ok, a - am, 0)
        b = (dt * da).sum(0) / (dt * dt).sum(0)
    return np.where(ok, da - b * dt, np.nan), b


def zones(ny, nx, lat):
    import xarray as xr
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
        print(f"[zones] {exc}; latitude bands")
        L = lat[:, None] * np.ones((1, nx))
        return {"N_SAF": L > -45, "SAF_SB": (L <= -45) & (L > -60), "S_SB": L <= -60}


def main():
    p = os.path.join(RUN_DIR, "anomalies.npz")
    if not os.path.exists(p):
        sys.exit(f"no {p}: set RUN_DIR to the c{K} PCMCI folder")
    d = np.load(p)
    names = [str(n) for n in d["names"]]
    X = d["feats"].astype("float64")
    y = d["chl"].astype("float64")
    lat, lon = d["lat"], d["lon"]
    D, T, ny, nx = X.shape
    t = np.arange(T) / 12.0
    yd, ytr = detrend(y.reshape(T, -1), t)
    Xb = X.reshape(D, T, -1)
    Xd = np.empty_like(Xb)
    Xtr = np.empty((D, Xb.shape[2]))
    for k in range(D):
        Xd[k], Xtr[k] = detrend(Xb[k], t)
    del X, Xb
    L = max(LAGS)
    cols, ctr, cname = [], [], []
    for lag in LAGS:
        for k in range(D):
            cols.append(Xd[k, L - lag:T - lag])
            ctr.append(Xtr[k])
            cname.append(names[k])
    A = np.stack(cols, -1)
    Y = yd[L:]
    ok = np.isfinite(Y) & np.all(np.isfinite(A), -1)
    A = np.where(ok[..., None], A, 0.0)
    Y = np.where(ok, Y, 0.0)
    n = ok.sum(0)
    AtA = np.einsum("tbp,tbq->bpq", A, A)
    Aty = np.einsum("tbp,tb->bp", A, Y)
    good = (n >= MIN_MONTHS) & np.isfinite(ytr)
    beta = np.full(Aty.shape, np.nan)
    beta[good] = np.linalg.solve(AtA[good] + 1e-9 * np.eye(AtA.shape[1]), Aty[good][..., None])[..., 0]
    fit = np.einsum("tbp,bp->tb", A, np.nan_to_num(beta))
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - ((Y - fit) ** 2).sum(0) / (Y ** 2).sum(0)
    r2[~good] = np.nan
    contrib = beta.T * np.array(ctr)
    per_driver = {nm: np.zeros(contrib.shape[1]) for nm in names}
    for j, cn in enumerate(cname):
        per_driver[cn] = per_driver[cn] + contrib[j]
    pred = np.nansum(contrib, 0)
    pred[~good] = np.nan
    shp = (ny, nx)
    act2 = np.where(good, ytr, np.nan).reshape(shp)
    pred2, r22 = pred.reshape(shp), r2.reshape(shp)
    per2 = {k: np.where(good, v, np.nan).reshape(shp) for k, v in per_driver.items()}

    keep = np.ones(shp, bool)
    if MASK.lower() != "none" and os.path.exists(MASK):
        m = np.load(MASK)
        r, sig = m["r"], m["sig"].astype(bool)
        if r.shape == shp:
            keep = (sig & np.isfinite(r) & (r > 0)) | ~np.isfinite(r)
            print(f"mask {MASK}: {int(keep.sum())} of {keep.size} blocks kept")
        else:
            print(f"[mask] shape {r.shape} differs from {shp}; no mask")
    zn = zones(ny, nx, lat)

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for z, m in zn.items():
        m = m & keep & np.isfinite(act2)

        def zm(a):
            return float(np.nanmean(a[m]))
        a_, p_ = zm(act2), zm(pred2)
        row = dict(zone=z, blocks=int(m.sum()), actual_sd_per_yr=a_, predicted_sd_per_yr=p_,
                   share_predicted=p_ / a_ if a_ else np.nan, median_R2=float(np.nanmedian(r22[m])))
        row.update({f"contrib_{k}": zm(v) for k, v in per2.items()})
        rows.append(row)
    with open(os.path.join(OUT, "chl_trend_attribution.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from corr_bsose_sat import setup_ax, wrap
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(20, 5.6))
    lim = 0.16
    for i, (letter, title, f) in enumerate([("a", "Chl-a trend", act2),
                                            ("b", "from the drivers' trends", pred2),
                                            ("c", "residual (a - b)", act2 - pred2)]):
        ax = fig.add_subplot(1, 4, i + 1, projection=ccrs.SouthPolarStereo())
        setup_ax(ax, letter, title)
        lc, fc = wrap(lon, np.where(keep, f, np.nan))
        mm = ax.pcolormesh(lc, lat, fc, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="nearest", transform=pc)
    fig.colorbar(mm, ax=fig.axes, shrink=0.6, location="bottom", pad=0.05
                 ).set_label("trend ($\\sigma$ yr$^{-1}$)")
    ax = fig.add_subplot(1, 4, 4)
    labs = ["actual", "predicted"] + names
    x = np.arange(len(rows))
    wdt = 0.8 / len(labs)
    cols = ["k", "0.5"] + list(plt.cm.tab10.colors[:len(names)])
    for j, lb in enumerate(labs):
        key = {"actual": "actual_sd_per_yr", "predicted": "predicted_sd_per_yr"}.get(lb, f"contrib_{lb}")
        ax.bar(x - 0.4 + (j + 0.5) * wdt, [r[key] for r in rows], wdt, color=cols[j], label=lb)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r["zone"] for r in rows])
    ax.set_ylabel("trend ($\\sigma$ yr$^{-1}$)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(-0.15, 1.03, "d", transform=ax.transAxes, fontsize=16, fontweight="bold")
    out = os.path.join(OUT, "chl_trend_attribution")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
