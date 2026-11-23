"""BGC-Argo chlorophyll (CHLA_ADJUSTED, QC 1/2/5/8, mean 0-25 dbar) against
B-SOSE (top 25 m) and the satellite, at the same cell and month.
Output in /work/bb1596/b384036/b384036/causality/results_argo_chl/."""
import csv
import glob
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
import corr_bsose_sat as cs

ARGO_DIR = os.environ.get("ARGO_DIR", CAUS + "/argo/sprof")
PROFILES = os.environ.get("PROFILES", CAUS + "/argo/argo_profiles_chl_0_25dbar.csv")
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_argo_chl")
ZONES_FILE = os.environ.get("ZONES_FILE", CAUS + "/acc_zones3_bsose.nc")
TOP = 25.0
MIN_LEVELS = int(os.environ.get("MIN_LEVELS", 3))
GOOD_QC = {"1", "2", "5", "8"}
BOXES = {"SE Pacific subtropics": (-42, -32, 230, 275),
         "S Atlantic subtropics": (-42, -32, 330, 375)}


def qc_str(a):
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x)
                         for x in a.ravel()]).reshape(a.shape)
    return a.astype(str)


def read_float(path):
    out = []
    try:
        ds = xr.open_dataset(path)
    except Exception as exc:
        print(f"  {os.path.basename(path)}: {exc}")
        return out
    with ds:
        if "CHLA_ADJUSTED" not in ds:
            return out
        chl = ds["CHLA_ADJUSTED"].values.astype("float64")
        qc = qc_str(ds["CHLA_ADJUSTED_QC"].values) if "CHLA_ADJUSTED_QC" in ds \
            else np.full(chl.shape, "1")
        pres = ds["PRES_ADJUSTED"].values.astype("float64") if "PRES_ADJUSTED" in ds \
            else np.full(chl.shape, np.nan)
        if "PRES" in ds:
            pres = np.where(np.isfinite(pres), pres, ds["PRES"].values.astype("float64"))
        lat = ds["LATITUDE"].values.astype("float64")
        lon = ds["LONGITUDE"].values.astype("float64")
        juld = pd.to_datetime(ds["JULD"].values)
    good = np.isfinite(chl) & np.isin(qc, list(GOOD_QC)) & np.isfinite(pres) \
        & (pres >= 0) & (pres <= TOP)
    for i in range(chl.shape[0]):
        g = good[i]
        if g.sum() < MIN_LEVELS or pres[i][g].min() > 10.0:
            continue
        if not (np.isfinite(lat[i]) and np.isfinite(lon[i])) or pd.isna(juld[i]):
            continue
        out.append((juld[i], lat[i], lon[i], float(np.mean(np.clip(chl[i][g], 0, None)))))
    return out


def stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n = int(ok.sum())
    if n < 10:
        return n, np.nan, np.nan
    la, lb = np.log10(a[ok]), np.log10(b[ok])
    return n, float(np.corrcoef(la, lb)[0, 1]), float(10 ** np.median(lb - la))


def anom_r(df, col):
    d = df[np.isfinite(df[col]) & (df[col] > 0) & (df["argo"] > 0)].copy()
    if len(d) < 30:
        return np.nan
    for c in ("argo", col):
        d["l_" + c] = np.log10(d[c])
        d["a_" + c] = d["l_" + c] - d.groupby("month")["l_" + c].transform("median")
    return float(np.corrcoef(d["a_argo"], d["a_" + col])[0, 1])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(PROFILES):
        prof = pd.read_csv(PROFILES, parse_dates=["time"])[["wmo", "time", "lat", "lon", "argo"]]
        files = [PROFILES]
        print(f"profiles from {PROFILES}", flush=True)
    else:
        files = sorted(glob.glob(os.path.join(ARGO_DIR, "*_Sprof.nc")))
        if not files:
            sys.exit(f"neither {PROFILES} nor Sprof files in {ARGO_DIR}")
        rows = []
        for f in files:
            wmo = os.path.basename(f).split("_")[0]
            rows += [(wmo,) + r for r in read_float(f)]
        prof = pd.DataFrame(rows, columns=["wmo", "time", "lat", "lon", "argo"])
    prof = prof[(prof.lat <= -30) & (prof.time >= "2013-01-01") & (prof.time < "2025-01-01")]
    print(f"{len(prof)} profiles with 0-{TOP:g} dbar chlorophyll from "
          f"{prof.wmo.nunique()} floats in {len(files)} files", flush=True)
    if prof.empty:
        sys.exit("no usable profiles")
    mod, yc, xc, times = cs.bsose_chl()
    sat = cs.satellite_on_grid(yc, xc, times)
    t_idx = ((prof.time.dt.year - 2013) * 12 + prof.time.dt.month - 1).values
    iy = np.abs(yc[None, :] - prof.lat.values[:, None]).argmin(axis=1)
    lon360 = np.mod(prof.lon.values, 360.0)
    ix = np.abs(((np.mod(xc, 360.0)[None, :] - lon360[:, None] + 180) % 360) - 180).argmin(axis=1)
    inside = (t_idx >= 0) & (t_idx < len(times)) & (np.abs(yc[iy] - prof.lat.values) < 0.2)
    prof = prof[inside].copy()
    t_idx, iy, ix = t_idx[inside], iy[inside], ix[inside]
    prof["bsose"] = mod[t_idx, iy, ix]
    prof["satellite"] = sat[t_idx, iy, ix]
    prof["month"] = prof.time.dt.month
    prof["lon360"] = np.mod(prof.lon.values, 360.0)
    del mod, sat
    if os.path.exists(ZONES_FILE):
        with xr.open_dataset(ZONES_FILE) as z:
            zone = z["zone"].transpose("YC", "XC").values
            fv = str(z["zone"].attrs.get("flag_values", "")).split()
            fm = str(z["zone"].attrs.get("flag_meanings", "")).split()
        names = {int(v): {"ACC": "SAF_SB"}.get(m, m) for v, m in zip(fv, fm)}
        prof["region"] = [names.get(int(k), "none") for k in zone[iy, ix]]
    else:
        prof["region"] = "All"
    prof = prof[np.isfinite(prof.bsose)]
    prof.to_csv(os.path.join(OUTDIR, "argo_matchups.csv"), index=False)
    print(f"{len(prof)} match-ups with B-SOSE, "
          f"{int(np.isfinite(prof.satellite).sum())} also with the satellite", flush=True)

    regions = ["All"] + [r for r in ("N_SAF", "SAF_SB", "S_SB") if r in set(prof.region)]
    sel = {r: (prof if r == "All" else prof[prof.region == r]) for r in regions}
    for name, (la0, la1, lo0, lo1) in BOXES.items():
        lo = np.where(prof.lon360 < lo0, prof.lon360 + 360, prof.lon360) if lo1 > 360 \
            else prof.lon360
        sel[name] = prof[((prof.lat >= la0) & (prof.lat <= la1) & (lo >= lo0) & (lo <= lo1)).values]

    out = []
    print(f"\n{'region':22s} {'pair':18s} {'n':>6s} {'r(log)':>7s} {'ratio':>6s} {'r(anom)':>8s}")
    for reg, d in sel.items():
        for col, lab in (("bsose", "B-SOSE vs Argo"), ("satellite", "satellite vs Argo")):
            n, r, ratio = stats(d.argo.values, d[col].values)
            ra = anom_r(d, col) if len(d) else np.nan
            out.append(dict(region=reg, pair=lab, n=n, r_log=round(r, 3),
                            median_ratio=round(ratio, 3), r_anom=round(ra, 3)))
            print(f"{reg:22s} {lab:18s} {n:6d} {r:7.2f} {ratio:6.2f} {ra:8.2f}")
    with open(os.path.join(OUTDIR, "argo_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("\nratio: median of product / Argo; r(anom): log10 values minus each "
          "product's monthly median in that region")

    fig, axes = plt.subplots(2, len(regions), figsize=(4.2 * len(regions), 8), dpi=180,
                             squeeze=False)
    for j, reg in enumerate(regions):
        d = sel[reg]
        for i, (col, colr) in enumerate((("bsose", "#1f77b4"), ("satellite", "#d62728"))):
            ax = axes[i, j]
            ok = (d.argo > 0) & (d[col] > 0)
            ax.scatter(d.argo[ok], d[col][ok], s=3, alpha=0.3, color=colr, lw=0)
            ax.plot([0.01, 20], [0.01, 20], "k-", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.01, 20)
            ax.set_ylim(0.01, 20)
            n, r, ratio = stats(d.argo.values, d[col].values)
            ax.set_title(f"{reg}: n={n}, r={r:.2f}, ratio={ratio:.2f}", fontsize=10)
            ax.set_xlabel("Argo Chl 0-25 m (mg m$^{-3}$)")
            ax.set_ylabel(("B-SOSE" if col == "bsose" else "satellite") + " Chl (mg m$^{-3}$)")
            ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "argo_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)

    keys = list(sel)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), dpi=180, squeeze=False)
    for j, reg in enumerate(keys):
        ax = axes[0, j]
        d = sel[reg]
        for col, lab, colr, mk in (("argo", "Argo", "k", "o"), ("bsose", "B-SOSE", "#1f77b4", "s"),
                                   ("satellite", "satellite", "#d62728", "^")):
            dd = d[np.isfinite(d[col])]
            med = dd.groupby("month")[col].median().reindex(range(1, 13))
            cnt = dd.groupby("month")[col].size().reindex(range(1, 13)).fillna(0)
            med[cnt < 5] = np.nan
            ax.plot(range(1, 13), med.values, marker=mk, color=colr, label=lab)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_title(f"{reg} ({len(d)} profiles)", fontsize=10)
        ax.set_ylabel("median Chl (mg m$^{-3}$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Monthly medians at the float match-ups (months with >= 5 profiles)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "argo_seasonal.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUTDIR}/argo_matchups.csv, argo_stats.csv, argo_scatter.pdf, argo_seasonal.pdf")


if __name__ == "__main__":
    main()
