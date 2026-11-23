"""Independent check, ACC only (zone 2): trends of oceQsw, surfTflx, Theta 0-25 m,
MLD and the biomass-weighted Llim, FElim over 0-25 m, from the raw B-SOSE files.
    python /work/bb1596/b384036/b384036/causality/check_acc_light.py   (compute node)
Output: printed table, /work/bb1596/b384036/b384036/causality/results_check_acc/
"""
import os
import csv
import numpy as np
import xarray as xr
from scipy.stats import linregress

CAUS = "/work/bb1596/b384036/b384036/causality"
BS = CAUS + "/bsose"
NAME = BS + "/{}_bsoseI156_2013to2024_monthly.nc"
OUT = CAUS + "/results_check_acc"
ZONE = int(os.environ.get("ZONE", 2))
NT, Y0 = 144, 2013
SEAS = {"DJF": [11, 0, 1], "MAM": [2, 3, 4], "JJA": [5, 6, 7], "SON": [8, 9, 10]}


def var_of(ds, prefer=None):
    if prefer and prefer in ds:
        return prefer
    skip = {"drF", "hFacC", "rA", "Depth", "maskC", "PHrefC", "drC"}
    c = [v for v in ds.data_vars if v not in skip and ds[v].ndim >= 3]
    return max(c, key=lambda v: ds[v].size)


def main():
    os.makedirs(OUT, exist_ok=True)
    with xr.open_dataset(CAUS + "/acc_zones3_bsose.nc") as z:
        zone = z["zone"].transpose("YC", "XC").values
    with xr.open_dataset(NAME.format("Theta"), decode_times=False) as g:
        yc = g["YC"].values
        z0 = np.abs(g["Z"].values)
        keep = np.flatnonzero(z0 <= 25.0)
        dz = (g["drF"].isel(Z=keep) * g["hFacC"].isel(Z=keep)).values.astype("float64")
        wet = g["hFacC"].isel(Z=0).values > 0
    if zone.shape != wet.shape:
        raise SystemExit(f"zone file {zone.shape} differs from the grid {wet.shape}")
    m = (zone == ZONE) & wet
    A = np.where(m, np.cos(np.deg2rad(yc))[:, None], 0.0)
    print(f"zone {ZONE}: {int(m.sum())} cells", flush=True)

    def amean(a2):
        ok = np.isfinite(a2) & (A[None] > 0)
        return np.where(ok, a2 * A[None], 0).sum((1, 2)) / np.where(ok, A[None], 0).sum((1, 2))

    S, attrs = {}, {}
    for word, key, prefer in [("oceQsw", "shortwave", None), ("surfTflx", "heat_flux", None),
                              ("MLD", "MLD", "BLGMLD")]:
        with xr.open_dataset(NAME.format(word), decode_times=False) as ds:
            v = var_of(ds, prefer)
            attrs[key] = {k: str(ds[v].attrs.get(k, "")) for k in ("units", "long_name", "description")}
            out = []
            for t0 in range(0, NT, 12):
                a = ds[v].isel(time=slice(t0, t0 + 12)).values.astype("float64")
                a = a[:, 0] if a.ndim == 4 else a
                if key == "MLD":
                    a = np.abs(a)
                    a[a == 0] = np.nan
                out.append(amean(a))
            S[key] = np.concatenate(out)
    with xr.open_dataset(NAME.format("Theta"), decode_times=False) as ds:
        out = []
        for t0 in range(0, NT, 12):
            a = ds["THETA"].isel(Z=keep, time=slice(t0, t0 + 12)).values.astype("float64")
            ok = np.isfinite(a) & (dz[None] > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                a = np.where(ok, a * dz[None], 0).sum(1) / np.where(ok, dz[None], 0).sum(1)
            out.append(amean(a))
        S["theta_0_25"] = np.concatenate(out)
    with xr.open_dataset(NAME.format("BioMass"), decode_times=False) as db:
        vb = var_of(db)
        kb = np.flatnonzero(np.abs(db["Z"].values) <= 25.0)
        for word, key in [("Llim", "gamma_light"), ("FElim", "gamma_Fe")]:
            with xr.open_dataset(NAME.format(word), decode_times=False) as dl:
                vl = var_of(dl)
                attrs[key] = {k: str(dl[vl].attrs.get(k, "")) for k in ("units", "long_name", "description")}
                out = []
                for t0 in range(0, NT, 12):
                    L = dl[vl].isel(Z=kb, time=slice(t0, t0 + 12)).values.astype("float64")
                    B = np.abs(db[vb].isel(Z=kb, time=slice(t0, t0 + 12)).values.astype("float64"))
                    w = B * dz[None][:, :len(kb)]
                    ok = np.isfinite(L) & np.isfinite(w) & (w > 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        g2 = np.where(ok, L * w, 0).sum(1) / np.where(ok, w, 0).sum(1)
                    out.append(amean(g2))
                S[key] = np.concatenate(out)

    print("\nATTRIBUTES (check the sign convention of the fluxes):")
    for k, a in attrs.items():
        print(f"  {k:12s} {a}")

    rows = []
    yrs = Y0 + np.arange(NT // 12)
    for key, s in S.items():
        mon = s.reshape(-1, 12)
        an = (mon - mon.mean(0)).ravel()
        r = linregress(np.arange(NT) / 12, an)
        row = dict(variable=key, mean=float(s.mean()), year_trend=r.slope, year_p=r.pvalue)
        for sn, idx in SEAS.items():
            if sn == "DJF":
                sv = np.array([np.mean([s[12 * (i - 1) + 11], s[12 * i], s[12 * i + 1]])
                               for i in range(1, NT // 12)])
                yy = yrs[1:]
            else:
                sv = mon[:, idx].mean(1)
                yy = yrs
            rr = linregress(yy, sv)
            row[f"{sn}_trend"], row[f"{sn}_p"] = rr.slope, rr.pvalue
        rows.append(row)
    with open(os.path.join(OUT, "check_acc_light.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nTRENDS per year (p in brackets), ACC mean:")
    print(f"{'variable':12s} {'mean':>10s} {'full year':>16s} " + " ".join(f"{s:>16s}" for s in SEAS))
    for r in rows:
        print(f"{r['variable']:12s} {r['mean']:10.4g} {r['year_trend']:9.3g} ({r['year_p']:.2f}) "
              + " ".join(f"{r[s + '_trend']:9.3g} ({r[s + '_p']:.2f})" for s in SEAS))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, s) in zip(axs.flat, S.items()):
        mon = s.reshape(-1, 12)
        ax.plot(yrs, mon.mean(1), "-o", c="k", label="annual")
        for sn, c in zip(["MAM", "JJA", "SON"], ["tab:orange", "tab:blue", "tab:green"]):
            ax.plot(yrs, mon[:, SEAS[sn]].mean(1), "--", c=c, label=sn)
        ax.set_title(key)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "check_acc_light.pdf"), bbox_inches="tight")
    print(f"\nwrote {OUT}/check_acc_light.pdf, .csv")


if __name__ == "__main__":
    main()
