"""Is the iron gained north of the SAF exported from the ACC (Ekman transport)?
    python /work/bb1596/b384036/b384036/causality/iron_lateral.py     (compute node)
TEST 1: trends of Fe 0-50 m and Chl-a vs latitude distance from the SAF.
TEST 2: northward Ekman Fe flux across the SAF, V_ek = -tau_x/(rho0 f), times the
0-50 m Fe of the ACC block just south of the SAF; trend split into wind and iron
parts; compared with the 0-50 m Fe inventory growth north of the SAF and the
vertical diffusive supply. Not included: geostrophic/eddy transport.
Output: /work/bb1596/b384036/b384036/causality/results_iron_lateral/
"""
import os
import sys
import csv
import warnings
import numpy as np
import xarray as xr

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
import iron_supply as IS  # noqa: E402

warnings.filterwarnings("ignore")
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_iron_lateral")
SRC_CACHE = os.environ.get("SRC_CACHE", CAUS + "/results_iron_sources/cache")
SUPPLY_NPZ = os.environ.get("SUPPLY_NPZ", CAUS + "/results_iron_supply/iron_supply.npz")
K, NT, YEAR0 = IS.K, IS.NT, IS.YEAR0
NY = NT // 12
RHO0, OMEGA, R = 1027.5, 7.2921e-5, 6.371e6
H_UP = 50.0
SPY = 365.25 * 86400
SECTORS = {"Atlantic": (290.0, 20.0), "Indian": (20.0, 147.0), "Pacific": (147.0, 290.0)}
DRAKE = 292.0


def taux_blocks():
    p = os.path.join(IS.BSOSE_DIR, IS.NAME.format("oceTAUX"))
    with xr.open_dataset(p, decode_times=False) as ds:
        da = ds["oceTAUX"]
        xd = [d for d in da.dims if d.upper().startswith("X")][0]
        yd = [d for d in da.dims if d.upper().startswith("Y")][0]
        da = da.transpose("time", yd, xd)
        out = []
        for t0 in range(0, NT, 12):
            a = da.isel(time=slice(t0, t0 + 12)).values.astype("float32")
            a = 0.5 * (a + np.roll(a, -1, axis=2))
            out.append(IS.blocks(a))
    return np.concatenate(out, 0)


def sector_of(lon):
    out = np.empty(lon.shape, object)
    for n, (a, b) in SECTORS.items():
        m = ((lon >= a) | (lon < b)) if a > b else ((lon >= a) & (lon < b))
        out[m] = n
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fe = np.concatenate([np.load(os.path.join(SRC_CACHE, f"year_{y}.npz"))["Fe_up"]
                         for y in range(YEAR0, YEAR0 + NY)], 0).astype("float64")
    S = dict(np.load(SUPPLY_NPZ))
    lat, lon = S["lat"], S["lon"]
    ny, nx = fe.shape[1:]
    chl, fv = S["Chl"], S["F"].astype("float64")
    zn = IS.zones(np.repeat(lat, K), ny, nx)
    nsaf, acc = zn["N_SAF"], zn["SAF_SB"]
    tau = taux_blocks().astype("float64")

    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())
    dx = R * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)
    area = (dx * R * np.deg2rad(dlat))[:, None] * np.ones((1, nx))
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))

    saf_row = np.full(nx, -1)
    for j in range(nx):
        rows = np.flatnonzero(nsaf[1:, j] & acc[:-1, j]) + 1
        if rows.size:
            saf_row[j] = rows.min()
    ok_col = saf_row > 0
    lat_saf = np.where(ok_col, 0.5 * (lat[np.clip(saf_row, 0, ny - 1)]
                                      + lat[np.clip(saf_row - 1, 0, ny - 1)]), np.nan)
    print(f"SAF found at {int(ok_col.sum())} of {nx} longitudes; "
          f"latitude {np.nanmin(lat_saf):.1f} to {np.nanmax(lat_saf):.1f}, mean {np.nanmean(lat_saf):.1f}")

    dist = lat[:, None] - lat_saf[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        tFe = 100 * IS.trend(IS.deseason(fe)[0])[0] / np.nanmean(fe, 0)
    sFe, _ = IS.std_trend(fe)
    sC, _ = IS.std_trend(chl)
    bins = np.arange(-8, 21, 1.0)
    bc = 0.5 * (bins[1:] + bins[:-1])
    sec = sector_of(np.broadcast_to(lon[None, :], (ny, nx)))
    inzone = (nsaf | acc) & np.isfinite(dist)

    def prof(a, extra=None):
        m0 = inzone if extra is None else inzone & extra
        return np.array([np.nanmean(a[m0 & (dist >= lo) & (dist < hi)])
                         if (m0 & (dist >= lo) & (dist < hi)).sum() >= 20 else np.nan
                         for lo, hi in zip(bins[:-1], bins[1:])])

    jj = np.flatnonzero(ok_col)
    ia = saf_row[jj] - 1
    vek = -tau[:, ia, jj] / (RHO0 * f[ia][None, :])
    fe_acc = fe[:, ia, jj]
    w = dx[ia][None, :]
    flux_col = vek * fe_acc * w
    Ftot = np.nansum(flux_col, 1)
    vclim = IS.deseason(vek[:, :, None])[1][:, :, 0]
    feclim = IS.deseason(fe_acc[:, :, None])[1][:, :, 0]
    F_wind = np.nansum(vek * feclim * w, 1)
    F_iron = np.nansum(vclim * fe_acc * w, 1)
    secj = sector_of(lon[jj])
    Fsec = {n: np.nansum(flux_col[:, secj == n], 1) for n in SECTORS}

    def tr(series):
        return float(IS.trend(IS.deseason(series[:, None, None])[0])[0][0, 0])

    inv = np.nansum(np.where(nsaf[None], fe * H_UP * area[None], 0), (1, 2))
    dIdt = tr(inv) / SPY
    vert = np.nansum(np.where(nsaf[None], fv * area[None], 0), (1, 2))
    rows = [dict(quantity="Ekman Fe flux across SAF, mean (mol/s)", value=float(Ftot.mean())),
            dict(quantity="Ekman Fe flux, trend (mol/s per yr)", value=tr(Ftot)),
            dict(quantity="  wind part (mol/s per yr)", value=tr(F_wind)),
            dict(quantity="  iron part (mol/s per yr)", value=tr(F_iron)),
            dict(quantity="vertical diffusive Fe supply into N_SAF, mean (mol/s)", value=float(np.nanmean(vert))),
            dict(quantity="vertical diffusive Fe supply, trend (mol/s per yr)", value=tr(vert)),
            dict(quantity="0-50 m Fe inventory N_SAF, mean (mol)", value=float(inv.mean())),
            dict(quantity="0-50 m Fe inventory N_SAF, growth rate (mol/s)", value=dIdt),
            dict(quantity="Ekman Fe flux increase over the record (mol/s)", value=tr(Ftot) * (NY - 1)),
            dict(quantity="vertical supply increase over the record (mol/s)", value=tr(vert) * (NY - 1))]
    rows += [dict(quantity=f"Ekman Fe flux trend, {n} (% of its mean per yr)",
                  value=100 * tr(v) / float(v.mean()) if v.mean() else np.nan) for n, v in Fsec.items()]
    rows += [dict(quantity=f"Fe 0-50 m trend {lo:+.0f} to {hi:+.0f} deg from SAF (%/yr)", value=float(v))
             for lo, hi, v in zip(bins[:-1], bins[1:], prof(tFe)) if np.isfinite(v)]
    with open(os.path.join(OUTDIR, "iron_lateral.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["quantity", "value"])
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"{r['quantity']:58s} {r['value']: .3g}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0, 0]
    ax.plot(bc, prof(sFe), "-o", c="tab:brown", lw=2, label="Fe 0-50 m, all")
    ax.plot(bc, prof(sC), "-o", c="tab:green", lw=2, label="Chl-a, all")
    for (n, c) in zip(SECTORS, ["tab:blue", "tab:orange", "tab:purple"]):
        ax.plot(bc, prof(sFe, sec == n), "--", c=c, lw=1, label=f"Fe, {n}")
    ax.axvline(0, c="k", lw=1)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("degrees latitude from the SAF (negative: ACC, positive: north of the SAF)")
    ax.set_ylabel("trend ($\\sigma$ yr$^{-1}$)")
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.1, 1.03, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axs[0, 1]
    ann = lambda s: s[:NY * 12].reshape(NY, 12).mean(1)  # noqa: E731
    yrs = YEAR0 + np.arange(NY)
    ax.plot(yrs, ann(Ftot), "-o", c="k", lw=2, label="total")
    ax.plot(yrs, ann(F_wind), "--", c="tab:cyan", label="wind only (mean Fe)")
    ax.plot(yrs, ann(F_iron), "--", c="tab:brown", label="iron only (mean wind)")
    for (n, v), c in zip(Fsec.items(), ["tab:blue", "tab:orange", "tab:purple"]):
        ax.plot(yrs, ann(v), ":", c=c, label=n)
    ax.set_ylabel("northward Ekman Fe flux across the SAF (mol s$^{-1}$)")
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.1, 1.03, "b", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axs[1, 0]
    labs = ["Ekman flux\nincrease\n(wind part)", "Ekman flux\nincrease\n(iron part)",
            "vertical supply\nincrease", "0-50 m inventory\ngrowth rate"]
    vals = [tr(F_wind) * (NY - 1), tr(F_iron) * (NY - 1), tr(vert) * (NY - 1), dIdt]
    ax.bar(range(4), vals, color=["tab:cyan", "tab:blue", "tab:brown", "k"])
    ax.set_title(f"means: Ekman flux {Ftot.mean():.3g}, vertical supply {np.nanmean(vert):.3g} mol s$^{{-1}}$",
                 fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=9)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_ylabel("change 2013-2024 / growth rate (mol s$^{-1}$)")
    ax.text(-0.1, 1.03, "c", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axs[1, 1]
    dl = (lon[jj] - DRAKE) % 360
    lb = np.arange(0, 361, 15)
    lc = 0.5 * (lb[1:] + lb[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        trc = np.array([100 * tr(np.nansum(flux_col[:, (dl >= a) & (dl < b)], 1))
                        / np.nanmean(np.nansum(flux_col[:, (dl >= a) & (dl < b)], 1))
                        if ((dl >= a) & (dl < b)).any() else np.nan for a, b in zip(lb[:-1], lb[1:])])
    ax.plot(lc, trc, "-o", c="k", label="Ekman Fe flux across the SAF (% / yr)")
    dlall = (lon - DRAKE) % 360
    fe_n = [np.nanmean(np.where(nsaf & ((dlall >= a) & (dlall < b))[None, :], tFe, np.nan))
            for a, b in zip(lb[:-1], lb[1:])]
    ax.plot(lc, fe_n, "--o", c="tab:brown", label="Fe 0-50 m north of the SAF (% / yr)")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("degrees east of Drake Passage")
    ax.set_ylabel("trend (% yr$^{-1}$)")
    ax.legend(frameon=False, fontsize=8)
    ax.text(-0.1, 1.03, "d", transform=ax.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUTDIR, "iron_lateral")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=170)
    print(f"wrote {out}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
