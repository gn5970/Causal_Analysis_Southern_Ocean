"""Where does the extra iron in B-SOSE come from? South America, sea ice, depth?
  1. LOGIN node:   STAGE=remote python /work/bb1596/b384036/b384036/causality/iron_sources.py
  2. COMPUTE node: STAGE=plot   python /work/bb1596/b384036/b384036/causality/iron_sources.py
Remote: Fe 0-50 m, Fe 500-1000 m (drift control), DFrINO3 at the first interface
below the MLD (nitrate supply). Local: SurfFe, SIArea, SIHeff, FElim/BioMass (0-25 m,
biomass-weighted), Chl and Fe supply from results_iron_supply/iron_supply.npz.
Output: /work/bb1596/b384036/b384036/causality/results_iron_sources/iron_sources.pdf/.png/.csv
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
STAGE = os.environ.get("STAGE", "remote")
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_iron_sources")
SUPPLY_NPZ = os.environ.get("SUPPLY_NPZ", CAUS + "/results_iron_supply/iron_supply.npz")
K, NT, YEAR0 = IS.K, IS.NT, IS.YEAR0
NY = NT // 12
UP, DEEP = (0.0, 50.0), (500.0, 1000.0)
SECTORS = {"Atlantic": (290.0, 20.0), "Indian": (20.0, 147.0), "Pacific": (147.0, 290.0)}
DRAKE = 292.0


def layer_mean(A, ks, drf, hfac):
    w = (drf[ks][:, None, None] * hfac[ks]).astype("float64")
    ok = np.isfinite(A) & (w[None] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(ok, A * w[None], 0).sum(1) / np.where(ok, w[None], 0).sum(1)).astype("float32")


def remote():
    os.makedirs(os.path.join(OUTDIR, "cache"), exist_ok=True)
    z, zl, drf, hfac, yc, xc, ra = IS.grid()
    iface_k = sorted({int(np.argmin(np.abs(zl - d))) for d in IS.IFACES})
    up_k = [k for k in range(z.size) if UP[0] <= z[k] <= UP[1]]
    deep_k = [k for k in range(z.size) if DEEP[0] < z[k] <= DEEP[1]]
    print(f"Fe 0-50 m: {len(up_k)} levels, 500-1000 m: {len(deep_k)} levels; "
          f"NO3 interfaces (m): {', '.join(f'{zl[k]:.0f}' for k in iface_k)}", flush=True)
    with xr.open_dataset(os.path.join(IS.BSOSE_DIR, IS.NAME.format("MLD")), decode_times=False) as ds:
        mld = np.abs(ds["BLGMLD"].isel(time=slice(0, NT)).values.astype("float32"))
    dI = np.array([zl[k] for k in iface_k])
    for y in range(YEAR0, YEAR0 + NY):
        c = os.path.join(OUTDIR, "cache", f"year_{y}.npz")
        if os.path.exists(c):
            continue
        t0 = (y - YEAR0) * 12
        fe_lev = sorted(set(up_k) | set(deep_k))
        fl = {k: i for i, k in enumerate(fe_lev)}
        FE = IS.read_levels("Fe", fe_lev, t0)
        fe_up = layer_mean(FE[:, [fl[k] for k in up_k]], up_k, drf, hfac)
        fe_deep = layer_mean(FE[:, [fl[k] for k in deep_k]], deep_k, drf, hfac)
        del FE
        fno3 = IS.read_levels("DFrINO3", iface_k, t0) / ra[None, None]
        for i, k in enumerate(iface_k):
            fno3[:, i, ~((hfac[k - 1] > 0) & (hfac[k] > 0))] = np.nan
        m = mld[t0:t0 + 12]
        idx = np.clip((dI[None, :, None, None] < m[:, None]).sum(1), 0, len(iface_k) - 1)
        f_ml = np.take_along_axis(fno3, idx[:, None], 1)[:, 0]
        np.savez(c + ".tmp.npz", Fe_up=IS.blocks(fe_up), Fe_deep=IS.blocks(fe_deep),
                 F_NO3=IS.blocks(f_ml))
        os.replace(c + ".tmp.npz", c)
        print(f"  {y}: done", flush=True)
    print("remote stage complete", flush=True)


def local_2d(word, var=None):
    p = os.path.join(IS.BSOSE_DIR, IS.NAME.format(word))
    with xr.open_dataset(p, decode_times=False) as ds:
        v = var or [x for x in ds.data_vars if ds[x].ndim >= 3 and x not in IS.SKIP][0]
        da = ds[v]
        out = []
        for t0 in range(0, NT, 12):
            a = da.isel(time=slice(t0, t0 + 12)).values.astype("float32")
            if a.ndim == 4:
                a = a[:, 0]
            out.append(IS.blocks(a))
        attrs = dict(da.attrs)
    print(f"{word}: {v} {attrs.get('units', '')} {attrs.get('long_name', '')}", flush=True)
    return np.concatenate(out, 0), attrs


def gamma_fe(word="FElim"):
    pf = os.path.join(IS.BSOSE_DIR, IS.NAME.format(word))
    pb = os.path.join(IS.BSOSE_DIR, IS.NAME.format("BioMass"))
    with xr.open_dataset(pf, decode_times=False) as df, xr.open_dataset(pb, decode_times=False) as db:
        vf = [x for x in df.data_vars if df[x].ndim == 4][0]
        vb = [x for x in db.data_vars if db[x].ndim == 4][0]
        keep = np.flatnonzero(np.abs(df["Z"].values) <= 25.0)
        g = db if "drF" in db else df
        w = (g["drF"].isel(Z=keep) * g["hFacC"].isel(Z=keep)).values.astype("float64")
        out = []
        for t0 in range(0, NT, 12):
            f = df[vf].isel(Z=keep, time=slice(t0, t0 + 12)).values.astype("float64")
            b = np.abs(db[vb].isel(Z=keep, time=slice(t0, t0 + 12)).values.astype("float64"))
            wb = b * w[None]
            ok = np.isfinite(f) & np.isfinite(wb) & (wb > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.append(IS.blocks((np.where(ok, f * wb, 0).sum(1)
                                      / np.where(ok, wb, 0).sum(1)).astype("float32")))
    return np.concatenate(out, 0)


def pct_trend(a):
    b, p = IS.trend(IS.deseason(a)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * b / np.nanmean(a, 0), p


def annual(a, m):
    s = np.nanmean(np.where(m[None], a, np.nan), (1, 2))
    return s[:NY * 12].reshape(NY, 12).mean(1)


def sector_masks(lon, ny):
    L = np.broadcast_to(lon[None, :], (ny, lon.size))
    return {n: (((L >= a) | (L < b)) if a > b else ((L >= a) & (L < b)))
            for n, (a, b) in SECTORS.items()}


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from corr_bsose_sat import setup_ax, wrap

    parts = [dict(np.load(os.path.join(OUTDIR, "cache", f"year_{y}.npz")))
             for y in range(YEAR0, YEAR0 + NY)]
    R = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    S = dict(np.load(SUPPLY_NPZ))
    lat, lon = S["lat"], S["lon"]
    ny, nx = S["F"].shape[1:]
    surf, sattrs = local_2d("SurfFe")
    ice, _ = local_2d("SIArea")
    heff, _ = local_2d("SIHeff")
    gfe = gamma_fe()
    glt = gamma_fe("Llim")
    zn = IS.zones(np.repeat(lat, K), ny, nx)
    sec = sector_masks(lon, ny)

    an, _ = IS.deseason(surf)
    share_ia = float(np.nanvar(an) / np.nanvar(surf))
    same = np.nanmax(np.abs(np.diff(surf[:NY * 12].reshape(NY, 12, ny, nx), axis=0)))
    print(f"SurfFe: interannual share of variance {share_ia:.3f}; max change between years "
          f"{same:.3g} (0 = the same climatology every year)", flush=True)

    tS, _ = pct_trend(surf)
    tU, _ = pct_trend(R["Fe_up"])
    tD, _ = pct_trend(R["Fe_deep"])
    tI, _ = IS.trend(IS.deseason(ice)[0])
    tC, _ = IS.std_trend(S["Chl"])
    sU, _ = IS.std_trend(R["Fe_up"])
    sG, _ = IS.std_trend(gfe)
    sL, _ = IS.std_trend(glt)
    sS, _ = IS.std_trend(surf)
    tFfe, _ = IS.trend(IS.deseason(S["F"])[0])
    tFn, _ = IS.trend(IS.deseason(R["F_NO3"])[0])

    rows = []
    for zname, zm_ in zn.items():
        for sname, sm in [("all", np.ones((ny, nx), bool))] + list(sec.items()):
            m = zm_ & sm

            def zmean(a):
                return float(np.nanmean(np.where(m, a, np.nan)))
            rows.append(dict(zone=zname, sector=sname, blocks=int(m.sum()),
                             SurfFe_pct_yr=zmean(tS), Fe0_50_pct_yr=zmean(tU),
                             Fe500_1000_pct_yr=zmean(tD), Chl_sd_yr=zmean(tC),
                             gammaFe_sd_yr=zmean(sG), gammaLight_sd_yr=zmean(sL), SIArea_per_yr=zmean(tI),
                             Fe_supply_pct_yr=100 * zmean(tFfe) / zmean(np.nanmean(S["F"], 0)),
                             NO3_supply_pct_yr=100 * zmean(tFn) / zmean(np.nanmean(R["F_NO3"], 0))))
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "iron_sources.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(19, 17))

    def pmap(i, letter, title, f, lim, lab):
        ax = fig.add_subplot(3, 3, i, projection=ccrs.SouthPolarStereo())
        setup_ax(ax, letter, title)
        lc, fc = wrap(lon, f)
        mm = ax.pcolormesh(lc, lat, fc, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest", transform=pc)
        fig.colorbar(mm, ax=ax, shrink=0.7).set_label(lab)

    pmap(1, "a", "Surface Fe input (SurfFe)", tS, 3, "trend (% yr$^{-1}$)")
    pmap(2, "b", "Fe, 0-50 m", tU, 3, "trend (% yr$^{-1}$)")
    pmap(3, "c", "Fe, 500-1000 m", tD, 3, "trend (% yr$^{-1}$)")
    pmap(4, "d", "Sea-ice concentration", tI, 0.02, "trend (yr$^{-1}$)")
    pmap(5, "e", "Chl-a, 0-25 m", tC, 0.16, "trend ($\\sigma$ yr$^{-1}$)")

    ax = fig.add_subplot(3, 3, 6)
    zr = [r for r in rows if r["sector"] == "all"]
    x = np.arange(len(zr))
    ax.bar(x - 0.2, [r["Fe_supply_pct_yr"] for r in zr], 0.4, color="tab:brown", label="Fe supply")
    ax.bar(x + 0.2, [r["NO3_supply_pct_yr"] for r in zr], 0.4, color="tab:blue", label="NO$_3$ supply")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r["zone"] for r in zr])
    ax.set_ylabel("trend of the diffusive supply (% yr$^{-1}$)")
    ax.legend(frameon=False)
    ax.text(-0.15, 1.03, "f", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = fig.add_subplot(3, 3, 7)
    acc = zn["SAF_SB"]
    dl = (lon - DRAKE) % 360
    bins = np.arange(0, 361, 15)
    xc_ = 0.5 * (bins[1:] + bins[:-1])
    for zname, ls in [("SAF_SB", "-"), ("N_SAF", "--")]:
        zmask = zn[zname]
        for f, lab, c in [(sU, "Fe 0-50 m", "tab:brown"), (sG, "$\\gamma_{Fe}$", "tab:red"),
                          (tC, "Chl-a", "tab:green")]:
            prof = [np.nanmean(np.where(zmask & ((dl >= a) & (dl < b))[None, :], f, np.nan))
                    for a, b in zip(bins[:-1], bins[1:])]
            ax.plot(xc_, prof, ls, marker="o", ms=3, c=c, label=f"{lab}, {zname}")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("degrees east of Drake Passage (solid: ACC, dashed: north of SAF)")
    ax.set_ylabel("trend ($\\sigma$ yr$^{-1}$)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(-0.15, 1.03, "g", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = fig.add_subplot(3, 3, 8)
    yrs = YEAR0 + np.arange(NY)
    ssb = zn["S_SB"]

    def z_(a):
        return (a - a.mean()) / a.std() if a.std() > 0 else a * 0
    ax.plot(yrs, z_(annual(ice, ssb)), "-o", c="tab:cyan", label="sea-ice concentration")
    ax.plot(yrs, z_(annual(heff, ssb)), "--", c="tab:cyan", label="sea-ice thickness")
    ax.plot(yrs, z_(annual(R["Fe_up"], ssb)), "-o", c="tab:brown", label="Fe 0-50 m")
    ax.plot(yrs, z_(annual(S["Chl"], ssb)), "-o", c="tab:green", label="Chl-a")
    ax.plot(yrs, z_(annual(glt, ssb)), "-s", c="gold", label="$\\gamma_{light}$")
    ax.plot(yrs, z_(annual(gfe, ssb)), "-s", c="tab:red", label="$\\gamma_{Fe}$")
    ax.set_ylabel("yearly mean, standardized (south of SB)")
    ax.legend(frameon=False, fontsize=9)
    ax.text(-0.15, 1.03, "h", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = fig.add_subplot(3, 3, 9)
    for (n, sm), c in zip(sec.items(), ["tab:blue", "tab:orange", "tab:purple"]):
        a = annual(R["Fe_up"], acc & sm)
        ax.plot(yrs, 100 * (a / a[0] - 1), "-o", c=c, label=n)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_ylabel("Fe 0-50 m in the ACC (% change from 2013)")
    ax.legend(frameon=False)
    ax.text(-0.15, 1.03, "i", transform=ax.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(OUTDIR, "iron_sources")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=170)
    print(f"wrote {out}.pdf/.png/.csv  (SurfFe units: {sattrs.get('units', '?')})")


if __name__ == "__main__":
    remote() if STAGE == "remote" else plot()
