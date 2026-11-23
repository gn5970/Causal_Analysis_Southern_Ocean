"""Figure 2: seasonally resolved trends (latitude x calendar month) in B-SOSE,
2013-2024:
  a  NPP                        column integral over the top NPP_DEPTH m (150)
  b  phytoplankton growth rate  mu = int NPP dV / int biomass dV
  c  light limitation           gamma_light = int Llim biomass dV / int biomass dV
  d  iron limitation            gamma_Fe    = int FElim biomass dV / int biomass dV

    python /work/bb1596/b384036/b384036/causality/fig2_trends.py

INPUT in /work/bb1596/b384036/b384036/causality/bsose/ (download_bsose.py):
NPP_, BioMass_, Llim_, FElim_bsoseI156_2013to2024_monthly.nc

THE INTEGRALS. dV = dz h dA: layer thickness (drF) x wet fraction (hFacC) x
cell area (dA proportional to cos(latitude) on the regular B-SOSE grid).
They run over the levels within the top W_DEPTH m (60, all that is
downloaded of the biomass and the limitation factors) and over all ocean
cells of a latitude band (LAT_STEP degrees, 2.5) -- so b, c and d are the
biomass-weighted values of the band, and the weighting is by biomass only.
a is the area-weighted band mean of the NPP column integral.

TREND, per band and calendar month: one value per year, least-squares slope
over 2013-2024, per decade; stippled where p < 0.05 (t test). Fitting each
calendar month separately is the deseasonalized trend. NORM=std divides the
slope by the standard deviation of that month's values, so the four panels
share one scale (standard deviations per decade).

UNITS. NPP mol m-3 s-1 -> mg C m-2 d-1; mu (NPP / biomass) mol ... s-1 /
mol ... -> d-1, valid when NPP and biomass are in the same element (C);
the units the files carry are printed, check them.

OUTPUT in /work/bb1596/b384036/b384036/causality/results_fig2/
  fig2_trends.pdf, fig2_trends.npz (slopes, p, band x month x year values)
"""
import os
import sys
import glob
import numpy as np
import xarray as xr
from scipy.stats import t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUS = "/work/bb1596/b384036/b384036/causality"
BSOSE_DIR = os.environ.get("BSOSE_DIR", CAUS + "/bsose")
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_fig2")
W_DEPTH = float(os.environ.get("W_DEPTH", 60.0))
NPP_DEPTH = float(os.environ.get("NPP_DEPTH", 150.0))
LAT_STEP = float(os.environ.get("LAT_STEP", 2.5))
NORM = os.environ.get("NORM", "none")
NT = int(os.environ.get("NT", 144))
YEAR0 = 2013
WORDS = {"npp": os.environ.get("NPP_WORD", "NPP"),
         "bio": os.environ.get("BIO_WORD", "BioMass"),
         "llim": os.environ.get("LLIM_WORD", "Llim"),
         "felim": os.environ.get("FELIM_WORD", "FElim")}
SKIP = {"drF", "hFacC", "rA", "Depth", "drC", "dxG", "dyG", "rAw", "rAs",
        "rAz", "dxC", "dyC", "maskC", "maskW", "maskS", "PHrefC"}


def find(word):
    for f in sorted(glob.glob(os.path.join(BSOSE_DIR, "*.nc"))):
        b = os.path.basename(f)
        if word.lower() in b.lower().replace("-", "_").replace(".", "_").split("_") \
                and "snap" not in b.lower():
            return f
    sys.exit(f"no file with '{word}' in its name in {BSOSE_DIR}")


def zdim(da):
    return next((d for d in da.dims if d.upper().startswith("Z")), None)


class Field:
    """One B-SOSE 3D field, its levels within dmax and their dz x h."""

    def __init__(self, word, dmax, geom_src=None):
        self.path = find(word)
        self.ds = xr.open_dataset(self.path, decode_times=False)
        c = [v for v in self.ds.data_vars if v not in SKIP and self.ds[v].ndim >= 3]
        if not c:
            sys.exit(f"{self.path}: no data variable")
        self.var = max(c, key=lambda v: (self.ds[v].ndim, self.ds[v].size))
        da = self.ds[self.var]
        self.zd = zdim(da)
        if self.zd is None:
            sys.exit(f"{self.path}: {self.var} has no depth axis")
        self.z = np.abs(self.ds[self.zd].values.astype("float64"))
        self.keep = np.flatnonzero(self.z <= dmax)
        g = self.ds if ("drF" in self.ds and "hFacC" in self.ds) else geom_src
        if g is None:
            sys.exit(f"{self.path}: no drF/hFacC for the layer weights")
        zg = zdim(g["hFacC"])
        zgv = np.abs(g[zg].values.astype("float64"))
        gi = [int(np.argmin(np.abs(zgv - self.z[k]))) for k in self.keep]
        if np.max(np.abs(zgv[gi] - self.z[self.keep])) > 0.5:
            sys.exit(f"{self.path}: levels do not match the geometry's")
        self.dzh = (g["drF"].values.astype("float32")[gi][:, None, None]
                    * g["hFacC"].isel({zg: gi}).values.astype("float32"))
        self.units = str(da.attrs.get("units", "?"))
        self.yc = self.ds["YC"].values.astype("float64")
        print(f"{word}: {os.path.basename(self.path)}, {self.var} [{self.units}], "
              f"{len(self.keep)} levels to {self.z[self.keep].max():.1f} m", flush=True)

    def year(self, t0, nlev=None):
        k = self.keep if nlev is None else self.keep[:nlev]
        v = self.ds[self.var].isel(time=slice(t0, t0 + 12), **{self.zd: k}).values
        v = v.astype("float32")
        v[~np.isfinite(v)] = 0.0
        return v                                           # (12, k, Y, X)


def trend(y, yrs):
    ok = np.isfinite(y)
    n = ok.sum(axis=-1)
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


def per_s(u):
    u = u.lower().replace(" ", "")
    return any(t in u for t in ("/s", "s-1", "s^-1", "s**-1"))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    bio = Field(WORDS["bio"], W_DEPTH)
    npp = Field(WORDS["npp"], NPP_DEPTH, bio.ds)
    llim = Field(WORDS["llim"], W_DEPTH, bio.ds)
    felim = Field(WORDS["felim"], W_DEPTH, bio.ds)
    nw = len(bio.keep)
    for f in (npp, llim, felim):
        if len(f.keep) < nw or np.max(np.abs(f.z[f.keep[:nw]] - bio.z[bio.keep])) > 0.5:
            sys.exit(f"{f.path}: its top levels differ from the biomass file's")
    nw_npp = nw                                   # NPP levels within W_DEPTH, for mu

    yc = bio.yc
    edges = np.arange(-90.0, -30.0 + LAT_STEP / 2, LAT_STEP)
    centres = 0.5 * (edges[1:] + edges[:-1])
    band = np.digitize(yc, edges) - 1
    onehot = np.zeros((centres.size, yc.size))
    ok_rows = (band >= 0) & (band < centres.size)
    onehot[band[ok_rows], np.flatnonzero(ok_rows)] = 1.0
    area = np.cos(np.deg2rad(yc))[:, None]                  # (Y, 1), relative dA

    c_npp = 12.011e3 * 86400.0 if ("mol" in npp.units.lower() and per_s(npp.units)) else 1.0
    c_mu = 86400.0 if ("mol" in bio.units.lower() and per_s(npp.units)) else 1.0
    u_npp = "mg C m$^{-2}$ d$^{-1}$" if c_npp != 1.0 else f"{npp.units} m"
    u_mu = "d$^{-1}$" if c_mu != 1.0 else f"({npp.units}) / ({bio.units})"
    print(f"NPP x {c_npp:g} -> {u_npp}; mu x {c_mu:g} -> {u_mu}", flush=True)

    nyear = NT // 12
    yrs = YEAR0 + np.arange(nyear, dtype=float)
    vals = {k: np.full((centres.size, 12, nyear), np.nan) for k in ("npp", "mu", "llim", "felim")}
    wet_col = (npp.dzh.sum(axis=0) > 0)                     # ocean columns
    A = np.where(wet_col, area, 0.0)                          # (Y, X)

    def band_sum(f2d):                                        # (12, Y, X) -> (nb, 12)
        return onehot @ f2d.sum(axis=2).T

    for iy in range(nyear):
        t0 = 12 * iy
        B = bio.year(t0)                                      # (12, k, Y, X)
        Bw = (B * bio.dzh[None]).sum(axis=1)                  # int B dz   (12, Y, X)
        den = band_sum(Bw * A)
        N = npp.year(t0)
        col = (N * npp.dzh[None]).sum(axis=1)                 # int NPP dz to NPP_DEPTH
        vals["npp"][:, :, iy] = c_npp * band_sum(col * A) / np.maximum(onehot @ A.sum(axis=1), 1e-30)[:, None]
        colw = (N[:, :nw_npp] * npp.dzh[None, :nw_npp]).sum(axis=1)
        del N, col
        with np.errstate(invalid="ignore", divide="ignore"):
            vals["mu"][:, :, iy] = np.where(den > 0, c_mu * band_sum(colw * A) / den, np.nan)
        for key, fld in (("llim", llim), ("felim", felim)):
            L = fld.year(t0, nw)
            num = band_sum((L * B * bio.dzh[None]).sum(axis=1) * A)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals[key][:, :, iy] = np.where(den > 0, 1.0 - num / den, np.nan)
            del L
        del B
        print(f"\r  year {YEAR0 + iy}", end="", flush=True)
    print()
    empty = onehot @ A.sum(axis=1) <= 0
    for k in vals:
        vals[k][empty] = np.nan

    res = {k: trend(v, yrs) for k, v in vals.items()}
    np.savez(os.path.join(OUTDIR, "fig2_trends.npz"), lat=centres, edges=edges, years=yrs,
             w_depth=W_DEPTH, npp_depth=NPP_DEPTH, norm=NORM,
             **{f"{k}_vals": vals[k] for k in vals},
             **{f"{k}_slope": res[k][0] for k in res}, **{f"{k}_p": res[k][1] for k in res})

    panels = [("npp", "NPP", f"NPP trend\n({u_npp} per year)"),
              ("mu", r"$\mu$", f"Growth rate trend\n({u_mu} per year)"),
              ("llim", r"light limitation, $1-\gamma_{light}$", "Light limitation trend\n(per year)"),
              ("felim", r"iron limitation, $1-\gamma_{Fe}$", "Iron limitation trend\n(per year)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    months = np.arange(12)
    for ax, (k, name, clab), letter in zip(axes.flat, panels, "abcd"):
        s, p, sd = res[k]
        if NORM == "std":
            with np.errstate(invalid="ignore", divide="ignore"):
                s = s / sd
            clab = clab.split("\n")[0] + "\n(SD per year)"
        lim = float(os.environ.get("CLIM", 0.3))     # same colour scale in all panels
        m = ax.pcolormesh(np.arange(13) - 0.5, edges, s, cmap=plt.cm.RdBu_r,
                          vmin=-lim, vmax=lim, shading="flat")
        yy, xx = np.nonzero(np.isfinite(p) & (p < 0.05))
        ax.scatter(months[xx], centres[yy], s=3, c="k", lw=0)
        ax.set_xticks(months)
        ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_xlabel("Month")
        ax.set_ylabel("Latitude (\u00b0)")
        ax.set_ylim(-90, -30)
        cb = fig.colorbar(m, ax=ax, extend="both", ticks=np.linspace(-lim, lim, 7))
        cb.ax.tick_params(labelsize=14)
        cb.set_label(clab, fontsize=13)
        ax.text(0.5, 0.06, name, transform=ax.transAxes, ha="center", fontsize=15,
                bbox=dict(facecolor="white", edgecolor="0.6"))
        ax.text(-0.15, 1.04, letter, transform=ax.transAxes, fontsize=18, fontweight="bold")
    fig.suptitle(f"B-SOSE {YEAR0}-{YEAR0 + nyear - 1}: trend per calendar month, "
                 f"{LAT_STEP:g}\u00b0 bands; NPP over 0-{NPP_DEPTH:g} m; mu and "
                 f"limitation biomass-weighted over 0-{W_DEPTH:g} m; dots: p < 0.05",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig2_trends.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}\n      {os.path.join(OUTDIR, 'fig2_trends.npz')}")


if __name__ == "__main__":
    main()
