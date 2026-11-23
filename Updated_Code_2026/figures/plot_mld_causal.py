"""Maps and seasonality of the drivers of the mixed-layer depth (PCMCI,
TARGET=MLD runs of bsose_pcmci.py), by SSH region.

    python /work/bb1596/b384036/b384036/causality/plot_mld_causal.py FOLDER [FOLDER ...]

FOLDER, e.g. /work/bb1596/b384036/b384036/causality/results_mld_PCMCI_c8_2013to2024,
holds pcmci_full.npz, pcmci_DJF.npz, ... and anomalies.npz.

TOP ROW     the three most important drivers, full year, lag LAG (0): the
            causal strength in every block where the link is significant
            (p <= ALPHA, 0.05), blank elsewhere; black lines: the region
            boundaries (N_SAF | SAF_SB | S_SB from acc_zones3_bsose.nc).
            "Most important" = the largest share of blocks with the link
            over all regions (DRIVERS=wind,surfTflx,oceQsw to choose).
BOTTOM ROW  one panel per region: per season (DJF ... SON) and driver,
            METRIC=mean (default): the mean causal strength over the blocks
            where the link is significant, as in the earlier MLD figure;
            METRIC=share: the share of blocks with the link (%).

No mask: B-SOSE's MLD and temperature agree with ORAS5 almost everywhere.

OUTPUT in FOLDER: mld_causal_lag0_mean.pdf (or _share), and
mld_causal_lag0_table.csv with both numbers per region, season and driver.
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
import summarize_links as sl  # noqa: E402

ZONES = os.environ.get("ZONES_FILE", CAUS + "/acc_zones3_bsose.nc")
ALPHA = float(os.environ.get("ALPHA", 0.05))
LAG = int(os.environ.get("LAG", 0))
METRIC = os.environ.get("METRIC", "mean")
DRIVERS = os.environ.get("DRIVERS", "")
VMAX = float(os.environ.get("VMAX", 0.6))
MIN_FITTED = getattr(sl, "MIN_FITTED", 20)   # fewer blocks fitted: no value
MIN_BLOCKS = getattr(sl, "MIN_BLOCKS", 10)   # fewer blocks with a link: no mean
COLORS = {"MLD": "#1f77b4", "salt": "#9467bd", "N": "#2ca02c",
          "wind": "#ff7f0e", "Theta": "#b15928", "surfTflx": "#d62728",
          "surfSflx": "#17becf", "oceQsw": "#4d4d4d"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]
REGIONS = ["N_SAF", "SAF_SB", "S_SB"]
LONG = {"wind": "wind stress amplitude", "surfTflx": "surface heat flux",
        "surfSflx": "surface freshwater (salt) flux", "oceQsw": "short-wave radiation",
        "N": "stratification (N$^2$)", "Theta": "temperature", "salt": "salinity",
        "MLD": "MLD"}


def load(folder, s):
    f = os.path.join(folder, f"pcmci_{s}.npz")
    if not os.path.exists(f):
        return None
    with np.load(f) as z:
        return dict(names=[str(n) for n in z["names"]], val=z["val"], p=z["p"])


def stats(m, k, mask):
    """share (%) of fitted blocks with the link, and mean strength there."""
    v, p = m["val"][k, LAG], m["p"][k, LAG]
    fitted = mask & np.isfinite(p)
    link = fitted & (p <= ALPHA) & np.isfinite(v)
    nf = int(fitted.sum())
    share = 100.0 * link.sum() / nf if nf >= MIN_FITTED else np.nan
    mean = float(np.mean(v[link])) if link.sum() >= MIN_BLOCKS else np.nan
    return share, mean, nf


def plot(folder):
    tag = os.path.basename(os.path.normpath(folder))
    maps = {s: load(folder, s) for s in ["full"] + SEASONS}
    if maps["full"] is None:
        print(f"{folder}: no pcmci_full.npz, skipped")
        return
    names = maps["full"]["names"]
    ny, nx = maps["full"]["val"].shape[2:]
    lat = sl.block_lat(folder, ny)
    with np.load(os.path.join(folder, "anomalies.npz")) as z:
        lon = np.asarray(z["lon"], float)
    lon = lon if lon.ndim == 1 else lon.mean(axis=0)
    sl.ZONE_FILE = ZONES
    masks = sl.region_masks(folder, lat, ny, nx)
    missing = [s for s in SEASONS if maps[s] is None]
    if missing:
        print(f"  {tag}: no pcmci file for {', '.join(missing)}")

    # table
    rows = []
    for reg in ["All"] + REGIONS:
        for s in ["full"] + SEASONS:
            if maps[s] is None:
                continue
            for k, f in enumerate(names):
                share, mean, nf = stats(maps[s], k, masks[reg])
                rows.append(dict(region=reg, season=s, driver=f, lag=LAG,
                                 blocks_fitted=nf, share_pct=round(share, 1),
                                 mean_strength=round(mean, 3)))
    with open(os.path.join(folder, f"mld_causal_lag{LAG}_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if DRIVERS:
        top = [d for d in DRIVERS.split(",") if d in names]
    else:
        share_all = {f: stats(maps["full"], k, masks["All"])[0] for k, f in enumerate(names)}
        top = sorted(names, key=lambda f: -np.nan_to_num(share_all[f]))[:3]
    print(f"{tag}: top drivers (lag {LAG}, full year, share of blocks): "
          + ", ".join(top))

    import cartopy.crs as ccrs
    try:
        from cartopy.io import shapereader
        shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
        coast = True
    except Exception:
        coast = False
    zone_id = np.zeros((ny, nx))
    for i, reg in enumerate(REGIONS, 1):
        zone_id[masks[reg]] = i
    lonw = np.concatenate([lon, [lon[0] + 360.0]])

    def wrapf(f):
        return np.concatenate([f, f[:, :1]], axis=1)

    fig = plt.figure(figsize=(22, 13.5), dpi=200)
    letters = "abcdef"
    for j, f in enumerate(top):
        k = names.index(f)
        v, p = maps["full"]["val"][k, LAG], maps["full"]["p"][k, LAG]
        field = np.where((p <= ALPHA) & np.isfinite(v), v, np.nan)
        ax = fig.add_axes([0.02 + 0.33 * j, 0.47, 0.28, 0.47],
                          projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
        if coast:
            ax.coastlines(resolution="110m", lw=0.6)
        ax.gridlines(lw=0.3, alpha=0.5)
        m = ax.pcolormesh(lonw, lat, wrapf(field), cmap=plt.cm.RdBu_r,
                          vmin=-VMAX, vmax=VMAX, transform=ccrs.PlateCarree(),
                          shading="nearest")
        ax.contour(lonw, lat, wrapf(zone_id), levels=[1.5, 2.5], colors="k",
                   linewidths=0.9, transform=ccrs.PlateCarree())
        cb = fig.colorbar(m, ax=ax, shrink=0.8, pad=0.03, extend="both")
        cb.set_label(f"{f}, full year, lag {LAG}", fontsize=11)
        sh = stats(maps["full"], k, masks["All"])[0]
        ax.set_title(f"Causal influence of {LONG.get(f, f)} anomalies\non the MLD "
                     f"(PCMCI; {sh:.0f}% of blocks)", fontsize=14)
        ax.text(-0.04, 1.02, letters[j], transform=ax.transAxes, fontsize=20,
                fontweight="bold")

    ylab = ("mean causal strength\n(blocks with the link)" if METRIC == "mean"
            else "% of blocks with\ndriver --> MLD")
    x = np.arange(len(SEASONS))
    allv = []
    axes = []
    for j, reg in enumerate(REGIONS):
        ax = fig.add_axes([0.06 + 0.32 * j, 0.07, 0.26, 0.31])
        axes.append(ax)
        for k, f in enumerate(names):
            ys = [stats(maps[s], k, masks[reg])[1 if METRIC == "mean" else 0]
                  if maps[s] is not None else np.nan for s in SEASONS]
            allv += [y for y in ys if np.isfinite(y)]
            ax.plot(x, ys, "-o", color=COLORS.get(f, None), lw=2 if f in top else 1.2,
                    ms=6 if f in top else 4, label=f)
        if METRIC == "share":
            ax.axhline(100 * ALPHA, color="k", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(SEASONS, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(f"{reg} ({int(masks[reg].sum())} blocks)", fontsize=14)
        ax.text(-0.12, 1.04, letters[3 + j], transform=ax.transAxes, fontsize=20,
                fontweight="bold")
        if j == 0:
            ax.set_ylabel(ylab, fontsize=12)
    if allv:
        lo, hi = min(allv), max(allv)
        pad = 0.08 * (hi - lo or 1)
        for ax in axes:
            ax.set_ylim(0 if METRIC == "share" else lo - pad, hi + pad)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(names), fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag}: drivers of the mixed-layer depth anomalies, lag {LAG}",
                 fontsize=16, y=0.995)
    out = os.path.join(folder, f"mld_causal_lag{LAG}_{METRIC}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}\n        {os.path.join(folder, f'mld_causal_lag{LAG}_table.csv')}")


if __name__ == "__main__":
    folders = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not folders:
        sys.exit(__doc__)
    for fo in folders:
        plot(fo)
