"""Correlation map of B-SOSE vs satellite Chl-a with the mask used for PCMCI.
    COARSEN=4 python /work/bb1596/b384036/b384036/causality/plot_mask_paper.py
Reads results_corr_bsose_sat_c{k}/corr_bsose_sat_c{k}_cal8.npz (sat_coverage_mask.py).
  coloured   r of the kept blocks (r > 0, p < 0.01)
  grey       excluded blocks (r <= 0 or p >= 0.01)
  hatched    kept without satellite test (< 8 calendar months south of 50S)
  black line edge of the kept area
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
from corr_bsose_sat import setup_ax, wrap  # noqa: E402

K = int(os.environ.get("COARSEN", 4))
MIN_CAL = int(os.environ.get("MIN_CAL", 8))
DIR = os.environ.get("CORR_DIR", f"{CAUS}/results_corr_bsose_sat_c{K}")
NPZ = os.path.join(DIR, f"corr_bsose_sat_c{K}_cal{MIN_CAL}.npz")
OUT = os.path.join(DIR, f"mask_corr_c{K}")


def main():
    import cartopy.crs as ccrs
    if not os.path.exists(NPZ):
        sys.exit(f"missing {NPZ}: run sat_coverage_mask.py with COARSEN={K} first")
    z = np.load(NPZ)
    r, sig, lat, lon = z["r"], z["sig"].astype(bool), z["lat"], z["lon"]
    untested = z["untested_cal"].astype(bool)
    kept = np.isfinite(r) & sig & (r > 0)
    excl = np.isfinite(r) & ~kept
    n = kept.sum() + excl.sum() + untested.sum()
    print(f"kept {100 * kept.sum() / n:.0f}%, excluded {100 * excl.sum() / n:.0f}%, "
          f"kept without test {100 * untested.sum() / n:.0f}% of {n} blocks")

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    setup_ax(ax, "", "")
    lc, rk = wrap(lon, np.where(kept, r, np.nan))
    m = ax.pcolormesh(lc, lat, rk, cmap="RdBu_r", vmin=-1, vmax=1,
                      shading="nearest", transform=pc)
    _, ex = wrap(lon, np.where(excl, 1.0, np.nan))
    ax.pcolormesh(lc, lat, ex, cmap="Greys", vmin=0, vmax=2.5,
                  shading="nearest", transform=pc)
    _, un = wrap(lon, untested.astype(float))
    ax.contourf(lc, lat, un, levels=[0.5, 1.5], colors="none", hatches=["////"],
                transform=pc)
    _, kk = wrap(lon, (kept | untested).astype(float))
    if K > 1:
        ax.contour(lc, lat, kk, levels=[0.5], colors="k", linewidths=0.7, transform=pc)
    ax.set_aspect("equal")
    cb = fig.colorbar(m, ax=ax, shrink=0.7, pad=0.06)
    cb.set_label("Correlation B-SOSE vs satellite Chl-a (r)", fontsize=12)
    ax.legend(handles=[Patch(facecolor=plt.cm.Greys(1 / 2.5),
                             label=f"excluded ({100 * excl.sum() / n:.0f}%)"),
                       Patch(facecolor="white", edgecolor="k", hatch="////",
                             label=f"kept, no satellite test ({100 * untested.sum() / n:.0f}%)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=250)
    print(f"wrote {OUT}.pdf, .png")


if __name__ == "__main__":
    main()
