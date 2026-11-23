"""Two-source mask for the B-SOSE PCMCI summaries.

    python /work/bb1596/b384036/b384036/causality/sat_coverage_mask.py

The satellite-vs-B-SOSE correlation (corr_bsose_sat.py) is a fair test only
where the satellite sees most of the seasonal cycle. In the seasonal ice
zone it sees roughly November-March only, so there the correlation tests
the timing within summer, not the seasonal cycle, and the verdict is not
used: those blocks are made "untested" (r = NaN). summarize_links.py
--mask-keep-untested then keeps them (validated instead by BGC-Argo,
whose 60-80S seasonal cycles B-SOSE matches, r = 0.80-0.93).

A block counts as covered in a calendar month when the satellite has a
value there in at least MIN_YEARS of the years. Blocks covered in fewer
than MIN_CAL calendar months AND south of LAT_MAX become untested;
everywhere else the satellite verdict stands (e.g. the SE Pacific gyre
stays rejected).

INPUT   results_corr_bsose_sat_c8/corr_bsose_sat_c8.npz  (corr_bsose_sat.py)
OUTPUT  results_corr_bsose_sat_c8/corr_bsose_sat_c8_cal8.npz  same keys plus
        ncal (calendar months covered) and untested_cal (the blocks
        reclassified); mask_blocks_c8_cal8.pdf, the map of the mask
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CAUS = "/work/bb1596/b384036/b384036/causality"
sys.path.insert(0, CAUS)
from corr_bsose_sat import satellite_on_grid, blocks, BSOSE_DIR, BSOSE_NAME  # noqa: E402

COARSEN = int(os.environ.get("COARSEN", 8))
NT = int(os.environ.get("NT", 144))
MIN_CAL = int(os.environ.get("MIN_CAL", 8))
MIN_YEARS = int(os.environ.get("MIN_YEARS", 2))
LAT_MAX = float(os.environ.get("LAT_MAX", -50.0))
CORR_DIR = os.environ.get("CORR_DIR", f"{CAUS}/results_corr_bsose_sat_c{COARSEN}")
CORR_NPZ = os.path.join(CORR_DIR, f"corr_bsose_sat_c{COARSEN}.npz")
OUT_NPZ = os.path.join(CORR_DIR, f"corr_bsose_sat_c{COARSEN}_cal{MIN_CAL}.npz")
OUT_PDF = os.path.join(CORR_DIR, f"mask_blocks_c{COARSEN}_cal{MIN_CAL}.pdf")


def calendar_coverage(sb, times):
    """(T, ny, nx) block satellite -> (ny, nx) calendar months with data in
    at least MIN_YEARS years."""
    month = pd.DatetimeIndex(times).month.values
    ok = np.isfinite(sb)
    ncal = np.zeros(sb.shape[1:], int)
    for m in range(1, 13):
        ncal += ok[month == m].sum(axis=0) >= MIN_YEARS
    return ncal


def main():
    k = COARSEN
    z = dict(np.load(CORR_NPZ))
    path = os.path.join(BSOSE_DIR, BSOSE_NAME.format("Chl"))
    with xr.open_dataset(path, decode_times=False) as ds:
        yc, xc = ds["YC"].values, ds["XC"].values
        ocean = (ds["hFacC"].isel(Z=0).values > 0)
    times = pd.date_range("2013-01-01", periods=NT, freq="MS")
    sat = satellite_on_grid(yc, xc, times)
    sat[:, ~ocean] = np.nan
    sb = blocks(sat, k)
    del sat
    ncal = calendar_coverage(sb, times)
    if ncal.shape != z["r"].shape:
        sys.exit(f"grid {ncal.shape} differs from {CORR_NPZ} {z['r'].shape}")

    lat2 = np.broadcast_to(z["lat"][:, None], ncal.shape)
    tested = np.isfinite(z["r"])
    was_rej = tested & ~(z["sig"].astype(bool) & (z["r"] > 0))
    untested_cal = tested & (ncal < MIN_CAL) & (lat2 < LAT_MAX)
    r = np.where(untested_cal, np.nan, z["r"])
    p = np.where(untested_cal, np.nan, z["p"])
    sig = z["sig"].astype(bool) & ~untested_cal
    z.update(r=r, p=p, sig=sig, ncal=ncal, untested_cal=untested_cal,
             min_cal=MIN_CAL, lat_max=LAT_MAX)
    np.savez(OUT_NPZ, **z)

    print(f"calendar months covered (>= {MIN_YEARS} years), blocks south of {LAT_MAX:g}: "
          f"median {int(np.median(ncal[lat2 < LAT_MAX]))}")
    print(f"reclassified as untested (< {MIN_CAL} calendar months, south of {LAT_MAX:g}): "
          f"{int(untested_cal.sum())} blocks, of which {int((untested_cal & was_rej).sum())} "
          f"were rejected and {int((untested_cal & ~was_rej).sum())} kept; "
          f"northernmost at {lat2[untested_cal].max() if untested_cal.any() else float('nan'):.1f}")
    cls = np.where(~np.isfinite(r), 0, np.where(sig & (r > 0), 2, 1))
    for i, name in enumerate(["untested", "rejected", "kept"]):
        print(f"  {name:9s} {int((cls == i).sum()):6d} blocks")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.pcolormesh(z["lon"], z["lat"], cls.astype(float),
                  cmap=ListedColormap(["0.8", "tab:red", "tab:green"]),
                  vmin=-0.5, vmax=2.5, shading="auto")
    ax.contour(z["lon"], z["lat"], ncal, levels=[MIN_CAL - 0.5], colors="k", linewidths=0.6)
    ax.set_title(f"Mask: green kept, red rejected, grey untested (incl. < {MIN_CAL} "
                 f"calendar months of satellite south of {abs(LAT_MAX):g}S); "
                 f"black: {MIN_CAL}-month coverage line")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_NPZ}\n      {OUT_PDF}")


if __name__ == "__main__":
    main()
