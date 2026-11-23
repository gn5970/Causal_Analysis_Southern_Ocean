"""How often, and with which sign, each driver --> Chl link is found.

    python summarize_links.py RESULTS_FOLDER [RESULTS_FOLDER ...]

For every results folder of bsose_pcmci.py / oras5_pcmci.py (any method:
PCMCI+, LPCMCI, PCMCI) it reads the saved maps (pcmci_full.npz,
pcmci_DJF.npz, ...) and writes into that folder

  links_share.pdf   share of blocks in which the causal graph has
                    driver --> Chl, by region (rows), lag (columns),
                    season (x) and driver (bars); the dashed line is the
                    5 % that chance alone gives at alpha = 0.05
  links_sign.pdf    among those blocks, the median signed partial
                    correlation (above zero: more driver, more Chl; below:
                    less Chl). Needs anomalies.npz (see below). The share
                    of those blocks with a positive sign is in the table
  links_table.csv   the numbers of both, with the number of blocks

The share answers "is the link there"; the mean strength in the
seasonal_mean_importance figures cannot, because it is averaged only
over the blocks that passed the test (so it sits just above the test's
threshold), and with GPDC it has no sign.

SIGN. GPDC's strength is a distance correlation, always positive. The
sign is taken from the partial correlation of driver(t - tau) with
Chl(t), controlling for Chl(t - 1), in the same block anomalies PCMCI
used, over the same target months (the season's). The anomalies are in
anomalies.npz, which bsose_pcmci.py / oras5_pcmci.py write before the
causal step; for a run finished before that was added:

    ANOM_ONLY=1 COARSEN=8 METHOD=LPCMCI python bsose_pcmci.py

(same settings as the run, minutes, writes only anomalies.npz).

REGIONS. By default latitude bands, as in the PCMCI scripts:
Subtropics 45-30S, ACC 60-45S, HighLat south of 60S, and All. With

    python summarize_links.py --zones /path/acc_zones_bsose.nc FOLDER ...

the regions are the SSH zones from make_acc_zones.py, named as in the
zone file (acc_zones3_bsose.nc: N_SAF, ACC, S_SB), and All. A block belongs to the zone that holds most
of its model cells (at least half of them; otherwise to none). The
block size is read from the folder name (_c8_ and so on). The figures
and the table then end in _zones.

    --ymax 100   the same y axis (0-100 %) in every links_share figure,
                 so that runs can be compared by eye (default: each
                 figure scaled to its own largest bar)
"""
import csv
import glob
import os
import sys
import warnings

import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.05
SEASONS = {"full": None, "DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8], "SON": [9, 10, 11]}
REGIONS = {"All": (-90, -30), "Subtropics": (-45, -30), "ACC": (-60, -45),
           "HighLat": (-90, -60)}
COLORS = {"MLD": "#1f77b4", "salt": "#9467bd", "N": "#2ca02c",
          "wind": "#ff7f0e", "wek": "#ff7f0e", "Theta": "#b15928"}
MIN_BLOCKS = 10          # fewer blocks with a link: no sign is shown


ZONE_FILE = None                   # set by --zones
YMAX = None                        # set by --ymax
MASK_FILE = None                   # set by --mask: corr_bsose_sat npz
MASK_RMIN = 0.0                    # set by --mask-rmin
MASK_KEEP_UNTESTED = False
MIN_MONTHS = 0                     # set by --min-months: complete target months a block needs         # --mask-keep-untested
ZONE_NAMES = {4: "STZ", 1: "SAZ", 2: "PFZ", 3: "AZ"}


def block_zones(folder, ny, nx, lat=None):
    """Zone of each block: the zone of most of its model cells, with the
    first block row matched to the zone file by latitude."""
    import xarray as xr
    m = re.search(r"_c(\d+)_", os.path.basename(os.path.normpath(folder)))
    if not m:
        sys.exit(f"{folder}: no _c<k>_ in the name, block size unknown")
    k = int(m.group(1))
    global ZONE_NAMES
    with xr.open_dataset(ZONE_FILE) as z:
        zone = z["zone"].transpose("YC", "XC")
        yz = zone["YC"].values
        zone = zone.values
        fv = str(z["zone"].attrs.get("flag_values", "")).split()
        fm = str(z["zone"].attrs.get("flag_meanings", "")).split()
    if fv and len(fv) == len(fm):
        # the region names the zone file carries (e.g. N_SAF ACC S_SB)
        ZONE_NAMES = {int(v): m for v, m in zip(fv, fm) if int(v) != 0}
    if "oras5grid" in os.path.basename(os.path.normpath(folder)):
        # blocks of the ORAS5 grid: zone of the B-SOSE cell nearest each block centre
        with xr.open_dataset(ZONE_FILE) as z2:
            xz = z2["zone"].transpose("YC", "XC")["XC"].values.astype(float)
        with np.load(os.path.join(folder, "anomalies.npz")) as a:
            blat, blon = np.asarray(a["lat"], float), np.asarray(a["lon"], float)
        iy = np.abs(yz[None, :] - blat[:, None]).argmin(1)
        ix = np.abs(((xz[None, :] - blon[:, None] + 180) % 360) - 180).argmin(1)
        out = zone[iy][:, ix]
        out[np.abs(yz[iy] - blat) > 0.5] = 0
        print("  ORAS5 grid: zone of the nearest B-SOSE cell to each block centre")
        return out
    i0 = 0
    if lat is not None and yz.size >= k:
        means = np.convolve(yz, np.ones(k) / k, mode="valid")
        i0 = int(np.argmin(np.abs(means - lat[0])))
        if i0:
            print(f"  first block row starts at zone-file row {i0}")
    zone = zone[i0:]
    if zone.shape[0] < ny * k or zone.shape[1] < nx * k:
        sys.exit(f"{ZONE_FILE}: grid {zone.shape} is smaller than {ny} x "
                 f"{nx} blocks of {k}")
    b = zone[:ny * k, :nx * k].reshape(ny, k, nx, k)
    counts = np.stack([(b == z).sum(axis=(1, 3)) for z in ZONE_NAMES])
    best = counts.argmax(axis=0)
    out = np.where(counts.max(axis=0) >= (k * k) / 2,
                   np.array(list(ZONE_NAMES))[best], 0)
    return out


def region_masks(folder, lat, ny, nx):
    """{region: (ny, nx) bool} -- latitude bands or the SSH zones."""
    if ZONE_FILE:
        z = block_zones(folder, ny, nx, lat)
        masks = {"All": np.ones((ny, nx), bool)}
        masks.update({{"ACC": "SAF_SB"}.get(n, n): z == k for k, n in ZONE_NAMES.items()})
        return masks
    return {r: np.broadcast_to(((lat > lo) & (lat <= hi))[:, None], (ny, nx))
            for r, (lo, hi) in REGIONS.items()}


def block_lat(folder, ny):
    """Latitude of each block row: from anomalies.npz, else trend_maps."""
    for pat in ("anomalies.npz", "trend_maps*.npz"):
        for f in sorted(glob.glob(os.path.join(folder, pat))):
            with np.load(f) as z:
                lat = np.asarray(z["lat"], float)
            lat = lat if lat.ndim == 1 else lat.mean(axis=1)
            if lat.size == ny:
                return lat
    sys.exit(f"{folder}: no anomalies.npz or trend_maps*.npz with {ny} "
             "latitudes, so the regions cannot be drawn")


def partial_r(x, y, z):
    """Partial correlation of x and y given z, along axis 0, per block,
    over the samples where all three are finite. (n, Y, X) -> (Y, X)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = ok.sum(axis=0)

    def centred(a):
        a = np.where(ok, a, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ok, a - a.sum(axis=0) / n, 0.0)

    x, y, z = centred(x), centred(y), centred(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = (z * z).sum(axis=0)
        rx = x - z * (x * z).sum(axis=0) / zz
        ry = y - z * (y * z).sum(axis=0) / zz
        r = (rx * ry).sum(axis=0) / np.sqrt((rx * rx).sum(axis=0)
                                            * (ry * ry).sum(axis=0))
    return np.where(n >= 12, r, np.nan)


def signed_r(anom, names, months_sel, tau_max):
    """r[driver][tau] (Y, X): partial correlation of driver(t - tau)
    with Chl(t) given Chl(t - 1), t over the target months."""
    feats, chl = anom["feats"].astype("float64"), anom["chl"].astype("float64")
    T = chl.shape[0]
    month = (np.arange(T) + int(anom["month0"]) - 1) % 12 + 1
    target = np.ones(T, bool) if months_sel is None else np.isin(month, months_sel)
    out = {}
    for k, f in enumerate(names):
        out[f] = []
        for tau in range(tau_max + 1):
            t = np.flatnonzero(target & (np.arange(T) >= max(tau, 1)))
            out[f].append(partial_r(feats[k][t - tau], chl[t], chl[t - 1]))
    return out


def summarize(folder):
    tag = os.path.basename(os.path.normpath(folder))
    maps = {}
    for s in SEASONS:
        f = os.path.join(folder, f"pcmci_{s}.npz")
        if os.path.exists(f):
            with np.load(f) as z:
                maps[s] = dict(names=[str(n) for n in z["names"]],
                               val=z["val"], p=z["p"],
                               setup=str(z["setup"]) if "setup" in z else "")
    if "full" not in maps:
        print(f"{folder}: no pcmci_full.npz, skipped")
        return
    names = maps["full"]["names"]
    nlag = maps["full"]["val"].shape[1]
    ny = maps["full"]["val"].shape[2]
    nx = maps["full"]["val"].shape[3]
    if MIN_MONTHS:
        # drop blocks with fewer than MIN_MONTHS complete months (Chl and all
        # drivers present) among the target months of each season
        af = os.path.join(folder, "anomalies.npz")
        if not os.path.exists(af):
            sys.exit(f"--min-months needs {af}")
        with np.load(af) as a:
            okm = np.isfinite(a["chl"]) & np.all(np.isfinite(a["feats"]), axis=0)
        mon = np.arange(okm.shape[0]) % 12 + 1
        for s_, m_ in maps.items():
            sm = SEASONS[s_]
            nm = (okm if sm is None else okm & np.isin(mon, sm)[:, None, None]).sum(0)
            drop = nm < MIN_MONTHS
            m_["val"] = np.where(drop[None, None], np.nan, m_["val"])
            m_["p"] = np.where(drop[None, None], np.nan, m_["p"])
            print(f"  {s_}: {int((~drop).sum())} blocks with >= {MIN_MONTHS} complete months")
    lat = block_lat(folder, ny)
    masks = region_masks(folder, lat, ny, nx)
    if MASK_FILE:
        with np.load(MASK_FILE) as mz:
            r_m, sig_m = mz["r"], mz["sig"].astype(bool)
        keep = sig_m & np.isfinite(r_m) & (r_m > MASK_RMIN)
        if MASK_KEEP_UNTESTED:
            keep |= ~np.isfinite(r_m)
        if r_m.shape != (ny, nx) and "oras5grid" in os.path.basename(os.path.normpath(folder)):
            # mask on the ORAS5 grid: each block takes the verdict of the
            # mask cell nearest its centre (mask lat/lon from the npz)
            with np.load(MASK_FILE) as mz:
                mlat, mlon = np.asarray(mz["lat"], float), np.asarray(mz["lon"], float)
            with np.load(os.path.join(folder, "anomalies.npz")) as a:
                blat, blon = np.asarray(a["lat"], float), np.asarray(a["lon"], float)
            iy = np.abs(mlat[None, :] - blat[:, None]).argmin(1)
            ix = np.abs(((mlon[None, :] - blon[:, None] + 180) % 360) - 180).argmin(1)
            far = np.abs(mlat[iy] - blat) > 1.0
            keep = keep[iy][:, ix]
            keep[far] = MASK_KEEP_UNTESTED
            print("  ORAS5 grid: mask verdict of the nearest mask cell to each block centre")
        elif r_m.shape != (ny, nx):
            sys.exit(f"{MASK_FILE}: grid {r_m.shape} differs from the run's "
                     f"{ny} x {nx} blocks (run corr_bsose_sat.py with the same COARSEN)")
        before = {r: int(m.sum()) for r, m in masks.items()}
        masks = {r: m & keep for r, m in masks.items()}
        print(f"  mask {os.path.basename(MASK_FILE)} (significant, r > {MASK_RMIN:g}"
              + (", untested kept" if MASK_KEEP_UNTESTED else "") + "): "
              + ", ".join(f"{r} {int(masks[r].sum())} of {before[r]}" for r in masks))
    regions = list(masks)
    suffix = (("_zones" if ZONE_FILE else "") + (f"_min{MIN_MONTHS}" if MIN_MONTHS else "")
              + (("_masked" + ("_keepuntested" if MASK_KEEP_UNTESTED else "")) if MASK_FILE else ""))
    if ZONE_FILE:
        print(f"{tag}: regions from {os.path.basename(ZONE_FILE)}: "
              + ", ".join(f"{r} {int(masks[r].sum())} blocks" for r in regions))
    anom_f = os.path.join(folder, "anomalies.npz")
    anom = dict(np.load(anom_f)) if os.path.exists(anom_f) else None
    if anom is not None and list(map(str, anom["names"])) != names:
        print(f"  {anom_f}: drivers differ from the run; no sign")
        anom = None
    if anom is None:
        print(f"{tag}: no anomalies.npz, so links_sign.pdf is not drawn "
              "(see the top of this script)")

    # share[region][lag][season][driver], sign likewise
    rows = []
    share = {r: [[{} for _ in SEASONS] for _ in range(nlag)] for r in regions}
    pos = {r: [[{} for _ in SEASONS] for _ in range(nlag)] for r in regions}
    medr = {r: [[{} for _ in SEASONS] for _ in range(nlag)] for r in regions}
    for si, s in enumerate(SEASONS):
        if s not in maps:
            continue
        m = maps[s]
        rr = signed_r(anom, names, SEASONS[s], nlag - 1) if anom is not None else None
        for k, f in enumerate(names):
            for tau in range(nlag):
                val, p = m["val"][k, tau], m["p"][k, tau]
                fitted = np.isfinite(val)
                linked = fitted & (p <= ALPHA)
                for r in regions:
                    reg = masks[r]
                    nf = int((fitted & reg).sum())
                    nl = int((linked & reg).sum())
                    sh = nl / nf if nf else np.nan
                    sp = np.nan
                    med = np.nan
                    if rr is not None and nl >= MIN_BLOCKS:
                        rv = rr[f][tau][linked & reg]
                        rv = rv[np.isfinite(rv)]
                        if rv.size >= MIN_BLOCKS:
                            sp = float((rv > 0).mean())
                            med = float(np.median(rv))
                    share[r][tau][si][f] = sh
                    pos[r][tau][si][f] = sp
                    medr[r][tau][si][f] = med
                    rows.append(dict(run=tag, region=r, lag=tau, season=s,
                                     driver=f, blocks_fitted=nf,
                                     blocks_linked=nl,
                                     share_linked=round(sh, 4) if nf else "",
                                     share_positive=("" if np.isnan(sp)
                                                     else round(sp, 3)),
                                     median_partial_r=("" if np.isnan(med)
                                                       else round(med, 3))))

    with open(os.path.join(folder, f"links_table{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def figure(data, ylabel, ref, ylim, out, title, scale=100):
        seas = list(SEASONS)
        fig, axes = plt.subplots(len(regions), nlag,
                                 figsize=(4.6 * nlag, 2.9 * len(regions)),
                                 sharey=True, squeeze=False, dpi=180)
        width = 0.8 / len(names)
        x = np.arange(len(seas))
        for ri, r in enumerate(regions):
            for tau in range(nlag):
                ax = axes[ri, tau]
                for k, f in enumerate(names):
                    v = [data[r][tau][si].get(f, np.nan) for si in range(len(seas))]
                    ax.bar(x + (k - (len(names) - 1) / 2) * width,
                           np.asarray(v, float) * scale, width,
                           color=COLORS.get(f, "0.5"), label=f)
                ax.axhline(ref * scale, color="k", ls="--", lw=0.9)
                ax.set_ylim(*ylim)
                ax.set_xticks(x)
                ax.set_xticklabels(seas, fontsize=10)
                ax.grid(axis="y", alpha=0.3)
                if ri == 0:
                    ax.set_title(f"lag {tau}", fontsize=13)
                if tau == 0:
                    ax.set_ylabel(f"{r}\n{ylabel}", fontsize=11)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(names), fontsize=12,
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{tag}: {title}", fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(os.path.join(folder, out), bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {os.path.join(folder, out)}")

    top = YMAX if YMAX else max(20.0, 100 * np.nanmax([v for r in share for t in share[r]
                                      for d in t for v in d.values()]
                                     + [0]) * 1.1)
    figure(share, "% of blocks with\ndriver --> Chl", ALPHA, (0, top),
           "links_share" + suffix + ".pdf",
           f"share of blocks where the graph has driver --> Chl "
           f"(dashed: {100 * ALPHA:g} % by chance)")
    if anom is not None:
        figure(medr, "median partial r\nwhere linked", 0.0, (-1, 1),
               "links_sign" + suffix + ".pdf",
               "sign of the links: median partial correlation of "
               "driver(t-lag) with Chl(t) given Chl(t-1), over the blocks "
               f"with the link (blank: fewer than {MIN_BLOCKS} blocks)",
               scale=1)
    print(f"  wrote {os.path.join(folder, f'links_table{suffix}.csv')}")

    # a few lines in the log: the strongest links overall
    full = [r for r in rows if r["season"] == "full" and r["region"] == "All"]
    full.sort(key=lambda r: -(r["share_linked"] or 0))
    print(f"{tag} ({maps['full']['setup']}), whole year, all blocks:")
    for r in full[:8]:
        sgn = ("" if r["share_positive"] == "" else
               f", {100 * r['share_positive']:.0f}% positive, median r "
               f"{r['median_partial_r']:+.2f}")
        print(f"  {r['driver']:6s} lag {r['lag']}: "
              f"{100 * r['share_linked']:.0f}% of {r['blocks_fitted']} "
              f"blocks{sgn}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--zones" in args:
        i = args.index("--zones")
        ZONE_FILE = args[i + 1]
        if not os.path.exists(ZONE_FILE):
            sys.exit(f"no zone file {ZONE_FILE}")
        del args[i:i + 2]
    if "--mask" in args:
        i = args.index("--mask")
        MASK_FILE = args[i + 1]
        if not os.path.exists(MASK_FILE):
            sys.exit(f"no mask file {MASK_FILE}")
        del args[i:i + 2]
    if "--mask-keep-untested" in args:
        MASK_KEEP_UNTESTED = True
        args.remove("--mask-keep-untested")
    if "--mask-rmin" in args:
        i = args.index("--mask-rmin")
        MASK_RMIN = float(args[i + 1])
        del args[i:i + 2]
    if "--min-months" in args:
        i = args.index("--min-months")
        MIN_MONTHS = int(args[i + 1])
        del args[i:i + 2]
    if "--ymax" in args:
        i = args.index("--ymax")
        YMAX = float(args[i + 1])
        del args[i:i + 2]
    folders = [a for a in args if os.path.isdir(a)]
    if not folders:
        sys.exit(__doc__)
    warnings.simplefilter("ignore", RuntimeWarning)
    for f in folders:
        summarize(f)
