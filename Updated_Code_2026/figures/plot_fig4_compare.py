"""Fig. 4 comparison: B-SOSE vs satellite + ORAS5, share of blocks with
driver --> Chl, one figure per lag. Reads the summarize_links.py tables.
    python /work/bb1596/b384036/b384036/causality/plot_fig4_compare.py
Writes fig4_compare_lag0/1/2.pdf/.png in OUTDIR.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAUS = "/work/bb1596/b384036/b384036/causality"
BSOSE_CSV = os.environ.get(
    "BSOSE_CSV", CAUS + "/results_final_PCMCI_c1_bsose_2013to2024/links_table_zones_masked_keepuntested.csv")
OBS_CSV = os.environ.get(
    "OBS_CSV", CAUS + "/results_obs_PCMCI_oras5grid_c4_2013to2024/links_table_zones_min24.csv")
OUTDIR = os.environ.get("OUTDIR", CAUS + "/results_fig4_compare")
MIN_BLOCKS = int(os.environ.get("MIN_BLOCKS", 100))
YMAX = float(os.environ.get("YMAX", 100))
YMAX_OBS = float(os.environ.get("YMAX_OBS", 60))
ALPHA = 0.05
SEASONS = ["full", "DJF", "MAM", "JJA", "SON"]
SEASON_LABEL = {"full": "Year", "DJF": "DJF", "MAM": "MAM", "JJA": "JJA", "SON": "SON"}
ZONES = [("N_SAF", "N_SAF"), ("SAF_SB", "ACC (SAF$-$SB)"), ("S_SB", "S_SB")]
DRIVERS = ["wind", "salt", "N", "MLD", "Theta"]
LABEL = {"wind": "wind stress", "salt": "salinity", "N": "$N^2$", "MLD": "MLD", "Theta": r"$\theta$"}
COLORS = {"MLD": "#1f77b4", "salt": "#9467bd", "N": "#2ca02c", "wind": "#ff7f0e", "Theta": "#b15928"}
COLS = [("B-SOSE (2013$-$2024)", BSOSE_CSV), ("Satellite + ORAS5 (2013$-$2024)", OBS_CSV)]


def load(path):
    d = pd.read_csv(path)
    d["region"] = d["region"].replace({"ACC": "SAF_SB"})
    d["share"] = pd.to_numeric(d["share_linked"], errors="coerce")
    return d


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = [load(p) for _, p in COLS]
    lags = sorted(set(data[0]["lag"]).intersection(data[1]["lag"]))
    for lag in lags:
        fig, axes = plt.subplots(len(ZONES), 2, figsize=(11, 2.9 * len(ZONES)), sharey="col",
                                 squeeze=False, dpi=200)
        width = 0.8 / len(DRIVERS)
        x = np.arange(len(SEASONS))
        for ci, ((ctitle, _), d) in enumerate(zip(COLS, data)):
            for ri, (zkey, zlab) in enumerate(ZONES):
                ax = axes[ri, ci]
                dz = d[(d.region == zkey) & (d.lag == lag)]
                for k, drv in enumerate(DRIVERS):
                    vals = []
                    for s in SEASONS:
                        q = dz[(dz.season == s) & (dz.driver == drv)]
                        ok = len(q) and q.blocks_fitted.iloc[0] >= MIN_BLOCKS
                        vals.append(100 * q.share.iloc[0] if ok else np.nan)
                    ax.bar(x + (k - (len(DRIVERS) - 1) / 2) * width, vals, width,
                           color=COLORS[drv], label=LABEL[drv])
                for si, s in enumerate(SEASONS):
                    q = dz[(dz.season == s) & (dz.driver == DRIVERS[0])]
                    if not len(q) or q.blocks_fitted.iloc[0] < MIN_BLOCKS:
                        ax.text(si, 3, "n.t.", ha="center", va="bottom", fontsize=8, color="0.4")
                ax.axhline(100 * ALPHA, color="k", ls="--", lw=0.8)
                ax.set_ylim(0, YMAX if ci == 0 else YMAX_OBS)
                ax.set_xticks(x)
                ax.set_xticklabels([SEASON_LABEL[s] for s in SEASONS], fontsize=10)
                ax.grid(axis="y", alpha=0.3)
                nb = dz[(dz.season == "full") & (dz.driver == DRIVERS[0])].blocks_fitted
                if len(nb):
                    ax.text(0.99, 0.97, f"{int(nb.iloc[0]):,} blocks", transform=ax.transAxes,
                            ha="right", va="top", fontsize=8, color="0.3")
                if ri == 0:
                    ax.set_title(ctitle, fontsize=12)
                if ci == 0:
                    ax.set_ylabel(f"{zlab}\n% of blocks with\ndriver $\\rightarrow$ Chl-a", fontsize=10)
                ax.text(-0.02 if ci == 0 else 0.0, 1.02, "abcdefghij"[ri * 2 + ci], transform=ax.transAxes,
                        fontsize=12, fontweight="bold", va="bottom", ha="right" if ci == 0 else "left")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(DRIVERS), fontsize=11, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"PCMCI links driver $\\rightarrow$ Chl-a, lag {lag} month{'s' if lag != 1 else ''} "
                     f"(dashed: {100 * ALPHA:g}% expected by chance)", fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUTDIR, f"fig4_compare_lag{lag}.{ext}"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUTDIR}/fig4_compare_lag{lag}.pdf, .png")

    print(f"\nfull year, % of blocks with driver --> Chl ({' / '.join(c for c, _ in COLS)})")
    for lag in lags:
        print(f"lag {lag}")
        for zkey, zlab in ZONES:
            parts = []
            for drv in DRIVERS:
                v = []
                for d in data:
                    q = d[(d.region == zkey) & (d.lag == lag) & (d.season == "full") & (d.driver == drv)]
                    v.append(f"{100 * q.share.iloc[0]:.0f}" if len(q) and np.isfinite(q.share.iloc[0]) else "-")
                parts.append(f"{drv} {v[0]}/{v[1]}")
            print(f"  {zkey:7s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
