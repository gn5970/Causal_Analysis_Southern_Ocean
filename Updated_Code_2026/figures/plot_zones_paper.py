"""Methods figure: the three dynamical zones.
    python /work/bb1596/b384036/b384036/causality/plot_zones_paper.py
a  map of N_SAF, ACC (SAF-SB), S_SB with the SAF and SB SSH contours; hatched: excluded by the Chl-a mask
b  latitude of the two boundaries against longitude
Settings: ZONES_FILE, MASK (MASK=none to skip), OUT
"""
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CAUS = "/work/bb1596/b384036/b384036/causality"
ZONES_FILE = os.environ.get("ZONES_FILE", CAUS + "/acc_zones3_bsose.nc")
MASK = os.environ.get("MASK", CAUS + "/results_corr_bsose_sat_c1/corr_bsose_sat_c1_cal8.npz")
OUT = os.environ.get("OUT", CAUS + "/results_zones/zones_map")
LAT_NORTH = -30.0
COLORS = {1: "#f4a582", 2: "#4393c3", 3: "#d1e5f0"}
LABELS = {1: "N_SAF (north of the SAF)", 2: "ACC (between SAF and SB)", 3: "S_SB (south of the SB)"}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with xr.open_dataset(ZONES_FILE) as z:
        zone = z["zone"].transpose("YC", "XC").values.astype(int)
        ssh = z["ssh_mean"].transpose("YC", "XC").values if "ssh_mean" in z else None
        lat = z["YC"].values.astype(float)
        lon = z["XC"].values.astype(float)
        attrs = dict(z.attrs)
    lon180 = ((lon + 180) % 360) - 180
    o = np.argsort(lon180)
    lon180, zone = lon180[o], zone[:, o]
    ssh = ssh[:, o] if ssh is not None else None
    sel = lat <= LAT_NORTH
    lat, zone = lat[sel], zone[sel]
    ssh = ssh[sel] if ssh is not None else None

    lev = {}
    for k, v in attrs.items():
        if k.startswith("ssh_"):
            name = k[4:].upper()
            if name.startswith("SAF"):
                lev["SAF"] = float(v)
            elif name.startswith("SB"):
                lev["SB"] = float(v)
    print(f"boundary SSH levels from {os.path.basename(ZONES_FILE)}: {lev or 'none found (zone edges drawn instead)'}")

    keep = None
    if MASK.lower() != "none" and os.path.exists(MASK):
        m = np.load(MASK)
        r, sig = m["r"], m["sig"].astype(bool)
        mlat, mlon = np.asarray(m["lat"], float), ((np.asarray(m["lon"], float) + 180) % 360) - 180
        excl = np.isfinite(r) & ~(sig & (r > 0))
        iy = np.abs(mlat[None, :] - lat[:, None]).argmin(1)
        ix = np.abs(((mlon[None, :] - lon180[:, None] + 180) % 360) - 180).argmin(1)
        keep = ~excl[iy][:, ix]
        keep[np.abs(mlat[iy] - lat) > 0.5] = True

    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(zone, float)
    print("\nzone      share of area   latitude 5-95 %")
    for k in (1, 2, 3):
        mk = zone == k
        la = np.broadcast_to(lat[:, None], zone.shape)[mk]
        print(f"  {LABELS[k][:6]:7s} {100 * w[mk].sum() / w[zone > 0].sum():8.0f}%      "
              f"{np.percentile(la, 5):6.1f} to {np.percentile(la, 95):6.1f}")
    acc = zone == 2
    north = np.array([lat[acc[:, j]].max() if acc[:, j].any() else np.nan for j in range(lon180.size)])
    south = np.array([lat[acc[:, j]].min() if acc[:, j].any() else np.nan for j in range(lon180.size)])
    print(f"\nSAF (northern edge of the ACC zone): mean {np.nanmean(north):.1f}, range "
          f"{np.nanmin(north):.1f} to {np.nanmax(north):.1f}")
    print(f"SB  (southern edge of the ACC zone): mean {np.nanmean(south):.1f}, range "
          f"{np.nanmin(south):.1f} to {np.nanmax(south):.1f}")

    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 6.2), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, LAT_NORTH], crs=pc)
    zplot = np.where(zone > 0, zone, np.nan)
    ax.pcolormesh(lon180, lat, zplot, cmap=ListedColormap([COLORS[1], COLORS[2], COLORS[3]]),
                  vmin=0.5, vmax=3.5, shading="nearest", transform=pc, rasterized=True)
    if keep is not None:
        ex = np.where(~keep & (zone > 0), 1.0, np.nan)
        ax.contourf(lon180, lat, ex, levels=[0.5, 1.5], colors="none", hatches=["////"], transform=pc)
    handles = [Patch(color=COLORS[k], label=LABELS[k]) for k in (1, 2, 3)]
    if ssh is not None and lev:
        for name, c in (("SAF", "#b2182b"), ("SB", "#542788")):
            if name in lev:
                ax.contour(lon180, lat, ssh, levels=[lev[name]], colors=c, linewidths=1.2, transform=pc)
                handles.append(Line2D([], [], color=c, lw=1.5, label=f"{name} (SSH = {lev[name]:.2f} m)"))
    else:
        ax.contour(lon180, lat, zone, levels=[1.5, 2.5], colors="k", linewidths=0.8, transform=pc)
    if keep is not None:
        handles.append(Patch(facecolor="none", edgecolor="0.3", hatch="////",
                             label="excluded by the Chl-a mask"))
    try:
        from cartopy.io import shapereader
        shapereader.natural_earth(resolution="110m", category="physical", name="land")
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.LAND, facecolor="0.75", zorder=3)
        ax.coastlines(resolution="110m", lw=0.4, zorder=4)
    except Exception:
        pass
    ax.gridlines(lw=0.3, alpha=0.5, ylocs=[-70, -60, -50, -40, -30])
    ax.set_title("a   Dynamical zones", loc="left", fontsize=12)

    bx = fig.add_subplot(gs[0, 1])
    bx.fill_between(lon180, north, LAT_NORTH, color=COLORS[1], step="mid", lw=0)
    bx.fill_between(lon180, south, north, color=COLORS[2], step="mid", lw=0)
    bx.fill_between(lon180, -80, south, color=COLORS[3], step="mid", lw=0)
    bx.plot(lon180, north, color="#b2182b", lw=1.0, label=f"SAF, mean {np.nanmean(north):.1f}$^\\circ$")
    bx.plot(lon180, south, color="#542788", lw=1.0, label=f"SB, mean {np.nanmean(south):.1f}$^\\circ$")
    bx.axhline(np.nanmean(north), color="#b2182b", ls="--", lw=0.7)
    bx.axhline(np.nanmean(south), color="#542788", ls="--", lw=0.7)
    bx.set_xlim(-180, 180)
    bx.set_ylim(-75, -35)
    bx.set_xticks(np.arange(-180, 181, 60))
    bx.set_xlabel("Longitude ($^\\circ$E)")
    bx.set_ylabel("Latitude ($^\\circ$)")
    bx.legend(loc="lower right", fontsize=9, frameon=True)
    bx.set_title("b   Latitude of the zone boundaries", loc="left", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"\nwrote {OUT}.pdf, .png")


if __name__ == "__main__":
    main()
