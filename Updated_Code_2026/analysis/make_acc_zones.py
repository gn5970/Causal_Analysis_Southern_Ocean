"""ACC zones from the model's own sea-surface height.

    python make_acc_zones.py
    FRONTS_DIR=/path/to/orsi python make_acc_zones.py     calibrate on Orsi files

The fronts of the ACC follow SSH contours (Sokolov & Rintoul 2009; Kim
& Orsi 2014), so the zones are taken from B-SOSE's own time-mean SSH:

  1. SSH averaged over the whole record (one mask for all months: a mask
     moving with the SSH would itself carry the eddies and the SAM).
  2. The SSH values of the three fronts read off in Drake Passage (mean
     over DP_LON +/- 1 deg), at the latitudes the fronts have there:
     Subantarctic Front (SAF), Polar Front (PF), southern boundary (SBdy,
     Orsi et al. 1995). Drake Passage is where all ACC water passes a
     narrow, well-observed section, so it calibrates the contours without
     assuming anything elsewhere.
  3. Zones by SSH:
        STZ  Subtropical Zone      north of the STF
        SAZ  Subantarctic Zone     between the STF and the SAF
        PFZ  Polar Frontal Zone    between the SAF and the PF
        AZ   Antarctic Zone        between the PF and the SBdy
     each kept only where it is part of the connected circumpolar band,
     which drops shelves, the Weddell and Ross gyres and isolated patches
     in the same SSH range. Cells south of the SBdy, or dropped, are 0.

CALIBRATION, default (CALIB=sr09): the method of Sokolov & Rintoul
(2009, JGR 114, C11019) applied to B-SOSE's own SSH. Fronts are
streamlines along which |grad SSH| concentrates, each with an SSH label
nearly constant around the belt and in time (their Table 1, nine
branches: SAF-N/M/S, PF-N/M/S, SACCF-N/S, SB). B-SOSE's SSH has its own
reference, so the labels are mapped by SSH = scale x label + offset, the
offset (and, with SR_SCALE=fit, the scale: full-depth SSH against their
2500 dbar dynamic height) chosen so the labels sit on the maxima of the
area-weighted mean |grad SSH| as a function of SSH. Calibrated on the
whole ACC, with no external front positions. The zones are bounded by
SAF_BRANCH (SAF-M), PF_BRANCH (PF-M) and SB_BRANCH (SB). The STF has
almost no SSH signature (SR09), so SAZ and STZ stay together unless an
Orsi stf.txt (FRONTS_DIR) or STF_LAT is given. Panel c of the figure
shows the gradient profile with the fitted labels: check that the
labels sit on its peaks.

CALIBRATION with CALIB=orsi: with the Orsi et al. (1995) front files in FRONTS_DIR
(stf.txt, saf.txt, pf.txt, sbdy.txt: longitude and latitude columns, as
distributed by the Australian Antarctic Data Centre), each front's SSH
value is the median of the time-mean SSH at that front's positions ALL
AROUND the Southern Ocean: one section can be off by a meander, the
whole path cannot. The Drake Passage value is printed beside it for
comparison. Without the files: SAF, PF and SBdy from Drake Passage
(DP_LON) at SAF_LAT / PF_LAT / SBDY_LAT, and the STF, which does not
pass Drake Passage, at STF_LON / STF_LAT. These defaults are
APPROXIMATE and must be checked before the zones go in the paper; the
log marks them and prints every front's latitude at six longitudes.

The SSH is smoothed with a Gaussian of SMOOTH_DEG (0.5 deg) before the
contours are taken, so standing eddies do not fray the zone edges.

Writes, next to the SSH file's folder (WORK):
  acc_zones_bsose.nc   zone (YC, XC): 0 none, 1 SAZ, 2 PFZ, 3 AZ, 4 STZ, on the
                       native B-SOSE grid in the file's own order (the
                       order bsose_pcmci.py and oras5_pcmci.py use), with
                       the time-mean SSH and the three contour values
  acc_zones_bsose.pdf  map: mean SSH, the three contours, the zones, and
                       for comparison the 45-60S band and (if given) the
                       Orsi fronts
"""
import glob
import re
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import ndimage

WORK = os.environ.get("WORK", "/work/bb1596/b384036/b384036/causality")
BSOSE_DIR = os.environ.get("BSOSE_DIR", os.path.join(WORK, "bsose"))
SSH_FILE = os.environ.get("SSH_FILE")
GRID_FILE = os.environ.get(
    "GRID_FILE", os.path.join(BSOSE_DIR, "Chl_bsoseI156_2013to2024_monthly.nc"))
FRONTS_DIR = os.environ.get("FRONTS_DIR")
DP_LON = float(os.environ.get("DP_LON", -65.0))
# APPROXIMATE front latitudes at 65W -- check against Orsi et al. (1995)
STF_LON = float(os.environ.get("STF_LON", 20.0))
SMOOTH_DEG = float(os.environ.get("SMOOTH_DEG", 0.5))
FRONTS = ("STF", "SAF", "PF", "SBdy")
LATS = {"STF": float(os.environ.get("STF_LAT", -41.0)),     # at STF_LON
        "SAF": float(os.environ.get("SAF_LAT", -56.8)),
        "PF": float(os.environ.get("PF_LAT", -58.7)),
        "SBdy": float(os.environ.get("SBDY_LAT", -61.8))}
CIRCUMPOLAR = 0.9          # a band must cover this share of longitudes
# Calibration: "sr09" (default) applies the method of Sokolov & Rintoul
# (2009) to B-SOSE's own SSH; "orsi" takes the Orsi front files in
# FRONTS_DIR (or, without them, one section per front).
CALIB = os.environ.get("CALIB", "sr09").lower()
# Sokolov & Rintoul (2009, Table 1): SSH labels of the ACC front
# branches, m, relative to 2500 dbar
SR09 = {"SAF-N": 1.97, "SAF-M": 1.71, "SAF-S": 1.47, "PF-N": 1.29,
        "PF-M": 1.15, "PF-S": 1.04, "SACCF-N": 0.94, "SACCF-S": 0.84,
        "SB": 0.75}
# which branch bounds each zone
SAF_BRANCH = os.environ.get("SAF_BRANCH", "SAF-M")
PF_BRANCH = os.environ.get("PF_BRANCH", "PF-M")
SB_BRANCH = os.environ.get("SB_BRANCH", "SB")
# the labels may be shifted (any reference) and, if SR_SCALE=fit,
# stretched (full-depth SSH against 2500 dbar dynamic height)
SR_SCALE = os.environ.get("SR_SCALE", "fit")
SCALE_RANGE = (0.8, 1.6)
# ZONE_SET: "acc3" (default) -- three regions from two fronts, SAF and SB:
#              north of the SAF / the ACC (SAF to SB) / south of the SB;
#           "fronts4" -- STZ, SAZ, PFZ, AZ from STF, SAF, PF and SB.
ZONE_SET = os.environ.get("ZONE_SET", "acc3").lower()
if ZONE_SET == "acc3":
    ZONES = {1: "N_SAF", 2: "ACC", 3: "S_SB"}
    ZLABEL = {1: "north of the SAF", 2: "ACC (SAF to SB)",
              3: "south of the SB"}
    ZFRONTS = ("SAF", "SBdy")
    OUT_DEFAULT = "acc_zones3_bsose"
else:
    ZONES = {4: "STZ", 1: "SAZ", 2: "PFZ", 3: "AZ"}
    ZLABEL = dict(ZONES)
    ZFRONTS = ("STF", "SAF", "PF", "SBdy")
    OUT_DEFAULT = "acc_zones_bsose"
OUT = os.environ.get("OUT", os.path.join(WORK, OUT_DEFAULT))
CHECK_LONS = (-65, 0, 20, 90, 140, -170)


def find_ssh():
    if SSH_FILE:
        return SSH_FILE
    for w in ("SSH", "ETAN"):
        hits = sorted(p for p in glob.glob(os.path.join(BSOSE_DIR, f"{w}_*.nc"))
                      if "snap" not in os.path.basename(p).lower())
        if hits:
            return hits[0]
    sys.exit(f"no SSH_*.nc or ETAN_*.nc in {BSOSE_DIR} (download_bsose.py SSH)")


_FRONTS = None
FRONT_PATTERNS = {               # file / layer / attribute names, lower case
    "STF": ("stf", "subtropical"),
    "SAF": ("saf", "subantarctic"),
    "PF": ("pf", "polar_front", "polar front", "polarfront"),
    "SBdy": ("sbdy", "sb_", "southern_boundary", "southern boundary",
             "boundary"),
}
SKIP = ("saccf", "southern_acc", "southern acc")


def _which_front(text):
    t = text.lower()
    if any(k in t for k in SKIP):
        return None
    for f, keys in FRONT_PATTERNS.items():
        for k in keys:
            if k == "pf":
                if re.search(r"(^|[^a-z])pf([^a-z]|$)", t):
                    return f
            elif k in t:
                return f
    return None


def _to_lonlat(x, y, prj_text):
    """Shapefile coordinates to lon/lat. Already geographic if they fit;
    otherwise reprojected from the .prj (or, without one, EPSG:3031)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.nanmax(np.abs(x)) <= 360.5 and np.nanmax(np.abs(y)) <= 90.5:
        return x, y
    from pyproj import CRS, Transformer
    src = CRS.from_wkt(prj_text) if prj_text else CRS.from_epsg(3031)
    tr = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)
    return tr.transform(x, y)


def _read_shp(path, prj_text):
    """{front: (lon, lat)} from one shapefile: the file name names the
    front, or else an attribute of each record does."""
    import shapefile
    out = {}
    r = shapefile.Reader(path)
    fields = [f[0] for f in r.fields[1:]]
    by_name = _which_front(os.path.basename(path))
    for rec, shp in zip(r.records(), r.shapes()):
        f = by_name
        if f is None:
            f = next((g for g in (_which_front(str(v)) for v in rec) if g), None)
        if f is None or not shp.points:
            continue
        pts = np.array(shp.points, float)
        lo, la = _to_lonlat(pts[:, 0], pts[:, 1], prj_text)
        old = out.get(f, (np.empty(0), np.empty(0)))
        out[f] = (np.r_[old[0], lo], np.r_[old[1], la])
    return out, fields


def _read_txt(path):
    rows = []
    for line in open(path, errors="replace"):
        parts = line.replace(",", " ").split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    a = np.array(rows) if rows else np.empty((0, 2))
    return a[:, 0], a[:, 1]


def load_fronts():
    """All Orsi fronts found in FRONTS_DIR, which may be a folder or a
    .zip (the AADC 'Fronts of the ACC - GIS data' download), holding
    shapefiles or text files. {front: (lon -180..180, lat)}."""
    global _FRONTS
    if _FRONTS is not None:
        return _FRONTS
    _FRONTS = {}
    if not FRONTS_DIR:
        return _FRONTS
    src = FRONTS_DIR
    if src.lower().endswith(".zip"):
        import zipfile
        dest = os.path.splitext(src)[0] + "_unzipped"
        if not os.path.isdir(dest):
            zipfile.ZipFile(src).extractall(dest)
        src = dest
    files = sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True))
    for f in files:
        low = f.lower()
        if low.endswith(".shp"):
            prj = os.path.splitext(f)[0] + ".prj"
            prj_text = open(prj).read() if os.path.exists(prj) else None
            got, fields = _read_shp(f, prj_text)
            for k, v in got.items():
                old = _FRONTS.get(k, (np.empty(0), np.empty(0)))
                _FRONTS[k] = (np.r_[old[0], v[0]], np.r_[old[1], v[1]])
            print(f"  fronts file {os.path.relpath(f, src)}: "
                  + (", ".join(f"{k} {v[0].size} points" for k, v in got.items())
                     or f"no front recognised (fields {fields})"), flush=True)
        elif low.endswith((".txt", ".csv", ".dat")):
            k = _which_front(os.path.basename(f))
            if k:
                lo, la = _read_txt(f)
                old = _FRONTS.get(k, (np.empty(0), np.empty(0)))
                _FRONTS[k] = (np.r_[old[0], lo], np.r_[old[1], la])
                print(f"  fronts file {os.path.relpath(f, src)}: {k} "
                      f"{lo.size} points", flush=True)
    for k, (lo, la) in list(_FRONTS.items()):
        ok = np.isfinite(lo) & np.isfinite(la)
        _FRONTS[k] = ((((lo[ok] + 180.0) % 360.0) - 180.0), la[ok])
    missing = [f for f in FRONT_PATTERNS if f not in _FRONTS]
    print(f"  Orsi fronts found: {sorted(_FRONTS)}"
          + (f"; NOT found: {missing}" if missing else ""), flush=True)
    return _FRONTS


def front_points(name):
    """(lon, lat) of an Orsi front, or None."""
    return load_fronts().get(name)


def smooth(ssh, lat):
    """Gaussian of SMOOTH_DEG, land-aware (normalised), periodic in x."""
    if SMOOTH_DEG <= 0:
        return ssh
    dy = float(np.abs(np.diff(lat)).mean())
    sig = SMOOTH_DEG / dy
    ok = np.isfinite(ssh)
    num = ndimage.gaussian_filter(np.where(ok, ssh, 0.0), sig,
                                  mode=("nearest", "wrap"))
    den = ndimage.gaussian_filter(ok.astype(float), sig,
                                  mode=("nearest", "wrap"))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(ok, out, np.nan)


def contour_lat(ssh, lat, lonw, value, lon0):
    """Northernmost latitude at lon0 (+/-1 deg) where the SSH profile
    crosses `value` going south (the front's position there)."""
    cols = np.abs(((lonw - lon0 + 180.0) % 360.0) - 180.0) <= 1.0
    prof = np.nanmean(ssh[:, cols], axis=1)
    o = np.argsort(lat)[::-1]                 # north to south
    la, pr = lat[o], prof[o]
    for i in range(len(la) - 1):
        if np.isfinite(pr[i]) and np.isfinite(pr[i + 1]) \
                and pr[i] >= value > pr[i + 1]:
            f = (pr[i] - value) / (pr[i] - pr[i + 1])
            return la[i] + f * (la[i + 1] - la[i])
    return np.nan


def front_lat_from_file(name):
    """Latitude of an Orsi front at DP_LON (mean of its points within
    1 deg of longitude), or None."""
    pts = front_points(name)
    if pts is None:
        return None
    near = np.abs(((pts[0] - DP_LON + 180.0) % 360.0) - 180.0) <= 1.0
    return float(np.mean(pts[1][near])) if near.any() else None


def read_mean_ssh(path):
    with xr.open_dataset(path, decode_times=False) as ds:
        name = next((v for v in ("ETAN", "SSH") if v in ds), None)
        if name is None:
            name = max(ds.data_vars, key=lambda v: ds[v].ndim)
        v = ds[name]
        ydim = next(d for d in v.dims if d.startswith("Y"))
        xdim = next(d for d in v.dims if d.startswith("X"))
        mean = v.mean("time").transpose(ydim, xdim).values.astype("float64")
        lat = ds[ydim].values.astype("float64")
        lon = ds[xdim].values.astype("float64")
        nt = v.sizes["time"]
    print(f"SSH: {os.path.basename(path)}, {name}, mean of {nt} months, "
          f"grid {lat.size} x {lon.size}", flush=True)
    return mean, lat, lon


def wet_mask(shape):
    if os.path.exists(GRID_FILE):
        with xr.open_dataset(GRID_FILE, decode_times=False) as g:
            h = g["hFacC"]
            wet = h.isel({h.dims[0]: 0}).transpose("YC", "XC").values > 0
        if wet.shape == shape:
            return wet
    return None


def circumpolar(mask):
    """Only the connected parts of `mask` that, together with the cells
    across the dateline seam, reach at least CIRCUMPOLAR of the
    longitudes."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    # join labels that meet across the periodic seam
    parent = list(range(n + 1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(lab[:, 0], lab[:, -1]):
        if a and b:
            parent[root(a)] = root(b)
    roots = np.array([root(i) for i in range(n + 1)])
    lab = roots[lab]
    keep = np.zeros(n + 1, bool)
    nx = mask.shape[1]
    for r in np.unique(lab[lab > 0]):
        cols = (lab == r).any(axis=0).sum()
        keep[r] = cols >= CIRCUMPOLAR * nx
    return keep[lab]


def ssh_gradient(ssh, lat, lon):
    """|grad SSH| in m per 100 km; lon must be the native, regularly
    spaced longitudes (periodic)."""
    R = 6.371e6
    phi = np.deg2rad(lat)
    dlam = np.deg2rad(float(np.abs(np.diff(lon)).mean()))
    gx = (np.roll(ssh, -1, axis=1) - np.roll(ssh, 1, axis=1)) / (
        2 * dlam * R * np.cos(phi)[:, None])
    gy = np.gradient(ssh, R * phi, axis=0)
    return np.hypot(gx, gy) * 1e5


MIN_DEPTH = float(os.environ.get("MIN_DEPTH", 2000.0))


def deep_mask(shape, lat):
    """Cells deeper than MIN_DEPTH (2000 m, as SR09): the shelf and slope,
    where the Antarctic Slope Current makes the strongest SSH gradients
    of all, are left out of the calibration. Depth from the grid file if
    it has it (Depth, or R_low), otherwise everything north of 65S."""
    if os.path.exists(GRID_FILE):
        with xr.open_dataset(GRID_FILE, decode_times=False) as g:
            for name in ("Depth", "R_low", "rLowC"):
                if name in g:
                    d = np.abs(g[name].transpose("YC", "XC").values)
                    if d.shape == shape:
                        print(f"  calibration on cells deeper than "
                              f"{MIN_DEPTH:g} m ({name} in "
                              f"{os.path.basename(GRID_FILE)})", flush=True)
                        return d > MIN_DEPTH
    print("  no depth in the grid file: calibration on cells north of 65S",
          flush=True)
    return np.broadcast_to((lat > -65.0)[:, None], shape)


def sr09_calibration(ssh, lat, lon):
    """Sokolov & Rintoul (2009) on B-SOSE: the area-weighted mean |grad
    SSH| as a function of SSH value, G(h), peaks where the jets are; the
    nine SR09 labels are mapped onto B-SOSE by h = scale * label + offset,
    with offset (and scale) chosen to put the labels on the peaks of G."""
    g = ssh_gradient(ssh, lat, lon)
    reg = (np.isfinite(ssh) & np.isfinite(g) & (lat <= -35.0)[:, None]
           & deep_mask(ssh.shape, lat))
    h, gv = ssh[reg], g[reg]
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], ssh.shape)[reg]
    lo, hi = np.percentile(h, [1, 99.5])
    edges = np.arange(lo, hi + 0.01, 0.01)
    mid = 0.5 * (edges[:-1] + edges[1:])
    k = np.digitize(h, edges) - 1
    G = np.full(mid.size, np.nan)
    for i in range(mid.size):
        m = k == i
        if w[m].sum() > 0:
            G[i] = np.average(gv[m], weights=w[m])
    G = np.convolve(np.nan_to_num(G), np.ones(3) / 3, mode="same")
    labels = np.array(list(SR09.values()))
    scales = (np.arange(SCALE_RANGE[0], SCALE_RANGE[1] + 1e-9, 0.01)
              if SR_SCALE == "fit" else [float(SR_SCALE)])
    best = (-np.inf, None, None)
    for sc in scales:
        for off in np.arange(lo - sc * labels.min(), hi - sc * labels.max(), 0.005):
            score = np.mean(np.interp(sc * labels + off, mid, G))
            if score > best[0]:
                best = (score, sc, off)
    score, sc, off = best
    if sc is None:
        sys.exit("the B-SOSE SSH range is narrower than the SR09 labels; "
                 "set SR_SCALE=fit or a smaller fixed scale")
    vals = {n: sc * v + off for n, v in SR09.items()}
    if SR_SCALE == "fit" and (abs(sc - SCALE_RANGE[0]) < 0.011
                              or abs(sc - SCALE_RANGE[1]) < 0.011):
        print(f"  WARNING: the fitted scale {sc:.2f} is at the edge of "
              f"{SCALE_RANGE}: the labels do not match B-SOSE's jets well. "
              "Check panel c, or use CALIB=orsi with the Orsi fronts.",
              flush=True)
    # the local peaks of G, for the log and the figure
    pk = [i for i in range(1, mid.size - 1) if G[i] >= G[i - 1] and G[i] >= G[i + 1]
          and G[i] > np.nanpercentile(G, 60)]
    print(f"  SR09 labels on B-SOSE: h = {sc:.2f} x label {off:+.3f} m "
          f"(mean |grad SSH| at the labels {score:.3f} m/100 km; "
          f"over the whole range {np.nanmean(G):.3f})", flush=True)
    print("    branch   label   B-SOSE SSH   |grad| there", flush=True)
    for n, v in SR09.items():
        print(f"    {n:8s} {v:5.2f}   {vals[n]:+8.3f}    "
              f"{np.interp(vals[n], mid, G):.3f}", flush=True)
    print("    peaks of |grad SSH| at B-SOSE SSH: "
          + ", ".join(f"{mid[i]:+.2f}" for i in pk), flush=True)
    return vals, (mid, G, sc, off)


def main():
    path = find_ssh()
    ssh, lat, lon = read_mean_ssh(path)
    wet = wet_mask(ssh.shape)
    if wet is None:
        wet = np.isfinite(ssh) & (ssh != 0.0)
    ssh = np.where(wet, ssh, np.nan)
    lonw = ((lon + 180.0) % 360.0) - 180.0

    ssh = smooth(ssh, lat)

    # ---- calibration: Orsi path if given, else one section per front
    def ssh_at(lon0, lat0):
        cols = np.abs(((lonw - lon0 + 180.0) % 360.0) - 180.0) <= 1.0
        prof = np.nanmean(ssh[:, cols], axis=1)
        ok = np.isfinite(prof)
        o = np.argsort(lat[ok])
        return float(np.interp(lat0, lat[ok][o], prof[ok][o]))

    iy_of = lambda la: np.abs(lat[:, None] - la[None, :]).argmin(axis=0)
    ix_of = lambda lo: np.abs(((lonw[:, None] - lo[None, :] + 180.0) % 360.0)
                              - 180.0).argmin(axis=0)
    values, used = {}, {}
    sr = None
    if CALIB == "sr09":
        branches, sr = sr09_calibration(ssh, lat, lon)
        for f, b in (("SAF", SAF_BRANCH), ("PF", PF_BRANCH), ("SBdy", SB_BRANCH)):
            values[f] = branches[b]
            used[f] = (f"SR09 {b}", None)
        # the STF has almost no SSH signature (SR09): only from an Orsi
        # file or an explicit STF_LAT; otherwise SAZ and STZ stay together
        if front_points("STF") is not None or "STF_LAT" in os.environ:
            fronts_left = ["STF"]
        else:
            fronts_left = []
            values["STF"] = np.inf
            used["STF"] = ("none: SAZ and STZ together, as SR09", None)
            print("  STF: not taken from SSH (SR09); the SAZ reaches north "
                  "to the domain edge. FRONTS_DIR with stf.txt, or STF_LAT, "
                  "separates the STZ", flush=True)
    else:
        fronts_left = list(FRONTS)
    for f in fronts_left:
        pts = front_points(f)
        sec_lon = STF_LON if f == "STF" else DP_LON
        if pts is not None:
            v = ssh[iy_of(pts[1]), ix_of(pts[0])]
            v = v[np.isfinite(v)]
            values[f] = float(np.median(v))
            sec_lat = front_lat_from_file(f) if f != "STF" else None
            dp = (f"; at {sec_lon:g}E alone: {ssh_at(sec_lon, sec_lat):+.3f} m"
                  if sec_lat is not None else "")
            used[f] = ("Orsi path, median of %d points" % v.size, None)
            print(f"  {f:5s} SSH {values[f]:+.3f} m (median along the whole "
                  f"Orsi {f}, {v.size} points{dp})", flush=True)
        else:
            values[f] = ssh_at(sec_lon, LATS[f])
            used[f] = ("default (check!)", LATS[f])
            print(f"  {f:5s} SSH {values[f]:+.3f} m at {sec_lon:g}E, "
                  f"{LATS[f]:.2f} deg (default, check!)", flush=True)
    if not (values["STF"] > values["SAF"] > values["PF"] > values["SBdy"]):
        sys.exit(f"the front SSH values are not ordered STF > SAF > PF > "
                 f"SBdy ({values}); check the front latitudes")
    print("  front latitude at   " + "".join(f"{l:>8g}E" for l in CHECK_LONS))
    for f in [f for f in FRONTS if np.isfinite(values[f])]:
        print(f"    {f:5s}            " + "".join(
            f"{contour_lat(ssh, lat, lonw, values[f], l):9.1f}"
            for l in CHECK_LONS), flush=True)

    zone = np.zeros(ssh.shape, "int8")
    if ZONE_SET == "acc3":
        # north of the SAF and the ACC band: their connected circumpolar
        # parts; south of the SB: every wet cell there (shelves, gyres)
        zone[circumpolar(ssh >= values["SAF"])] = 1
        zone[circumpolar((ssh < values["SAF"]) & (ssh >= values["SBdy"]))] = 2
        zone[wet & (ssh < values["SBdy"])] = 3
    else:
        # ---- zones, each its connected circumpolar band
        zone[circumpolar(ssh >= values["STF"])] = 4
        zone[circumpolar((ssh < values["STF"]) & (ssh >= values["SAF"]))] = 1
        zone[circumpolar((ssh < values["SAF"]) & (ssh >= values["PF"]))] = 2
        zone[circumpolar((ssh < values["PF"]) & (ssh >= values["SBdy"]))] = 3
    zone[~wet] = 0
    for k, n in ZONES.items():
        m = zone == k
        if not m.any():
            print(f"  {n:4s}        0 cells -- empty: check the {n} fronts",
                  flush=True)
            continue
        la = np.broadcast_to(lat[:, None], m.shape)[m]
        print(f"  {n:6s} {int(m.sum()):8d} cells, latitude "
              f"{np.percentile(la, 5):.1f} to {np.percentile(la, 95):.1f} "
              "(5-95 %)", flush=True)

    xr.Dataset(
        {"zone": (("YC", "XC"), zone,
                  {"flag_values": " ".join(str(k) for k in [0] + list(ZONES)),
                   "flag_meanings": " ".join(["none"] + list(ZONES.values())),
                   "zone_set": ZONE_SET}),
         "ssh_mean": (("YC", "XC"), ssh.astype("float32"))},
        coords={"YC": lat, "XC": lon},
        attrs={"source": os.path.basename(path),
               **{f"ssh_{f}": float(values[f]) for f in FRONTS
                  if np.isfinite(values[f])},
               "calib": CALIB, "saf_branch": SAF_BRANCH,
               "pf_branch": PF_BRANCH, "sb_branch": SB_BRANCH,
               "calibration": str({k: v[0] for k, v in used.items()}),
               "smooth_deg": SMOOTH_DEG, "DP_lon": DP_LON,
               "STF_lon": STF_LON}).to_netcdf(OUT + ".nc")
    print(f"wrote {OUT}.nc", flush=True)

    # ---- map: SSH zones, the 45-60S band, and the Orsi fronts if given
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    order = np.argsort(lonw)
    L, Z, S = lonw[order], zone[:, order], ssh[:, order]
    zcol = ({0: "#efefef", 1: "#f4a582", 2: "#4393c3", 3: "#d1e5f0",
             4: "#ffffff"} if ZONE_SET == "acc3" else
            {0: "#efefef", 1: "#f4a582", 2: "#92c5de", 3: "#2166ac",
             4: "#fee8c8"})
    cmap = matplotlib.colors.ListedColormap([zcol[k] for k in range(5)])
    fcol = {"STF": "#e08214", "SAF": "#b2182b", "PF": "#1b7837",
            "SBdy": "#542788"}
    proj = ccrs.SouthPolarStereo()
    ncol = 3 if sr is not None else 2
    fig = plt.figure(figsize=(7.5 * ncol, 7.6), dpi=200)

    def base(ax):
        ax.set_extent([-180, 180, -80, -30], crs=ccrs.PlateCarree())
        gl = ax.gridlines(lw=0.4, color="0.5", alpha=0.6, ylocs=[-70, -60, -50, -40],
                          xlocs=range(-180, 181, 60), draw_labels=False)
        for la in (-70, -60, -50, -40):
            ax.text(0, la, f"{-la}S", fontsize=8, color="0.35", ha="center",
                    va="bottom", transform=ccrs.PlateCarree())

    def fronts(ax):
        for f in [f for f in ZFRONTS if np.isfinite(values[f])]:
            ax.contour(L, lat, S, levels=[values[f]], colors=fcol[f],
                       linewidths=1.1, transform=ccrs.PlateCarree())

    ax = fig.add_subplot(1, ncol, 1, projection=proj)
    base(ax)
    ax.pcolormesh(L, lat, np.ma.masked_where(~np.isfinite(S), Z), cmap=cmap,
                  vmin=-0.5, vmax=4.5, shading="nearest", rasterized=True,
                  transform=ccrs.PlateCarree())
    fronts(ax)
    ax.plot([DP_LON, DP_LON], [-63, -54], color="k", lw=2.2,
            transform=ccrs.PlateCarree())
    ax.set_title("a   regions from the time-mean B-SOSE SSH",
                 loc="left", fontsize=13)
    handles = ([Patch(color=zcol[k], label=ZLABEL[k]) for k in ZONES]
               + ([] if ZONE_SET == "acc3" else
                  [Patch(color=zcol[0], label="outside / south of SBdy")])
               + [Line2D([], [], color=fcol[f], lw=1.6,
                         label=f"{f}: SSH {values[f]:+.2f} m") for f in ZFRONTS
                  if np.isfinite(values[f])]
               + [Line2D([], [], color="k", lw=2.2,
                         label=f"Drake Passage calibration ({-DP_LON:g}W)")])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.02),
              ncol=2, fontsize=9, frameon=False)

    ax = fig.add_subplot(1, ncol, 2, projection=proj)
    base(ax)
    band = ((lat > -60) & (lat <= -45))[:, None] & np.isfinite(S)
    ax.pcolormesh(L, lat, np.ma.masked_where(~np.isfinite(S), band.astype(float)),
                  cmap=matplotlib.colors.ListedColormap(["#efefef", "#bdbdbd"]),
                  shading="nearest", rasterized=True,
                  transform=ccrs.PlateCarree())
    fronts(ax)
    handles = [Patch(color="#bdbdbd", label="45-60S band")]
    for f in ZFRONTS:
        pts = front_points(f)
        if pts is None:
            continue
        ax.plot(pts[0], pts[1], ".", ms=1.0, color=fcol[f], alpha=0.8,
                transform=ccrs.PlateCarree())
        handles.append(Line2D([], [], color=fcol[f], ls=":", lw=2,
                              label=f"Orsi et al. (1995) {f}"))
    handles += [Line2D([], [], color=fcol[f], lw=1.6, label=f"{f} SSH contour")
                for f in ZFRONTS if np.isfinite(values[f])]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.02),
              ncol=2, fontsize=9, frameon=False)
    ax.set_title("b   comparison: 45-60S band"
                 + (" and Orsi et al. (1995) fronts (dotted)" if load_fronts() else ""),
                 loc="left", fontsize=13)
    if sr is not None:
        mid, G, sc, off = sr
        ax = fig.add_subplot(1, ncol, 3)
        ax.plot(G, mid, color="k", lw=1.4)
        for n, v in SR09.items():
            h = sc * v + off
            f = ("SAF" if n == SAF_BRANCH else "PF" if n == PF_BRANCH
                 else "SBdy" if n == SB_BRANCH else None)
            ax.axhline(h, color=fcol[f] if f else "0.6",
                       lw=1.6 if f else 0.8, ls="-" if f else "--")
            ax.text(ax.get_xlim()[1] if False else np.nanmax(G) * 1.02, h,
                    n, va="center", fontsize=8,
                    color=fcol[f] if f else "0.4")
        ax.set_xlim(0, np.nanmax(G) * 1.25)
        ax.set_ylim(mid.min(), mid.max())
        ax.set_xlabel("mean |grad SSH| (m per 100 km)", fontsize=11)
        ax.set_ylabel("B-SOSE time-mean SSH (m)", fontsize=11)
        ax.set_title(f"c   Sokolov & Rintoul (2009) labels on B-SOSE\n"
                     f"SSH = {sc:.2f} x label {off:+.2f} m",
                     loc="left", fontsize=12)
        ax.grid(alpha=0.3)
    try:
        from cartopy.io import shapereader
        shapereader.natural_earth(resolution="110m", category="physical",
                                  name="coastline")
        for a in fig.axes:
            if hasattr(a, "coastlines"):
                a.coastlines(resolution="110m", lw=0.5)
    except Exception:
        pass
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and .png", flush=True)


if __name__ == "__main__":
    main()
