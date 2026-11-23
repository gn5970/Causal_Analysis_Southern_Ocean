"""Download ESA OC-CCI monthly chlorophyll (Level 3, merged sensors) for the
Southern Ocean and average it to 0.25 deg (mean of valid pixels in 6x6 native
blocks). Run on a Levante LOGIN node (compute nodes have no internet).
Restartable: finished months are kept in CACHE.
OUTPUT /work/bb1596/b384036/b384036/causality/ESA_OCCCI_v6_monthly_L3_south30S_0p25deg_mean_1998_2025.nc
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr

CAUS = "/work/bb1596/b384036/b384036/causality"
ESA_URL = os.environ.get("ESA_URL", "https://www.oceancolour.org/thredds/dodsC/CCI_ALL-v6.0-MONTHLY")
YEAR0 = int(os.environ.get("YEAR0", 1998))
YEAR1 = int(os.environ.get("YEAR1", 2025))
LAT_NORTH = float(os.environ.get("LAT_NORTH", -30.0))
K = int(os.environ.get("K", 6))
MIN_VALID = int(os.environ.get("MIN_VALID", 1))
OUT = os.environ.get("OUT", f"{CAUS}/ESA_OCCCI_v6_monthly_L3_south30S_0p25deg_mean_{YEAR0}_{YEAR1}.nc")
CACHE = os.environ.get("CACHE", f"{CAUS}/esa_occci_cache_0p25deg")
RETRIES = 5


def block_mean(a, k):
    ny, nx = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:ny, :nx].reshape(ny // k, k, nx // k, k)
    ok = np.isfinite(b)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n >= MIN_VALID, s / n, np.nan)
    return m.astype("float32"), n.astype("int16")


def main():
    os.makedirs(CACHE, exist_ok=True)
    print(f"opening {ESA_URL}", flush=True)
    ds = xr.open_dataset(ESA_URL, decode_times=True)
    v = ds["chlor_a"]
    lat = ds["lat"].values.astype("float64")
    lon = ds["lon"].values.astype("float64")
    t = pd.to_datetime(ds["time"].values)
    print(f"  server: {t.size} months, {t[0]:%Y-%m} to {t[-1]:%Y-%m}; grid {lat.size} x {lon.size}", flush=True)
    for k in ("title", "product_version", "processing_level", "id", "history"):
        if k in ds.attrs:
            print(f"  {k}: {str(ds.attrs[k])[:200]}")
    if t[-1].year < YEAR1:
        print(f"  NOTE: the server ends {t[-1]:%Y-%m}; the file will stop there, not in {YEAR1}", flush=True)

    rows = np.flatnonzero(lat <= LAT_NORTH)
    r0, r1 = rows.min(), rows.max() + 1
    r1 = r0 + ((r1 - r0) // K) * K
    lat_sub = lat[r0:r1]
    lat_b = lat_sub[: (lat_sub.size // K) * K].reshape(-1, K).mean(1)
    nlon = (lon.size // K) * K
    lon_b = lon[:nlon].reshape(-1, K).mean(1)
    lon_b = ((lon_b + 180.0) % 360.0) - 180.0
    print(f"  rows {r0}..{r1 - 1} ({lat_sub.min():.2f} to {lat_sub.max():.2f}), "
          f"-> {lat_b.size} x {lon_b.size} cells of {K} x {K} pixels", flush=True)

    sel = np.flatnonzero((t.year >= YEAR0) & (t.year <= YEAR1))
    for n_done, i in enumerate(sel, 1):
        f = os.path.join(CACHE, f"{t[i]:%Y%m}.npz")
        if os.path.exists(f):
            continue
        for attempt in range(1, RETRIES + 1):
            try:
                a = v.isel(time=i, lat=slice(r0, r1)).values.astype("float32")
                break
            except Exception as exc:
                print(f"  {t[i]:%Y-%m} attempt {attempt} failed: {exc}", flush=True)
                if attempt == RETRIES:
                    sys.exit("giving up; run again later, finished months are kept")
                time.sleep(30 * attempt)
        a[~(np.isfinite(a) & (a > 0) & (a < 1e3))] = np.nan
        m, n = block_mean(a, K)
        np.savez(f + ".part.npz", chl=m, n=n)
        os.replace(f + ".part.npz", f)
        print(f"  {t[i]:%Y-%m} ({n_done}/{sel.size}): native {100 * np.isfinite(a).mean():.0f}% valid, "
              f"0.25deg {100 * np.isfinite(m).mean():.0f}% of cells", flush=True)

    chl = np.stack([np.load(os.path.join(CACHE, f"{t[i]:%Y%m}.npz"))["chl"] for i in sel])
    nv = np.stack([np.load(os.path.join(CACHE, f"{t[i]:%Y%m}.npz"))["n"] for i in sel])
    oy, ox = np.argsort(lat_b), np.argsort(lon_b)
    chl, nv = chl[:, oy][:, :, ox], nv[:, oy][:, :, ox]
    times = pd.DatetimeIndex([pd.Timestamp(t[i].year, t[i].month, 1) for i in sel])
    out = xr.Dataset(
        {"chlor_a": (("time", "lat", "lon"), chl,
                     {"units": "mg m-3", "long_name": "chlorophyll-a, mean of valid 4 km pixels in the cell"}),
         "n_valid": (("time", "lat", "lon"), nv,
                     {"long_name": f"valid 4 km pixels in the cell (of {K * K})"})},
        coords={"time": times, "lat": lat_b[oy], "lon": lon_b[ox]},
        attrs={"source": ESA_URL,
               "product": "ESA Ocean Colour CCI, monthly, Level 3 merged sensors (L3S), not gap-filled",
               "processing": f"south of {LAT_NORTH} deg; mean of valid pixels in {K}x{K} native (1/24 deg) "
                             f"blocks = 0.25 deg; cell missing if fewer than {MIN_VALID} valid pixels; "
                             "no temporal gap filling",
               "history": f"{pd.Timestamp.now():%Y-%m-%d} esa_occci_download.py"})
    tmp = OUT + ".part"
    out.to_netcdf(tmp, encoding={"chlor_a": {"zlib": True, "complevel": 4},
                                 "n_valid": {"zlib": True, "complevel": 4}})
    os.replace(tmp, OUT)
    cov = np.isfinite(chl)
    print(f"\nwrote {OUT}\n  {times[0]:%Y-%m} to {times[-1]:%Y-%m}, {times.size} months, "
          f"{lat_b.size} x {lon_b.size}; {100 * cov.mean():.1f}% of cell-months with data")
    mon = times.month
    print("  coverage by month: " + " ".join(f"{100 * cov[mon == m].mean():.0f}" for m in range(1, 13))
          + "  (J..D, %)")


if __name__ == "__main__":
    main()
