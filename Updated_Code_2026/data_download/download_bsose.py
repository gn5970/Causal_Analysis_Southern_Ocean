"""Fetch the B-SOSE monthly fields that obs_causal_pipeline.py reads.

    python download_bsose.py --list         what the server has, nothing fetched
    python download_bsose.py                Chl, Theta, Salt, MLD, oceTAUX, oceTAUY
    python download_bsose.py SSH            named fields only
    python download_bsose.py --dry-run      what would be fetched, and how

The files come from the B-SOSE server (BSOSE_URL, iteration 156 by
default) and land in <PAPER5_DIR>/bsose, where the pipeline looks for
them (BSOSE_DIR overrides both). Run it where there is an outbound
route: a login node or the laptop, not a compute node.

SIZE. The 2D fields (MLD, stresses, SSH) are well under a gigabyte each
and are downloaded whole. The 3D fields (Chl, Theta, Salt) are tens of
gigabytes each at all 52 levels, and the pipeline uses the top 50 m. So
for those the script first tries to read ONLY the top DEPTH_MAX metres
straight from the server (HTTP range requests; needs `pip install
fsspec aiohttp h5netcdf`). If the server or the file format does not
allow that, it says so, downloads the whole file into a staging
directory, cuts it to the top DEPTH_MAX metres and deletes the full
copy. Either way what is kept is the same small file, with the name the
server gave it, so the months can be read off the name
(..._2013to2024_...).

Finished files are skipped, and an interrupted whole-file download
resumes where it stopped.

This script has not been run against the real server from where it was
written: the names it expects are matched as whole words (Chl, Theta,
...) and `--list` shows what is actually there. If a field is reported
as not found, look at the listing and name the file's word on the
command line, or adjust FIELDS below.
"""

import os
import re
import sys
import urllib.request
from pathlib import Path

BSOSE_URL = os.environ.get(
    "BSOSE_URL", "https://sose.ucsd.edu/SO6/ITER156/MONTHLY/")
DATA_DIR = Path(os.environ.get(
    "PAPER5_DIR", "/work/bb1596/b384036/b384036/causality"))
BSOSE_DIR = Path(os.environ.get("BSOSE_DIR", DATA_DIR / "bsose"))
STAGING = BSOSE_DIR / "_staging"
DEPTH_MAX = float(os.environ.get("DEPTH_MAX", 60.0))          # keep levels down to here; the pipeline uses 50

# word in the file name: is it a 3D field?
FIELDS = {"Llim": True, "FElim": True,   # limitation factors, 3D
          "Chl": True, "Theta": True, "Salt": True,
          "MLD": False, "oceTAUX": False, "oceTAUY": False, "SSH": False,
          # the model's 2D chlorophyll diagnostic, 0.75 GB against 38 GB
          # for the 3D field; used by the pipeline with BSOSE_CHL=CHLsat
          "CHLsat": False}
DEFAULT = ["Chl", "Theta", "Salt", "MLD", "oceTAUX", "oceTAUY", "SSH"]
ALIASES = {"SSH": ["SSH", "ETAN"]}


def words(name: str):
    return re.split(r"[_.\-]", name.lower())


def parse_index(html: str):
    """File names in a directory-listing page."""
    names = re.findall(r'href="([^"?/][^"]*\.(?:nc|h5|nc4))"', html,
                       flags=re.IGNORECASE)
    return sorted(dict.fromkeys(names))


def listing():
    req = urllib.request.Request(BSOSE_URL,
                                 headers={"User-Agent": "download_bsose"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return parse_index(r.read().decode("utf-8", "replace"))


def remote_size(url: str):
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "download_bsose"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def pick(names, word: str):
    # Monthly MEANS only. The server also holds ..._monthlySnaps.nc for
    # some fields: instantaneous states at the month boundaries, which
    # are a different quantity and must not be mistaken for the means.
    hits = [n for n in names if word.lower() in words(n)
            and "snap" not in n.lower()]
    monthly = [n for n in hits if "monthly" in n.lower()]
    return (monthly or hits or [None])[0]


def download_whole(url: str, dest: Path) -> None:
    """Whole file, resuming a partial one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    total = remote_size(url)
    if total and have >= total:
        part.replace(dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "download_bsose"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:        # server ignored the range
            have = 0
        mode = "ab" if have else "wb"
        done = have
        with open(part, mode) as out:
            while chunk := r.read(1 << 22):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e9:6.2f} of {total / 1e9:.2f} GB",
                          end="", flush=True)
    print()
    part.replace(dest)


def depth_dim(ds):
    return next((d for d in ds.dims if d.upper().startswith("Z")), None)


def write_top(ds, out: Path) -> None:
    """Write the levels above DEPTH_MAX of `ds` to `out`, a year at a time.

    The source is opened undecoded, so values, time stamps and
    attributes are copied exactly as they are. A year at a time, because
    twelve years of even eight levels do not fit in memory at once.
    """
    import numpy as np
    import netCDF4

    zdim = depth_dim(ds)
    if zdim is not None:
        zvar = ds[zdim].values if zdim in ds.variables else None
        if zvar is None:
            raise RuntimeError(f"no coordinate for the depth dimension {zdim}")
        keep = np.flatnonzero(np.abs(zvar) <= DEPTH_MAX)
        if keep.size == 0:
            raise RuntimeError(f"no level within {DEPTH_MAX:g} m")
        zdims = [d for d in ds.dims if d.upper().startswith("Z")]
        # every vertical dimension of the same length is cut alike
        sel = {d: keep for d in zdims if ds.sizes[d] == ds.sizes[zdim]}
        ds = ds.isel(sel)
    tdim = "time" if "time" in ds.dims else None
    tmp = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    if tdim is None:
        ds.load().to_netcdf(tmp)
        tmp.replace(out)
        return
    nt = ds.sizes[tdim]
    tvars = [v for v in ds.variables if tdim in ds[v].dims]
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars
           if ds[v].ndim >= 3}
    start = 0
    if tmp.exists():                 # resume: redo the last year written
        try:
            with netCDF4.Dataset(tmp) as nc:
                start = max(0, nc.dimensions[tdim].size // 12 * 12 - 12)
            print(f"    resuming {tmp.name} at month {start} of {nt}", flush=True)
        except Exception:
            tmp.unlink()
            start = 0
    for t0 in range(start, nt, 12):
        chunk = ds.isel({tdim: slice(t0, t0 + 12)}).load()
        if t0 == 0:
            chunk.to_netcdf(tmp, unlimited_dims=[tdim], encoding=enc)
        else:
            with netCDF4.Dataset(tmp, "a") as nc:
                nc.set_auto_maskandscale(False)
                n0 = t0
                for v in tvars:
                    axis = ds[v].dims.index(tdim)
                    idx = [slice(None)] * ds[v].ndim
                    idx[axis] = slice(n0, n0 + chunk.sizes[tdim])
                    nc.variables[v][tuple(idx)] = chunk[v].values
        print(f"\r    months {min(t0 + 12, nt)} of {nt}", end="", flush=True)
    print()
    tmp.replace(out)


def subset_remote(url: str, out: Path) -> bool:
    """Read only the top levels over HTTP. False if that is not possible."""
    try:
        import fsspec
        import xarray as xr
        import h5netcdf  # noqa: F401  (the engine xarray is asked for)
    except ImportError as exc:
        print(f"    reading the top levels remotely needs {exc.name}; "
              "falling back to the whole file")
        return False
    import time
    for attempt in range(1, 7):      # a dropped connection resumes the .part
        try:
            with fsspec.open(url, "rb", block_size=8 << 20).open() as f:
                with xr.open_dataset(f, engine="h5netcdf",
                                     decode_cf=False) as ds:
                    write_top(ds, out)
            return True
        except Exception as exc:
            print(f"\n    remote read failed ({type(exc).__name__}: {exc}); "
                  f"attempt {attempt} of 6", flush=True)
            time.sleep(30 * attempt)
    print("    could not read it remotely; falling back to the whole file "
          "(the .part is kept and resumed)")
    return False


def subset_local(path: Path, out: Path) -> None:
    import xarray as xr
    try:
        ds = xr.open_dataset(path, decode_cf=False)
    except Exception:
        ds = xr.open_dataset(path, engine="h5netcdf", decode_cf=False)
    with ds:
        write_top(ds, out)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    print(f"server       {BSOSE_URL}")
    print(f"destination  {BSOSE_DIR}")
    try:
        names = listing()
    except Exception as exc:
        print(f"Could not read the server's listing: {type(exc).__name__}: "
              f"{exc}")
        return 1
    if "--list" in sys.argv:
        for n in names:
            size = remote_size(BSOSE_URL + n)
            print(f"  {size / 1e9:8.2f} GB  {n}" if size else f"  {'?':>11s}  {n}")
        print(f"{len(names)} files")
        return 0

    failed = []
    for word in (args or DEFAULT):
        name = next((n for w in ALIASES.get(word, [word])
                     if (n := pick(names, w)) is not None), None)
        if name is None:
            print(f"  {word}: no file with that word in its name. "
                  "Run with --list to see what is there.")
            failed.append(word)
            continue
        three_d = FIELDS.get(word, "--3d" in sys.argv)
        out = BSOSE_DIR / name
        if out.exists():
            print(f"  {word}: {name} present, skipped")
            continue
        url = BSOSE_URL + name
        size = remote_size(url)
        gb = f"{size / 1e9:.1f} GB" if size else "size unknown"
        if dry:
            print(f"  {word}: would fetch {name} ({gb} on the server; "
                  + (f"kept: top {DEPTH_MAX:g} m only)" if three_d
                     else "kept whole)"))
            continue
        print(f"  {word}: {name} ({gb} on the server)", flush=True)
        try:
            if three_d:
                print(f"    reading the top {DEPTH_MAX:g} m from the "
                      "server...", flush=True)
                if not subset_remote(url, out):
                    staged = STAGING / name
                    if not staged.exists():
                        download_whole(url, staged)
                    print(f"    cutting to the top {DEPTH_MAX:g} m...")
                    subset_local(staged, out)
                    staged.unlink()
            else:
                download_whole(url, out)
            print(f"    kept {out.name}, {out.stat().st_size / 1e9:.2f} GB")
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed.append(word)
    if dry:
        return 0
    if failed:
        print(f"\nNot fetched: {failed}. Rerun to retry; finished files "
              "are skipped.")
        return 1
    print("\nDone. Check with: ANALYSIS=bsose python obs_causal_pipeline.py "
          "--check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
