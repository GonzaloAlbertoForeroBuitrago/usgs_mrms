from __future__ import annotations

import gzip
import random
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import zarr
from numcodecs import Blosc

from .config import PipelineConfig
from .exceptions import MissingOptionalDependency
from .geo import build_mask_and_lonlat_from_basin


def _require_gdal() -> object:
    try:
        from osgeo import gdal  # type: ignore
    except Exception as e:
        raise MissingOptionalDependency(
            "GDAL (osgeo) is required to read MRMS GRIB2. "
            "Install via conda-forge: conda install -c conda-forge gdal"
        ) from e
    gdal.UseExceptions()
    return gdal


def as_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def hours_from_windows(df: pd.DataFrame, start_col: str, end_col: str) -> pd.DatetimeIndex:
    if df.empty:
        return pd.DatetimeIndex([], tz="UTC")

    s_all = pd.to_datetime(df[start_col], errors="coerce").dropna().map(as_utc)
    e_all = pd.to_datetime(df[end_col], errors="coerce").dropna().map(as_utc)

    ranges: list[pd.DatetimeIndex] = []
    for s, e in zip(s_all, e_all):
        if e < s:
            continue
        ranges.append(pd.date_range(s.floor("h"), e.floor("h"), freq="h", tz="UTC"))

    if not ranges:
        return pd.DatetimeIndex([], tz="UTC")

    times = ranges[0]
    if len(ranges) > 1:
        times = times.append(ranges[1:])
    return pd.DatetimeIndex(times).unique().sort_values()


def radaronly_aws_url(cfg: PipelineConfig, ts) -> str:
    ts = as_utc(ts).floor("h")
    d = ts.strftime("%Y%m%d")
    hms = ts.strftime("%H%M%S")
    fn = f"MRMS_RadarOnly_QPE_01H_00.00_{d}-{hms}.grib2.gz"
    return f"{cfg.aws_radaronly}/{d}/{fn}"


def radaronly_mt_url(cfg: PipelineConfig, ts) -> str:
    ts = as_utc(ts).floor("h")
    dpath = f"{ts.year:04d}/{ts.month:02d}/{ts.day:02d}"
    ymd = ts.strftime("%Y%m%d")
    hms = ts.strftime("%H%M%S")
    fn = f"RadarOnly_QPE_01H_00.00_{ymd}-{hms}.grib2.gz"
    return f"{cfg.mtarchive}/{dpath}/mrms/ncep/RadarOnly_QPE_01H/{fn}"


def robust_get(sess: requests.Session, url: str, timeout: int, max_tries: int = 6) -> Optional[requests.Response]:
    for attempt in range(1, max_tries + 1):
        try:
            return sess.get(url, timeout=timeout)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
        ):
            wait = min(60.0, (2**attempt) + random.uniform(0, 1.5))
            time.sleep(wait)
    return None


def first_available_radaronly(cfg: PipelineConfig, times: pd.DatetimeIndex, max_checks: int = 80, widen: int = 48) -> tuple[pd.Timestamp, bytes]:
    sess = requests.Session()
    sess.headers.update(cfg.http_headers_mrms)

    def try_one(ts0: pd.Timestamp) -> Optional[tuple[pd.Timestamp, bytes]]:
        ts0 = as_utc(ts0)

        u_aws = radaronly_aws_url(cfg, ts0)
        r = robust_get(sess, u_aws, cfg.http_timeout_mrms, max_tries=3)
        if r is not None and r.status_code == 200 and r.content[:2] == b"\x1f\x8b":
            return ts0, r.content

        u_mt = radaronly_mt_url(cfg, ts0)
        r = robust_get(sess, u_mt, cfg.http_timeout_mrms, max_tries=3)
        if r is not None and r.status_code == 200 and r.content[:2] == b"\x1f\x8b":
            return ts0, r.content

        return None

    try:
        for ts in times[:max_checks]:
            got = try_one(ts)
            if got is not None:
                return got
            time.sleep(random.uniform(*cfg.sleep_between))

        if len(times) > 0:
            mid = times[int(len(times) / 2)]
            probe = pd.date_range(mid - pd.Timedelta(hours=widen), mid + pd.Timedelta(hours=widen), freq="h", tz="UTC")
            for ts in probe:
                got = try_one(ts)
                if got is not None:
                    return got
                time.sleep(random.uniform(*cfg.sleep_between))

        raise RuntimeError("Could not find a RadarOnly hour for georeferencing.")
    finally:
        sess.close()


def looks_like_zarr_group(path: Path) -> bool:
    return path.exists() and path.is_dir() and ((path / ".zgroup").exists() or (path / "zarr.json").exists())


def init_zarr(times: pd.DatetimeIndex, out_path: Path) -> zarr.hierarchy.Group:
    out_path = Path(out_path)

    if out_path.exists() and not looks_like_zarr_group(out_path):
        shutil.rmtree(out_path)

    if looks_like_zarr_group(out_path):
        return zarr.open_group(out_path.as_posix(), mode="r+", zarr_version=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    idx = pd.DatetimeIndex(times)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    times_np = idx.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    root = zarr.open_group(out_path.as_posix(), mode="w", zarr_version=2)
    root.create("time", shape=(len(times_np),), chunks=(len(times_np),), dtype="datetime64[ns]", compressor=None)
    root["time"][:] = times_np
    root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    return root


def ensure_pixel_arrays(cfg: PipelineConfig, root: zarr.hierarchy.Group, mask: dict) -> None:
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    rows = mask["rows"]
    cols = mask["cols"]
    lon = mask["lon_pix"]
    lat = mask["lat_pix"]
    n_pixels = rows.size

    if "rain" not in root:
        root.create(
            "rain",
            shape=(root["time"].shape[0], n_pixels),
            chunks=(min(cfg.time_chunk, root["time"].shape[0]), min(cfg.pixel_chunk, n_pixels)),
            dtype=cfg.dtype,
            compressor=compressor,
            fill_value=np.nan,
        )
        root["rain"].attrs["_ARRAY_DIMENSIONS"] = ["time", "pixel"]

    for name, arr, dims in [
        ("row", rows.astype(np.int32), ["pixel"]),
        ("col", cols.astype(np.int32), ["pixel"]),
        ("lon", lon.astype(cfg.dtype), ["pixel"]),
        ("lat", lat.astype(cfg.dtype), ["pixel"]),
    ]:
        if name not in root:
            root.create(
                name,
                shape=(n_pixels,),
                chunks=(min(cfg.pixel_chunk, n_pixels),),
                dtype=arr.dtype,
                compressor=compressor if name in ("lon", "lat") else None,
            )
        root[name][:] = arr
        root[name].attrs["_ARRAY_DIMENSIONS"] = dims


def resume_fill_rain(cfg: PipelineConfig, out_path: Path, mask: dict, missing_csv: Path) -> Tuple[int, int]:
    gdal = _require_gdal()

    root = zarr.open_group(out_path.as_posix(), mode="r+", zarr_version=2)
    ensure_pixel_arrays(cfg, root, mask)

    rain = root["rain"]
    tarr = root["time"][:]  # datetime64[ns]
    times = pd.to_datetime(tarr)  # naive; interpret as UTC hours
    n = len(times)

    done = np.zeros(n, dtype=bool)
    block = 256
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        chunk = rain[i0:i1, :]
        done[i0:i1] = np.isfinite(chunk).any(axis=1)

    missing_idx = np.where(~done)[0]
    if missing_idx.size == 0:
        return 0, 0
    start_idx = int(missing_idx[0])

    sess = requests.Session()
    sess.headers.update(cfg.http_headers_mrms)

    missing_rows: list[tuple[str, str, str]] = []
    aws_ok = 0
    mt_ok = 0

    rows = mask["rows"]
    cols = mask["cols"]

    try:
        for i in range(start_idx, n):
            if np.isfinite(rain[i, :]).any():
                continue

            ts = pd.Timestamp(times[i]).tz_localize("UTC")

            url_aws = radaronly_aws_url(cfg, ts)
            r = robust_get(sess, url_aws, cfg.http_timeout_mrms, max_tries=6)
            url = url_aws
            src = "aws"

            if (r is None) or (r.status_code != 200):
                url_mt = radaronly_mt_url(cfg, ts)
                r = robust_get(sess, url_mt, cfg.http_timeout_mrms, max_tries=6)
                url = url_mt
                src = "mt"

            if (r is None) or (r.status_code != 200):
                rain[i, :] = np.nan
                code = "no_response" if r is None else f"http_{r.status_code}"
                missing_rows.append((str(ts), url, code))
            else:
                if src == "aws":
                    aws_ok += 1
                else:
                    mt_ok += 1

                raw = gzip.decompress(r.content)
                vs = f"/vsimem/mrms_{i}.grib2"
                gdal.FileFromMemBuffer(vs, raw)

                ds = gdal.Open(vs)
                if ds is None:
                    rain[i, :] = np.nan
                    missing_rows.append((str(ts), url, "gdal_open_failed"))
                else:
                    arr2d = ds.ReadAsArray()
                    rain[i, :] = arr2d[rows, cols].astype(cfg.dtype, copy=False)

                ds = None
                gdal.Unlink(vs)

            if (i + 1) % cfg.debug_every_n == 0 or (i + 1) == n:
                print(f"  [rain] {i+1}/{n} aws_ok={aws_ok} mt_ok={mt_ok} missing_new={len(missing_rows)}")

            time.sleep(random.uniform(*cfg.sleep_between))
    finally:
        sess.close()

    if missing_rows:
        miss_df = pd.DataFrame(missing_rows, columns=["time_utc", "url", "reason"])
        if missing_csv.exists():
            old = pd.read_csv(missing_csv)
            miss_df = pd.concat([old, miss_df], ignore_index=True).drop_duplicates(subset=["time_utc"], keep="last")
        miss_df.to_csv(missing_csv, index=False)

    try:
        zarr.consolidate_metadata(out_path.as_posix())
    except Exception:
        pass

    return aws_ok, mt_ok


def build_zarr_radaronly_from_windows(
    cfg: PipelineConfig,
    windows_csv: Path,
    basin_json: Path,
    out_zarr: Path,
    missing_csv: Path,
) -> Tuple[int, int, int]:
    top = pd.read_csv(windows_csv, parse_dates=["date_peak", "start_rain", "end_rain"])
    times = hours_from_windows(top, "start_rain", "end_rain")
    if len(times) == 0:
        return 0, 0, 0

    _, gz_bytes = first_available_radaronly(cfg, times, max_checks=80, widen=48)
    mask = build_mask_and_lonlat_from_basin(basin_json, gz_bytes, dtype=cfg.dtype)

    init_zarr(times, out_zarr)
    aws_ok, mt_ok = resume_fill_rain(cfg, out_zarr, mask, missing_csv)
    return int(len(times)), int(mask["rows"].size), int(aws_ok + mt_ok)