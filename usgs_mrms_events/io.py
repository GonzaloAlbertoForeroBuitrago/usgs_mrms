from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def now_utc_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def date_windows(start: str, end: str, window_days: int) -> list[tuple[str, str]]:
    dt_start = datetime.strptime(start, "%Y-%m-%d")
    dt_end = datetime.strptime(end, "%Y-%m-%d")
    out: list[tuple[str, str]] = []
    cur = dt_start
    delta = timedelta(days=window_days)
    while cur <= dt_end:
        w_end = min(cur + delta, dt_end)
        out.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end + timedelta(days=1)
    return out


def resolve_iana_timezone(lon: Optional[float], lat: Optional[float], fallback_abbrev: Optional[str] = None) -> str:
    """Resolve IANA tz from lon/lat (timezonefinder), else fallback from abbrev."""
    try:
        from timezonefinder import TimezoneFinder  # type: ignore

        if (lon is not None) and (lat is not None):
            tf = TimezoneFinder()
            tz = tf.timezone_at(lat=float(lat), lng=float(lon))
            if tz:
                return tz
    except Exception:
        pass

    ab = (fallback_abbrev or "").upper().strip()
    map_abbrev = {
        "CST": "America/Chicago",
        "CDT": "America/Chicago",
        "MST": "America/Denver",
        "MDT": "America/Denver",
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles",
        "AKST": "America/Anchorage",
        "AKDT": "America/Anchorage",
        "HST": "Pacific/Honolulu",
    }
    return map_abbrev.get(ab, "UTC")


def load_stage_with_utc_local(parquet_fp: Path, tz_iana: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_fp)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if ZoneInfo is not None and tz_iana and tz_iana.upper() != "UTC":
        try:
            df["datetime_local"] = df["datetime"].dt.tz_convert(ZoneInfo(tz_iana))
        except Exception:
            df["datetime_local"] = df["datetime"]
    else:
        df["datetime_local"] = df["datetime"]

    return df


INVENTORY_COLUMNS = [
    "monitoring_location_number",
    "monitoring_location_name",
    "state_name",
    "county_name",
    "altitude",
    "contributing_drainage_area",
    "time_zone_abbreviation",
    "id",
    "hydrologic_unit_code",
    "lon",
    "lat",
    "basin_json_path",
    "stage_parquet_path",
    "site_meta_json_path",
    "events_top_csv",
    "events_windows_csv",
    "rain_zarr_path",
    "timestamp_utc",
]


def append_inventory_row(inventory_csv: Path, row: dict) -> None:
    inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not inventory_csv.exists()
    with inventory_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INVENTORY_COLUMNS})
