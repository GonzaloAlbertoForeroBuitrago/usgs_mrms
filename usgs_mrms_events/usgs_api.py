from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests

from .config import PipelineConfig
from .io import date_windows, now_utc_iso
from .paths import normalize_site_id


def get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 60,
    headers: Optional[dict] = None,
    max_retries: int = 5,
) -> dict:
    """
    Send a GET request and return JSON.
    If USGS responds with 429, wait and retry a few times.
    """
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=timeout)

        # Success
        if r.status_code == 200:
            return r.json()

        # Rate limit: wait and retry
        if r.status_code == 429:
            wait_s = min(60, 2 ** attempt)
            print(f"[USGS 429] attempt {attempt}/{max_retries}, waiting {wait_s}s")
            time.sleep(wait_s)
            continue

        # Any other HTTP error
        r.raise_for_status()

    raise RuntimeError(f"USGS request failed after {max_retries} retries: {url}")


def fetch_monitoring_location(cfg: PipelineConfig, site_id: str) -> dict:
    sid = normalize_site_id(site_id)
    params = {"f": "json", "filter": f"agency_code='USGS' AND monitoring_location_number='{sid}'"}
    data = get_json(cfg.ogc_monitoring_locations, params=params, timeout=cfg.http_timeout_usgs, headers=cfg.http_headers_usgs)
    feats = data.get("features", [])
    if not feats:
        raise RuntimeError(f"Monitoring location not found for site_id={sid}")
    return feats[0]


def extract_inventory_row(feature: dict) -> dict:
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", [None, None])

    lon = coords[0] if isinstance(coords, list) and len(coords) >= 2 else None
    lat = coords[1] if isinstance(coords, list) and len(coords) >= 2 else None

    return {
        "monitoring_location_number": str(props.get("monitoring_location_number") or ""),
        "monitoring_location_name": props.get("monitoring_location_name"),
        "state_name": props.get("state_name"),
        "county_name": props.get("county_name"),
        "altitude": props.get("altitude"),
        "contributing_drainage_area": props.get("contributing_drainage_area"),
        "time_zone_abbreviation": props.get("time_zone_abbreviation"),
        "id": props.get("id"),
        "hydrologic_unit_code": props.get("hydrologic_unit_code"),
        "lon": lon,
        "lat": lat,
        "uses_daylight_savings": props.get("uses_daylight_savings"),
    }


def download_basin_json(cfg: PipelineConfig, site_id: str, out_json: Path, done_marker: Path, *, overwrite: bool) -> None:
    if done_marker.exists() and out_json.exists() and not overwrite:
        return
    if out_json.exists() and not overwrite and not done_marker.exists():
        done_marker.write_text(now_utc_iso(), encoding="utf-8")
        return

    sid = normalize_site_id(site_id)
    url = f"{cfg.basin_gages_endpoint}/{sid}"
    params = {"f": "json"}

    try:
        data = get_json(url, params=params, timeout=cfg.http_timeout_usgs, headers=cfg.http_headers_usgs)
    except requests.exceptions.HTTPError as e:
        if getattr(e.response, "status_code", None) == 404:
            raise RuntimeError(f"Basin JSON not found for site_id={sid} (404)") from e
        raise

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    done_marker.write_text(now_utc_iso(), encoding="utf-8")


def discover_time_series_id(cfg: PipelineConfig, site_id: str) -> Optional[str]:
    sid = normalize_site_id(site_id)
    params = {
        "f": "json",
        "monitoring_location_id": f"USGS-{sid}",
        "parameter_code": cfg.param_stage,
        "properties": "id",
        "limit": "1",
    }
    data = get_json(cfg.ogc_ts_meta, params=params, timeout=cfg.http_timeout_usgs, headers=cfg.http_headers_usgs)
    feats = data.get("features", [])
    if not feats:
        return None
    return feats[0].get("id") or (feats[0].get("properties", {}) or {}).get("id")


def build_continuous_url(cfg: PipelineConfig, site_id: str, ts_id: Optional[str], start_dt: str, end_dt: str) -> str:
    sid = normalize_site_id(site_id)
    start_iso = f"{start_dt}T00:00:00Z"
    end_iso = f"{end_dt}T23:59:59Z"

    params: dict[str, str] = {
        "f": "json",
        "datetime": f"{start_iso}/{end_iso}",
        "properties": "time,value",
        "limit": "20000",
    }
    if ts_id:
        params["time_series_id"] = ts_id
    else:
        params["monitoring_location_id"] = f"USGS-{sid}"
        params["parameter_code"] = cfg.param_stage

    req = requests.Request("GET", cfg.ogc_continuous, params=params).prepare()
    assert req.url is not None
    return req.url


def paged_features(cfg: PipelineConfig, url: str) -> Iterable[dict]:
    next_url = url

    while next_url:
        payload = get_json(
            next_url,
            timeout=cfg.http_timeout_usgs,
            headers=cfg.http_headers_usgs,
        )

        for feat in payload.get("features", []):
            yield feat

        next_url = next(
            (lk.get("href") for lk in payload.get("links", []) if lk.get("rel") == "next" and lk.get("href")),
            None,
        )

        if next_url:
            time.sleep(0.2)


def fetch_stage_window(cfg: PipelineConfig, site_id: str, ts_id: Optional[str], start_dt: str, end_dt: str) -> Optional[pd.DataFrame]:
    url = build_continuous_url(cfg, site_id, ts_id, start_dt, end_dt)
    rows: list[tuple[str, Any]] = []
    for feat in paged_features(cfg, url):
        p = feat.get("properties", {}) or {}
        t = p.get("time")
        v = p.get("value")
        if t is not None and v is not None:
            rows.append((t, v))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["datetime", "Stage_ft"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    df["Stage_ft"] = pd.to_numeric(df["Stage_ft"], errors="coerce")
    df = df.dropna(subset=["datetime", "Stage_ft"])
    if df.empty:
        return None
    return df.sort_values("datetime").drop_duplicates("datetime", keep="last")


def download_stage_parquet(
    cfg: PipelineConfig,
    site_id: str,
    out_parquet: Path,
    done_marker: Path,
    *,
    start_date: str,
    end_date: str,
    overwrite: bool,
) -> int:
    if done_marker.exists() and out_parquet.exists() and not overwrite:
        try:
            return int(len(pd.read_parquet(out_parquet, columns=["datetime"])))
        except Exception:
            return -1

    if out_parquet.exists() and not overwrite and not done_marker.exists():
        done_marker.write_text(now_utc_iso(), encoding="utf-8")
        try:
            return int(len(pd.read_parquet(out_parquet, columns=["datetime"])))
        except Exception:
            return -1

    ts_id = discover_time_series_id(cfg, site_id)

    parts: list[pd.DataFrame] = []
    for w_start, w_end in date_windows(start_date, end_date, cfg.window_days):
        dfw = fetch_stage_window(cfg, site_id, ts_id, w_start, w_end)
        if dfw is not None and not dfw.empty:
            parts.append(dfw)
        time.sleep(0.35 + random.uniform(0.0, 0.2))

    if not parts:
        return 0

    df_all = pd.concat(parts, ignore_index=True).sort_values("datetime").drop_duplicates("datetime")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_parquet, index=False, engine="pyarrow")
    done_marker.write_text(now_utc_iso(), encoding="utf-8")
    return int(len(df_all))