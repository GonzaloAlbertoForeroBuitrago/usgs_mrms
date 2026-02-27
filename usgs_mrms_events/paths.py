from __future__ import annotations

import re
from pathlib import Path


def normalize_site_id(site_id: str | int) -> str:
    """Normalize USGS site_id.

    - Accepts digits-only ids (keeps leading zeros)
    - Accepts 'USGS-08165500' and strips prefix
    """
    s = str(site_id).strip()
    s = s.replace("USGS-", "").replace("usgs-", "").strip()
    if not re.fullmatch(r"\d{1,15}", s):
        raise ValueError(f"site_id must be digits-only (1..15 chars). Got: {site_id!r}")
    return s


def prefixes(site_id: str) -> tuple[str, str]:
    s = normalize_site_id(site_id)
    p2 = s[:2]
    p4 = s[:4] if len(s) >= 4 else s
    return p2, p4


def safe_state_folder(state_name: str | None) -> str:
    if not state_name:
        return "UNKNOWN_STATE"
    return str(state_name).strip().upper().replace(" ", "_")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_station_paths(base_dir: Path, site_id: str, state_name: str | None) -> dict[str, Path]:
    """Deterministic folder layout.

    data/{events,basins_json,stage_parquet,site_meta,rain_zarr}/STATE/AA/AAAA/{site_id}.*
    """
    sid = normalize_site_id(site_id)
    p2, p4 = prefixes(sid)
    st = safe_state_folder(state_name)
    rel = Path(st) / p2 / p4

    basin_dir = base_dir / "basins_json"
    stage_dir = base_dir / "stage_parquet"
    meta_dir = base_dir / "site_meta"
    events_dir = base_dir / "events"
    rain_dir = base_dir / "rain_zarr"

    paths: dict[str, Path] = {
        "basin_json": basin_dir / rel / f"{sid}.json",
        "stage_parquet": stage_dir / rel / f"{sid}.parquet",
        "site_meta_json": meta_dir / rel / f"{sid}_monitoring_location.json",
        "events_top_csv": events_dir / rel / f"{sid}_top_events.csv",
        "events_windows_csv": events_dir / rel / f"{sid}_rain_windows.csv",
        "rain_zarr": rain_dir / rel / f"{sid}.zarr",
        "rain_missing_csv": rain_dir / rel / f"{sid}_missing_radaronly_hours.csv",
        "done_meta": meta_dir / rel / f"{sid}.meta.done",
        "done_basin": basin_dir / rel / f"{sid}.basin.done",
        "done_stage": stage_dir / rel / f"{sid}.stage.done",
        "done_events": events_dir / rel / f"{sid}.events.done",
        "done_rain": rain_dir / rel / f"{sid}.rain.done",
        "inventory_csv": base_dir / "stations_inventory.csv",
    }

    for p in paths.values():
        _ensure_parent(p)
    return paths
