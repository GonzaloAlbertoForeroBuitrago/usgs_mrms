from __future__ import annotations

import boto3
from dotenv import load_dotenv
import json
import shutil
import os
from pathlib import Path
from typing import Any

from .logger import get_logger, setup_logging, site_logger
from .config import PipelineConfig
from .events import postprocess_events_and_windows
from .io import append_inventory_row, now_utc_iso
from .paths import build_station_paths, normalize_site_id
from .mrms import build_zarr_radaronly_from_windows
from .usgs_api import (
    download_basin_json,
    download_stage_parquet,
    extract_inventory_row,
    fetch_monitoring_location,
)

log = get_logger("usgs_mrms_events.pipeline")

# Load AWS credentials from .env file
load_dotenv()
key = os.getenv("KEY")
secret = os.getenv("SECRET")
bucket_name = os.getenv("BUCKET_NAME")

def upload_to_s3(path):
    s3 = boto3.resource("s3", aws_access_key_id=key, aws_secret_access_key=secret, region_name="us-east-1")
    bucket = s3.Bucket("tgf-mentorship-gonzalo")
    
    bucket.upload_file(path, path)

def run_site(
    *,
    site_id: str | int,
    start_date: str = "2019-04-01",
    end_date: str = "2026-01-30",
    base_dir: str | Path = "data",
    overwrite: bool = False,
    config: PipelineConfig | None = None,
    upload: bool = True
) -> dict[str, Any]:
    """
    Run the unified pipeline for a single USGS site.

    User inputs:
      - site_id
      - start_date (YYYY-MM-DD)
      - end_date (YYYY-MM-DD)
      - overwrite (optional)
      - base_dir (optional)

    Everything else is controlled by PipelineConfig defaults.
    """
    sid = normalize_site_id(site_id)
    cfg = config or PipelineConfig(base_dir=Path(base_dir).resolve())

    # Ensure root logging exists (single-site runs)
    setup_logging(log_dir=cfg.log_dir)

    # Per-site logger (also used for multi-process runs)
    slog = site_logger(sid, site_logs_dir=Path(cfg.log_dir) / "sites")

    slog.info(f"[{sid}] starting | base_dir={cfg.base_dir}")

    # Step A: monitoring-location metadata (defines state + folder layout)
    feat = fetch_monitoring_location(cfg, sid)
    inv = extract_inventory_row(feat)

    paths = build_station_paths(cfg.base_dir, sid, inv.get("state_name"))

    if overwrite or (not paths["site_meta_json"].exists()) or (not paths["done_meta"].exists()):
        paths["site_meta_json"].write_text(json.dumps(feat, indent=2), encoding="utf-8")
        paths["done_meta"].write_text(now_utc_iso(), encoding="utf-8")

    # Step B: basin JSON
    download_basin_json(cfg, sid, paths["basin_json"], paths["done_basin"], overwrite=overwrite)

    # Step C: stage parquet
    stage_rows = download_stage_parquet(
        cfg,
        sid,
        paths["stage_parquet"],
        paths["done_stage"],
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
    )

    # Step D: events + rain windows
    n_events, n_windows, tz_iana = postprocess_events_and_windows(
        cfg,
        inv,
        paths["stage_parquet"],
        paths["events_top_csv"],
        paths["events_windows_csv"],
        paths["done_events"],
        overwrite=overwrite,
    )

    # Step E: rainfall zarr (resume-safe)
    if overwrite and paths["rain_zarr"].exists():
        shutil.rmtree(paths["rain_zarr"], ignore_errors=True)
        if paths["done_rain"].exists():
            paths["done_rain"].unlink(missing_ok=True)

    if (not paths["done_rain"].exists()) or overwrite:
        hours_n, pixels_n, files_ok = build_zarr_radaronly_from_windows(
            cfg,
            windows_csv=paths["events_windows_csv"],
            basin_json=paths["basin_json"],
            out_zarr=paths["rain_zarr"],
            missing_csv=paths["rain_missing_csv"],
        )
        paths["done_rain"].write_text(now_utc_iso(), encoding="utf-8")
        if upload:
            upload_to_s3(paths["done_rain"])
        
    else:
        hours_n = -1
        pixels_n = -1
        files_ok = -1

    print(f"[{sid}] meta ✓  basin ✓  stage ✓ ({stage_rows} rows)  events ✓ ({n_events})  rain ✓")
    slog.info(
        f"[{sid}] completed | stage_rows={stage_rows} events={n_events} windows={n_windows} "
        f"rain_hours={hours_n} rain_pixels={pixels_n} files_ok={files_ok} tz={tz_iana}"
    )

    inv_row = {
        **inv,
        "basin_json_path": str(paths["basin_json"]),
        "stage_parquet_path": str(paths["stage_parquet"]),
        "site_meta_json_path": str(paths["site_meta_json"]),
        "events_top_csv": str(paths["events_top_csv"]),
        "events_windows_csv": str(paths["events_windows_csv"]),
        "rain_zarr_path": str(paths["rain_zarr"]),
        "timestamp_utc": now_utc_iso(),
    }
    append_inventory_row(paths["inventory_csv"], inv_row)

    return {
        "site_id": sid,
        "state": inv.get("state_name"),
        "tz_iana": tz_iana,
        "stage_rows": stage_rows,
        "n_events": n_events,
        "n_windows": n_windows,
        "rain_hours": hours_n,
        "rain_pixels": pixels_n,
        "rain_files_ok": files_ok,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def run_many(
    site_ids: list[str | int],
    *,
    start_date: str = "2019-04-01",
    end_date: str = "2026-01-30",
    base_dir: str | Path = "data",
    overwrite: bool = False,
    config: PipelineConfig | None = None,
    upload: bool = True
) -> dict[str, int]:
    ok = 0
    fail = 0

    for i, sid in enumerate(site_ids, start=1):
        try:
            print(f"[{i}/{len(site_ids)}] {sid}")
            run_site(
                site_id=sid,
                start_date=start_date,
                end_date=end_date,
                base_dir=base_dir,
                overwrite=overwrite,
                config=config,
                upload=upload
            )
            ok += 1
        except Exception as e:
            print(f"[{sid}] FAILED: {type(e).__name__}: {e}")
            fail += 1

    return {"ok": ok, "fail": fail}