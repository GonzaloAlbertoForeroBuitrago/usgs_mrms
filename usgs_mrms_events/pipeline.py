from __future__ import annotations

import json
import os
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv

from .config import PipelineConfig
from .events import postprocess_events_and_windows
from .io import append_inventory_row, now_utc_iso
from .logger import get_logger, setup_logging, site_logger
from .mrms import build_zarr_radaronly_from_windows
from .paths import build_station_paths, ensure_path_parent, normalize_site_id
from .usgs_api import (
    download_basin_json,
    download_stage_parquet,
    extract_inventory_row,
    fetch_monitoring_location,
)

log = get_logger("usgs_mrms_events.pipeline")

# Load AWS credentials from .env file
ENV_PATH = Path(__file__).resolve().parent.parent / "env" / "aws_credentials.env"
load_dotenv(ENV_PATH)


def _build_s3_bucket():
    key = os.getenv("KEY")
    secret = os.getenv("SECRET")
    bucket_name = os.getenv("BUCKET_NAME") or "tgf-mentorship-gonzalo"
    if not key or not secret or not bucket_name:
        return None

    try:
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name="us-east-1",
        )
        return s3.Bucket(bucket_name)
    except Exception:
        return None


def upload_to_s3(path: Path, slog=None) -> bool:
    bucket = _build_s3_bucket()
    if bucket is None:
        if slog:
            slog.warning(f"S3 upload skipped; bucket/credentials unavailable for {path}")
        return False

    try:
        bucket.upload_file(str(path), str(path))
        if slog:
            slog.info(f"Uploaded to S3: {path}")
        return True
    except Exception as e:
        if slog:
            slog.exception(f"S3 upload failed for {path}: {e}")
        return False


def _run_site_wrapper(args):
    site_id, start_date, end_date, base_dir, overwrite, config, upload = args
    return run_site(
        site_id=site_id,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir,
        overwrite=overwrite,
        config=config,
        upload=upload,
    )


def _result_payload(
    sid: str,
    inv: dict[str, Any] | None,
    paths: dict[str, Path] | None,
    *,
    status: str,
    reason: str | None = None,
    tz_iana: str | None = None,
    stage_rows: int | None = None,
    n_events: int | None = None,
    n_windows: int | None = None,
    rain_hours: int | None = None,
    rain_pixels: int | None = None,
    rain_files_ok: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "site_id": sid,
        "state": (inv or {}).get("state_name"),
        "tz_iana": tz_iana,
        "status": status,
        "reason": reason,
        "error": error,
        "stage_rows": stage_rows,
        "n_events": n_events,
        "n_windows": n_windows,
        "rain_hours": rain_hours,
        "rain_pixels": rain_pixels,
        "rain_files_ok": rain_files_ok,
        "paths": {k: str(v) for k, v in (paths or {}).items()},
    }


def run_site(
    *,
    site_id: str | int,
    start_date: str = "2019-04-01",
    end_date: str = "2026-01-30",
    base_dir: str | Path = "data",
    overwrite: bool = False,
    config: PipelineConfig | None = None,
    upload: bool = True,
) -> dict[str, Any]:
    """Run the unified pipeline for a single USGS site.

    Important behavior:
    - If basin.json or stage.parquet is missing, the station is skipped and the batch continues.
    - Events and rain/zarr folders are created only when they are truly needed.
    - Any station-specific failure returns a status payload instead of stopping the full run.
    """
    sid = normalize_site_id(site_id)
    cfg = config or PipelineConfig(base_dir=Path(base_dir).resolve())

    setup_logging(log_dir=cfg.log_dir)
    slog = site_logger(sid, site_logs_dir=Path(cfg.log_dir) / "sites")
    slog.info(f"[{sid}] starting | base_dir={cfg.base_dir}")

    inv: dict[str, Any] | None = None
    paths: dict[str, Path] | None = None

    # Step A: monitoring-location metadata (defines state + folder layout)
    try:
        feat = fetch_monitoring_location(cfg, sid)
        inv = extract_inventory_row(feat)
        paths = build_station_paths(cfg.base_dir, sid, inv.get("state_name"))

        ensure_path_parent(paths["site_meta_json"])
        ensure_path_parent(paths["done_meta"])
        if overwrite or (not paths["site_meta_json"].exists()) or (not paths["done_meta"].exists()):
            paths["site_meta_json"].write_text(json.dumps(feat, indent=2), encoding="utf-8")
            paths["done_meta"].write_text(now_utc_iso(), encoding="utf-8")
    except Exception as e:
        slog.exception(f"[{sid}] metadata failed: {e}")
        return _result_payload(sid, inv, paths, status="failed_meta", error=str(e))

    # Step B: basin JSON
    basin_ok = False
    try:
        ensure_path_parent(paths["basin_json"])
        ensure_path_parent(paths["done_basin"])
        download_basin_json(cfg, sid, paths["basin_json"], paths["done_basin"], overwrite=overwrite)
        basin_ok = paths["basin_json"].exists()
        if not basin_ok:
            slog.warning(f"[{sid}] basin_json missing after download attempt: {paths['basin_json']}")
    except Exception as e:
        slog.exception(f"[{sid}] basin download failed: {e}")

    # Step C: stage parquet
    stage_rows = 0
    stage_ok = False
    try:
        ensure_path_parent(paths["stage_parquet"])
        ensure_path_parent(paths["done_stage"])
        stage_rows = download_stage_parquet(
            cfg,
            sid,
            paths["stage_parquet"],
            paths["done_stage"],
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
        )
        stage_ok = paths["stage_parquet"].exists() and stage_rows > 0
        if not stage_ok:
            if paths["stage_parquet"].exists() and stage_rows <= 0:
                try:
                    paths["stage_parquet"].unlink(missing_ok=True)
                except Exception:
                    pass
            slog.warning(
                f"[{sid}] stage_parquet missing or empty after download attempt: "
                f"{paths['stage_parquet']} | rows={stage_rows}"
            )
    except Exception as e:
        slog.exception(f"[{sid}] stage download failed: {e}")

    # Critical gate: do not proceed to events/rain unless BOTH inputs exist.
    if not basin_ok or not stage_ok:
        reasons: list[str] = []
        if not basin_ok:
            reasons.append("missing_basin_json")
        if not stage_ok:
            reasons.append("missing_stage_parquet")
        reason = ",".join(reasons)
        print(f"[{sid}] SKIPPED: {reason}")
        slog.warning(f"[{sid}] skipped before events/rain: {reason}")
        return _result_payload(
            sid,
            inv,
            paths,
            status="skipped_missing_inputs",
            reason=reason,
            stage_rows=stage_rows,
        )

    # Step D: events + rain windows
    try:
        ensure_path_parent(paths["events_top_csv"])
        ensure_path_parent(paths["events_windows_csv"])
        ensure_path_parent(paths["done_events"])
        n_events, n_windows, tz_iana = postprocess_events_and_windows(
            cfg,
            inv,
            paths["stage_parquet"],
            paths["events_top_csv"],
            paths["events_windows_csv"],
            paths["done_events"],
            overwrite=overwrite,
        )
    except Exception as e:
        slog.exception(f"[{sid}] events processing failed: {e}")
        return _result_payload(
            sid,
            inv,
            paths,
            status="failed_events",
            error=str(e),
            stage_rows=stage_rows,
        )

    if not paths["events_windows_csv"].exists() or n_windows <= 0:
        slog.warning(f"[{sid}] no events windows generated; skipping rain zarr")
        return _result_payload(
            sid,
            inv,
            paths,
            status="skipped_no_windows",
            reason="no_rain_windows",
            tz_iana=tz_iana,
            stage_rows=stage_rows,
            n_events=n_events,
            n_windows=n_windows,
        )

    # Step E: rainfall zarr (resume-safe)
    try:
        ensure_path_parent(paths["rain_zarr"])
        ensure_path_parent(paths["rain_missing_csv"])
        ensure_path_parent(paths["done_rain"])

        if overwrite and paths["rain_zarr"].exists():
            shutil.rmtree(paths["rain_zarr"], ignore_errors=True)
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
                try:
                    upload_to_s3(paths["done_rain"], slog=slog)
                    # Delete the zarr folder after successful upload
                    shutil.rmtree(paths["rain_zarr"], ignore_errors=True)
                except Exception as e:
                    slog.exception(f"[{sid}] upload to S3 failed: {e}")

        else:
            hours_n = -1
            pixels_n = -1
            files_ok = -1
    except Exception as e:
        slog.exception(f"[{sid}] rain zarr build failed: {e}")
        if paths["rain_zarr"].exists() and not paths["done_rain"].exists():
            shutil.rmtree(paths["rain_zarr"], ignore_errors=True)
        return _result_payload(
            sid,
            inv,
            paths,
            status="failed_rain",
            error=str(e),
            tz_iana=tz_iana,
            stage_rows=stage_rows,
            n_events=n_events,
            n_windows=n_windows,
        )

    print(f"[{sid}] meta ✓ basin ✓ stage ✓ ({stage_rows} rows) events ✓ ({n_events}) rain ✓")
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

    return _result_payload(
        sid,
        inv,
        paths,
        status="ok",
        tz_iana=tz_iana,
        stage_rows=stage_rows,
        n_events=n_events,
        n_windows=n_windows,
        rain_hours=hours_n,
        rain_pixels=pixels_n,
        rain_files_ok=files_ok,
    )


def run_many(
    site_ids: list[str | int],
    *,
    start_date: str = "2019-04-01",
    end_date: str = "2026-01-30",
    base_dir: str | Path = "data",
    overwrite: bool = False,
    config: PipelineConfig | None = None,
    upload: bool = True,
    workers: int | None = None,
) -> dict[str, int]:
    cfg = config or PipelineConfig(base_dir=Path(base_dir).resolve())
    worker_count = workers or min(max(1, cfg.default_workers), cpu_count(), cfg.max_workers_cap)

    tasks = [(sid, start_date, end_date, base_dir, overwrite, cfg, upload) for sid in site_ids]

    ok = 0
    skip = 0
    fail = 0

    print(f"[run_many] sites={len(site_ids)} workers={worker_count}")

    with Pool(worker_count) as pool:
        for result in pool.imap_unordered(_run_site_wrapper, tasks):
            status = result.get("status", "failed_unknown")
            sid = result.get("site_id", "unknown")
            if status == "ok":
                ok += 1
            elif status.startswith("skipped"):
                skip += 1
                print(f"[{sid}] {status}: {result.get('reason')}")
            else:
                fail += 1
                print(f"[{sid}] {status}: {result.get('error')}")

    return {"ok": ok, "skip": skip, "fail": fail, "workers": worker_count}
