from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LogPaths:
    log_dir: Path
    run_log: Path
    site_logs_dir: Path


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_log_paths(log_dir: Path, *, run_id: Optional[str] = None) -> LogPaths:
    run_id = run_id or utc_run_id()
    log_dir = Path(log_dir).expanduser().resolve()
    _ensure_dir(log_dir)
    site_logs_dir = log_dir / "sites"
    _ensure_dir(site_logs_dir)
    run_log = log_dir / f"pipeline_{run_id}.log"
    return LogPaths(log_dir=log_dir, run_log=run_log, site_logs_dir=site_logs_dir)


def setup_logging(
    *,
    log_dir: Path,
    run_id: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 25_000_000,
    backup_count: int = 5,
) -> LogPaths:
    """
    Configure root logging once (file + optional console).
    Safe for single-process. For multi-process, prefer per-site logs created by `site_logger(...)`.
    """
    paths = build_log_paths(log_dir, run_id=run_id)

    root = logging.getLogger()
    root.setLevel(level)

    # Prevent duplicate handlers if setup_logging is called more than once
    if getattr(root, "_usgs_mrms_events_configured", False):
        return paths

    fmt = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=str(paths.run_log),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # Mark configured
    setattr(root, "_usgs_mrms_events_configured", True)

    # Make logs deterministic in UTC
    os.environ.setdefault("TZ", "UTC")

    return paths


def get_logger(name: str = "usgs_mrms_events") -> logging.Logger:
    return logging.getLogger(name)


def site_logger(site_id: str, *, site_logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Dedicated per-site logger writing to a site-specific file (safer for multi-process runs).
    """
    lg = logging.getLogger(f"usgs_mrms_events.site.{site_id}")
    lg.setLevel(level)

    if getattr(lg, "_configured", False):
        return lg

    site_logs_dir = Path(site_logs_dir).expanduser().resolve()
    _ensure_dir(site_logs_dir)

    fmt = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fp = site_logs_dir / f"{site_id}.log"
    fh = RotatingFileHandler(filename=str(fp), maxBytes=25_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    setattr(lg, "_configured", True)
    return lg