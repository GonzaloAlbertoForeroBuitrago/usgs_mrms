from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import PipelineConfig
from .exceptions import MissingOptionalDependency
from .io import load_stage_with_utc_local, now_utc_iso, resolve_iana_timezone


def detect_top_events(stage_df: pd.DataFrame, *, top_n: int, percentile: int) -> pd.DataFrame:
    """
    Detect and return the top-N events by peak magnitude using HydroEventDetector.

    Requires optional dependency: hydro-event-detector
    """
    try:
        from hydro_event_detector import HydroEventDetector  # type: ignore
    except Exception as e:
        raise MissingOptionalDependency(
            "HydroEventDetector is required for event detection. "
            "Install with: pip install 'usgs-mrms-events[events]' "
            "or: pip install Hydro-Event-Detector --no-deps"
        ) from e

    s = pd.to_numeric(stage_df["Stage_ft"], errors="coerce")
    t = stage_df["datetime"]
    df = pd.DataFrame({"Stage_ft": s.values}, index=pd.DatetimeIndex(t))
    df = df[~df.index.duplicated(keep="first")].sort_index()

    valid = df["Stage_ft"].dropna()
    if valid.empty:
        raise ValueError("No valid Stage_ft values after cleaning.")

    datetimes_naive = pd.to_datetime(valid.index).tz_localize(None)
    values = pd.to_numeric(valid.values, errors="coerce")

    hed = HydroEventDetector(datetimes_naive, values)
    hed.baseflow_lyne_hollick()
    hed.detect_events()
    hed.create_events_dataframe()
    hed.filter_events(percentile)
    hed.create_events_dataframe()

    filtered_df = hed.dataframe
    if filtered_df is None or filtered_df.empty:
        raise ValueError("No events found after filtering. Try lowering event_filter_percentile.")

    top = (
        filtered_df.sort_values("flow_peak", ascending=False)
        .loc[:, ["date_peak", "flow_peak"]]
        .head(top_n)
        .reset_index(drop=True)
    )
    top["date_peak"] = pd.to_datetime(top["date_peak"])
    return top


def build_rain_windows(top_events: pd.DataFrame, *, pre_days: float, post_days: float) -> pd.DataFrame:
    out = top_events.copy()
    out["date_peak"] = pd.to_datetime(out["date_peak"])
    out["start_rain"] = out["date_peak"] - pd.Timedelta(days=pre_days)
    out["end_rain"] = out["date_peak"] + pd.Timedelta(days=post_days)
    return out


def postprocess_events_and_windows(
    cfg: PipelineConfig,
    inv: dict,
    stage_parquet: Path,
    events_top_csv: Path,
    events_windows_csv: Path,
    done_marker: Path,
    *,
    overwrite: bool,
) -> Tuple[int, int, str]:
    """
    Build top events CSV and rain windows CSV.

    Returns: (n_events, n_windows, tz_iana)
    """
    if done_marker.exists() and events_top_csv.exists() and events_windows_csv.exists() and not overwrite:
        tz_iana = resolve_iana_timezone(inv.get("lon"), inv.get("lat"), inv.get("time_zone_abbreviation"))
        try:
            n_events = int(len(pd.read_csv(events_top_csv)))
        except Exception:
            n_events = -1
        try:
            n_windows = int(len(pd.read_csv(events_windows_csv)))
        except Exception:
            n_windows = -1
        return n_events, n_windows, tz_iana

    tz_iana = resolve_iana_timezone(inv.get("lon"), inv.get("lat"), inv.get("time_zone_abbreviation"))
    stage_df = load_stage_with_utc_local(stage_parquet, tz_iana)

    top_events = detect_top_events(stage_df, top_n=cfg.top_n_events, percentile=cfg.event_filter_percentile)
    windows = build_rain_windows(top_events, pre_days=cfg.rain_pre_days, post_days=cfg.rain_post_days)

    if overwrite or (not events_top_csv.exists()):
        top_events.to_csv(events_top_csv, index=False)
    if overwrite or (not events_windows_csv.exists()):
        windows.to_csv(events_windows_csv, index=False)

    done_marker.write_text(now_utc_iso(), encoding="utf-8")
    return int(len(top_events)), int(len(windows)), tz_iana