from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the USGS→events→MRMS pipeline.

    Keep this as the *single source of truth* for constants/endpoints.
    """

    # Base output
    base_dir: Path = Path("data").resolve()
    # Logs (default: <base_dir>/logs)
    log_dir: Optional[Path] = None
    # Shared MRMS cache (default: <base_dir>/_mrms_cache)
    mrms_cache_dir: Optional[Path] = None

    # Step tuning
    window_days: int = 1000
    param_stage: str = "00065"
    top_n_events: int = 60
    event_filter_percentile: int = 50
    rain_pre_days: float = 0.5
    rain_post_days: float = 0.25

    # HTTP
    http_timeout_usgs: int = 60
    http_timeout_mrms: int = 30

    http_headers_usgs: dict[str, str] = None  # type: ignore[assignment]
    http_headers_mrms: dict[str, str] = None  # type: ignore[assignment]
    usgs_api_key: Optional[str] = None

    # MRMS RadarOnly
    aws_radaronly: str = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_01H_00.00"
    mtarchive: str = "https://mtarchive.geol.iastate.edu"

    dtype: str = "float32"
    time_chunk: int = 48
    pixel_chunk: int = 2048
    sleep_between_min: float = 0.02
    sleep_between_max: float = 0.06
    debug_every_n: int = 50

    # Parallel defaults
    default_workers: int = 4
    max_workers_cap: int = 8

    @property
    def sleep_between(self) -> tuple[float, float]:
        """Compatibility tuple for random sleep."""
        return (self.sleep_between_min, self.sleep_between_max)

    # USGS endpoints
    ogc_monitoring_locations: str = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items"
    ogc_ts_meta: str = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items"
    ogc_continuous: str = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
    basin_gages_endpoint: str = "https://api.water.usgs.gov/fabric/pygeoapi/collections/gagesii-basins/items"

    def __post_init__(self) -> None:  # dataclass hook (even frozen)
        object.__setattr__(self, "base_dir", Path(self.base_dir).expanduser().resolve())

        # Default log_dir is <base_dir>/logs if not set
        resolved_log_dir = Path(self.log_dir).expanduser().resolve() if self.log_dir else (self.base_dir / "logs")
        object.__setattr__(self, "log_dir", resolved_log_dir)

        # Default shared MRMS cache outside per-station folders
        resolved_cache_dir = (
            Path(self.mrms_cache_dir).expanduser().resolve()
            if self.mrms_cache_dir
            else (self.base_dir / "_mrms_cache")
        )
        object.__setattr__(self, "mrms_cache_dir", resolved_cache_dir)

        resolved_usgs_api_key = self.usgs_api_key or os.getenv("USGS_API_KEY")
        object.__setattr__(self, "usgs_api_key", resolved_usgs_api_key)

        resolved_http_headers_usgs = self.http_headers_usgs or {
            "User-Agent": "usgs-mrms-events/0.1",
            "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.1",
            "Accept-Encoding": "gzip",
        }
        resolved_http_headers_usgs = dict(resolved_http_headers_usgs)

        if resolved_usgs_api_key:
            resolved_http_headers_usgs["X-Api-Key"] = resolved_usgs_api_key

        object.__setattr__(self, "http_headers_usgs", resolved_http_headers_usgs)

       

        object.__setattr__(
            self,
            "http_headers_mrms",
            self.http_headers_mrms
            or {
                "User-Agent": "usgs-mrms-events/0.1",
                "Accept": "*/*",
                "Accept-Encoding": "gzip",
            },
        )
