from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for the USGS→events→MRMS pipeline.

    Keep this as the *single source of truth* for constants/endpoints.
    """

    # Base output
    base_dir: Path = Path("data").resolve()

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

    # MRMS RadarOnly
    aws_radaronly: str = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_01H_00.00"
    mtarchive: str = "https://mtarchive.geol.iastate.edu"

    dtype: str = "float32"
    time_chunk: int = 48
    pixel_chunk: int = 2048
    sleep_between_min: float = 0.02
    sleep_between_max: float = 0.06
    debug_every_n: int = 50
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
        object.__setattr__(
            self,
            "http_headers_usgs",
            self.http_headers_usgs
            or {
                "User-Agent": "usgs-mrms-events/0.1",
                "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.1",
                "Accept-Encoding": "gzip",
            },
        )
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
