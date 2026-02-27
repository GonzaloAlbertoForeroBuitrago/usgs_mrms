from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import MissingOptionalDependency


def _require_geo_stack() -> tuple[Any, Any, Any, Any]:
    """
    Lazily import heavy GIS stack. Raises a clean error if missing.
    Returns (gpd, shape, gdal, ogr, osr).
    """
    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:
        raise MissingOptionalDependency(
            "geopandas is required for basin geometry handling. "
            "Install with: pip install 'usgs-mrms-events[geo]'"
        ) from e

    try:
        from shapely.geometry import shape  # type: ignore
    except Exception as e:
        raise MissingOptionalDependency(
            "shapely is required for basin geometry handling. "
            "Install with: pip install 'usgs-mrms-events[geo]'"
        ) from e

    try:
        from osgeo import gdal, ogr, osr  # type: ignore
    except Exception as e:
        raise MissingOptionalDependency(
            "GDAL (osgeo) is required to read MRMS GRIB2 and rasterize masks. "
            "Install via conda-forge (recommended): conda install -c conda-forge gdal"
        ) from e

    gdal.UseExceptions()
    return gpd, shape, gdal, ogr, osr


def load_basin_polygon_from_json(basin_json_fp: Path):
    gpd, shape, *_ = _require_geo_stack()
    data = json.loads(basin_json_fp.read_text(encoding="utf-8"))
    geom = data.get("geometry")
    if geom is None:
        raise RuntimeError(f"Basin JSON has no geometry: {basin_json_fp}")
    return shape(geom)


def build_mask_and_lonlat_from_basin(basin_json_fp: Path, sample_gz_bytes: bytes, *, dtype: str) -> dict[str, Any]:
    gpd, shape, gdal, ogr, osr = _require_geo_stack()

    poly = load_basin_polygon_from_json(basin_json_fp)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")

    vs = "/vsimem/_sample.grib2"
    raw = gzip.decompress(sample_gz_bytes)
    gdal.FileFromMemBuffer(vs, raw)

    ds = gdal.Open(vs)
    if ds is None:
        gdal.Unlink(vs)
        raise RuntimeError("GDAL could not open sample RadarOnly GRIB2.")

    try:
        gt = ds.GetGeoTransform()
        proj_wkt = ds.GetProjection()
        nx, ny = ds.RasterXSize, ds.RasterYSize

        geom_wkt = gdf.geometry.union_all().wkt

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        drv = ogr.GetDriverByName("Memory")
        dsv = drv.CreateDataSource("mem")
        lyr = dsv.CreateLayer("basin", srs=srs, geom_type=ogr.wkbPolygon)

        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetGeometry(ogr.CreateGeometryFromWkt(geom_wkt))
        lyr.CreateFeature(feat)

        mask_ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(gt)
        mask_ds.SetProjection(proj_wkt)

        gdal.RasterizeLayer(mask_ds, [1], lyr, burn_values=[1])
        mask = mask_ds.ReadAsArray().astype(bool)

        rows, cols = np.where(mask)
        if rows.size == 0:
            raise RuntimeError("Basin mask is empty on MRMS grid.")

        lon_pix = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
        lat_pix = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]

        return {
            "rows": rows.astype(np.int32),
            "cols": cols.astype(np.int32),
            "lon_pix": lon_pix.astype(dtype),
            "lat_pix": lat_pix.astype(dtype),
            "gt": gt,
            "proj_wkt": proj_wkt,
            "nx": int(nx),
            "ny": int(ny),
        }
    finally:
        ds = None
        gdal.Unlink(vs)