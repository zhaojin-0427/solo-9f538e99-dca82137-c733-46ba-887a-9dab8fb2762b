"""Multi-day accumulation of published daily rainfall mosaics.

Inputs are the immutable GeoTIFF files of already-published *daily* mosaic
versions (see :mod:`app.rasters`).  The first day defines the target grid
(CRS, transform, width, height); later days may use a different CRS,
resolution or footprint and are regridded onto that target.

For every target pixel each day contributes an area-weighted mean of the
valid source pixels that intersect it.  Regridding uses GDAL's average
resampling, which weights every contributing source pixel by the overlapping
area (so it stays exact for same-CRS, aligned, whole-pixel relationships and
degrades gracefully for arbitrary CRS/resolution combinations).  A parallel
reprojection of the valid/invalid mask yields the fraction of the target
pixel covered by valid source pixels.

When any day covers less than :data:`ACCUM_COVERAGE_THRESHOLD` (90%) of a
target pixel, the pixel is written as NoData (NaN); otherwise the daily means
are summed.  The result is a single-band float32 GeoTIFF with NaN NoData.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

from .rasters import crs_equal
from .store import SelectedVersion

# Fraction of a target pixel that every day must cover with valid pixels.
ACCUM_COVERAGE_THRESHOLD = 0.9
# Floating-point slack so exactly-90% coverage still counts as sufficient.
COVERAGE_EPSILON = 1e-9
# Whole-pixel alignment check, matching the mosaic service's tolerance.
ALIGN_TOLERANCE_PIXELS = 1e-3
OUTPUT_DTYPE = "float32"
OUTPUT_NODATA = float("nan")


class AccumulationRasterError(Exception):
    """A source image is missing, unreadable or structurally unsuitable."""

    def __init__(self, message: str, *, day: str | None = None):
        super().__init__(message)
        self.message = message
        self.day = day

    def as_detail(self) -> dict:
        detail: dict[str, object] = {
            "code": "source_raster_error",
            "message": self.message,
        }
        if self.day is not None:
            detail["day"] = self.day
        return detail


@dataclass(frozen=True)
class DailySource:
    selection: SelectedVersion
    data: np.ndarray  # float32, NaN where the source is NoData
    valid: np.ndarray  # uint8 0/1 mask
    transform: Affine
    crs: CRS
    width: int
    height: int
    source_nodata: float | None


@dataclass(frozen=True)
class AccumulationInfo:
    width: int
    height: int
    transform: Affine
    crs: CRS
    bounds: tuple[float, float, float, float]
    res_x: float
    res_y: float
    dtype: str
    nodata: float


@dataclass(frozen=True)
class DailyContribution:
    day: str
    version: int
    published_at: str
    sha256: str
    # Mean valid-coverage fraction over all target pixels for this day.
    covered_pixel_fraction: float
    # Fraction of target pixels that met the 90% coverage rule for this day.
    qualifying_pixel_fraction: float


@dataclass(frozen=True)
class AccumulationResult:
    info: AccumulationInfo
    contributions: tuple[DailyContribution, ...]
    output: np.ndarray
    nodata_pixel_fraction: float


def _bounds_of(transform: Affine, width: int, height: int) -> tuple[float, float, float, float]:
    res_x = float(transform.a)
    res_y = float(-transform.e)
    return (
        float(transform.c),
        float(transform.f - height * res_y),
        float(transform.c + width * res_x),
        float(transform.f),
    )


def load_daily_source(selection: SelectedVersion) -> DailySource:
    """Open one published version file and validate its structure."""

    try:
        with rasterio.open(selection.path) as ds:
            if ds.driver != "GTiff":
                raise AccumulationRasterError(
                    f"source for day {selection.day} is not a GeoTIFF "
                    f"(driver {ds.driver!r})",
                    day=selection.day,
                )
            if ds.count != 1:
                raise AccumulationRasterError(
                    f"source for day {selection.day} must have exactly one band, "
                    f"got {ds.count}",
                    day=selection.day,
                )
            if not ds.dtypes[0].startswith(("float", "int", "uint")):
                raise AccumulationRasterError(
                    f"source for day {selection.day} has unsupported pixel type "
                    f"{ds.dtypes[0]!r}",
                    day=selection.day,
                )
            if ds.crs is None:
                raise AccumulationRasterError(
                    f"source for day {selection.day} declares no CRS",
                    day=selection.day,
                )
            if ds.width < 1 or ds.height < 1:
                raise AccumulationRasterError(
                    f"source for day {selection.day} has non-positive dimensions",
                    day=selection.day,
                )
            coefficients = (
                ds.transform.a,
                ds.transform.b,
                ds.transform.c,
                ds.transform.d,
                ds.transform.e,
                ds.transform.f,
            )
            if not all(math.isfinite(float(v)) for v in coefficients):
                raise AccumulationRasterError(
                    f"source for day {selection.day} has a non-finite transform",
                    day=selection.day,
                )
            if not math.isclose(ds.transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
                ds.transform.e * -1, abs(ds.transform.e), abs_tol=1e-12
            ) or ds.transform.a <= 0 or ds.transform.e >= 0:
                raise AccumulationRasterError(
                    f"source for day {selection.day} must use a north-up, positive-x grid",
                    day=selection.day,
                )
            raw = ds.read(1)
            transform = ds.transform
            crs = ds.crs
            width = ds.width
            height = ds.height
            source_nodata = ds.nodata
    except AccumulationRasterError:
        raise
    except Exception as exc:
        # Missing files are caught earlier in the store, but an unreadable or
        # corrupt TIFF (truncated bytes, broken header, ...) lands here.
        raise AccumulationRasterError(
            f"source image for day {selection.day} version {selection.version} "
            f"could not be read: {exc}",
            day=selection.day,
        ) from exc

    data = raw.astype(np.float32, copy=True)
    if source_nodata is not None:
        if math.isnan(float(source_nodata)):
            valid = np.isfinite(data).astype(np.uint8)
        else:
            valid = (data != np.float32(source_nodata)).astype(np.uint8)
        data[valid == 0] = np.nan
    else:
        # No declared NoData: NaN/Inf readings still cannot be averaged.
        valid = np.isfinite(data).astype(np.uint8)
        data[valid == 0] = np.nan

    return DailySource(
        selection=selection,
        data=data,
        valid=valid,
        transform=transform,
        crs=crs,
        width=width,
        height=height,
        source_nodata=None if source_nodata is None else float(source_nodata),
    )


def _snap_length(length: float, full: float) -> float:
    """Snap a near-full overlap length to the full pixel dimension.

    Intersection math on large projected coordinates can leave a fully
    covered edge one ULP short; treat anything within a thousandth of a pixel
    as fully covered so aligned, whole-pixel relationships stay exact.
    """

    if abs(length - full) <= ALIGN_TOLERANCE_PIXELS * full:
        return full
    return length


def _exact_regrid_same_crs(
    source: DailySource, target: AccumulationInfo
) -> tuple[np.ndarray, np.ndarray]:
    """Exact area-weighted regrid for north-up grids sharing one CRS.

    Every source pixel is an axis-aligned rectangle in the shared CRS, so its
    intersection with each target pixel is computed with affine index math
    rather than a rasterising warper.  This stays pixel-exact for whole-pixel
    relationships and gives true overlap-area weights for arbitrary offsets
    and resolutions (including source pixels that straddle target pixels).
    """

    # Total geometric overlap weight and the valid-only overlap weight per
    # target pixel; coverage is valid weight relative to the target area.
    geometry_weight = np.zeros((target.height, target.width), dtype=np.float64)
    valid_weight = np.zeros((target.height, target.width), dtype=np.float64)
    weighted = np.zeros((target.height, target.width), dtype=np.float64)

    src = source.transform
    dst = target.transform
    src_res_x = float(src.a)
    src_res_y = float(-src.e)
    dst_res_x = float(dst.a)
    dst_res_y = float(-dst.e)

    # Target-pixel column/row spans touched by each source pixel, computed once
    # from the two affine transforms.
    # x coordinate of source pixel (r, c): x = src.c + (c + 0.5)*src_res_x etc.
    for r in range(source.height):
        src_top = src.f - r * src_res_y
        src_bottom = src_top - src_res_y
        row_top = (dst.f - src_top) / dst_res_y
        row_bottom = (dst.f - src_bottom) / dst_res_y
        r0 = max(0, int(np.floor(min(row_top, row_bottom) - ALIGN_TOLERANCE_PIXELS)))
        r1 = min(
            target.height,
            int(np.ceil(max(row_top, row_bottom) + ALIGN_TOLERANCE_PIXELS)),
        )
        if r0 >= r1:
            continue
        for c in range(source.width):
            src_left = src.c + c * src_res_x
            src_right = src_left + src_res_x
            col_left = (src_left - dst.c) / dst_res_x
            col_right = (src_right - dst.c) / dst_res_x
            c0 = max(0, int(np.floor(min(col_left, col_right) - ALIGN_TOLERANCE_PIXELS)))
            c1 = min(
                target.width,
                int(np.ceil(max(col_left, col_right) + ALIGN_TOLERANCE_PIXELS)),
            )
            if c0 >= c1:
                continue

            value = float(source.data[r, c])
            is_valid = bool(source.valid[r, c])
            for tr in range(r0, r1):
                dst_top = dst.f - tr * dst_res_y
                dst_bottom = dst_top - dst_res_y
                overlap_y = max(
                    0.0, min(src_top, dst_top) - max(src_bottom, dst_bottom)
                )
                if overlap_y <= 0.0:
                    continue
                overlap_y = _snap_length(overlap_y, dst_res_y)
                for tc in range(c0, c1):
                    dst_left = dst.c + tc * dst_res_x
                    dst_right = dst_left + dst_res_x
                    overlap_x = max(
                        0.0, min(src_right, dst_right) - max(src_left, dst_left)
                    )
                    if overlap_x <= 0.0:
                        continue
                    overlap_x = _snap_length(overlap_x, dst_res_x)
                    weight = (overlap_x * overlap_y) / (dst_res_x * dst_res_y)
                    if abs(weight - 1.0) <= ALIGN_TOLERANCE_PIXELS:
                        weight = 1.0
                    geometry_weight[tr, tc] += weight
                    if is_valid:
                        valid_weight[tr, tc] += weight
                        weighted[tr, tc] += weight * value

    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = valid_weight.astype(np.float32)
        np.clip(coverage, 0.0, 1.0, out=coverage)
        daily_mean = (weighted / valid_weight).astype(np.float32)
    no_overlap = (geometry_weight <= 0.0) & (valid_weight <= 0.0)
    daily_mean[no_overlap | ~np.isfinite(daily_mean)] = OUTPUT_NODATA
    return daily_mean, coverage


def _warp_across_crs(
    source: DailySource, target: AccumulationInfo
) -> tuple[np.ndarray, np.ndarray]:
    """GDAL average-warp for sources in a different CRS than the target."""

    daily_mean = np.full((target.height, target.width), OUTPUT_NODATA, dtype=np.float32)
    rasterio.warp.reproject(
        source=np.ascontiguousarray(source.data, dtype=np.float32),
        destination=daily_mean,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=OUTPUT_NODATA,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=OUTPUT_NODATA,
        # GDAL averages every source pixel intersecting a target pixel,
        # weighted by the overlapping area and ignoring src_nodata.
        resampling=Resampling.average,
    )

    coverage = np.zeros((target.height, target.width), dtype=np.float32)
    rasterio.warp.reproject(
        source=source.valid,
        destination=coverage,
        src_transform=source.transform,
        src_crs=source.crs,
        dst_transform=target.transform,
        dst_crs=target.crs,
        resampling=Resampling.average,
    )
    np.clip(coverage, 0.0, 1.0, out=coverage)
    return daily_mean, coverage


def _reproject_to_target(source: DailySource, target: AccumulationInfo) -> tuple[np.ndarray, np.ndarray]:
    """Return (area-weighted daily mean, valid coverage fraction) arrays."""

    same_grid = (
        crs_equal(source.crs, target.crs)
        and (source.width, source.height) == (target.width, target.height)
        and source.transform.almost_equals(target.transform)
    )
    if same_grid:
        # No geometry work needed: the daily mean is the source itself and the
        # coverage fraction is exactly 1.0 wherever a valid pixel exists.
        return source.data.copy(), source.valid.astype(np.float32)

    if crs_equal(source.crs, target.crs):
        return _exact_regrid_same_crs(source, target)
    return _warp_across_crs(source, target)


def build_accumulation(sources: list[DailySource]) -> AccumulationResult:
    """Accumulate daily means onto the first day's grid."""

    if not sources:
        raise ValueError("at least one daily source is required")

    first = sources[0]
    target = AccumulationInfo(
        width=first.width,
        height=first.height,
        transform=first.transform,
        crs=first.crs,
        bounds=_bounds_of(first.transform, first.width, first.height),
        res_x=float(first.transform.a),
        res_y=float(-first.transform.e),
        dtype=OUTPUT_DTYPE,
        nodata=OUTPUT_NODATA,
    )

    total_pixels = target.width * target.height
    # Accumulate in float64 to avoid compounding float32 rounding across days,
    # then cast back to the required output type.
    accumulator = np.zeros((target.height, target.width), dtype=np.float64)
    # Pixels rejected by any day; starts empty.
    rejected = np.zeros((target.height, target.width), dtype=bool)
    contributions: list[DailyContribution] = []

    for source in sources:
        daily_mean, coverage = _reproject_to_target(source, target)
        covered = coverage >= (ACCUM_COVERAGE_THRESHOLD - COVERAGE_EPSILON)
        # A target pixel with no finite reprojected reading is not usable even
        # if the mask fraction looked acceptable.
        covered &= np.isfinite(daily_mean)
        rejected |= ~covered
        accumulator += np.where(covered, daily_mean.astype(np.float64), 0.0)
        contributions.append(
            DailyContribution(
                day=source.selection.day,
                version=source.selection.version,
                published_at=source.selection.published_at,
                sha256=source.selection.sha256,
                covered_pixel_fraction=float(
                    coverage.sum(dtype=np.float64) / total_pixels
                ),
                qualifying_pixel_fraction=float(
                    np.count_nonzero(covered) / total_pixels
                ),
            )
        )

    output = accumulator.astype(np.float32)
    output[rejected] = OUTPUT_NODATA
    nodata_fraction = float(np.count_nonzero(rejected) / total_pixels)
    return AccumulationResult(
        info=target,
        contributions=tuple(contributions),
        output=output,
        nodata_pixel_fraction=nodata_fraction,
    )


def write_accumulation(result: AccumulationResult, destination: Path) -> AccumulationInfo:
    """Write the accumulation as a single-band float32 GeoTIFF (NaN NoData)."""

    info = result.info
    profile = {
        "driver": "GTiff",
        "width": info.width,
        "height": info.height,
        "count": 1,
        "dtype": OUTPUT_DTYPE,
        "crs": info.crs,
        "transform": info.transform,
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
        "zlevel": 6,
        "predictor": 3,  # floating-point predictor
        "BIGTIFF": "NO",
    }
    days = ",".join(item.day for item in result.contributions)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(result.output, 1)
        dst.update_tags(
            AREA_OR_POINT="Area",
            RAINMOSAIC_ALGORITHM="daily-area-weighted-mean-accumulation",
            RAINMOSAIC_COVERAGE_THRESHOLD=str(ACCUM_COVERAGE_THRESHOLD),
            RAINMOSAIC_DAYS=days,
            RAINMOSAIC_VERSIONS=",".join(
                str(item.version) for item in result.contributions
            ),
        )
    return info


def verify_accumulation(path: Path, expected: AccumulationInfo) -> None:
    """Re-open a freshly written accumulation and confirm its structure."""

    with rasterio.open(path) as ds:
        if ds.driver != "GTiff" or ds.count != 1:
            raise RuntimeError("written accumulation has unexpected driver or band count")
        if ds.dtypes[0] != OUTPUT_DTYPE:
            raise RuntimeError("written accumulation has unexpected dtype")
        if (ds.width, ds.height) != (expected.width, expected.height):
            raise RuntimeError("written accumulation has unexpected dimensions")
        if ds.nodata is None or not math.isnan(float(ds.nodata)):
            raise RuntimeError("written accumulation has unexpected NoData value")
        if ds.crs != expected.crs:
            raise RuntimeError("written accumulation has unexpected CRS")
        if not ds.transform.almost_equals(expected.transform):
            raise RuntimeError("written accumulation has unexpected transform")
        values = ds.read(1)
        if values.dtype != np.dtype(OUTPUT_DTYPE):
            raise RuntimeError("written accumulation band is not float32")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
