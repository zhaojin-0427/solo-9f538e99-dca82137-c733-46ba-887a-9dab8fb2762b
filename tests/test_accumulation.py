"""Accumulation math: area-weighted daily means, 90% coverage rule, NoData."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app.accumulation import (
    ACCUM_COVERAGE_THRESHOLD,
    AccumulationRasterError,
    DailySource,
    build_accumulation,
    load_daily_source,
    write_accumulation,
    verify_accumulation,
)
from app.store import SelectedVersion

NODATA = -32768


def _selection(day: str, version: int, path: Path) -> SelectedVersion:
    return SelectedVersion(
        day=day,
        version=version,
        path=path,
        published_at=f"{day}T03:00:00.000000Z",
        sha256="0" * 64,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )


def _write_tif(
    path: Path,
    values: np.ndarray,
    *,
    x0: float = 0.0,
    y0: float = 0.1,
    res: float = 0.01,
    crs: str = "EPSG:4326",
    dtype: str = "int16",
    nodata=NODATA,
) -> Path:
    array = np.asarray(values, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=from_origin(x0, y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
    return path


def _source(path: Path, day: str, version: int) -> DailySource:
    return load_daily_source(_selection(day, version, path))


def _two_sources(tmp_path: Path, day1, day2):
    p1 = _write_tif(tmp_path / "v1.tif", day1)
    p2 = _write_tif(tmp_path / "v2.tif", day2)
    return _source(p1, "2026-01-01", 1), _source(p2, "2026-01-02", 2)


def test_same_grid_daily_values_are_summed(tmp_path):
    day1 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    day2 = np.array([[10, 20], [30, 40]], dtype=np.int16)
    s1, s2 = _two_sources(tmp_path, day1, day2)

    result = build_accumulation([s1, s2])
    assert result.output.dtype == np.float32
    assert result.output.tolist() == [[11.0, 22.0], [33.0, 44.0]]
    assert result.nodata_pixel_fraction == 0.0
    assert result.info.width == 2 and result.info.height == 2
    assert result.info.crs == CRS.from_epsg(4326)
    assert result.info.dtype == "float32"
    assert math.isnan(result.info.nodata)
    assert [c.day for c in result.contributions] == ["2026-01-01", "2026-01-02"]
    assert [c.version for c in result.contributions] == [1, 2]
    assert [c.covered_pixel_fraction for c in result.contributions] == [1.0, 1.0]
    assert [c.qualifying_pixel_fraction for c in result.contributions] == [1.0, 1.0]


def test_three_day_accumulation(tmp_path):
    p1 = _write_tif(tmp_path / "v1.tif", np.full((2, 2), 1, dtype=np.int16))
    p2 = _write_tif(tmp_path / "v2.tif", np.full((2, 2), 2, dtype=np.int16))
    p3 = _write_tif(tmp_path / "v3.tif", np.full((2, 2), 3, dtype=np.int16))
    sources = [
        _source(p1, "2026-01-01", 1),
        _source(p2, "2026-01-02", 2),
        _source(p3, "2026-01-03", 3),
    ]
    result = build_accumulation(sources)
    assert result.output.tolist() == [[6.0, 6.0], [6.0, 6.0]]
    assert len(result.contributions) == 3


def test_target_grid_comes_from_first_day(tmp_path):
    # Day 1 covers a 3x2 area; day 2 is a smaller, aligned window inside it
    # (middle column of the bottom row only).
    day1 = np.ones((2, 3), dtype=np.int16)
    day2 = np.full((1, 1), 5, dtype=np.int16)
    p1 = _write_tif(tmp_path / "v1.tif", day1, x0=0.0, y0=0.02)
    p2 = _write_tif(tmp_path / "v2.tif", day2, x0=0.01, y0=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert (result.info.width, result.info.height) == (3, 2)
    out = result.output
    # Day 2 only covers one target pixel; the other five are NoData.
    assert np.isnan(out).sum() == 5
    assert out[1, 1] == pytest.approx(6.0)  # day1 mean 1 + day2 mean 5


def test_pixel_below_90_percent_coverage_on_any_day_becomes_nodata(tmp_path):
    # A 1x10 target row; day 2 validly covers 9 of 10 pixels -> 0.9 is exactly
    # the threshold and those nine pixels qualify, the tenth is NoData.
    day1 = np.ones((1, 10), dtype=np.int16)
    for valid_count, expect_finite in ((9, 9), (8, 8)):
        day2 = np.full((1, 10), NODATA, dtype=np.int16)
        day2[0, :valid_count] = 2
        p1 = _write_tif(tmp_path / f"a{valid_count}.tif", day1, y0=0.01)
        p2 = _write_tif(tmp_path / f"b{valid_count}.tif", day2, y0=0.01)
        result = build_accumulation(
            [_source(p1, "d1", 1), _source(p2, "d2", 2)]
        )
        assert np.isfinite(result.output).sum() == expect_finite
        assert result.contributions[1].covered_pixel_fraction == pytest.approx(
            valid_count / 10
        )
        assert result.contributions[1].qualifying_pixel_fraction == pytest.approx(
            valid_count / 10
        )

    # Explicit 0.9 boundary check.
    day2b = np.full((1, 10), NODATA, dtype=np.int16)
    day2b[0, :9] = 2
    p1 = _write_tif(tmp_path / "edge_a.tif", day1, y0=0.01)
    p2 = _write_tif(tmp_path / "edge_b.tif", day2b, y0=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert np.isfinite(result.output[0, 0])
    assert np.isnan(result.output[0, 9])
    assert result.contributions[1].covered_pixel_fraction == pytest.approx(0.9)


def test_area_weighted_mean_over_finer_source_grid(tmp_path):
    # Target pixel is 0.02 wide; day 2 supplies four 0.01 pixels 1,2,3,4.
    day1 = np.full((1, 1), 10, dtype=np.int16)
    day2 = np.array([[1, 2, 3, 4]], dtype=np.int16).reshape(2, 2)
    p1 = _write_tif(tmp_path / "v1.tif", day1, x0=0.0, y0=0.02, res=0.02)
    p2 = _write_tif(tmp_path / "v2.tif", day2, x0=0.0, y0=0.02, res=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    # Area-weighted mean of 1..4 over the whole target pixel is 2.5.
    assert result.output[0, 0] == pytest.approx(12.5)


def test_nodata_source_pixels_are_excluded_from_the_daily_mean(tmp_path):
    # Target (day 1) is one coarse 0.02-wide pixel; day 2 supplies four fine
    # 0.01 pixels with only one valid, so the valid coverage is 1/4 = 0.25 and
    # fails the 90% rule, making the target pixel NoData.
    day1 = np.full((1, 1), 1, dtype=np.int16)
    day2 = np.array([[4, NODATA], [NODATA, NODATA]], dtype=np.int16)
    p1 = _write_tif(tmp_path / "v1.tif", day1, x0=0.0, y0=0.02, res=0.02)
    p2 = _write_tif(tmp_path / "v2.tif", day2, x0=0.0, y0=0.02, res=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert np.isnan(result.output[0, 0])
    assert result.contributions[1].covered_pixel_fraction == pytest.approx(0.25)


def test_different_crs_is_regridded_onto_first_day_grid(tmp_path):
    # Day 1 in geographic CRS; day 2 in Web Mercator covering the same extent
    # (roughly 0..0.00018 deg at the equator ~ 0..20 m).
    day1 = np.full((1, 2), 1, dtype=np.int16)
    p1 = _write_tif(
        tmp_path / "v1.tif", day1, x0=0.0, y0=0.00009, res=0.00009, crs="EPSG:4326"
    )
    day2 = np.full((1, 2), 7, dtype=np.int16)
    p2 = _write_tif(
        tmp_path / "v2.tif", day2, x0=0.0, y0=10.0, res=10.0, crs="EPSG:3857"
    )
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    # Target keeps day 1's geographic grid and CRS.
    assert result.info.crs == CRS.from_epsg(4326)
    assert result.info.width == 2
    finite = np.isfinite(result.output)
    # Wherever day 2 actually covers, the value is 1 + 7 = 8.
    assert np.all(result.output[finite] == pytest.approx(8.0))


def test_float_source_with_nan_nodata_is_supported(tmp_path):
    day1 = np.array([[1.5, 2.5]], dtype=np.float32)
    day2 = np.array([[3.5, np.nan]], dtype=np.float32)
    p1 = _write_tif(tmp_path / "v1.tif", day1, dtype="float32", nodata=np.nan)
    p2 = _write_tif(tmp_path / "v2.tif", day2, dtype="float32", nodata=np.nan)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert result.output[0, 0] == pytest.approx(5.0)
    assert np.isnan(result.output[0, 1])


def test_output_geotiff_is_float32_with_nan_nodata(tmp_path):
    day1 = np.array([[1, NODATA], [3, 4]], dtype=np.int16)
    day2 = np.array([[5, 6], [7, 8]], dtype=np.int16)
    p1 = _write_tif(tmp_path / "v1.tif", day1)
    p2 = _write_tif(tmp_path / "v2.tif", day2)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    out_path = tmp_path / "accum.tif"
    info = write_accumulation(result, out_path)
    verify_accumulation(out_path, info)
    with rasterio.open(out_path) as ds:
        assert ds.dtypes[0] == "float32"
        assert math.isnan(float(ds.nodata))
        values = ds.read(1)
        assert values[0, 1] != values[0, 1]  # NaN
        assert values[0, 0] == pytest.approx(6.0)
        assert ds.crs == CRS.from_epsg(4326)
        assert ds.tags()["RAINMOSAIC_ALGORITHM"] == "daily-area-weighted-mean-accumulation"
        assert ds.tags()["RAINMOSAIC_DAYS"] == "d1,d2"


def test_corrupt_source_file_fails_the_whole_request(tmp_path):
    good = _write_tif(tmp_path / "v1.tif", np.ones((2, 2), dtype=np.int16))
    broken = tmp_path / "v2.tif"
    broken.write_bytes(b"not a geotiff at all")
    with pytest.raises(AccumulationRasterError) as exc:
        load_daily_source(_selection("2026-01-02", 2, broken))
    assert exc.value.day == "2026-01-02"
    # The good source alone must not produce any partial output.
    with pytest.raises(AccumulationRasterError):
        build_accumulation(
            [
                _source(good, "2026-01-01", 1),
                load_daily_source(_selection("2026-01-02", 2, broken)),
            ]
        )


def test_source_without_nodata_treats_non_finite_as_invalid(tmp_path):
    # int16 without a declared nodata: every finite pixel is valid.
    p = _write_tif(tmp_path / "v1.tif", np.array([[1, 2]], dtype=np.int16), nodata=None)
    source = _source(p, "d1", 1)
    assert source.valid.tolist() == [[1, 1]]
    assert np.isfinite(source.data).all()


def test_threshold_constant_is_90_percent():
    assert ACCUM_COVERAGE_THRESHOLD == 0.9


def test_build_requires_at_least_one_source():
    with pytest.raises(ValueError):
        build_accumulation([])


def test_area_weighted_mean_when_fine_pixels_straddle_one_target(tmp_path):
    # One coarse 0.03 target pixel; day 2 fills it with a 3x3 fine grid whose
    # overall mean is 6, so full coverage and an exact area-weighted mean.
    day1 = np.array([[100]], dtype=np.int16)
    day2 = np.array(
        [[3, 6, 9], [3, 6, 9], [3, 6, 9]], dtype=np.int16
    )
    p1 = _write_tif(tmp_path / "v1.tif", day1, x0=0.0, y0=0.03, res=0.03)
    p2 = _write_tif(tmp_path / "v2.tif", day2, x0=0.0, y0=0.03, res=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert result.output[0, 0] == pytest.approx(106.0)
    assert result.contributions[1].covered_pixel_fraction == pytest.approx(1.0)


def test_partial_horizontal_coverage_fraction_is_exact(tmp_path):
    day1 = np.array([[100]], dtype=np.int16)
    # A 3x2 fine grid covers only the left two thirds of the 0.03 target cell.
    day2 = np.array([[3, 6], [3, 6], [3, 6]], dtype=np.int16)
    p1 = _write_tif(tmp_path / "v1.tif", day1, x0=0.0, y0=0.03, res=0.03)
    p2 = _write_tif(tmp_path / "v2.tif", day2, x0=0.0, y0=0.03, res=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert np.isnan(result.output[0, 0])
    assert result.contributions[1].covered_pixel_fraction == pytest.approx(2 / 3)


def test_single_valid_pixel_among_nine_gives_one_ninth_coverage(tmp_path):
    day1 = np.array([[100]], dtype=np.int16)
    day2 = np.full((3, 3), NODATA, dtype=np.int16)
    day2[0, 0] = 5
    p1 = _write_tif(tmp_path / "v1.tif", day1, x0=0.0, y0=0.03, res=0.03)
    p2 = _write_tif(tmp_path / "v2.tif", day2, x0=0.0, y0=0.03, res=0.01)
    result = build_accumulation([_source(p1, "d1", 1), _source(p2, "d2", 2)])
    assert np.isnan(result.output[0, 0])
    assert result.contributions[1].covered_pixel_fraction == pytest.approx(1 / 9)
