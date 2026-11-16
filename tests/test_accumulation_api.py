"""API behaviour for multi-day rainfall accumulation."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import rasterio

from app import accumulation as accum_mod
from app.rasters import NODATA
from tests.conftest import make_geotiff, read_band, submit_payload

DAYS = ["2026-01-01", "2026-01-02", "2026-01-03"]


def _publish_day(client, day: str, values, *, expected_version: int, res: float = 0.01):
    """Publish one day whose mosaic equals ``values`` exactly.

    The array is split into two adjacent blocks that touch without overlap or
    a gap, so the mosaic endpoint reconstructs the full array on its grid.
    """

    array = np.asarray(values, dtype=np.int16)
    height, width = array.shape
    assert width % 2 == 0, "test arrays need an even width to split into blocks"
    half = width // 2
    block_a = make_geotiff(array[:, :half], x0=0.0, res=res)
    block_b = make_geotiff(array[:, half:], x0=half * res, res=res)
    data, files = submit_payload(
        block_a=block_a,
        block_b=block_b,
        expected_version=expected_version,
        day=day,
        a_time=f"{day}T01:00:00Z",
        b_time=f"{day}T02:00:00Z",
    )
    response = client.post("/api/v1/mosaics", data=data, files=files)
    assert response.status_code == 201, response.text
    return response.json()["version"]


def _publish_two_days(client, day1_values, day2_values):
    v1 = _publish_day(client, DAYS[0], day1_values, expected_version=0)
    v2 = _publish_day(client, DAYS[1], day2_values, expected_version=v1)
    return v1, v2


def test_accumulation_tiff_is_float32_nan_nodata_and_summed(client):
    day1 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    day2 = np.array([[10, 20], [30, 40]], dtype=np.int16)
    _publish_two_days(client, day1, day2)

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1]},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/tiff"
    assert "rainfall-accumulation-2026-01-01_2026-01-02.tif" in response.headers[
        "content-disposition"
    ]
    assert response.headers["x-accumulation-days"] == ",".join(DAYS[:2])
    assert response.headers["x-accumulation-versions"] == "1,2"
    assert response.headers["x-accumulation-coverage-threshold"] == "0.9"
    assert response.headers["cache-control"] == "no-store"

    values, meta = read_band(response.content)
    assert meta["dtype"] == "float32"
    assert np.isnan(meta["nodata"])
    assert values.tolist() == [[11.0, 22.0], [33.0, 44.0]]
    assert np.isfinite(values).all()


def test_accumulation_json_lists_days_versions_and_coverage(client):
    day1 = np.array([[1, 2]], dtype=np.int16)
    day2 = np.array([[3, NODATA]], dtype=np.int16)
    _publish_two_days(client, day1, day2)

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["start_day"] == DAYS[0]
    assert body["end_day"] == DAYS[1]
    assert body["days"] == DAYS[:2]
    assert body["coverage_threshold"] == 0.9
    assert [s["day"] for s in body["sources"]] == DAYS[:2]
    assert [s["version"] for s in body["sources"]] == [1, 2]
    assert all("published_at" in s and "sha256" in s for s in body["sources"])
    # Second day has one NoData target pixel -> mean coverage 0.5.
    assert body["sources"][1]["covered_pixel_fraction"] == 0.5
    assert body["sources"][1]["qualifying_pixel_fraction"] == 0.5
    assert body["image"]["dtype"] == "float32"
    assert body["image"]["nodata"] == "NaN"
    assert body["image"]["nodata_pixel_fraction"] == 0.5


def test_uses_newest_version_of_each_day_from_snapshot(client):
    day1 = np.array([[1, 1]], dtype=np.int16)
    day2 = np.array([[2, 2]], dtype=np.int16)
    v1, v2 = _publish_two_days(client, day1, day2)

    # Republish day 1 with larger values; it becomes version 3 (newest for that day).
    v3 = _publish_day(client, DAYS[0], np.array([[10, 10]], dtype=np.int16), expected_version=v2)
    assert v3 == 3

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "json"},
    )
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert [s["version"] for s in sources] == [3, 2]

    tiff = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1]},
    )
    values, _ = read_band(tiff.content)
    assert values.tolist() == [[12.0, 12.0]]


def test_missing_day_fails_entire_request_with_no_partial_image(client):
    _publish_day(client, DAYS[0], np.array([[1, 1]], dtype=np.int16), expected_version=0)
    # Nothing published for 2026-01-02.
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "json"},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "day_not_published"
    assert detail["day"] == DAYS[1]


def test_three_day_range_is_supported(client):
    versions = []
    expected = 0
    for index, day in enumerate(DAYS):
        values = np.full((2, 2), index + 1, dtype=np.int16)
        expected = _publish_day(client, day, values, expected_version=expected)
        versions.append(expected)

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[2], "format": "json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["days"] == DAYS
    assert [s["version"] for s in body["sources"]] == versions

    tiff = client.get(
        "/api/v1/accumulations", params={"start_day": DAYS[0], "end_day": DAYS[2]}
    )
    values, meta = read_band(tiff.content)
    assert meta["dtype"] == "float32"
    # 1 + 2 + 3 = 6 everywhere.
    assert values.tolist() == [[6.0, 6.0], [6.0, 6.0]]


def test_range_validation_rejects_wrong_span_and_order(client):
    # One day only.
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[0]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_range"

    # Four days.
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": "2026-01-04"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_range"

    # Reversed order.
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[1], "end_day": DAYS[0]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_range"

    # Malformed date.
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": "01-01-2026", "end_day": DAYS[1]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_day"

    # Bad format value.
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "png"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_format"


def test_missing_source_file_fails_with_410(client, settings):
    _publish_two_days(
        client,
        np.array([[1, 1]], dtype=np.int16),
        np.array([[2, 2]], dtype=np.int16),
    )
    # Remove the immutable file of day 2's version after publication.
    (settings.versions_dir / "v2.tif").unlink()

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1]},
    )
    assert response.status_code == 410
    detail = response.json()["detail"]
    assert detail["code"] == "source_image_missing"
    assert detail["day"] == DAYS[1]


def test_corrupt_source_file_fails_with_422(client, settings):
    _publish_two_days(
        client,
        np.array([[1, 1]], dtype=np.int16),
        np.array([[2, 2]], dtype=np.int16),
    )
    (settings.versions_dir / "v2.tif").write_bytes(b"broken tiff bytes")

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1]},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "source_raster_error"
    assert detail["day"] == DAYS[1]
    # No partial accumulation file is left behind.
    assert list(settings.tmp_dir.glob("accum-*.tif")) == []


def test_accumulation_ignores_versions_published_during_calculation(client, monkeypatch):
    day1 = np.array([[1, 1]], dtype=np.int16)
    day2 = np.array([[2, 2]], dtype=np.int16)
    v1, v2 = _publish_two_days(client, day1, day2)

    real_build = accum_mod.build_accumulation
    published_during = threading.Event()

    def build_then_publish(sources):
        # Simulate a new version of day 1 landing while pixels are computed.
        if not published_during.is_set():
            _publish_day(
                client,
                DAYS[0],
                np.array([[100, 100]], dtype=np.int16),
                expected_version=v2,
            )
            published_during.set()
        return real_build(sources)

    monkeypatch.setattr(accum_mod, "build_accumulation", build_then_publish)
    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "json"},
    )
    assert response.status_code == 200, response.text
    # The in-flight snapshot selected versions 1 and 2, not the concurrently
    # published version 3.
    assert [s["version"] for s in response.json()["sources"]] == [1, 2]

    monkeypatch.undo()
    # A request issued after the publication picks up the new latest version.
    later = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "json"},
    )
    assert later.status_code == 200
    assert [s["version"] for s in later.json()["sources"]] == [3, 2]
    values, _ = read_band(
        client.get(
            "/api/v1/accumulations",
            params={"start_day": DAYS[0], "end_day": DAYS[1]},
        ).content
    )
    assert values.tolist() == [[102.0, 102.0]]


def test_existing_mosaic_download_still_works_alongside_accumulation(client):
    _publish_two_days(
        client,
        np.array([[1, 2]], dtype=np.int16),
        np.array([[3, 4]], dtype=np.int16),
    )
    # Original publication endpoints are untouched.
    assert client.get("/api/v1/mosaics/current").status_code == 200
    download = client.get("/api/v1/mosaics/1/download")
    assert download.status_code == 200
    values, meta = read_band(download.content)
    assert meta["dtype"] == "int16"
    assert values.tolist() == [[1, 2]]

    accum = client.get(
        "/api/v1/accumulations", params={"start_day": DAYS[0], "end_day": DAYS[1]}
    )
    assert accum.status_code == 200


def test_different_crs_day_is_regridded_onto_first_day_grid(client, settings):
    # Day 1 in EPSG:4326 (the target grid); publish via the normal endpoint.
    _publish_day(client, DAYS[0], np.array([[1, 1]], dtype=np.int16), expected_version=0)

    # Hand-craft a day-2 version in EPSG:3857 by writing its file and row
    # directly through the store, since the mosaic endpoint only accepts one CRS.
    import sqlite3

    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    path = settings.versions_dir / "v2.tif"
    # Compute the target day-1 footprint in EPSG:3857 and build a day-2 source
    # over that same real-world area, so the two rasters fully overlap.
    from rasterio.warp import transform_bounds

    target_bounds_4326 = (0.0, 0.01, 0.02, 0.02)
    left, bottom, right, top = transform_bounds(
        CRS.from_epsg(4326), CRS.from_epsg(3857), *target_bounds_4326
    )
    res_x = (right - left) / 2
    res_y = (top - bottom) / 2
    array = np.full((2, 2), 7, dtype=np.int16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=from_origin(left, top, res_x, res_y),
        nodata=NODATA,
        compress="deflate",
    ) as ds:
        ds.write(array, 1)

    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO versions (
                version, day, status, published_at, width, height, crs_wkt,
                crs_label, transform, bounds, res_x, res_y, dtype, nodata,
                sha256, size_bytes, block_meta
            ) VALUES (?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2,
                DAYS[1],
                "2026-01-02T03:00:00.000000Z",
                2,
                2,
                CRS.from_epsg(3857).to_wkt(),
                "EPSG:3857",
                json.dumps([res_x, 0.0, left, 0.0, -res_y, top]),
                json.dumps([left, bottom, right, top]),
                res_x,
                res_y,
                "int16",
                NODATA,
                "a" * 64,
                path.stat().st_size,
                "[]",
            ),
        )
        conn.execute(
            "INSERT INTO state (key, value) VALUES ('current_version', '2') "
            "ON CONFLICT(key) DO UPDATE SET value = '2'"
        )
        conn.commit()
    finally:
        conn.close()

    response = client.get(
        "/api/v1/accumulations",
        params={"start_day": DAYS[0], "end_day": DAYS[1], "format": "json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Target grid remains day 1's EPSG:4326 grid.
    assert body["image"]["crs"] == "EPSG:4326"
    assert [s["version"] for s in body["sources"]] == [1, 2]

    tiff = client.get(
        "/api/v1/accumulations", params={"start_day": DAYS[0], "end_day": DAYS[1]}
    )
    values, meta = read_band(tiff.content)
    assert meta["crs"] == "EPSG:4326"
    assert meta["dtype"] == "float32"
    finite = np.isfinite(values)
    assert finite.any()
    assert np.all(values[finite] == pytest.approx(8.0))


def test_nodata_pixel_when_any_day_covers_less_than_90_percent(client):
    # 2x2 target; day 2 has NoData in one pixel -> that pixel gets < 90% cover.
    day1 = np.ones((2, 2), dtype=np.int16)
    day2 = np.array([[5, 5], [5, NODATA]], dtype=np.int16)
    _publish_two_days(client, day1, day2)

    response = client.get(
        "/api/v1/accumulations", params={"start_day": DAYS[0], "end_day": DAYS[1]}
    )
    values, _ = read_band(response.content)
    assert np.isnan(values[1, 1])
    assert values[0, 0] == pytest.approx(6.0)
    assert values[1, 0] == pytest.approx(6.0)
    assert values[0, 1] == pytest.approx(6.0)


def test_concurrent_accumulations_are_independent(client):
    _publish_two_days(
        client,
        np.ones((2, 2), dtype=np.int16),
        np.full((2, 2), 2, dtype=np.int16),
    )

    def call():
        return client.get(
            "/api/v1/accumulations", params={"start_day": DAYS[0], "end_day": DAYS[1]}
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = [future.result() for future in [pool.submit(call) for _ in range(4)]]
    assert all(r.status_code == 200 for r in responses)
    contents = {r.content for r in responses}
    assert len(contents) == 1  # identical inputs -> identical output bytes
