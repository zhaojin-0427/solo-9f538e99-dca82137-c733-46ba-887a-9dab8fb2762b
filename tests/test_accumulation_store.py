"""Store-level snapshot selection for multi-day accumulation."""

from __future__ import annotations

import pytest

from app import store
from app.store import AccumulationError


def _publish(settings, day: str, expected_version: int) -> int:
    """Insert a minimal ready version row + file without the raster pipeline."""

    import json
    import sqlite3

    store.init_store(settings)
    version = expected_version + 1
    path = settings.versions_dir / f"v{version}.tif"
    path.write_bytes(b"fake-tiff-bytes")
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(
            """
            INSERT INTO versions (
                version, day, status, published_at, width, height, crs_wkt,
                crs_label, transform, bounds, res_x, res_y, dtype, nodata,
                sha256, size_bytes, block_meta
            ) VALUES (?, ?, 'ready', ?, 1, 1, '', 'EPSG:4326',
                      ?, ?, 0.01, 0.01, 'int16', -32768, ?, ?, '[]')
            """,
            (
                version,
                day,
                f"{day}T00:00:00.000000Z",
                json.dumps([0.01, 0.0, 0.0, 0.0, -0.01, 0.01]),
                json.dumps([0.0, 0.0, 0.01, 0.01]),
                "0" * 64,
                path.stat().st_size,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return version


def test_selects_newest_version_per_day_in_order(settings):
    v1 = _publish(settings, "2026-01-01", 0)
    v2 = _publish(settings, "2026-01-02", v1)
    _publish(settings, "2026-01-01", v2)  # newer republish of day 1

    selected = store.select_daily_versions(settings, ["2026-01-01", "2026-01-02"])
    assert [(s.day, s.version) for s in selected] == [
        ("2026-01-01", 3),
        ("2026-01-02", 2),
    ]
    assert all(s.path.is_file() for s in selected)
    # The older day-1 version is simply not chosen, not deleted.
    assert (settings.versions_dir / "v1.tif").is_file()


def test_missing_day_raises_and_nothing_is_returned(settings):
    _publish(settings, "2026-01-01", 0)
    with pytest.raises(AccumulationError) as exc:
        store.select_daily_versions(settings, ["2026-01-01", "2026-01-02"])
    assert exc.value.code == "day_not_published"
    assert exc.value.day == "2026-01-02"


def test_missing_source_file_raises(settings):
    version = _publish(settings, "2026-01-01", 0)
    _publish(settings, "2026-01-02", version)
    (settings.versions_dir / "v2.tif").unlink()

    with pytest.raises(AccumulationError) as exc:
        store.select_daily_versions(settings, ["2026-01-01", "2026-01-02"])
    assert exc.value.code == "source_image_missing"
    assert exc.value.day == "2026-01-02"


def test_selection_is_one_snapshot_ignoring_later_publication(settings, monkeypatch):
    v1 = _publish(settings, "2026-01-01", 0)
    _publish(settings, "2026-01-02", v1)

    real_connect = store.connect
    calls = {"n": 0}

    class TrackingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, *a, **k):
            return self._inner.execute(*a, **k)

        def close(self):
            calls["n"] += 1
            return self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def tracking_connect(s):
        return TrackingConn(real_connect(s))

    monkeypatch.setattr(store, "connect", tracking_connect)
    selected = store.select_daily_versions(settings, ["2026-01-01", "2026-01-02"])
    assert [s.version for s in selected] == [1, 2]
    # Both day lookups share one connection, i.e. one database snapshot.
    assert calls["n"] == 1
