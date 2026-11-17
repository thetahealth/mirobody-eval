"""The indicator tables hold exactly what the per-row INSERT used to put there.

Pure logic: a throwaway user directory and an in-memory DuckDB, no network.

    pytest benchmark/data/eslbench/tools/test_retrieve_bulk_insert.py -v

`_hydrate_device_indicators` / `_hydrate_exam_indicators` load through one NDJSON read instead of
`executemany`. Each test builds the expected table the old way — `executemany` over the rows the
hydrate step should produce — and requires the new table to be identical, row for row, including
NULLs, empty strings, text that needs quoting and both timestamp spellings found in the data.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from benchmark.data.eslbench.tools import retrieve

USER = "user9001@demo"


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "_DOT_DATA", tmp_path)
    context = retrieve.ToolContext(user_email=USER)
    context.data_dir.mkdir(parents=True)
    return context


def _write(ctx, name: str, payload) -> None:
    (ctx.data_dir / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _table(con, table: str) -> list[tuple]:
    return con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()


def _old_way(ddl: str, table: str, rows: list[tuple]) -> list[tuple]:
    con = duckdb.connect()
    con.execute(ddl)
    placeholders = ", ".join("?" for _ in rows[0])
    con.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
    return _table(con, table)


def test_device_rows_match_the_per_row_insert(ctx):
    _write(
        ctx,
        "timeline.json",
        {
            "user_id": "u-1",
            "entries": [
                {
                    "entry_type": "device_indicator",
                    "time": "2024-03-01T07:48:39",
                    "indicator": "BodyWeight",
                    "device_type": "smart_scale",
                    "value": 92.0109,
                    "unit": "kg",
                },
                {
                    "entry_type": "device_indicator",
                    "time": "2024-03-02 00:00:00",
                    "indicator": "BodyWeight",
                    "device_type": "smart_scale",
                    "value": None,
                    "unit": "kg",
                },
                {
                    "entry_type": "device_indicator",
                    "time": "2024-03-03T08:00:00",
                    "indicator": "Steps",
                    "device_type": "smartwatch",
                    "value": 8454,
                    "unit": None,
                },
                {
                    "entry_type": "device_indicator",
                    "time": "2024-03-04T08:00:00",
                    "indicator": 'Note, "quoted"\nsecond line 中文',
                    "device_type": "smartwatch",
                    "value": "",
                    "unit": "-",
                },
                {
                    "entry_type": "exam_indicator",
                    "time": "2024-03-05T09:00:00",
                    "indicator": "LDL",
                    "value": 3.1,
                    "unit": "mmol/L",
                },
            ],
        },
    )
    expected = _old_way(
        "CREATE TABLE device_indicators(user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,"
        " device_type VARCHAR, value VARCHAR, unit VARCHAR)",
        "device_indicators",
        [
            ("u-1", "2024-03-01T07:48:39", "BodyWeight", "smart_scale", "92.0109", "kg"),
            ("u-1", "2024-03-02 00:00:00", "BodyWeight", "smart_scale", None, "kg"),
            ("u-1", "2024-03-03T08:00:00", "Steps", "smartwatch", "8454", None),
            ("u-1", "2024-03-04T08:00:00", 'Note, "quoted"\nsecond line 中文', "smartwatch", "", "-"),
        ],
    )

    con = duckdb.connect()
    retrieve._hydrate_device_indicators(con, ctx)

    assert _table(con, "device_indicators") == expected


def test_exam_rows_match_the_per_row_insert(ctx):
    _write(ctx, "timeline.json", {"user_id": "u-1", "entries": []})
    _write(
        ctx,
        "exam_data.json",
        [
            {
                "exam_date": "2024-03-05",
                "exam_type": "annual",
                "exam_location": "Clinic, Room 2",
                "indicators": {
                    "LDL": {"value": 3.1, "unit": "mmol/L"},
                    "HbA1c": {"value": None, "unit": "%", "timestamp": "2024-03-05T09:30:00"},
                    "Urine color": {"indicator_name": "尿色", "value": "yellow", "unit": None},
                },
            },
        ],
    )
    expected = _old_way(
        "CREATE TABLE exam_indicators(user_id VARCHAR, time TIMESTAMP, indicator VARCHAR,"
        " exam_type VARCHAR, exam_location VARCHAR, value VARCHAR, unit VARCHAR)",
        "exam_indicators",
        [
            ("u-1", "2024-03-05 00:00:00", "LDL", "annual", "Clinic, Room 2", "3.1", "mmol/L"),
            ("u-1", "2024-03-05T09:30:00", "HbA1c", "annual", "Clinic, Room 2", None, "%"),
            ("u-1", "2024-03-05 00:00:00", "尿色", "annual", "Clinic, Room 2", "yellow", None),
        ],
    )

    con = duckdb.connect()
    retrieve._hydrate_exam_indicators(con, ctx)

    assert _table(con, "exam_indicators") == expected


def test_no_rows_still_leaves_an_empty_table(ctx):
    con = duckdb.connect()

    retrieve._hydrate_device_indicators(con, ctx)
    retrieve._hydrate_exam_indicators(con, ctx)

    assert _table(con, "device_indicators") == []
    assert _table(con, "exam_indicators") == []
