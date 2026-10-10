"""Unit tests for ESL-Bench → mirobody row/document mapping.

Pure logic: no database, no network, no LLM.

    pytest generator/eslbench/test_mapping.py -v
"""

from __future__ import annotations

from datetime import datetime

import pytest

from .mapping import (
    SOURCE_DEVICE,
    SOURCE_EXAM,
    SOURCE_TABLE,
    MappingStats,
    events_document,
    exam_documents,
    profile_document,
    series_row,
    series_rows,
)

USER_ID = "42"
USER_DIR = "user5086_AT_demo"
EMAIL = "user5086@demo"

DEVICE_ENTRY = {
    "time": "2022-09-14T04:31:19",
    "entry_type": "device_indicator",
    "indicator": "24-HourAverageDiastolicBP",
    "device_type": "bp_monitor",
    "value": 76.8604,
    "unit": "mmHg",
}

EXAM_ENTRY = {
    "time": "2022-09-14T08:15:00",
    "entry_type": "exam_indicator",
    "exam_type": "routine",
    "exam_location": "hospital",
    "indicator": "1,25-DihydroxyvitaminD-1,25(OH)2D",
    "value": 110.0,
    "unit": "pmol/L",
}

EVENT_ENTRY = {
    "time": "2020-01-01T00:00:00",
    "end_time": "2022-10-04T00:00:00",
    "entry_type": "event",
    "event": {
        "event_id": "exercise_change_1",
        "event_type": "exercise_change",
        "event_name": "Gym strength and cardio training 4 days/week",
        "start_date": "2020-01-01",
        "duration_days": 2000,
        "interrupted": True,
        "interruption_date": "2022-10-04",
    },
}


class TestSeriesRow:
    def test_device_entry_maps_to_th_series_data_row(self):
        row = series_row(DEVICE_ENTRY, user_id=USER_ID, user_dir=USER_DIR)

        assert row["user_id"] == USER_ID
        assert row["indicator"] == "24-HourAverageDiastolicBP"
        assert row["value"] == "76.8604"
        assert row["source"] == SOURCE_DEVICE
        assert row["source_table"] == SOURCE_TABLE
        assert row["source_table_id"] == USER_DIR

    def test_point_measurement_has_equal_start_and_end(self):
        """Device/exam readings are instants, not intervals. The
        `(user_id, indicator, start_time, end_time)` unique key relies on this."""
        row = series_row(DEVICE_ENTRY, user_id=USER_ID, user_dir=USER_DIR)

        assert row["start_time"] == datetime(2022, 9, 14, 4, 31, 19)
        assert row["end_time"] == row["start_time"]
        assert row["start_time"].tzinfo is None, "column is `timestamp without time zone`"

    def test_unit_travels_in_fhir_mapping_info(self):
        """`health_indicator_service._fetch_indicator_data_batch` reads the unit
        from `fhir_mapping_info->>'unit'` — nowhere else. A row without it
        shows the agent a bare number."""
        row = series_row(DEVICE_ENTRY, user_id=USER_ID, user_dir=USER_DIR)

        assert row["fhir_mapping_info"] == '{"unit": "mmHg"}'

    def test_fhir_id_left_null_for_indicator_sync_to_resolve(self):
        """`IndicatorSyncTask.backfill_from_registry` fills this; guessing here
        would bypass its ambiguity guards."""
        row = series_row(DEVICE_ENTRY, user_id=USER_ID, user_dir=USER_DIR)

        assert row["fhir_id"] is None

    def test_device_comment_records_device_type(self):
        row = series_row(DEVICE_ENTRY, user_id=USER_ID, user_dir=USER_DIR)

        assert "bp_monitor" in row["comment"]

    def test_exam_entry_uses_exam_source_and_context(self):
        row = series_row(EXAM_ENTRY, user_id=USER_ID, user_dir=USER_DIR)

        assert row["source"] == SOURCE_EXAM
        assert row["indicator"] == "1,25-DihydroxyvitaminD-1,25(OH)2D"
        assert row["value"] == "110.0"
        assert "routine" in row["comment"]
        assert "hospital" in row["comment"]

    def test_event_entry_is_not_a_series_row(self):
        assert series_row(EVENT_ENTRY, user_id=USER_ID, user_dir=USER_DIR) is None

    def test_unknown_entry_type_is_skipped(self):
        entry = {"time": "2022-01-01T00:00:00", "entry_type": "something_new"}

        assert series_row(entry, user_id=USER_ID, user_dir=USER_DIR) is None

    @pytest.mark.parametrize("missing", ["time", "indicator", "value"])
    def test_entry_missing_a_required_field_is_skipped(self, missing):
        entry = {k: v for k, v in DEVICE_ENTRY.items() if k != missing}

        assert series_row(entry, user_id=USER_ID, user_dir=USER_DIR) is None

    def test_null_value_is_skipped_not_stored_as_string_none(self):
        entry = {**DEVICE_ENTRY, "value": None}

        assert series_row(entry, user_id=USER_ID, user_dir=USER_DIR) is None

    def test_missing_unit_yields_empty_unit_not_none(self):
        entry = {k: v for k, v in DEVICE_ENTRY.items() if k != "unit"}
        row = series_row(entry, user_id=USER_ID, user_dir=USER_DIR)

        assert row["fhir_mapping_info"] == '{"unit": ""}'

    def test_string_value_is_preserved(self):
        entry = {**DEVICE_ENTRY, "value": "positive"}
        row = series_row(entry, user_id=USER_ID, user_dir=USER_DIR)

        assert row["value"] == "positive"


class TestSeriesRows:
    def test_filters_events_and_keeps_indicators(self):
        rows = list(series_rows(
            [DEVICE_ENTRY, EVENT_ENTRY, EXAM_ENTRY], user_id=USER_ID, user_dir=USER_DIR,
        ))

        assert [r["source"] for r in rows] == [SOURCE_DEVICE, SOURCE_EXAM]

    def test_is_lazy(self):
        """Timelines run to ~75k entries; the caller batches inserts off this."""
        import types

        assert isinstance(series_rows([], user_id=USER_ID, user_dir=USER_DIR), types.GeneratorType)


class TestMappingStats:
    def test_accounts_for_every_entry(self):
        entries = [
            DEVICE_ENTRY,
            EXAM_ENTRY,
            EVENT_ENTRY,
            {**DEVICE_ENTRY, "value": None},
            {**DEVICE_ENTRY, "time": "not-a-date"},
            {"entry_type": "future_entry_kind"},
        ]
        stats = MappingStats()
        list(series_rows(entries, user_id=USER_ID, user_dir=USER_DIR, stats=stats))

        assert stats.entries == 6
        assert stats.device_rows == 1
        assert stats.exam_rows == 1
        assert stats.events == 1
        assert stats.null_value == 1
        assert stats.malformed == 1
        assert stats.other_entry_type == 1
        assert stats.rows == 2

    def test_counts_unique_key_collisions(self):
        """ESL-Bench occasionally records two lab values at the same instant;
        `(user_id, indicator, start_time, end_time)` can only hold one."""
        entries = [EXAM_ENTRY, {**EXAM_ENTRY, "value": 111.0}]
        stats = MappingStats()
        rows = list(series_rows(entries, user_id=USER_ID, user_dir=USER_DIR, stats=stats))

        assert len(rows) == 2, "both rows are yielded; the database collapses them"
        assert stats.key_collisions == 1

    def test_summary_names_what_was_dropped(self):
        stats = MappingStats()
        list(series_rows(
            [DEVICE_ENTRY, {**DEVICE_ENTRY, "value": None}],
            user_id=USER_ID, user_dir=USER_DIR, stats=stats,
        ))
        summary = stats.summary()

        assert "1 rows" in summary
        assert "null-valued" in summary

    def test_summary_omits_dropped_clause_when_nothing_dropped(self):
        stats = MappingStats()
        list(series_rows([DEVICE_ENTRY], user_id=USER_ID, user_dir=USER_DIR, stats=stats))

        assert "dropped" not in stats.summary()


class TestEventsDocument:
    def test_renders_dates_and_name(self):
        doc = events_document([EVENT_ENTRY, DEVICE_ENTRY], EMAIL)

        assert "Gym strength and cardio training 4 days/week" in doc
        assert "exercise_change" in doc
        assert "2020-01-01" in doc
        assert "2022-10-04" in doc, "interruption-adjusted end date must be present"

    def test_omits_causal_ground_truth(self):
        """`events.json` carries `affected_indicators` / `impact_reasoning` — the
        answer chain for Explanation questions. The document is built from the
        timeline's event view only, which is also what the ESL-Bench reference
        target exposes, so scores stay comparable and answers stay hidden."""
        entry = {
            **EVENT_ENTRY,
            "event": {
                **EVENT_ENTRY["event"],
                "affected_indicators": [{"indicator_name": "HRV", "expected_change": "+6"}],
                "impact_reasoning": "exercise raises vagal tone therefore HRV increases",
                "description": "Regular gym attendance 4 days a week.",
            },
        }
        doc = events_document([entry], EMAIL)

        assert "affected_indicators" not in doc
        assert "expected_change" not in doc
        assert "vagal tone" not in doc
        assert "HRV" not in doc

    def test_uninterrupted_event_still_renders(self):
        entry = {
            "time": "2021-05-02T00:00:00",
            "entry_type": "event",
            "event": {
                "event_id": "diet_change_9",
                "event_type": "diet_change",
                "event_name": "Low sodium diet",
                "start_date": "2021-05-02",
                "duration_days": 30,
            },
        }
        doc = events_document([entry], EMAIL)

        assert "Low sodium diet" in doc
        assert "2021-05-02" in doc

    def test_no_events_is_reported_not_silently_empty(self):
        doc = events_document([DEVICE_ENTRY], EMAIL)

        assert EMAIL in doc
        assert "no life events" in doc.lower()


class TestProfileDocument:
    PROFILE = {
        "metadata": {"profile_id": "user5086_AT_demo", "use_case": {"name": "Healthy control cohort"}},
        "demographics": {"age": 34, "gender": "Male", "occupation": "Sales Manager"},
        "personality": "High-energy sales manager.",
        "health_profile": {
            "summary": "Generally excellent health status.",
            "physical_measurements": {"height": 178, "weight": 79},
            "family_history": [{"relative": "Father", "condition": "Hypertension"}],
            "allergies_and_intolerances": ["Seasonal pollen"],
            "chronic_conditions": [],
        },
    }

    def test_renders_nested_values(self):
        doc = profile_document(self.PROFILE, EMAIL)

        assert "Sales Manager" in doc
        assert "178" in doc
        assert "Hypertension" in doc
        assert "Seasonal pollen" in doc

    def test_drops_generator_bookkeeping(self):
        """`metadata` describes how the synthetic user was generated (cohort
        label, generation params). It is not health data a real user would
        have, and no question keys on it."""
        doc = profile_document(self.PROFILE, EMAIL)

        assert "Healthy control cohort" not in doc
        assert "profile_id" not in doc

    def test_empty_list_does_not_vanish_silently(self):
        """"no chronic conditions" is itself an answer — several Lookup
        questions ask whether a condition is present."""
        doc = profile_document(self.PROFILE, EMAIL)

        assert "chronic_conditions" in doc.replace(" ", "_").lower() or "Chronic Conditions" in doc


class TestExamDocuments:
    EXAMS = [
        {
            "exam_date": "2022-09-14",
            "exam_type": "routine",
            "exam_location": "hospital",
            "overall_assessment": "Healthy adult male.",
            "recommendations": ["Maintain exercise"],
            "abnormal_findings": [],
            "generation_summary": {"seed": 1234},
            "indicators": {
                "Microalbuminuria-MAU": {
                    "indicator_name": "Microalbuminuria-MAU",
                    "value": 6.5,
                    "unit": "mg/L",
                    "reference_range": "5.525-7.475",
                    "status": "normal",
                    "clinical_significance": "Within normal range.",
                    "generation_metadata": {"annual_drift_rate_pct": 0.0},
                },
            },
        },
        {
            "exam_date": "2022-10-14",
            "exam_type": "follow-up",
            "exam_location": "hospital",
            "indicators": {
                "HbA1c": {"indicator_name": "HbA1c", "value": 5.4, "unit": "%", "status": "normal"},
            },
        },
    ]

    def test_one_document_per_exam(self):
        docs = exam_documents(self.EXAMS, EMAIL)

        assert len(docs) == 2
        assert [name for name, _ in docs] == ["exam_2022-09-14.md", "exam_2022-10-14.md"]

    def test_renders_clinical_context(self):
        (_, body), _ = exam_documents(self.EXAMS, EMAIL)

        assert "Microalbuminuria-MAU" in body
        assert "6.5" in body
        assert "mg/L" in body
        assert "5.525-7.475" in body
        assert "normal" in body
        assert "Within normal range." in body
        assert "Healthy adult male." in body
        assert "Maintain exercise" in body

    def test_drops_generator_bookkeeping(self):
        (_, body), _ = exam_documents(self.EXAMS, EMAIL)

        assert "generation_metadata" not in body
        assert "annual_drift_rate_pct" not in body
        assert "generation_summary" not in body
        assert "seed" not in body

    def test_exam_without_optional_sections_still_renders(self):
        _, (name, body) = exam_documents(self.EXAMS, EMAIL)

        assert name == "exam_2022-10-14.md"
        assert "HbA1c" in body

    def test_stays_within_inline_storage_limit(self):
        """DeepAgent's filesystem backend keeps text <= 256 KB inline and
        offloads anything larger to object storage — which a bare
        self-hosted deployment may not have configured. One document per
        exam keeps every document inline."""
        big = {
            "exam_date": "2023-01-01",
            "exam_type": "routine",
            "indicators": {
                f"Indicator{i}": {
                    "indicator_name": f"Indicator{i}",
                    "value": i,
                    "unit": "mg/L",
                    "reference_range": "1-2",
                    "status": "normal",
                    "clinical_significance": "x" * 2000,
                }
                for i in range(220)
            },
        }
        (_, body), = exam_documents([big], EMAIL)

        assert len(body.encode("utf-8")) < 256 * 1024
