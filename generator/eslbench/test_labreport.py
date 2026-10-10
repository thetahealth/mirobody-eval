"""Unit tests for synthetic lab-report generation.

Pure logic: no database, no network, no LLM.

    pytest generator/eslbench/test_labreport.py -v
"""

from __future__ import annotations

import pytest

from .labreport import (
    MAX_ROWS,
    SYNTHETIC_BANNER,
    LabReport,
    build_report,
    render_pdf,
    select_measurements,
)

PANEL = {
    "exam_date": "2025-10-15",
    "exam_type": "routine",
    "exam_location": "hospital",
    "indicators": {
        "TotalCholesterol-TC": {
            "indicator_name": "TotalCholesterol-TC", "value": 5.4, "unit": "mmol/L",
            "reference_range": "3.0-5.18", "status": "abnormal",
        },
        "Triglycerides-TG": {
            "indicator_name": "Triglycerides-TG", "value": 2.4, "unit": "mmol/L",
            "reference_range": "0.4-1.7", "status": "abnormal",
        },
        "High-DensityLipoprotein-HDL": {
            "indicator_name": "High-DensityLipoprotein-HDL", "value": 1.15,
            "unit": "mmol/L", "reference_range": "1.29-1.55", "status": "abnormal",
        },
        "HemoglobinA1c-HbA1c": {
            "indicator_name": "HemoglobinA1c-HbA1c", "value": 7.2, "unit": "%",
            "reference_range": "4.0-6.0", "status": "abnormal",
        },
        "UrineGlucose-GLU": {
            "indicator_name": "UrineGlucose-GLU", "value": 0.0, "unit": "mmol/L",
            "reference_range": "0-0.8", "status": "normal",
        },
        "T-Score": {
            "indicator_name": "T-Score", "value": -1.2, "unit": "",
            "reference_range": ">-1.0", "status": "abnormal",
        },
    },
}

PROFILE = {
    "demographics": {"age": 58, "gender": "Female"},
    "health_profile": {"summary": "Type 2 diabetes, suboptimal control."},
}


class TestSelectMeasurements:
    def test_the_panel_leads_the_way_a_report_prints_it(self):
        """Total cholesterol and triglycerides head a lipid order. Ranking used
        to be `lipid|cholesterol` plus alphabetical, which led with whatever
        sorted first — oxidised LDL, small dense LDL — assays a routine panel
        does not open with."""
        names = [m.name for m in select_measurements(PANEL, limit=4)]

        assert names[0] == "Total Cholesterol"
        assert names[1] == "Triglycerides"

    def test_limit_is_respected(self):
        assert len(select_measurements(PANEL, limit=3)) == 3

    def test_fills_up_with_non_lipid_markers(self):
        rows = select_measurements(PANEL, limit=6)

        assert len(rows) == 6
        assert "Hemoglobin A1c" in [m.name for m in rows]

    def test_carries_units_and_reference_ranges(self):
        tc = next(m for m in select_measurements(PANEL, limit=6) if m.name == "Total Cholesterol")

        assert tc.value == "5.4"
        assert tc.unit == "mmol/L"
        assert tc.reference_range == "3.0-5.18"
        assert tc.abnormal is True

    def test_normal_rows_are_not_flagged(self):
        glu = next(m for m in select_measurements(PANEL, limit=6) if m.name == "Urine Glucose")

        assert glu.abnormal is False

    def test_asking_for_more_than_the_panel_has_returns_what_exists(self):
        assert len(select_measurements(PANEL, limit=MAX_ROWS)) == len(PANEL["indicators"])

    def test_rejects_a_limit_above_what_one_page_holds(self):
        with pytest.raises(ValueError, match="MAX_ROWS"):
            select_measurements(PANEL, limit=MAX_ROWS + 1)

    def test_names_print_as_analytes_not_as_identifiers(self):
        """ESL-Bench ids carry CamelCase words, an optional section prefix and a
        trailing abbreviation. A report prints none of that scaffolding.

        This is not cosmetic. While the abbreviation stayed attached, exactly
        one of ESL-Bench's 190 indicator names survived mirobody's resolver —
        `Total Cholesterol-TC` matches no LOINC entry — so every row of the
        sample report came out unresolved and the standardization half of the
        demo showed nothing at all. Stripping it takes that to 80.
        """
        names = [m.name for m in select_measurements(PANEL, limit=6)]

        assert "Total Cholesterol" in names          # TotalCholesterol-TC
        assert "High-Density Lipoprotein" in names   # internal hyphen survives
        assert "Hemoglobin A1c" in names             # HbA1c is not split apart
        assert "T-Score" in names                    # `Score` is a word, not an abbreviation

    def test_narrative_findings_are_not_printed_as_results(self):
        """Panels mix measurements with imaging impressions. The Result column
        is one number wide, so `Mild hepatic steatosis` printed there arrives
        clipped to `Mild hepat` — and no resolver can code it either."""
        panel = {
            **PANEL,
            "indicators": {
                "AbdominalUltrasound-ABD-US": {
                    "indicator_name": "AbdominalUltrasound-ABD-US",
                    "value": "Mild hepatic steatosis", "unit": "",
                    "reference_range": "", "status": "abnormal",
                },
                "BodyMassIndex-BMI": {
                    "indicator_name": "BodyMassIndex-BMI", "value": 27.1,
                    "unit": "kg/m2", "reference_range": "18.5-24.0", "status": "abnormal",
                },
            },
        }
        rows = select_measurements(panel, limit=6)

        assert [m.name for m in rows] == ["Body Mass Index"]

    def test_section_prefixes_are_dropped(self):
        """A report prints its section once as a heading, never glued onto each
        row. `PhysicalExamination-SystolicBloodPressure` resolves only once the
        prefix is gone."""
        panel = {
            **PANEL,
            "indicators": {
                "PhysicalExamination-SystolicBloodPressure": {
                    "indicator_name": "PhysicalExamination-SystolicBloodPressure",
                    "value": 148, "unit": "mmHg", "reference_range": "<130",
                    "status": "abnormal",
                },
                "Lipid-FreeFattyAcids": {
                    "indicator_name": "Lipid-FreeFattyAcids", "value": 0.62,
                    "unit": "mmol/L", "reference_range": "0.1-0.6", "status": "abnormal",
                },
            },
        }
        names = [m.name for m in select_measurements(panel, limit=6)]

        assert names == ["Free Fatty Acids", "Systolic Blood Pressure"]

    def test_prefixed_duplicate_rows_are_dropped(self):
        """Panels carry the same analyte twice under a prefixed alias
        (`LDL/HDLRatio` and `Lipid-LDL/HDLRatio`). Printing both looks like a
        rendering fault.

        Dropping the section prefix makes the pair render *identically*, which
        an alias test requiring the two names to differ would wave through."""
        panel = {
            **PANEL,
            "indicators": {
                "LDL/HDLRatio": {
                    "indicator_name": "LDL/HDLRatio", "value": 2.78, "unit": "",
                    "reference_range": "<2.5", "status": "abnormal",
                },
                "Lipid-LDL/HDLRatio": {
                    "indicator_name": "Lipid-LDL/HDLRatio", "value": 2.78, "unit": "",
                    "reference_range": "<2.5", "status": "abnormal",
                },
            },
        }
        rows = select_measurements(panel, limit=6)

        assert len(rows) == 1

    def test_distinct_analytes_sharing_a_value_are_both_kept(self):
        """Deduplication keys on the name relationship too — two unrelated
        analytes that happen to read the same are not duplicates."""
        panel = {
            **PANEL,
            "indicators": {
                "Triglycerides-TG": {
                    "indicator_name": "Triglycerides-TG", "value": 2.4, "unit": "mmol/L",
                    "reference_range": "0.4-1.7", "status": "abnormal",
                },
                "TotalCholesterol-TC": {
                    "indicator_name": "TotalCholesterol-TC", "value": 2.4, "unit": "mmol/L",
                    "reference_range": "3.0-5.18", "status": "abnormal",
                },
            },
        }

        assert len(select_measurements(panel, limit=6)) == 2


class TestBuildReport:
    def test_carries_the_panel_date_and_type(self):
        report = build_report(PANEL, PROFILE, subject="user5086@demo")

        assert report.collected == "2025-10-15"
        assert "routine" in report.exam_type.lower()

    def test_subject_line_is_synthetic_not_a_person(self):
        report = build_report(PANEL, PROFILE, subject="user5086@demo")

        assert "user5086@demo" in report.subject
        assert "58" in report.subject
        assert "Female" in report.subject

    def test_report_is_labelled_synthetic(self):
        """A realistic-looking lab report must say on its face that it is not a
        real record — for the reader, and so nobody can pass it off as one."""
        report = build_report(PANEL, PROFILE, subject="user5086@demo")

        assert SYNTHETIC_BANNER in report.banner

    def test_missing_demographics_do_not_break_the_report(self):
        report = build_report(PANEL, {}, subject="user5086@demo")

        assert "user5086@demo" in report.subject


class TestRenderPdf:
    def report(self, **kw) -> LabReport:
        return build_report(PANEL, PROFILE, subject="user5086@demo", **kw)

    def test_produces_a_valid_pdf_envelope(self):
        pdf = render_pdf(self.report())

        assert pdf.startswith(b"%PDF-1.4")
        assert pdf.rstrip().endswith(b"%%EOF")
        assert b"/Type /Catalog" in pdf
        assert b"startxref" in pdf

    def test_xref_offsets_point_at_their_objects(self):
        """A wrong byte offset yields a file that opens in some readers and
        fails in others — worth checking rather than eyeballing."""
        pdf = render_pdf(self.report())

        start = int(pdf.rsplit(b"startxref", 1)[1].split(b"%%EOF")[0].strip())
        xref = pdf[start:]
        assert xref.startswith(b"xref")

        lines = xref.split(b"\n")
        count = int(lines[1].split()[1])
        for obj_no in range(1, count):
            offset = int(lines[1 + obj_no + 1].split()[0])
            assert pdf[offset:].startswith(f"{obj_no} 0 obj".encode()), (
                f"object {obj_no} is not at the offset the xref claims"
            )

    def test_measurements_appear_in_the_content_stream(self):
        pdf = render_pdf(self.report())

        assert b"Total Cholesterol" in pdf
        assert b"5.4" in pdf
        assert b"mmol/L" in pdf

    def test_synthetic_banner_is_on_the_page(self):
        pdf = render_pdf(self.report())

        assert SYNTHETIC_BANNER.encode() in pdf

    def test_parentheses_in_text_are_escaped(self):
        """Unescaped ( ) in a PDF literal string corrupts the content stream."""
        panel = {
            **PANEL,
            "indicators": {
                "Vitamin D (25-OH)": {
                    "indicator_name": "Vitamin D (25-OH)", "value": 30, "unit": "ng/mL",
                    "reference_range": "30-100", "status": "normal",
                },
            },
        }
        pdf = render_pdf(build_report(panel, PROFILE, subject="user5086@demo"))

        assert rb"Vitamin D \(25-OH\)" in pdf

    def test_non_latin_characters_do_not_corrupt_the_stream(self):
        """The standard-14 fonts are WinAnsi; anything outside it is
        transliterated rather than written as raw bytes."""
        panel = {
            **PANEL,
            "indicators": {
                "血脂-TC": {
                    "indicator_name": "血脂-TC", "value": 5.4, "unit": "mmol/L",
                    "reference_range": "3.0-5.18", "status": "abnormal",
                },
            },
        }
        pdf = render_pdf(build_report(panel, PROFILE, subject="user5086@demo"))

        assert "血脂".encode() not in pdf
        assert pdf.startswith(b"%PDF-1.4")

    def test_long_analyte_names_are_clipped_to_their_column(self):
        """PDF text does not wrap or clip on its own — an over-long name would
        print straight through the Result column."""
        panel = {
            **PANEL,
            "indicators": {
                "Lipid-SmallDenseLow-DensityLipoproteinCholesterolFraction": {
                    "indicator_name": "Lipid-SmallDenseLow-DensityLipoproteinCholesterolFraction",
                    "value": 1.1, "unit": "mmol/L", "reference_range": "<0.9",
                    "status": "abnormal",
                },
            },
        }
        report = build_report(panel, PROFILE, subject="user5086@demo")
        pdf = render_pdf(report)

        assert b"..." in pdf, "the over-long name should have been truncated"
        assert report.measurements[0].name.encode() not in pdf, "full name must not be drawn"

    def test_the_identifier_form_never_reaches_the_page(self):
        """A report prints `Triglycerides`, not `Triglycerides-TG`. The trailing
        abbreviation is what kept these names from standardizing."""
        pdf = render_pdf(self.report())

        assert b"Triglycerides" in pdf
        assert b"Triglycerides-TG" not in pdf

    def test_declared_stream_length_matches_the_stream(self):
        pdf = render_pdf(self.report())

        head, rest = pdf.split(b"stream\n", 1)
        body = rest.split(b"\nendstream", 1)[0]
        declared = int(head.rsplit(b"/Length ", 1)[1].split(b" ")[0].split(b">>")[0])

        assert declared == len(body)
