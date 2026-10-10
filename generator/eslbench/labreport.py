"""Render one ESL-Bench exam panel as a synthetic lab-report PDF.

This is the demo's opening beat: a file a newcomer can drop into mirobody and
watch get parsed, unit-normalised, LOINC-coded and then reasoned over. It pairs
with :mod:`generator.eslbench.seed_mirobody` — seed the user's *earlier* panels,
hand them the *latest* one as a PDF, and "what's my lipid trend?" has real
history to trend against instead of a single point.

The PDF is written by hand rather than through a rendering library: mirobody
already ships no PDF *writer*, and adding one for a sample file is not worth a
dependency. Only the standard-14 fonts are used, so nothing is embedded. Text
is transliterated to WinAnsi — the PDF is a Latin-script document.

Extraction does not depend on a text layer either way: mirobody parses PDFs
through ``unified_file_extract`` (multimodal), not through a text scraper. A
text-layer PDF is simply smaller and crisper than a rasterised one.

**The output is labelled synthetic on its face.** It looks like a lab report,
which is exactly why it says it is not one — a realistic-looking medical record
should never be mistakable for a real patient's.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#-----------------------------------------------------------------------------

SYNTHETIC_BANNER = "SYNTHETIC SAMPLE - GENERATED DATA, NOT A REAL PATIENT RECORD"

# One A4 page at 11pt leading holds roughly this many measurement rows once the
# header and footer are laid out. Exceeding it raises rather than silently
# dropping rows off the bottom of the page.
MAX_ROWS = 34

# The analytes a real lipid-and-glucose order prints, in the order it prints
# them. Ranking by position in this list rather than by a broad `lipid|glucose`
# regex plus alphabetical order, which surfaced whatever sorted first —
# oxidised LDL, small dense LDL, LDL particle number. Those are send-out
# subfraction assays; leading a routine panel with them is not what a lab does,
# and they are also the least standardisable rows in the dataset.
#
# The list deliberately keeps rows that do not resolve (HDL, LDL, the ratio):
# a panel where every row standardises would misrepresent what this step does.
_PANEL_ORDER = tuple(re.compile(p, re.I) for p in (
    r"^TotalCholesterol",
    r"^Triglycerides",
    r"^High-DensityLipoprotein",
    r"^Low-DensityLipoprotein",
    r"^Non-HDLCholesterol",
    r"^Cholesterol/HDLRatio",
    r"^ApolipoproteinA",
    r"^ApolipoproteinB",
    r"^FastingBloodGlucose",
    r"^PostprandialBloodGlucose",
    r"^GlycatedHemoglobin",
    r"^(DiabetesScreening-)?Insulin$",
))

_PAGE_W, _PAGE_H = 595, 842
_MARGIN = 48
_LEADING = 15

# Column x-offsets, in points from the left margin.
_COL_NAME, _COL_VALUE, _COL_UNIT, _COL_REF, _COL_FLAG = 0, 250, 320, 390, 480


@dataclass
class Measurement:
    name: str
    value: str
    unit: str
    reference_range: str
    abnormal: bool


@dataclass
class LabReport:
    subject: str
    collected: str
    exam_type: str
    banner: str
    measurements: list[Measurement] = field(default_factory=list)
    note: str = ""

#-----------------------------------------------------------------------------
# Panel → report

_CAMEL_BOUNDARIES = (
    re.compile(r"(?<=[a-z0-9])(?=[A-Z])"),      # ...ityLipo -> ...ity Lipo
    re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])"),    # ...HDLRatio -> ...HDL Ratio
)


# Hyphen-delimited segments no longer than this are treated as abbreviations
# and left intact: splitting `HbA1c` into `Hb A1c` reads worse than not
# splitting at all, while `TotalCholesterol` clearly needs it.
_ABBREVIATION_MAX = 6

# Panel headings, not part of an analyte's printed name. ESL-Bench prefixes some
# ids with the section they belong to; a real report prints the section once as a
# heading and never glues it onto the row.
_SECTION_PREFIXES = frozenset({
    "Lipid", "PhysicalExamination", "DiabetesScreening", "Urinalysis",
})


def _is_abbreviation(segment: str) -> bool:
    """Whether a hyphen-delimited segment is a trailing lab abbreviation.

    ``TC``, ``FBG``, ``HbA1c``, ``US``, ``eGFR``, ``hs``, ``25(OH)D`` — short,
    and carrying no more than two lowercase letters. An English word of that
    length cannot: ``Score`` has four, ``Ratio`` four, ``Density`` six. One
    shape test is enough; a lookup table of known abbreviations turned out to be
    entirely redundant with it.

    Only the *trailing* segments are tested, so ``HDL`` being strippable does
    not endanger ``Non-HDLCholesterol`` — stripping stops at the first segment
    that is a word, and the analyte's own name is to the left of it.
    """
    if not segment or len(segment) > _ABBREVIATION_MAX + 2:
        return False
    if segment.count("(") != segment.count(")"):
        return False        # `OH)` of `Vitamin D (25-OH)`, not an abbreviation:
                            # stripping it would strand the opening bracket
    letters = [c for c in segment if c.isalpha()]
    if not letters:
        return True                                     # 25(OH)D, 1,25(OH)2D, -1
    return sum(c.islower() for c in letters) <= 2


def display_name(raw: str) -> str:
    """Turn an ESL-Bench indicator id into the name a lab report would print.

    ``TotalCholesterol-TC`` → ``Total Cholesterol``,
    ``Lipid-FreeFattyAcids`` → ``Free Fatty Acids``,
    ``PhysicalExamination-SystolicBloodPressure`` → ``Systolic Blood Pressure``.

    ESL-Bench ids are machine identifiers: CamelCase words, an optional section
    prefix, and an optional trailing abbreviation. A report prints none of that
    scaffolding — it prints the analyte. Both affixes are dropped and the
    remaining CamelCase is spaced out.

    Printing the id nearly verbatim, as this used to, cost the demo its whole
    point: of ESL-Bench's 190 indicator names exactly **one** survived
    ``mirobody.engine.resolve`` as ``Total Cholesterol-TC`` and friends, versus
    80 as plain analyte names. The extractor was reading the page correctly and
    every reading still landed unresolved.

    Names that do not resolve even so — ``High-Density Lipoprotein`` (a report
    would say ``HDL Cholesterol``), imaging, ECG intervals, derived ratios — are
    left alone. Rewriting them into whatever the resolver happens to accept
    would make the sample prove nothing about the resolver.
    """
    segments = raw.split("-")

    while len(segments) > 1 and _is_abbreviation(segments[-1]):
        segments.pop()
    if len(segments) > 1 and segments[0] in _SECTION_PREFIXES:
        segments.pop(0)

    spaced = []
    for segment in segments:
        if len(segment) > _ABBREVIATION_MAX:
            for pattern in _CAMEL_BOUNDARIES:
                segment = pattern.sub(" ", segment)
        spaced.append(segment)
    return re.sub(r"\s{2,}", " ", "-".join(spaced)).strip()


def _is_alias_of(a: str, b: str) -> bool:
    """Whether two printed names denote the same analyte.

    ``Lipid-LDL/HDLRatio`` vs ``LDL/HDLRatio`` — same analyte, carried twice by
    the panel. Equality counts: now that :func:`display_name` drops the section
    prefix, that pair renders to one identical string, and an alias test that
    demanded the two differ would let the duplicate straight through.
    """
    x, y = a.lower().replace(" ", ""), b.lower().replace(" ", "")
    return x == y or x.endswith(y) or y.endswith(x)


def _is_quantity(value: str) -> bool:
    """Whether a panel entry is a measured quantity rather than a narrative.

    ESL-Bench panels mix the two: ``BodyMassIndex-BMI`` carries ``27.1``, while
    ``AbdominalUltrasound-ABD-US`` carries ``Mild hepatic steatosis``. An
    analyte table has one column for the result, sized for a number — a
    radiologist's impression lands there clipped to ``Mild hepat``, and it is
    not a reading a resolver could ever code. Narrative findings belong to an
    impressions section, which this one-page report does not have.
    """
    try:
        float(value)
    except ValueError:
        return False
    return True


def select_measurements(panel: dict, *, limit: int = 12) -> list[Measurement]:
    """Pick *limit* measurements from an exam panel, in printed-report order.

    Panels carry ~200 indicators; a lab report the size of a phone book makes a
    poor demo, so this narrows to a believable single-page panel, ordered by
    :data:`_PANEL_ORDER`. Anything the list does not name falls to the end in
    alphabetical order, so a panel missing some of those rows still fills up.
    Aliased duplicates are collapsed — see :func:`_is_alias_of`.
    """
    if limit > MAX_ROWS:
        raise ValueError(f"limit {limit} exceeds MAX_ROWS ({MAX_ROWS}) for a single page")

    indicators = panel.get("indicators") or {}

    def rank(name: str) -> int:
        for i, pattern in enumerate(_PANEL_ORDER):
            if pattern.match(name):
                return i
        return len(_PANEL_ORDER)

    ordered = sorted(indicators.items(), key=lambda kv: (rank(kv[0]), kv[0]))

    out: list[Measurement] = []
    for name, raw in ordered:
        if len(out) >= limit:
            break
        raw = raw if isinstance(raw, dict) else {}
        canonical = str(raw.get("indicator_name") or name)
        value = str(raw.get("value") if raw.get("value") is not None else "")
        unit = str(raw.get("unit") or "")

        if not _is_quantity(value):
            continue

        if any(
            m.value == value and m.unit == unit and _is_alias_of(m.name, display_name(canonical))
            for m in out
        ):
            continue

        out.append(
            Measurement(
                name=display_name(canonical),
                value=value,
                unit=unit,
                reference_range=str(raw.get("reference_range") or ""),
                abnormal=str(raw.get("status") or "").strip().lower() not in ("", "normal"),
            )
        )
    return out


def build_report(
    panel: dict, profile: dict, *, subject: str, limit: int = 12
) -> LabReport:
    """Assemble a :class:`LabReport` from one exam panel and the user's profile."""
    demographics = (profile or {}).get("demographics") or {}
    age = demographics.get("age")
    gender = demographics.get("gender")

    parts = [subject]
    if age is not None:
        parts.append(f"{age}y")
    if gender:
        parts.append(str(gender))

    return LabReport(
        subject=" / ".join(parts),
        collected=str(panel.get("exam_date") or "unknown"),
        exam_type=f"{panel.get('exam_type') or 'routine'} ({panel.get('exam_location') or 'clinic'})",
        banner=SYNTHETIC_BANNER,
        measurements=select_measurements(panel, limit=limit),
        note=(
            "Generated from the ESL-Bench synthetic longitudinal dataset "
            "(healthmemoryarena/ESL-Bench). No real person is described."
        ),
    )

#-----------------------------------------------------------------------------
# PDF writing

_WINANSI_FALLBACK = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "…": "...",
    "µ": "u",
}


def _to_winansi(text: str) -> str:
    """Coerce *text* into the standard-14 fonts' encoding.

    Anything outside it is transliterated (or dropped) rather than emitted as
    raw bytes, which would corrupt the content stream.
    """
    out: list[str] = []
    for ch in text:
        if ch in _WINANSI_FALLBACK:
            out.append(_WINANSI_FALLBACK[ch])
            continue
        if ord(ch) < 128:
            out.append(ch)
            continue
        folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        out.append(folded if folded else "?")
    return "".join(out)


def _escape(text: str) -> str:
    """Escape a PDF literal string. Backslash first, or it doubles the others."""
    return (
        _to_winansi(text)
        .replace("\\", r"\\")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


# Approximate Helvetica advance widths, in 1/1000 em. Enough to keep a long
# analyte name from running into the next column; not a substitute for the real
# AFM tables, which would be a lot of table for one sample file.
_NARROW = set("ijltfrI.,:;'|!()[]{}/ ")
_WIDE = set("ABCDEFGHJKLMNOPQRSTUVWXYZmw@%&")


def text_width(text: str, size: int) -> float:
    """Approximate rendered width of *text* in points."""
    total = 0.0
    for ch in text:
        if ch in _NARROW:
            total += 280
        elif ch in _WIDE:
            total += 720
        else:
            total += 556
    return total * size / 1000.0


def fit(text: str, max_width: float, size: int) -> str:
    """Truncate *text* with an ellipsis so it renders within *max_width*."""
    if text_width(text, size) <= max_width:
        return text
    ellipsis = "..."
    budget = max_width - text_width(ellipsis, size)
    out = ""
    for ch in text:
        if text_width(out + ch, size) > budget:
            break
        out += ch
    return out.rstrip() + ellipsis


def _text(x: float, y: float, size: int, content: str, *, bold: bool = False) -> str:
    font = "F2" if bold else "F1"
    return f"BT /{font} {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({_escape(content)}) Tj ET\n"


def _content_stream(report: LabReport) -> str:
    if len(report.measurements) > MAX_ROWS:
        raise ValueError(
            f"{len(report.measurements)} measurements exceed MAX_ROWS ({MAX_ROWS}) "
            f"for a single page"
        )

    left = _MARGIN
    y = _PAGE_H - _MARGIN
    out: list[str] = []

    out.append(_text(left, y, 9, report.banner, bold=True))
    y -= _LEADING * 1.6

    out.append(_text(left, y, 18, "Laboratory Report", bold=True))
    y -= _LEADING * 1.5

    out.append(_text(left, y, 10, f"Subject:   {report.subject}"))
    y -= _LEADING
    out.append(_text(left, y, 10, f"Collected: {report.collected}"))
    y -= _LEADING
    out.append(_text(left, y, 10, f"Exam:      {report.exam_type}"))
    y -= _LEADING * 1.8

    header = (("Analyte", _COL_NAME), ("Result", _COL_VALUE), ("Unit", _COL_UNIT),
              ("Reference", _COL_REF), ("Flag", _COL_FLAG))
    for label, dx in header:
        out.append(_text(left + dx, y, 10, label, bold=True))
    y -= _LEADING * 0.4

    # A rule under the table header, drawn as a thin filled rectangle.
    out.append(f"0.5 w {left} {y:.1f} m {_PAGE_W - _MARGIN} {y:.1f} l S\n")
    y -= _LEADING

    # Each cell is clipped to its column so a long analyte name cannot run into
    # the Result column — PDF text has no automatic clipping.
    padding = 8
    widths = {
        _COL_NAME: _COL_VALUE - _COL_NAME - padding,
        _COL_VALUE: _COL_UNIT - _COL_VALUE - padding,
        _COL_UNIT: _COL_REF - _COL_UNIT - padding,
        _COL_REF: _COL_FLAG - _COL_REF - padding,
        _COL_FLAG: _PAGE_W - 2 * _MARGIN - _COL_FLAG,
    }

    for m in report.measurements:
        for value, dx in (
            (m.name, _COL_NAME), (m.value, _COL_VALUE), (m.unit, _COL_UNIT),
            (m.reference_range, _COL_REF), ("H/L" if m.abnormal else "", _COL_FLAG),
        ):
            if value:
                out.append(_text(
                    left + dx, y, 10, fit(value, widths[dx], 10),
                    bold=m.abnormal and dx == _COL_VALUE,
                ))
        y -= _LEADING

    y -= _LEADING
    out.append(_text(left, y, 8, report.note))
    y -= _LEADING * 0.8
    out.append(_text(left, y, 8, "Flag: H/L marks a result outside the reference range."))

    return "".join(out)


def render_pdf(report: LabReport) -> bytes:
    """Serialise *report* as a single-page PDF."""
    stream = _content_stream(report).encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_W} {_PAGE_H}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    stamp = datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R "
        f"/Info << /Title (Synthetic Laboratory Report) "
        f"/Producer (mirobody-eval) /CreationDate ({stamp}) >> >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")

    return bytes(out)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    """把某个已下载用户的一次体检面板渲染成 PDF。

    纯本地：读 prepare_data 下载的 JSON，写一个 PDF，不碰数据库。

        python -m generator.eslbench.labreport --users user5086@demo -o samples/lab_report.pdf
    """
    import argparse
    import json

    from .prepare_data import DATA_DIR

    parser = argparse.ArgumentParser(
        prog="python -m generator.eslbench.labreport",
        description="从 ESL-Bench 体检面板生成合成化验单 PDF（PHI-free）",
    )
    parser.add_argument("--users", required=True, help="合成用户邮箱，如 user5086@demo")
    parser.add_argument("-o", "--output", default="samples/lab_report.pdf", help="PDF 输出路径")
    parser.add_argument("--exam-index", type=int, default=-1, help="第几次体检（默认 -1，即最新一次）")
    parser.add_argument("--rows", type=int, default=12, help="打印多少个分析物")

    args = parser.parse_args()

    user_dir = args.users.replace("@", "_AT_")
    user_root = DATA_DIR / user_dir
    exam_path = user_root / "exam_data.json"
    if not exam_path.exists():
        raise SystemExit(f"{exam_path} 不存在。先跑: python -m generator.eslbench.prepare_data")

    exams = json.loads(exam_path.read_text(encoding="utf-8"))
    if not exams:
        raise SystemExit(f"{args.users} 没有体检面板")
    if not -len(exams) <= args.exam_index < len(exams):
        raise SystemExit(f"--exam-index {args.exam_index} 越界：{args.users} 共 {len(exams)} 次体检")

    profile_path = user_root / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}

    report = build_report(exams[args.exam_index], profile, subject=args.users, limit=args.rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_pdf(report))

    print(f"{out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"  受检者: {report.subject}")
    print(f"  采样日: {report.collected}  ({report.exam_type})")
    print(f"  分析物: {len(report.measurements)} 项")
    print(
        f"\n让这份 PDF 带来库里还没有的数据（"
        f"否则单点无趋势可言）:\n"
        f"    python -m generator.eslbench.seed_mirobody --users {args.users} --hold-out-exams 1"
    )


if __name__ == "__main__":
    main()
