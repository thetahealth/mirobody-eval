"""ESL-Bench → mirobody row/document mapping. Pure functions, no I/O.

Two destinations, chosen by what the agent can actually query:

* **Indicator readings** (``device_indicator`` / ``exam_indicator`` timeline
  entries) → ``th_series_data`` rows. This is the agent's native structured
  path: ``search_health_indicators`` finds the indicator, then
  ``fetch_health_data`` reads values out of this table.

* **Life events, profile, exam reports** → markdown documents destined for
  ``th_files``, which DeepAgent auto-mirrors read-only into ``/library/`` so
  ``read_file`` / ``grep`` can reach them. None of these are time-series, so
  none of them belong in ``th_series_data``.

**What is deliberately withheld.** ``events.json`` carries
``affected_indicators`` and ``impact_reasoning`` — an explicit
event→indicator causal chain with expected magnitudes. That is the answer key
for the Explanation dimension, so events are rendered from the *timeline's*
event view instead, which exposes only identity and dates. This also matches
the data surface the ESL-Bench reference target is given (``timeline.json``,
``exam_data.json``, ``profile.json`` — never ``events.json``), keeping scores
comparable to the published leaderboard. Generator bookkeeping
(``profile.metadata``, ``generation_metadata``, ``generation_summary``) is
dropped for the same reason: it describes how the user was synthesised, not
the user.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

#-----------------------------------------------------------------------------

SOURCE_TABLE = "eslbench"
SOURCE_DEVICE = "eslbench.device"
SOURCE_EXAM = "eslbench.exam"

# Per-indicator prose in an exam record. Cap it so a 200-indicator panel can't
# push one document past the 256 KB inline-storage threshold in DeepAgent's
# filesystem backend (beyond which it needs object storage that a bare
# self-hosted deployment may not have configured).
_CLINICAL_NOTE_MAX = 400

# Keys describing the synthesis process rather than the user.
_BOOKKEEPING_KEYS = frozenset({"generation_metadata", "generation_summary"})


def _parse_time(raw: Any) -> datetime | None:
    """ISO-8601 → naive datetime (``th_series_data`` columns are tz-less)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    return dt.replace(tzinfo=None)


def _date_of(raw: Any) -> str:
    dt = _parse_time(raw)
    return dt.date().isoformat() if dt else ""


def _humanize(key: str) -> str:
    return key.replace("_", " ").strip().title()

#-----------------------------------------------------------------------------
# th_series_data rows

def series_row(entry: dict, *, user_id: str, user_dir: str) -> dict | None:
    """One timeline entry → one ``th_series_data`` row, or ``None`` to skip.

    Skipped: ``event`` entries (they become a document instead), unknown
    entry types, and entries missing a time / indicator / value — a row with
    any of those absent is unqueryable, and inventing a placeholder would put
    fiction in front of the agent.

    ``start_time == end_time`` because these are instantaneous readings; the
    ``(user_id, indicator, start_time, end_time)`` unique key depends on it.
    ``fhir_id`` is left ``None`` for
    :class:`~mirobody.task.indicator_sync.IndicatorSyncTask` to resolve —
    it has ambiguity guards this mapping has no business second-guessing.
    """
    entry_type = entry.get("entry_type")
    if entry_type == "device_indicator":
        source = SOURCE_DEVICE
        comment = f"ESL-Bench device reading ({entry.get('device_type') or 'unknown device'})"
    elif entry_type == "exam_indicator":
        source = SOURCE_EXAM
        comment = (
            f"ESL-Bench exam reading "
            f"({entry.get('exam_type') or 'exam'} @ {entry.get('exam_location') or 'unknown'})"
        )
    else:
        return None

    start_time = _parse_time(entry.get("time"))
    indicator = entry.get("indicator")
    value = entry.get("value")

    if start_time is None or not isinstance(indicator, str) or not indicator.strip():
        return None
    if value is None:
        return None

    return {
        "user_id": user_id,
        "indicator": indicator.strip(),
        "value": str(value),
        "start_time": start_time,
        "end_time": start_time,
        "source": source,
        "source_table": SOURCE_TABLE,
        "source_table_id": user_dir,
        "comment": comment,
        "indicator_id": "",
        "task_id": "",
        "fhir_id": None,
        "fhir_mapping_info": json.dumps({"unit": entry.get("unit") or ""}),
    }


def _skip_reason(entry: dict) -> str | None:
    """Why :func:`series_row` would skip *entry*, or ``None`` if it maps."""
    if entry.get("entry_type") == "event":
        return "event"
    if entry.get("entry_type") not in ("device_indicator", "exam_indicator"):
        return "other_entry_type"
    if _parse_time(entry.get("time")) is None:
        return "malformed"
    indicator = entry.get("indicator")
    if not isinstance(indicator, str) or not indicator.strip():
        return "malformed"
    if entry.get("value") is None:
        return "null_value"
    return None


@dataclass
class MappingStats:
    """What :func:`series_rows` did with a timeline.

    Reported by the seed rather than kept internal: ESL-Bench timelines
    legitimately contain ``value: null`` readings (a day with no vigorous
    activity records null, not zero), and occasional duplicate lab entries at
    an identical timestamp. Both are dropped — nulls because
    ``th_series_data.value`` is text and storing ``"None"`` would show the
    agent a fake value, duplicates because mirobody's
    ``(user_id, indicator, start_time, end_time)`` unique key cannot hold two.
    Neither should vanish silently, or a 9%-short seed reads as complete.
    """

    entries: int = 0
    events: int = 0
    device_rows: int = 0
    exam_rows: int = 0
    null_value: int = 0
    malformed: int = 0
    other_entry_type: int = 0
    key_collisions: int = 0
    _seen_keys: set = field(default_factory=set, repr=False)

    @property
    def rows(self) -> int:
        return self.device_rows + self.exam_rows

    def summary(self) -> str:
        parts = [
            f"{self.rows} rows ({self.device_rows} device, {self.exam_rows} exam)",
            f"{self.events} events",
            f"from {self.entries} timeline entries",
        ]
        dropped = []
        if self.null_value:
            dropped.append(f"{self.null_value} null-valued readings")
        if self.malformed:
            dropped.append(f"{self.malformed} malformed")
        if self.other_entry_type:
            dropped.append(f"{self.other_entry_type} unrecognised entry types")
        if self.key_collisions:
            dropped.append(f"{self.key_collisions} collapsed by unique key")
        if dropped:
            parts.append("dropped: " + ", ".join(dropped))
        return "; ".join(parts)


def series_rows(
    entries: Iterable[dict],
    *,
    user_id: str,
    user_dir: str,
    stats: MappingStats | None = None,
) -> Iterator[dict]:
    """Lazily map a timeline's entries to ``th_series_data`` rows.

    Lazy on purpose: a timeline runs to ~75k entries and the caller streams
    these into batched inserts rather than materialising them all.

    Pass *stats* to account for every entry, including the dropped ones.
    Collision counting is only done when *stats* is supplied, since it needs
    to remember the keys already yielded.
    """
    for entry in entries:
        if stats is not None:
            stats.entries += 1

        reason = _skip_reason(entry)
        if reason is not None:
            if stats is not None:
                setattr(stats, "events" if reason == "event" else reason,
                        getattr(stats, "events" if reason == "event" else reason) + 1)
            continue

        row = series_row(entry, user_id=user_id, user_dir=user_dir)
        if row is None:  # pragma: no cover - _skip_reason already vetted it
            continue

        if stats is not None:
            key = (row["indicator"], row["start_time"])
            if key in stats._seen_keys:
                stats.key_collisions += 1
            else:
                stats._seen_keys.add(key)
            if row["source"] == SOURCE_DEVICE:
                stats.device_rows += 1
            else:
                stats.exam_rows += 1

        yield row

#-----------------------------------------------------------------------------
# Documents

def events_document(entries: Iterable[dict], email: str) -> str:
    """Render the timeline's life events as a markdown table.

    Columns are exactly the timeline event view: identity, type, and the
    resolved active window. ``end`` comes from the entry's ``end_time``,
    which upstream has already adjusted for interruption — so "which events
    were active on date X" is answerable from this table alone, without
    exposing the causal annotations that live in ``events.json``.
    """
    rows: list[str] = []
    for entry in entries:
        if entry.get("entry_type") != "event":
            continue
        event = entry.get("event") or {}
        name = str(event.get("event_name") or "").replace("|", "\\|")
        start = event.get("start_date") or _date_of(entry.get("time"))
        end = _date_of(entry.get("end_time"))
        interrupted = "yes" if event.get("interrupted") else "no"
        rows.append(
            f"| {event.get('event_id') or ''} | {event.get('event_type') or ''} | {name} "
            f"| {start} | {end} | {event.get('duration_days') or ''} | {interrupted} "
            f"| {event.get('interruption_date') or ''} |"
        )

    header = f"# Life events — {email}\n\n"
    if not rows:
        return header + "This record contains no life events.\n"

    return (
        header
        + f"{len(rows)} events, ordered as recorded in the timeline. "
        + "An event is active from `start` through `end` inclusive.\n\n"
        + "| event_id | type | name | start | end | duration_days | interrupted | interruption_date |\n"
        + "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )


def _render_value(key: str, value: Any, depth: int) -> list[str]:
    """Recursively render a profile value as markdown lines."""
    heading = "#" * min(depth + 1, 6)
    label = _humanize(key)

    if isinstance(value, dict):
        lines = [f"{heading} {label}", ""]
        for k, v in value.items():
            if k in _BOOKKEEPING_KEYS:
                continue
            lines.extend(_render_value(k, v, depth + 1))
        return lines

    if isinstance(value, list):
        if not value:
            # An empty list is an answer ("no chronic conditions"), so it is
            # stated rather than omitted.
            return [f"**{label}**: none recorded", ""]
        lines = [f"**{label}**:", ""]
        for item in value:
            if isinstance(item, dict):
                inner = "; ".join(f"{_humanize(k)}: {v}" for k, v in item.items())
                lines.append(f"- {inner}")
            else:
                lines.append(f"- {item}")
        lines.append("")
        return lines

    return [f"**{label}**: {value}", ""]


def profile_document(profile: dict, email: str) -> str:
    """Render ``profile.json`` as markdown, minus the ``metadata`` block."""
    lines = [f"# Health profile — {email}", ""]
    for key, value in profile.items():
        if key == "metadata" or key in _BOOKKEEPING_KEYS:
            continue
        lines.extend(_render_value(key, value, depth=1))
    return "\n".join(lines).rstrip() + "\n"


def _truncate(text: Any, limit: int) -> str:
    s = str(text or "").replace("|", "\\|").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def exam_documents(exams: Iterable[dict], email: str) -> list[tuple[str, str]]:
    """Render each exam record as its own ``(filename, markdown)`` pair.

    One document per exam rather than one combined file: a 7-exam history at
    ~200 indicators each would exceed the inline-storage threshold as a single
    document, and per-report files also match how a real user's uploads look.
    """
    docs: list[tuple[str, str]] = []

    for exam in exams:
        date = str(exam.get("exam_date") or "unknown")
        lines = [
            f"# Exam report {date} — {email}",
            "",
            f"**Type**: {exam.get('exam_type') or 'unknown'}  ",
            f"**Location**: {exam.get('exam_location') or 'unknown'}",
            "",
        ]

        if assessment := exam.get("overall_assessment"):
            lines += ["## Overall assessment", "", str(assessment), ""]

        if findings := exam.get("abnormal_findings"):
            lines += ["## Abnormal findings", ""]
            lines += [f"- {f}" for f in findings]
            lines.append("")

        if recommendations := exam.get("recommendations"):
            lines += ["## Recommendations", ""]
            lines += [f"- {r}" for r in recommendations]
            lines.append("")

        indicators = exam.get("indicators") or {}
        if indicators:
            lines += [
                "## Measurements",
                "",
                "| indicator | value | unit | reference range | status | note |",
                "|---|---|---|---|---|---|",
            ]
            for name, ind in indicators.items():
                ind = ind if isinstance(ind, dict) else {}
                lines.append(
                    f"| {_truncate(ind.get('indicator_name') or name, 120)} "
                    f"| {_truncate(ind.get('value'), 40)} "
                    f"| {_truncate(ind.get('unit'), 40)} "
                    f"| {_truncate(ind.get('reference_range'), 60)} "
                    f"| {_truncate(ind.get('status'), 30)} "
                    f"| {_truncate(ind.get('clinical_significance'), _CLINICAL_NOTE_MAX)} |"
                )
            lines.append("")

        docs.append((f"exam_{date}.md", "\n".join(lines).rstrip() + "\n"))

    return docs
