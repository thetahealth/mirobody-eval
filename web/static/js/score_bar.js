/* Score distribution for a set of cases, shared by the report page and the task page.
 *
 * A pass/fail grade is one of two segments. A continuous grade ("scored", e.g. HealthBench
 * or kg_qa) falls into one of five bands, so the bar shows where the scores lie instead of
 * one solid "scored" block. An ungraded case ("error") is its own segment and never reads as
 * a low score. Cases not graded yet (a running task) add to the total and leave empty track.
 *
 * items: [{ result, score }]  →  [{ key, label, cls?, color?, count, pct }], empty ones dropped
 */
const SCORE_BANDS = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1']
    .map((label, i) => ({ label, color: `var(--score-band-${i + 1})` }));  // colors: static/css/style.css

function scoreSegments(items) {
    const segs = [
        { key: 'pass', label: 'pass', cls: 'bg-success', count: 0 },
        { key: 'fail', label: 'fail', cls: 'bg-destructive', count: 0 },
        ...SCORE_BANDS.map(b => ({ key: b.label, label: b.label, color: b.color, count: 0 })),
        { key: 'error', label: 'ungraded', cls: 'bg-warning', count: 0 },
    ];
    for (const it of items) {
        if (it.result === 'pass') segs[0].count++;
        else if (it.result === 'fail') segs[1].count++;
        else if (it.result === 'error') segs[segs.length - 1].count++;
        else if (it.result === 'scored' && it.score != null) {
            const band = Math.min(SCORE_BANDS.length - 1, Math.max(0, Math.floor(it.score * SCORE_BANDS.length)));
            segs[2 + band].count++;
        }
    }
    const total = items.length;
    return segs
        .filter(s => s.count > 0)
        .map(s => ({ ...s, pct: total > 0 ? (s.count / total) * 100 : 0 }));
}
