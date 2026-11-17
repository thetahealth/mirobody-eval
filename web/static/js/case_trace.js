// Tool calls of one reply, shown as a step list under the answer in the case detail.
//
// A record carries name, args and the raw result text; nothing marks success or failure
// (a tool that fails usually returns its error as text). So nothing here guesses a status:
// each step's one-line summary shows the start of what the tool returned, and an error
// reads as an error there.
(function () {
  const CODE_ARGS = ["sql", "query", "code", "jmes", "path"];
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  const oneLine = (s) => String(s).replace(/\s+/g, " ").trim();
  const parseJson = (s) => { try { return JSON.parse(s); } catch { return undefined; } };
  const isScalar = (v) => v === null || ["string", "number", "boolean"].includes(typeof v);
  const pairs = (obj, n) => Object.entries(obj || {})
    .filter(([, v]) => isScalar(v) && v !== "")
    .slice(0, n)
    .map(([k, v]) => `${k}=${clip(oneLine(v), 40)}`)
    .join("  ");
  const codeArg = (args) => CODE_ARGS.find((k) => typeof args?.[k] === "string" && args[k].trim());

  // "sql=WITH s AS (SELECT …" style summary of the call.
  window.toolArgsSummary = (tc) => {
    const a = tc?.args || {};
    const k = codeArg(a);
    return clip(k ? oneLine(a[k]) : pairs(a, 3), 90) || "()";
  };

  // First thing the tool returned: a few top-level fields of a JSON object, the size of
  // a JSON array, or the first non-empty line of text.
  window.toolResultSummary = (tc) => {
    const r = String(tc?.result ?? "");
    if (!r.trim()) return "(empty)";
    const j = parseJson(r);
    if (Array.isArray(j)) return `${j.length} items`;
    if (j && typeof j === "object") return clip(pairs(j, 3) || `{${Object.keys(j).length} keys}`, 90);
    return clip(oneLine(r.split("\n").find((l) => l.trim()) || r), 90);
  };

  // Arguments for the expanded step: a code argument (SQL, JMESPath, ...) with its own
  // line breaks, the rest as indented JSON.
  window.toolArgsDetail = (tc) => {
    const a = { ...(tc?.args || {}) };
    const k = codeArg(a);
    const code = k ? a[k] : "";
    if (k) delete a[k];
    const rest = Object.keys(a).length ? JSON.stringify(a, null, 2) : "";
    return [code, rest].filter(Boolean).join("\n\n") || "()";
  };

  // Result for the expanded step, one entry per line (JSON indented first).
  window.toolResultLines = (tc) => {
    const r = String(tc?.result ?? "");
    const j = parseJson(r);
    return (j !== undefined && typeof j === "object" ? JSON.stringify(j, null, 2) : r).split("\n");
  };
})();
