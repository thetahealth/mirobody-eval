// Render a model's answer (or a user turn) as Markdown, for the case-detail dialogue.
//
// Model output is untrusted text. A dedicated Marked instance keeps it inert:
//   - raw HTML is shown as text, never parsed into elements;
//   - links keep only http(s) and mailto targets, anything else renders as its text;
//   - images are not fetched — an answer must not make the viewer's browser call out —
//     they render as a link to their source instead.
(function () {
  const escapeHtml = (s) => String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const safeHref = (href) => /^(https?:|mailto:)/i.test(String(href || "").trim());

  const md = new marked.Marked({ gfm: true, breaks: true });
  md.use({
    renderer: {
      html({ text }) { return escapeHtml(text); },
      link({ href, title, tokens }) {
        const label = this.parser.parseInline(tokens);
        if (!safeHref(href)) return label;
        const t = title ? ` title="${escapeHtml(title)}"` : "";
        return `<a href="${escapeHtml(href)}"${t} target="_blank" rel="noopener noreferrer">${label}</a>`;
      },
      image({ href, text }) {
        const label = escapeHtml(text || href || "image");
        return safeHref(href)
          ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">[${label}]</a>`
          : `[${label}]`;
      },
    },
  });

  window.renderAnswerMarkdown = (text) => (text ? md.parse(String(text)) : "");
})();
