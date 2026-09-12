"""Shared citation format and conservative, evidence-checked normalization."""
import re


CITATION_FORMAT_INSTRUCTIONS = """
EVIDENCE CITATION FORMAT (required inside each edge's evidence string)
- Use the standalone marker [snippet] for search results or [visit] for visited
  pages. Keep the source query, chunk label, and window OUTSIDE that marker.
- Format: [snippet] source=qX; chunk=Tool_Response #N; window=LO..HI; explanation.
  Copy source, chunk, and window from this target's Source_provenance; replace
  the placeholders with actual values. Do not invent a source or window.
- Correct syntax example: [snippet] source=q1; chunk=Tool_Response #2; window=0..6;
  the retrieved passage supplies the entity reused by the target query.
- Incorrect syntax: [q1 snippet Tool_Response #2 window 0..6].
- For a visit, use the same format with [visit]. For think evidence, keep the
  existing standalone [sN] sentence ID and name the source query where applicable.
  PK and soft-failure edges still require think evidence; a snippet marker cannot
  replace that requirement. Formatting does not establish evidential support.
"""

_COMBINED_CITATION = re.compile(
    r"\[(?P<source>q\d+)\s+(?P<kind>snippet|visit)\s+"
    r"(?P<chunk>[^\[\]\n]+?)\s+window\s+"
    r"(?P<lo>\d+)\s*\.\.\s*(?P<hi>\d+)\]", re.IGNORECASE,
)


def _chunk_key(label):
    return " ".join(str(label).strip().strip("[]").split()).casefold()


def _visible_windows(payload, source):
    """Recover the overlapping-window unions shown in Source_provenance."""
    groups = {}
    for token, candidates in payload.get("candidates_by_token", {}).items():
        for candidate in candidates:
            if candidate.get("source_qid") != source:
                continue
            for window in candidate.get("provenance", []):
                lo, hi = window.get("window_lo"), window.get("window_hi")
                if not isinstance(lo, int) or not isinstance(hi, int) or lo > hi:
                    continue
                key = (window.get("source_type", "snippet"),
                       _chunk_key(window.get("chunk_label", "")))
                groups.setdefault(key, []).append((lo, hi, {token}))
    for (kind, chunk), windows in groups.items():
        merged = []
        for lo, hi, tokens in sorted(windows, key=lambda w: (w[0], w[1])):
            if merged and lo <= merged[-1][1]:
                prev_lo, prev_hi, prev_tokens = merged[-1]
                merged[-1] = (prev_lo, max(prev_hi, hi), prev_tokens | tokens)
            else:
                merged.append((lo, hi, tokens))
        for lo, hi, tokens in merged:
            yield kind, chunk, lo, hi, tokens


def normalize_provenance_citations(text, source, payload, covered_signals):
    """Normalize only explicit references matching supplied provenance windows.

    No source, window, or explanation is inferred. If any combined reference
    cannot be checked, leave the entire string unchanged for normal rejection.
    Callers must still apply temporal, edge-kind, coverage, and parent-set gates.
    """
    matches = list(_COMBINED_CITATION.finditer(text))
    if not matches or not covered_signals:
        return text
    windows = list(_visible_windows(payload, source))
    for match in matches:
        key = (match["kind"].lower(), _chunk_key(match["chunk"]),
               int(match["lo"]), int(match["hi"]))
        if match["source"].casefold() != source.casefold() or not any(
                (kind, chunk, lo, hi) == key and tokens & set(covered_signals)
                for kind, chunk, lo, hi, tokens in windows):
            return text

    def replace(match):
        return (f'[{match["kind"].lower()}] source={match["source"]}; '
                f'chunk={match["chunk"]}; window={match["lo"]}..{match["hi"]}')

    return _COMBINED_CITATION.sub(replace, text)
