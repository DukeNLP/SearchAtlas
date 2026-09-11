"""Runtime configuration and attribution prompts."""
from __future__ import annotations
import json
import re
import os
from ..llm_compat import env_flag, parse_json_object

class _LazyClient:
    """Do not require credentials or open a client for offline parsing."""
    def __init__(self):
        self._value = None
    def __getattr__(self, name):
        if self._value is None:
            from openai import OpenAI
            if not API_KEY:
                raise RuntimeError("Set OPENAI_API_KEY before executing attribution calls")
            self._value = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=LLM_TIMEOUT_SEC)
        return getattr(self._value, name)


EDGE_POLICY_VERSION = "trace-grounded-20260911"

EDGE_POLICY = "strict"

API_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "180"))

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

LLM_CONNECTION_EXTRA_RETRIES = int(os.getenv("LLM_CONNECTION_EXTRA_RETRIES", "4"))

LLM_CONNECTION_WAIT_BASE_SEC = float(os.getenv("LLM_CONNECTION_WAIT_BASE_SEC", "8"))

LLM_JSON_MODE = env_flag(os.getenv("LLM_JSON_MODE", "0"))

LLM_USAGE_JSONL = os.getenv("LLM_USAGE_JSONL", "").strip()

LLM_MIN_COMPLETION_TOKENS = int(os.getenv("LLM_MIN_COMPLETION_TOKENS", "0"))

LLM_IO_MODE = "hardened"

ANSWER_DECOMPOSE_BASE_TOKENS = int(os.getenv("ANSWER_DECOMPOSE_BASE_TOKENS", "1500"))

ANSWER_DECOMPOSE_MAX_TOKENS = int(os.getenv("ANSWER_DECOMPOSE_MAX_TOKENS", "3000"))

try:
    LLM_EXTRA_BODY = parse_json_object(
        os.getenv("LLM_EXTRA_BODY_JSON", ""),
        label="LLM_EXTRA_BODY_JSON",
    )
except (json.JSONDecodeError, TypeError) as exc:
    print(f"WARNING: ignoring invalid LLM_EXTRA_BODY_JSON: {exc}")
    LLM_EXTRA_BODY = {}

client = _LazyClient()

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")

USE_MCP_TOOL_RE = re.compile(r"<use_mcp_tool>\s*([\s\S]*?)\s*</use_mcp_tool>")

SEARCH_TOOL_NAMES = {"search", "google_scholar"}

SEARCH_RESULT_HEADER_PATTERN = r"A Google (?:search|scholar) for '(.*?)' found (\d+) results?:"

SEARCH_RESULT_HEADER_RE = re.compile(
    rf"^{SEARCH_RESULT_HEADER_PATTERN}\s*$",
    re.MULTILINE | re.IGNORECASE,
)

EDGE_KINDS = [
    "constraint_use",
    "evidence_derived",
    "prior_knowledge_derived",
    "failure_derived",
]

STOPWORDS = set("""
a an the is are was were be been being have has had do does did will would shall
should can could may might must need ought dare to of in for on with at by from
as into through during before after above below between out off over under again
further then once here there when where why how all each every both few more most
other some such no nor not only own same so than too very just don doesn didn
what which who whom this that these those am if or and but because until while
about against its it he she they them their his her him i me my we our you your
""".split())

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
except Exception:
    _NLP = None

VISIT_RESULT_HEADER_PATTERN = r"The useful information in (\S+) for user goal (.*?) as follows:"

VISIT_RESULT_HEADER_RE = re.compile(
    rf"^{VISIT_RESULT_HEADER_PATTERN}\s*$",
    re.MULTILINE,
)

VISIT_SUMMARY_HARD_FAIL_PATTERNS = [
    (
        "visit summary page access failed",
        re.compile(r"\bcould not be accessed\b", re.I),
    ),
    (
        "visit summary page processing failed",
        re.compile(r"\bcould not be processed\b", re.I),
    ),
    (
        "visit summary page content missing from tool input",
        re.compile(r"\bno markdown content was provided in the input to process\b", re.I),
    ),
    (
        "visit summary page blocked or inaccessible",
        re.compile(
            r"\b(?:temporarily blocked|login page|sign[- ]?up/login page|access denied|forbidden|captcha-style prompt|content is inaccessible)\b",
            re.I,
        ),
    ),
    (
        "visit summary HTTP or missing-page error",
        re.compile(
            r"\b(?:403|404|422)\b|requested page could not be found|does not exist \(404 error\)",
            re.I,
        ),
    ),
]

_SOURCE_TYPE_PRIORITY = {"visit": 2, "snippet": 1, "think": 0}

SOFT_FAIL_PATTERNS = [
    re.compile(
        r"(?:did\s+not|didn't|does\s+not|doesn't|do\s+not|don't|"
        r"are\s+not|aren't|is\s+not|isn't|were\s+not|weren't)\s+"
        r"(?:(?:directly|actually|clearly|really)\s+)?"
        r"(?:yield|provide|give|contain|include|reveal|show(?:ing)?|mention|return)",
        re.I,
    ),
    re.compile(r"(?:insufficient|incomplete|lacking|missing)\s+(?:information|details|data|results|evidence)", re.I),
    re.compile(r"(?:no|without)\s+(?:specific|direct|concrete|relevant|useful)\s+(?:information|details|data|results|mention|evidence)", re.I),
    re.compile(r"(?:still\s+)?(?:need|require|missing)\s+(?:to\s+find|to\s+run|to\s+check|to\s+verify|more|additional|further|the\s+)", re.I),
    re.compile(r"(?:bridge\s+this\s+gap|crucial\s+gap|remaining\s+information\s+gap|still\s+a\s+\w+\s+gap|gap|gaps)\b", re.I),
    re.compile(r"(?:overlooked|missing|remaining)\s+(?:connection|clue|detail|piece|evidence)", re.I),
    re.compile(r"(?:one|another)\s+(?:key|critical|crucial)\s+(?:detail|clue|piece)\s+(?:is|remains)", re.I),
    re.compile(r"too\s+(?:broad|general|vague)", re.I),
    re.compile(r"not\s+(?:sufficient|enough|adequate|satisfactory)", re.I),
    re.compile(r"(?:failed\s+to|unable\s+to)\s+(?:find|locate|identify|determine|retrieve)", re.I),
    re.compile(r"(?:none\s+of\s+the\s+(?:results|searches|queries))", re.I),
    re.compile(r"(?:have\s+not|haven't)\s+(?:found|obtained|identified)", re.I),
    re.compile(
        r"(?:verify|check|determine|see)\s+(?:whether|if)\s+"
        r"(?:there\s+(?:are|were|is|was)\s+)?(?:any|another|other|more)\b",
        re.I,
    ),
]

SOFT_FAIL_RETRIEVAL_CONTEXT_RE = re.compile(
    r"\b(?:result|results|search|query|queries|snippet|page|pages|source|sources|"
    r"retrieval|visit|visited|tool|lookup|returned|yielded|found|located|obtained)\b",
    re.I,
)

SOFT_FAIL_EXPLICIT_INABILITY_RE = re.compile(
    r"\b(?:failed\s+to|unable\s+to|could\s+not|couldn't|have\s+not|haven't)\s+"
    r"(?:find|locate|identify|determine|retrieve|verify|confirm|access|obtain)\b",
    re.I,
)

SOFT_FAIL_COMPLETENESS_RE = re.compile(
    r"\b(?:verify|check|determine|see)\s+(?:whether|if)\s+"
    r"(?:there\s+(?:are|were|is|was)\s+)?(?:any|another|other|more)\b",
    re.I,
)

SOFT_FAIL_CROSS_BRANCH_SCOPE_RE = re.compile(
    r"\b(?:both|each|either|those|these|all)\s+"
    r"(?:date|dates|year|years|case|cases|candidate|candidates|branch|branches)\b",
    re.I,
)

YEAR_TOKEN_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")

LATEST_REF_PATTERNS = [
    re.compile(r"(?:previous|last|most\s+recent)\s+(?:search|query|results?)", re.I),
    re.compile(r"in\s+the\s+(?:previous|last)\s+search", re.I),
]

HARD_FAIL_PATTERNS = [
    re.compile(r"\b0\s+results?\b", re.I),
    re.compile(r"\bblocked\b", re.I),
    re.compile(r"\b(?:403|404)\b", re.I),
    re.compile(r"\blogin[\s\-_]*(?:required|wall)\b", re.I),
    re.compile(r"\baccess\s+denied\b", re.I),
    re.compile(r"\bno\s+results?\s+found\b", re.I),
    re.compile(r"\bforbidden\b", re.I),
]

Q0_DECOMPOSE_SYSTEM = """\
You decompose a user question into atomic Q0 units for a search-trajectory DAG.

Each unit is ONE testable constraint or requested field from the question.
Unit types:
  target_type     — the kind of entity being sought (e.g. "Mexican restaurant")
  location        — geographic constraint (e.g. "in New Mexico")
  attribute       — factual constraint (e.g. "hotel opened in 1955")
  numeric         — numeric range / threshold (e.g. "between 1990 and 1994")
  answer_field    — what the answer must provide (e.g. "founder full name")
  temporal        — time constraint (e.g. "published in 2015-2017")

Output JSON only:
{"units": [
  {"unit_id": "u1", "unit_type": "target_type", "q0_span": "soccer match"},
  {"unit_id": "u2", "unit_type": "attribute", "q0_span": "Brazilian referee"},
  ...
]}
Rules:
- Be exhaustive: capture ALL constraints and requested fields.
- q0_span must be a verbatim or near-verbatim substring of the question.
- Keep units atomic (one constraint each).
- Output valid JSON only, no commentary.
"""

Q0_MATCH_ARBITER_SYSTEM = """\
You judge whether a search query **operationalizes** (addresses / investigates) a
specific constraint unit from the original question Q0.

"Operationalizes" means the query is purposefully trying to find information
related to that constraint — even via paraphrase, synonym, or a narrower/broader
formulation.  A query that merely shares common English words (e.g. "restaurant",
"show", "city") WITHOUT targeting the constraint should be rejected.

For EACH (unit, query) pair in the input list, output ONE verdict object.

Output JSON only — an array of objects:
[
  {
    "pair_id": "<pair_id from input>",
    "match": true/false,
    "reason": "one-sentence explanation"
  },
  ...
]

Rules:
- Be strict: surface word overlap alone is NOT enough.
- Paraphrase / synonym IS enough if the intent aligns.
- Output valid JSON only, no commentary outside the JSON array.
"""

MINIMAL_ANSWER_EXTRACT_SYSTEM = """\
You compress a final answer down to the shortest text that directly answers the user's question.

You receive:
- The original question
- A longer final answer that may contain background, biography, justification, or extra context

Return JSON only:
{"minimal_answer": "..."}

Rules:
- Keep only information needed to directly answer the question.
- Preserve names, numbers, years, and essential qualifiers.
- Remove unrelated background, biography, and extra explanation.
- If the answer is already minimal, return it unchanged.
- Output valid JSON only.
"""

ANSWER_DECOMPOSE_SYSTEM = """\
You decompose a final answer into atomic answer units for a search-trajectory DAG.

Each unit is ONE independently verifiable factual claim in the answer.
Examples of answer units:
  - entity names (person, place, organization)
  - dates or years
  - numeric values
  - descriptive attributes

You also receive the original question to help identify which claims are relevant.

Output JSON only:
{"answer_units": [
  {"unit_id": "a1", "claim": "Ireland", "unit_type": "entity_name"},
  {"unit_id": "a2", "claim": "Romania", "unit_type": "entity_name"},
  ...
]}
Rules:
- Be exhaustive: capture ALL distinct factual claims.
- Each claim should be a short, verifiable string.
- unit_type: one of entity_name, date, number, attribute, description.
- Output valid JSON only, no commentary.
"""

ANSWER_SUPPORT_MATCH_SYSTEM = """\
You are the final validation gate for a query-to-Answer edge.

You receive:
- The original question
- The final answer's analysis as non-evidential context
- A small list of candidate (answer_unit, query_evidence) pairs

Mark support=true only when the TOOL EVIDENCE EXCERPT materially entails that
the answer unit fills the field or relation requested by the original question.
Alias, paraphrase, and a short inference within the excerpt are allowed.

Reject when:
- The excerpt is only topically related
- The excerpt merely mentions the answer token/entity without establishing the
  requested role, relation, date, value, or attribute
- The excerpt negates support, says the fact is missing/uncertain, or reports
  that no source mentions it
- Support exists only in the issued query text
- The claim is asserted only in the answer reasoning, not in the query evidence

Output JSON only:
{
  "verdicts": [
    {
      "pair_id": "...",
      "supports": true,
      "confidence": "high",
      "reason": "one-sentence explanation",
      "evidence_sentence": "best short supporting sentence or empty string"
    }
  ]
}

Rules:
- confidence must be high or uncertain
- Set confidence=high only for an explicit or clearly entailed claim in the excerpt
- The query text helps identify the retrieval action but is never evidence
- Use the answer analysis only as disambiguating context, not as evidence itself
"""

ANSWER_DISTRIBUTED_SUPPORT_SYSTEM = """\
You judge whether a final answer unit is supported by a small set of prior queries jointly.

Distributed support is allowed:
- one query anchors the entity
- another query supplies a date, place, school name, expansion, or attribute
- one query provides an acronym and another provides the full expansion

Every selected query must contribute through its attributable tool evidence.
Query text identifies the retrieval action but is never evidence.
Do not use the final answer reasoning alone as evidence.

Output JSON only:
{
  "selected_qids": ["q3", "q7"],
  "confidence": "high",
  "reason": "one-sentence explanation"
}

Rules:
- Return the minimal useful set of qids.
- Return [] if the provided query evidence does not materially support the unit.
- confidence must be high or uncertain.
"""

ANSWER_SIGNAL_GRANULARITY_RANK = {
    "whole_answer": 0,
    "unit_claim": 1,
    "subphrase": 2,
    "token": 3,
}

ANSWER_TARGETED_QUERY_PATTERNS = (
    "birthplace",
    "birth place",
    "born",
    "where born",
    "where was",
    "where is .* from",
    "origin",
    "hometown",
    "home town",
    "native of",
    "runtime",
    "running time",
    "duration",
    "length",
    "school",
    "university",
    "college",
    "alma mater",
)

PHASE1_SYSTEM_HYBRID = """\
You are a Query–DAG edge classifier for a web-search / deep-research trajectory.

You analyze ONE search turn t and output ONLY edges whose TARGET is a query issued in turn t.
You may output query→query edges and optional Prior_knowledge→query edges.
Never output Q0→query edges. Hard-failure retries are handled elsewhere; you may
emit only source-specific soft-failure edges.

INPUTS
You will receive:
- Q0 (background only)
- Query Index (prior queries)
- Turn-Level Status (anchors + failure candidates/notes)
- Current Think Block for turn t, split into sentences with ids [s1], [s2], ...
- New Queries in turn t (parallel)
- Hint Chart per new query q with:
  (1) token statuses: Q0_ONLY / Q0_SEEN / SEEN / NEW
  (2) Used_signals U(q): tokens that have admissible evidence (think and/or provenance)
  (3) Think_support: sentence hits for tokens
  (4) Source_provenance: provenance windows with source qid + type [VISIT]/[snippet] + window_text
IMPORTANT: covered_signals must be a SUBSET of the provided token-level U(q). Do not invent new tokens.

EDGE KINDS YOU MAY OUTPUT
1) evidence_derived:
   source = a prior query qX
   target = current-turn query q
   Allowed ONLY if at least one token in U(q) has a TOOL provenance window
   attributed to qX. A current think sentence may explain causal reuse, but a
   think-only mention cannot establish qX as the source.
   The earlier result must causally contribute a concrete entity, value, relation,
   or reformulation clue. Chronology, broad topic overlap, and generic search words
   are not contribution.

2) prior_knowledge_derived:
   source = "Prior_knowledge"
   target = current-turn query q
   Allowed ONLY if (per Hint Chart):
     token is NEW, token appears in Think_support, AND token has provenance_count = 0.

3) failure_derived (soft only):
   source = a prior query qX
   target = current-turn query q
   Allowed ONLY if a think sentence explicitly states qX was insufficient / missing required info
   and q is a direct attempt to fill that gap.
   Output failure_subtype="soft".
Do NOT output hard-failure retry edges or anchor-to-anchor transitions (they are deterministic elsewhere).

EDGE-KIND PRECEDENCE
When the current think block explicitly says that the source query's results
were insufficient and the target query repairs that gap, classify that
source-target relation as failure_derived even if the source Tool_Response
also contains overlapping target tokens. Do not emit evidence_derived for the
same source-target pair. Mere token overlap does not override the visible
failure/retry motivation.

EVIDENCE HARD GATE
Every emitted edge MUST cite:
- at least one think sentence id [s#] and/or
- at least one provenance window (including its source qid and [VISIT]/[snippet]).
If you cannot cite admissible evidence, DO NOT add the edge.
Do not rely on an automatic fallback: omitted parents remain omitted.

INFORMATIVE SIGNAL RULE (to reduce instability)
Define informative tokens as: tokens in U(q) that are NOT in a generic-template list
(founder/owner/born/year/info/details/near/location/restaurant/hotel/museum/etc.)
and have length ≥ 3, OR are numbers/dates, OR are URL-like.
Only informative tokens count toward coverage/MPSC.
If U(q) has NO informative tokens and there is no valid failure_derived / prior_knowledge_derived,
then output no_source_found for q.

PARENT SELECTION + TIE-BREAK (deterministic)
For each informative token in U(q), collect candidate parents from Source_provenance.
When multiple parents support the same token:
- Prefer [VISIT] over [snippet].
- Prefer the parent explicitly named or clearly referenced by the current think block.
- Otherwise use the most recent source by (source_turn, source_qid); this is the
  deterministic harness tie-break.

MINIMAL PARENT SET COVERAGE (MPSC)
For each target q:
- Select the smallest set of evidence_derived / prior_knowledge_derived parents whose union
  covers the informative tokens justified by the emitted edges.
- Each selected parent must cover at least one informative token not covered by others.
- Do not add redundant parents.
- Never invent a parent merely to force complete U(q) coverage.
Note: failure_derived edges are exempt from MPSC token-coverage (they are event edges).

PROHIBITIONS
- Never output Q0 edges.
- Never output edges between same-turn queries.
- Never hallucinate sources or evidence.
- Never use Prior_knowledge as a fallback when provenance exists.

OUTPUT (JSON ONLY)
{
  "edges": [
    {
      "source": "qX" | "Prior_knowledge",
      "target": "qY",
      "edge_kind": "evidence_derived" | "prior_knowledge_derived" | "failure_derived",
      "failure_subtype": "soft",
      "covered_signals": ["..."],
      "evidence": "...",
      "validation_reason": "retrieved_clue_reused" | "unsupported_signal_introduced" | "explicit_failure_repair",
      "confidence": "high" | "uncertain"
    }
  ],
  "no_source_found": ["qY", "..."]
}
Return valid JSON only.
"""
