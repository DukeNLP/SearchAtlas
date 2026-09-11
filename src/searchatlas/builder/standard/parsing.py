"""Parse tagged tool calls and chronological query occurrences."""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from . import settings

def parse_tool_call_json(blob: str) -> Optional[Dict[str, Any]]:
    blob = (blob or '').strip()
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        repaired = blob[:exc.pos].rstrip()
        trailer = blob[exc.pos:].strip()
        if not repaired or not trailer or any((ch != '}' for ch in trailer)):
            return None
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None

def _extract_xml_tag_text(block: str, tag: str) -> str:
    m = re.search(f'<{tag}>([\\s\\S]*?)</{tag}>', block, re.IGNORECASE)
    return m.group(1).strip() if m else ''

def extract_mcp_tool_calls(content: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not content:
        return out
    for m in settings.USE_MCP_TOOL_RE.finditer(content):
        block = m.group(1)
        server_name = _extract_xml_tag_text(block, 'server_name')
        tool_name = _extract_xml_tag_text(block, 'tool_name')
        arguments_blob = _extract_xml_tag_text(block, 'arguments')
        arguments = parse_tool_call_json(arguments_blob) if arguments_blob else None
        if not tool_name or not isinstance(arguments, dict):
            continue
        out.append({'server_name': server_name, 'tool_name': tool_name, 'arguments': arguments})
    return out

def extract_search_queries_from_tool_call(tc: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(tc, dict):
        return []
    if tc.get('name') not in settings.SEARCH_TOOL_NAMES:
        return []
    args = tc.get('arguments', {}) or {}
    raw = args.get('query')
    if raw is None:
        raw = args.get('search_queries')
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
    return out

def extract_search_queries_from_mcp_tool_call(tc: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(tc, dict):
        return []
    if tc.get('tool_name') != 'google_search':
        return []
    args = tc.get('arguments', {}) or {}
    raw = args.get('q')
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    return [text] if text else []

def extract_search_queries_from_response_headers(content: str) -> List[str]:
    if not content:
        return []
    return [m.group(1).strip() for m in settings.SEARCH_RESULT_HEADER_RE.finditer(content)]

def search_queries_in_message(content: str) -> List[str]:
    """Return only explicitly issued searches, in tool-call order."""
    calls = []
    for match in settings.TOOL_CALL_RE.finditer(content or ''):
        calls.append((match.start(), extract_search_queries_from_tool_call(parse_tool_call_json(match.group(1)))))
    for match in settings.USE_MCP_TOOL_RE.finditer(content or ''):
        for call in extract_mcp_tool_calls(match.group(0)):
            calls.append((match.start(), extract_search_queries_from_mcp_tool_call(call)))
    return [query for _, queries in sorted(calls, key=lambda item: item[0]) for query in queries]


def assistant_has_search_turn(messages: List[Dict[str, Any]], idx: int) -> bool:
    """Tool responses cannot create search actions that were not issued."""
    return bool(0 <= idx < len(messages) and messages[idx].get('role') == 'assistant'
                and search_queries_in_message(messages[idx].get('content') or ''))

def _fallback_singularize(w: str) -> str:
    """
    Simple plural→singular fallback (not perfect, but catches common cases):
      companies->company, queries->query, referees->referee, cards->card
    """
    w = w.lower()
    if len(w) <= 3:
        return w
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith(('ches', 'shes', 'xes', 'zes', 'sses')) and len(w) > 4:
        return w[:-2]
    if w.endswith('s') and (not w.endswith('ss')) and (len(w) > 3):
        return w[:-1]
    return w

def tokenize(text: str) -> List[str]:
    """
    Tokenize into alnum tokens INCLUDING digits.
    - lowercased
    - lemmatize (plural→singular) when spaCy is available
    - remove stopwords
    - keep length >= 2
    """
    text = text or ''
    if not text:
        return []
    toks: List[str] = []
    if settings._NLP is not None:
        doc = settings._NLP(text)
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            raw = t.text.lower()
            if re.fullmatch('[a-z0-9]+', raw) and re.search('[a-z]', raw) and re.search('[0-9]', raw):
                norm = raw
            elif t.like_num:
                norm = raw
            else:
                norm = (t.lemma_ or raw).lower()
            if not re.fullmatch('[a-z0-9]+', norm):
                continue
            if len(norm) < 2 and (not norm.isdigit()):
                continue
            if norm in settings.STOPWORDS:
                continue
            toks.append(norm)
        return toks
    raw = re.findall('[A-Za-z0-9]+', text.lower())
    for w in raw:
        if len(w) < 2 and (not w.isdigit()):
            continue
        w = _fallback_singularize(w)
        if w in settings.STOPWORDS:
            continue
        toks.append(w)
    return toks

def tokenize_set(text: str) -> Set[str]:
    return set(tokenize(text))

def extract_queries(messages: List[Dict]) -> Tuple[List[Dict], Dict[int, List[str]]]:
    """One search turn per assistant message; parallel calls share a turn."""
    queries = []
    turn_to_qids: Dict[int, List[str]] = {}
    turn = 0
    for message_idx, msg in enumerate(messages):
        if msg.get('role') != 'assistant':
            continue
        texts = search_queries_in_message(msg.get('content') or '')
        if not texts:
            continue
        turn += 1
        turn_to_qids[turn] = []
        for text in texts:
            qid = f'q{len(queries) + 1}'
            queries.append({'id': qid, 'text': text, 'turn': turn, 'message_idx': message_idx})
            turn_to_qids[turn].append(qid)
    return queries, turn_to_qids
