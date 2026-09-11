"""Decompose question constraints and match their query deployments."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from ..llm_compat import completion_token_budget, record_llm_event
from . import settings

def decompose_q0_units(question: str) -> List[Dict[str, str]]:
    """
    LLM call to decompose Q0 into atomic units.
    Returns [{unit_id, unit_type, q0_span}, ...].
    Falls back to simple tokenization if LLM fails.
    """
    from .attribution import call_llm_json
    from .parsing import tokenize
    prompt = f'Question:\n{question}\n\nDecompose into atomic Q0 units (JSON only):'
    result = call_llm_json(settings.Q0_DECOMPOSE_SYSTEM, prompt, max_tokens=1500, retries=2, call_name='q0_decompose')
    if result and 'units' in result:
        units = result['units']
        valid = []
        for u in units:
            if isinstance(u, dict) and u.get('unit_id') and u.get('q0_span'):
                u.setdefault('unit_type', 'attribute')
                valid.append(u)
        if valid:
            for u in valid:
                print(f'''      {u['unit_id']} ({u['unit_type']}): "{u['q0_span']}"''')
            return valid
    toks = tokenize(question)
    return [{'unit_id': f'u{i + 1}', 'unit_type': 'attribute', 'q0_span': tok} for i, tok in enumerate(toks)]

def match_q0_unit_to_query_tiered(unit: Dict[str, str], query_text: str) -> Tuple[Optional[str], int]:
    """
    Tiered Q0 unit → query matching.
    Returns (matched_span_or_None, tier).
      tier 1 — exact substring match           → deterministic accept
      tier 2 — token overlap ≥ 80%             → deterministic accept
      tier 3 — token overlap 40%-80%           → gray zone, needs LLM arbitration
      tier 0 — no match / overlap < 40%        → no rule-based match
    """
    from .parsing import tokenize_set
    q0_span = unit.get('q0_span', '').strip().lower()
    q_lower = query_text.lower()
    if not q0_span:
        return (None, 0)
    idx = q_lower.find(q0_span)
    if idx >= 0:
        return (query_text[idx:idx + len(q0_span)], 1)
    unit_toks = tokenize_set(q0_span)
    if not unit_toks:
        return (None, 0)
    q_toks = tokenize_set(query_text)
    overlap = unit_toks & q_toks
    ratio = len(overlap) / len(unit_toks)
    if ratio >= 0.8:
        return (', '.join(sorted(overlap)), 2)
    elif ratio >= 0.4:
        return (', '.join(sorted(overlap)), 3)
    else:
        return (None, 0)

def match_q0_unit_to_query(unit: Dict[str, str], query_text: str) -> Optional[str]:
    span, tier = match_q0_unit_to_query_tiered(unit, query_text)
    return span if tier >= 1 else None

def _llm_arbiter_q0_matches(question: str, pairs: List[Dict[str, str]]) -> Dict[str, bool]:
    """
    Batch LLM call to judge a list of (unit, query) candidate pairs.

    Args:
      question: the original Q0 text (for context)
      pairs: list of dicts with keys:
        pair_id, unit_id, unit_type, q0_span, qid, query_text

    Returns: {pair_id: True/False}
    """
    from .llm import _apply_llm_io_options, _completion_length_kwargs, _create_chat_completion, _format_llm_exception, _parse_model_json, strict_edge_policy
    if not pairs:
        return {}
    lines = [f'Q0: {question}\n', 'Pairs to judge:']
    for p in pairs:
        lines.append(f'''  pair_id={p['pair_id']}  unit={p['unit_id']}({p['unit_type']})  q0_span="{p['q0_span']}"  →  {p['qid']}: "{p['query_text']}"''')
    user_prompt = '\n'.join(lines)
    resp = None
    text = ''
    effective_max_tokens = completion_token_budget(3000, settings.LLM_MIN_COMPLETION_TOKENS)
    try:
        kwargs = dict(model=settings.MODEL, messages=[{'role': 'system', 'content': settings.Q0_MATCH_ARBITER_SYSTEM}, {'role': 'user', 'content': user_prompt}], temperature=settings.LLM_TEMPERATURE, timeout=settings.LLM_TIMEOUT_SEC)
        kwargs.update(_completion_length_kwargs(effective_max_tokens))
        _apply_llm_io_options(kwargs, json_object=False)
        resp = _create_chat_completion(kwargs)
        text = resp.choices[0].message.content or ''
        verdicts = _parse_model_json(text, expected=list)
        record_llm_event(settings.LLM_USAGE_JSONL, event='success', call_name='q0_match_arbiter', model=settings.MODEL, prompt_chars=len(settings.Q0_MATCH_ARBITER_SYSTEM) + len(user_prompt), max_tokens=effective_max_tokens, attempt=1, response=resp, response_chars=len(text), metadata={'edge_policy': settings.EDGE_POLICY, 'llm_io_mode': settings.LLM_IO_MODE})
        return {v['pair_id']: bool(v.get('match', False)) for v in verdicts}
    except Exception as e:
        record_llm_event(settings.LLM_USAGE_JSONL, event='parse_error' if resp is not None else 'request_error', call_name='q0_match_arbiter', model=settings.MODEL, prompt_chars=len(settings.Q0_MATCH_ARBITER_SYSTEM) + len(user_prompt), max_tokens=effective_max_tokens, attempt=1, response=resp, response_chars=len(text), error=_format_llm_exception(e), metadata={'edge_policy': settings.EDGE_POLICY, 'llm_io_mode': settings.LLM_IO_MODE})
        print(f'    ⚠ Q0 arbiter LLM error: {_format_llm_exception(e)}')
    fallback_value = not strict_edge_policy()
    return {p['pair_id']: fallback_value for p in pairs}

def build_q0_unit_edges(q0_units: List[Dict[str, str]], queries: List[Dict], question: str='') -> Tuple[List[Dict], Dict[str, str]]:
    """
    Tiered Q0-unit → query matching with LLM arbitration.

    Tier 1  substring match       → deterministic accept
    Tier 2  token overlap ≥ 80%   → deterministic accept
    Tier 3  token overlap 40-80%  → LLM arbitration (filter noise)
    Tier 4  unit has 0 matches    → LLM rescue (find paraphrases)

    Returns:
      edges: list of Q0->query edge dicts
      unit_first_use: {unit_id: qid} mapping
    """
    sorted_qs = sorted(queries, key=lambda q: (q['turn'], int(q['id'][1:]) if q['id'][1:].isdigit() else 999))
    qid_to_text = {q['id']: q['text'] for q in sorted_qs}
    accepted: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    gray_zone: List[Tuple[str, str, str]] = []
    for unit in q0_units:
        uid = unit['unit_id']
        for q in sorted_qs:
            span, tier = match_q0_unit_to_query_tiered(unit, q['text'])
            if tier in (1, 2):
                accepted[uid].append((q['id'], span))
            elif tier == 3:
                gray_zone.append((uid, q['id'], span))
    if gray_zone:
        unit_map = {u['unit_id']: u for u in q0_units}
        arbiter_pairs = []
        for i, (uid, qid, span) in enumerate(gray_zone):
            u = unit_map[uid]
            arbiter_pairs.append({'pair_id': f't3_{i}', 'unit_id': uid, 'unit_type': u.get('unit_type', ''), 'q0_span': u.get('q0_span', ''), 'qid': qid, 'query_text': qid_to_text.get(qid, '')})
        print(f'    Tier 3 arbiter: {len(arbiter_pairs)} gray-zone pairs → LLM')
        verdicts = _llm_arbiter_q0_matches(question, arbiter_pairs)
        for (uid, qid, span), pair_info in zip(gray_zone, arbiter_pairs):
            pid = pair_info['pair_id']
            if verdicts.get(pid, False):
                accepted[uid].append((qid, span))
                print(f'      ✓ Tier 3 accepted: {uid} → {qid}')
            else:
                print(f'      ✗ Tier 3 rejected: {uid} → {qid}')
    unmatched_uids = [u['unit_id'] for u in q0_units if u['unit_id'] not in accepted]
    if unmatched_uids and question:
        unit_map = {u['unit_id']: u for u in q0_units}
        rescue_pairs = []
        for uid in unmatched_uids:
            u = unit_map[uid]
            for q in sorted_qs:
                rescue_pairs.append({'pair_id': f"t4_{uid}_{q['id']}", 'unit_id': uid, 'unit_type': u.get('unit_type', ''), 'q0_span': u.get('q0_span', ''), 'qid': q['id'], 'query_text': q['text']})
        if rescue_pairs:
            print(f'    Tier 4 rescue: {len(unmatched_uids)} unmatched units × {len(sorted_qs)} queries = {len(rescue_pairs)} pairs → LLM')
            verdicts = _llm_arbiter_q0_matches(question, rescue_pairs)
            for rp in rescue_pairs:
                if verdicts.get(rp['pair_id'], False):
                    accepted[rp['unit_id']].append((rp['qid'], f"[paraphrase] {rp['q0_span']}"))
                    print(f"      ✓ Tier 4 rescued: {rp['unit_id']} → {rp['qid']}")
    unit_first_use: Dict[str, str] = {}
    by_qid: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'first_use_units': [], 'reuse_units': [], 'first_use_matches': [], 'reuse_matches': []})
    for unit in q0_units:
        uid = unit['unit_id']
        matched = accepted.get(uid, [])
        if not matched:
            continue
        qid_order = {q['id']: idx for idx, q in enumerate(sorted_qs)}
        matched_sorted = sorted(matched, key=lambda m: qid_order.get(m[0], 999999))
        first_qid, first_qspan = matched_sorted[0]
        unit_first_use[uid] = first_qid
        by_qid[first_qid]['first_use_units'].append(uid)
        by_qid[first_qid]['first_use_matches'].append({'unit_id': uid, 'unit_type': unit.get('unit_type', ''), 'q0_span': unit.get('q0_span', ''), 'q_span': first_qspan})
        for rqid, rqspan in matched_sorted[1:]:
            by_qid[rqid]['reuse_units'].append(uid)
            by_qid[rqid]['reuse_matches'].append({'unit_id': uid, 'unit_type': unit.get('unit_type', ''), 'q0_span': unit.get('q0_span', ''), 'q_span': rqspan, 'first_use_qid': first_qid})
    edges: List[Dict] = []
    for qid in sorted(by_qid.keys(), key=lambda x: int(x[1:]) if x.startswith('q') and x[1:].isdigit() else 999999):
        first_units = sorted(set(by_qid[qid]['first_use_units']))
        reuse_units = sorted(set(by_qid[qid]['reuse_units']))
        evidence_parts = []
        if first_units:
            evidence_parts.append(f"Q0 unit first-use: {', '.join(first_units[:10])}{('…' if len(first_units) > 10 else '')}")
        if reuse_units:
            evidence_parts.append(f"Q0 unit reuse: {', '.join(reuse_units[:10])}{('…' if len(reuse_units) > 10 else '')}")
        evidence = '; '.join(evidence_parts) if evidence_parts else 'Q0 unit linkage'
        edges.append({'source': 'Q0', 'target': qid, 'type': 'lead_to', 'edge_kind': 'constraint_use', 'evidence': evidence, 'confidence': 'high', 'phase': 'q0_unit_firstuse_reuse', 'metadata': {'first_use_units': first_units, 'reuse_units': reuse_units, 'first_use_matches': by_qid[qid]['first_use_matches'], 'reuse_matches': by_qid[qid]['reuse_matches']}})
    return (edges, unit_first_use)

def q0_edges_from_tokens_firstuse(question: str, queries: List[Dict]) -> List[Dict]:
    """
    Rule-based Q0 edges with first-use + reuse (no LLM).

    For each Q0 content token, find all queries that contain it.
      - first-use: earliest query containing the token
      - reuse: later queries containing the same token

    Build one Q0→q edge per target query, carrying both first-use and reuse
    token lists in evidence/metadata.

    IMPORTANT SAME-TURN RULE:
      - If a query only has reuse_toks, we normally suppress its Q0 edge,
        because earlier-turn q->q edges should explain that reuse.
      - BUT if the first-use of a reuse token happened in the SAME turn,
        we must keep Q0->q, because same-turn q->q edges are forbidden.
    """
    from .parsing import tokenize_set
    q0_token_set = tokenize_set(question)
    if not q0_token_set:
        return []
    edges = []
    sorted_qs = sorted(queries, key=lambda q: (q['turn'], int(q['id'][1:]) if q['id'][1:].isdigit() else 999))
    qid_to_turn = {q['id']: q['turn'] for q in sorted_qs}
    token_first_qid: Dict[str, str] = {}
    token_reuse_qids: Dict[str, List[str]] = defaultdict(list)
    for tok in q0_token_set:
        matched_qids: List[str] = []
        for q in sorted_qs:
            if tok in tokenize_set(q['text']):
                matched_qids.append(q['id'])
        if matched_qids:
            token_first_qid[tok] = matched_qids[0]
            token_reuse_qids[tok] = matched_qids[1:]
    qid_first_tokens: Dict[str, List[str]] = defaultdict(list)
    qid_reuse_tokens: Dict[str, List[str]] = defaultdict(list)
    for tok, qid in token_first_qid.items():
        qid_first_tokens[qid].append(tok)
    for tok, rqids in token_reuse_qids.items():
        for rqid in rqids:
            qid_reuse_tokens[rqid].append(tok)
    all_qids = sorted(set(qid_first_tokens.keys()) | set(qid_reuse_tokens.keys()), key=lambda x: int(x[1:]) if x.startswith('q') and x[1:].isdigit() else 999999)
    for qid in all_qids:
        first_toks = sorted(set(qid_first_tokens.get(qid, [])))
        reuse_toks = sorted(set(qid_reuse_tokens.get(qid, [])))
        curr_turn = qid_to_turn.get(qid, -1)
        keep_reuse_only_same_turn = False
        same_turn_reuse_toks: List[str] = []
        earlier_turn_reuse_toks: List[str] = []
        for tok in reuse_toks:
            first_qid = token_first_qid.get(tok)
            first_turn = qid_to_turn.get(first_qid, -999)
            if first_turn == curr_turn:
                keep_reuse_only_same_turn = True
                same_turn_reuse_toks.append(tok)
            else:
                earlier_turn_reuse_toks.append(tok)
        if not first_toks and (not keep_reuse_only_same_turn):
            continue
        evidence_parts = []
        if first_toks:
            evidence_parts.append(f"Q0 token first-use: {', '.join(first_toks[:12])}{('…' if len(first_toks) > 12 else '')}")
        if reuse_toks:
            evidence_parts.append(f"Q0 token reuse: {', '.join(reuse_toks[:12])}{('…' if len(reuse_toks) > 12 else '')}")
        if not first_toks and keep_reuse_only_same_turn:
            evidence_parts.append(f"same-turn reuse fallback: {', '.join(same_turn_reuse_toks[:12])}{('…' if len(same_turn_reuse_toks) > 12 else '')}")
        edges.append({'source': 'Q0', 'target': qid, 'type': 'lead_to', 'edge_kind': 'constraint_use', 'evidence': '; '.join(evidence_parts) if evidence_parts else 'Q0 token linkage', 'confidence': 'high', 'phase': 'kw_q0_firstuse_reuse', 'metadata': {'first_use_tokens': first_toks, 'reuse_tokens': reuse_toks, 'same_turn_reuse_tokens': sorted(same_turn_reuse_toks), 'earlier_turn_reuse_tokens': sorted(earlier_turn_reuse_toks), 'reuse_only_same_turn_fallback': not first_toks and keep_reuse_only_same_turn}})
    return edges

def q0_edges_from_tokens_dense(question: str, queries: List[Dict]) -> List[Dict]:
    from .parsing import tokenize_set
    q0_token_set = tokenize_set(question)
    if not q0_token_set:
        return []
    edges = []
    for q in queries:
        qid = q['id']
        overlap = sorted(tokenize_set(q['text']) & q0_token_set)
        if not overlap:
            continue
        edges.append({'source': 'Q0', 'target': qid, 'type': 'lead_to', 'edge_kind': 'constraint_use', 'evidence': f"Q0 token overlap: {', '.join(overlap)}", 'confidence': 'high', 'phase': 'kw_q0_dense', 'metadata': {'q0_tokens': overlap, 'num_overlap': len(overlap)}})
    return edges
