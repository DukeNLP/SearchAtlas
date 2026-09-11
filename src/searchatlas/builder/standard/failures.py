"""Identify explicit retrieval failures and attributable retries."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Any
from . import settings

def is_explicit_soft_failure_sentence(sentence: str) -> bool:
    sentence = sentence or ''
    return bool(settings.SOFT_FAIL_RETRIEVAL_CONTEXT_RE.search(sentence) or settings.SOFT_FAIL_EXPLICIT_INABILITY_RE.search(sentence) or settings.SOFT_FAIL_COMPLETENESS_RE.search(sentence))

def soft_failure_anchor_is_compatible(source_query: str, target_query: str, evidence_sentences: List[str]) -> bool:
    """Reject an anchor edge when the target switches to a conflicting year.

    A plural cross-branch statement such as "check both dates" remains
    compatible because the prior failure can motivate completion of the other
    branch. This guard only resolves explicit year conflicts; other cases stay
    available to the local classifier and strict detector.
    """
    source_years = set(settings.YEAR_TOKEN_RE.findall(source_query or ''))
    target_years = set(settings.YEAR_TOKEN_RE.findall(target_query or ''))
    if not source_years or not target_years or source_years & target_years:
        return True
    evidence_text = ' '.join(evidence_sentences or [])
    return bool(settings.SOFT_FAIL_CROSS_BRANCH_SCOPE_RE.search(evidence_text))

def detect_soft_failure(think_text: str, turn_idx: Optional[int]=None) -> Tuple[bool, List[str]]:
    """
    Turn-level soft failure: results exist but insufficient for the subgoal.
    Returns (is_soft_fail, evidence_sentences).
    """
    from .evidence import split_sentences
    from .llm import strict_edge_policy
    if not think_text:
        return (False, [])
    normalized_text = think_text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace('—', '-').replace('–', '-')
    sents = split_sentences(normalized_text)
    evidence_sents = []
    for sent in sents:
        for pat in settings.SOFT_FAIL_PATTERNS:
            if pat.search(sent):
                if strict_edge_policy() and (not is_explicit_soft_failure_sentence(sent)):
                    continue
                evidence_sents.append(sent.strip())
                break
    if evidence_sents:
        turn_label = f'turn {turn_idx}' if turn_idx is not None else 'unknown turn'
        print(f'[SOFT_FAIL] {turn_label}: {len(evidence_sents)} matched think sentence(s)')
        for i, sent in enumerate(evidence_sents):
            print(f'  [SOFT_FAIL_SENT {i}] {sent}')
    return (len(evidence_sents) > 0, evidence_sents)

def detect_hard_failures(turn_qids: List[str], result_counts: Dict[str, int], qid_search_doc: Dict[str, str], visit_summary_hard_failures: Optional[Dict[str, List[Dict[str, Any]]]]=None) -> Dict[str, Dict[str, Any]]:
    """
    Query-level hard failure: 0 results, blocked, 403/404, login wall, plus
    visit-summary tool failures.
    Returns structured info keyed by qid.
    """
    hard_fails: Dict[str, Dict[str, Any]] = {}
    for qid in turn_qids:
        search_evidences: List[str] = []
        rc = result_counts.get(qid)
        if rc is not None and rc == 0:
            search_evidences.append(f'{qid} returned 0 results')
        doc = qid_search_doc.get(qid, '')
        if doc:
            for pat in settings.HARD_FAIL_PATTERNS:
                if pat.search(doc):
                    search_evidences.append(f'tool output for {qid}: hard failure signal')
                    break
        visit_records = list((visit_summary_hard_failures or {}).get(qid, []))
        visit_evidences: List[str] = []
        for rec in visit_records:
            for ev in rec.get('evidences', []) or []:
                if ev not in visit_evidences:
                    visit_evidences.append(ev)
        origins: Set[str] = set()
        if search_evidences:
            origins.add('search')
        if visit_evidences:
            origins.add('visit_summary')
        if origins:
            failure_origin = next(iter(origins)) if len(origins) == 1 else 'mixed'
            hard_fails[qid] = {'evidences': list(dict.fromkeys(search_evidences + visit_evidences)), 'search_evidences': search_evidences, 'visit_summary_evidences': visit_evidences, 'visit_failures': visit_records, 'failure_origin': failure_origin}
    return hard_fails

def is_direct_retry(q_fail_text: str, q_retry_text: str, threshold: float=0.5) -> bool:
    """Check if q_retry is a direct retry/reformulation of q_fail (high token overlap)."""
    from .parsing import tokenize_set
    toks_fail = tokenize_set(q_fail_text)
    toks_retry = tokenize_set(q_retry_text)
    if not toks_fail or not toks_retry:
        return False
    overlap = toks_fail & toks_retry
    jaccard = len(overlap) / len(toks_fail | toks_retry)
    return jaccard >= threshold

def jaccard_similarity(toks_a: Set[str], toks_b: Set[str]) -> float:
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(len(toks_a | toks_b), 1)

def aggregate_failure_origin(hard_fails_prev: Dict[str, Dict[str, Any]]) -> str:
    origins = {info.get('failure_origin', 'search') for info in hard_fails_prev.values() if info.get('failure_origin')}
    if not origins:
        return 'search'
    if len(origins) == 1:
        return next(iter(origins))
    return 'mixed'

def select_localized_hard_retry(q_fail: str, fail_info: Dict[str, Any], new_qids: List[str], qid_to_text: Dict[str, str], query_url_metadata: Dict[str, Dict[str, List[str]]]) -> Tuple[Optional[str], Dict[str, Any]]:
    from .evidence import qid_sort_key
    from .parsing import tokenize_set
    fail_text = qid_to_text.get(q_fail, '')
    fail_toks = tokenize_set(fail_text)
    fail_meta = query_url_metadata.get(q_fail, {})
    fail_search_urls = set(fail_meta.get('search_urls', []) or [])
    fail_search_sites = set(fail_meta.get('search_sites', []) or [])
    fail_search_url_tokens = set(fail_meta.get('search_url_tokens', []) or [])
    visit_failures = fail_info.get('visit_failures', []) or []
    fail_visit_urls = {rec.get('normalized_visit_url', '') for rec in visit_failures if rec.get('normalized_visit_url')}
    fail_visit_sites = {rec.get('visit_site', '') for rec in visit_failures if rec.get('visit_site')}
    fail_visit_url_tokens: Set[str] = set()
    for rec in visit_failures:
        fail_visit_url_tokens.update(rec.get('url_tokens', []) or [])
    fail_urls = fail_search_urls | fail_visit_urls
    fail_sites = fail_search_sites | fail_visit_sites
    fail_url_tokens = fail_search_url_tokens | fail_visit_url_tokens
    failure_origin = fail_info.get('failure_origin', 'search')
    best_qid: Optional[str] = None
    best_key: Optional[Tuple[float, int]] = None
    best_meta: Dict[str, Any] = {}
    for q_retry in new_qids:
        retry_text = qid_to_text.get(q_retry, '')
        retry_toks = tokenize_set(retry_text)
        retry_meta = query_url_metadata.get(q_retry, {})
        retry_urls = set(retry_meta.get('search_urls', []) or []) | set(retry_meta.get('visit_urls', []) or [])
        retry_sites = set(retry_meta.get('search_sites', []) or []) | set(retry_meta.get('visit_sites', []) or [])
        retry_url_tokens = set(retry_meta.get('search_url_tokens', []) or []) | set(retry_meta.get('visit_url_tokens', []) or [])
        query_sim = jaccard_similarity(fail_toks, retry_toks)
        exact_url_hits = sorted(fail_urls & retry_urls)
        same_site_hits = sorted(fail_sites & retry_sites)
        entity_hits = sorted(fail_url_tokens & (retry_toks | retry_url_tokens))
        direct_text_retry = is_direct_retry(fail_text, retry_text)
        avoid_failed_site = bool(failure_origin in {'visit_summary', 'mixed'} and query_sim >= 0.35 and fail_visit_sites and retry_sites and fail_visit_sites.isdisjoint(retry_sites))
        if failure_origin == 'search':
            is_candidate = bool(direct_text_retry or exact_url_hits)
        elif failure_origin == 'visit_summary':
            is_candidate = bool(exact_url_hits or (entity_hits and query_sim >= 0.2) or direct_text_retry or avoid_failed_site)
        else:
            is_candidate = bool(direct_text_retry or exact_url_hits or (entity_hits and query_sim >= 0.2) or avoid_failed_site)
        if not is_candidate:
            continue
        score = 0.0
        match_signals: List[str] = []
        if exact_url_hits:
            score += 100.0
            match_signals.append('same_url')
        if same_site_hits:
            score += 35.0
            match_signals.append('same_site')
        if entity_hits:
            score += min(18.0, 6.0 * len(entity_hits))
            match_signals.append('same_entity')
        if direct_text_retry:
            score += 40.0 + 20.0 * query_sim
            match_signals.append('query_overlap')
        elif query_sim >= 0.25:
            score += 10.0 * query_sim
        if avoid_failed_site:
            score += 8.0
            match_signals.append('avoid_failed_site')
        qid_num = qid_sort_key(q_retry)[0]
        key = (score, qid_num)
        if best_key is None or key > best_key:
            best_key = key
            best_qid = q_retry
            best_meta = {'retry_score': round(score, 3), 'query_similarity': round(query_sim, 3), 'match_signals': match_signals, 'same_url_hits': exact_url_hits, 'same_site_hits': same_site_hits, 'entity_hits': entity_hits, 'avoid_failed_site': avoid_failed_site}
    return (best_qid, best_meta)

def build_deterministic_hard_failure_edges_for_turn(turn_idx: int, new_qids: List[str], turn_to_qids: Dict[int, List[str]], anchors: Dict[int, str], qid_to_text: Dict[str, str], hard_fails_prev: Dict[str, Dict[str, Any]], query_url_metadata: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    if not hard_fails_prev:
        return edges
    # A batch-level failure cannot be attached to an arbitrary first query.
    # Retain only a failed query paired with an evidence-compatible retry.
    for q_fail, fail_info in hard_fails_prev.items():
        best_retry, retry_meta = select_localized_hard_retry(q_fail, fail_info, new_qids, qid_to_text, query_url_metadata)
        if best_retry:
            edges.append({'source': q_fail, 'target': best_retry, 'type': 'lead_to', 'edge_kind': 'failure_derived', 'failure_subtype': 'hard', 'failure_origin': fail_info.get('failure_origin', 'search'), 'evidence': f"Hard failure: {'; '.join(fail_info.get('evidences', [])[:2])}. Direct retry.", 'confidence': 'high', 'phase': 'deterministic_hard_fail', 'metadata': {'failure_origin': fail_info.get('failure_origin', 'search'), **retry_meta}})
            print(f'    ✦ Turn {turn_idx}: hard failure retry {q_fail} -> {best_retry}')
    return edges
