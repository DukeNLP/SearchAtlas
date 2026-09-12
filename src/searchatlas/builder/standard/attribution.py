"""Select trace-supported parents for each query turn."""
from __future__ import annotations
import re
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from tqdm import tqdm
from ..llm_compat import completion_token_budget, record_llm_event
from ..citations import normalize_provenance_citations
from . import settings

def call_llm_json(system: str, user: str, max_tokens: int=3500, retries: int=3, call_name: str='call_llm_json') -> Optional[Dict]:
    from .llm import _apply_llm_io_options, _completion_length_kwargs, _create_chat_completion, _format_llm_exception, _is_connection_like_llm_error, _is_non_retryable_llm_error, _parse_model_json
    connection_retries_left = settings.LLM_CONNECTION_EXTRA_RETRIES
    effective_max_tokens = completion_token_budget(max_tokens, settings.LLM_MIN_COMPLETION_TOKENS)
    for attempt in range(retries):
        resp = None
        text = ''
        try:
            kwargs = dict(model=settings.MODEL, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], temperature=settings.LLM_TEMPERATURE, timeout=settings.LLM_TIMEOUT_SEC)
            kwargs.update(_completion_length_kwargs(effective_max_tokens))
            _apply_llm_io_options(kwargs, json_object=True)
            resp = _create_chat_completion(kwargs)
            text = resp.choices[0].message.content or ''
            parsed = _parse_model_json(text, expected=dict)
            record_llm_event(settings.LLM_USAGE_JSONL, event='success', call_name=call_name, model=settings.MODEL, prompt_chars=len(system) + len(user), max_tokens=effective_max_tokens, attempt=attempt + 1, response=resp, response_chars=len(text), metadata={'edge_policy': settings.EDGE_POLICY, 'llm_io_mode': settings.LLM_IO_MODE})
            return parsed
        except Exception as e:
            record_llm_event(settings.LLM_USAGE_JSONL, event='parse_error' if resp is not None else 'request_error', call_name=call_name, model=settings.MODEL, prompt_chars=len(system) + len(user), max_tokens=effective_max_tokens, attempt=attempt + 1, response=resp, response_chars=len(text), error=_format_llm_exception(e), metadata={'edge_policy': settings.EDGE_POLICY, 'llm_io_mode': settings.LLM_IO_MODE})
            print(f'    LLM error (attempt {attempt + 1}/{retries}): {_format_llm_exception(e)}')
            if _is_non_retryable_llm_error(e):
                break
            if _is_connection_like_llm_error(e) and connection_retries_left > 0:
                extra_idx = settings.LLM_CONNECTION_EXTRA_RETRIES - connection_retries_left
                wait_sec = settings.LLM_CONNECTION_WAIT_BASE_SEC * 2 ** extra_idx
                connection_retries_left -= 1
                print(f'    Connection-like LLM failure; extra retry {settings.LLM_CONNECTION_EXTRA_RETRIES - connection_retries_left}/{settings.LLM_CONNECTION_EXTRA_RETRIES} after {wait_sec:.1f}s')
                time.sleep(wait_sec)
                continue
            time.sleep(2 ** attempt)
    return None

def build_user_prompt_hybrid(t: int, question: str, qid_to_text: Dict[str, str], turn_to_qids: Dict[int, List[str]], result_counts: Dict[str, int], current_think: str, new_qids: List[str], per_query_payload: Dict[str, Dict[str, Any]], anchor_prev: Optional[str]=None, anchor_curr: Optional[str]=None, soft_fail_prev: bool=False, soft_fail_evidence: Optional[List[str]]=None, hard_fails_prev: Optional[Dict[str, Any]]=None) -> str:
    from .llm import strict_edge_policy
    parts: List[str] = []
    parts.append(f'## Q0 (for background only; do NOT output Q0 edges)\n{question}\n')
    if t > 1:
        parts.append('## Query Index')
        for prev_t in range(1, t):
            for qid in turn_to_qids.get(prev_t, []):
                rc = result_counts.get(qid)
                rc_str = ''
                if rc is not None:
                    rc_str = ' [0 results]' if rc == 0 else f' [{rc} results]'
                parts.append(f'''  {qid} (T{prev_t}): "{qid_to_text.get(qid, '?')}"{rc_str}''')
        parts.append('')
    parts.append(f'## Turn-Level Status')
    if anchor_curr:
        parts.append(f'  anchor(t={t}): {anchor_curr}')
    if anchor_prev:
        parts.append(f'  anchor(t={t - 1}): {anchor_prev}')
    if soft_fail_prev:
        parts.append(f'  SOFT_FAIL(t={t - 1}): TRUE (prior results insufficient)')
        if soft_fail_evidence:
            for i, s in enumerate(soft_fail_evidence[:5]):
                parts.append(f'    evidence[{i}]: {s[:200]}')
        if strict_edge_policy():
            parts.append(f'  -> Explicit soft failure accepted by the strict detector. A deterministic {anchor_prev} -> {anchor_curr} [failure_derived/soft] edge will be added. Do not emit an evidence_derived edge for that same pair merely because the failed result contains overlapping tokens; emit other source-specific parents only when they make a distinct concrete contribution.')
        else:
            parts.append(f'  -> Deterministic anchor edge already added: {anchor_prev} -> {anchor_curr} [failure_derived/soft]')
    else:
        parts.append(f'  SOFT_FAIL(t={t - 1}): FALSE')
    if hard_fails_prev:
        for qf, fail_info in hard_fails_prev.items():
            failure_origin = None
            if isinstance(fail_info, dict):
                evs = fail_info.get('evidences') or []
                failure_origin = fail_info.get('failure_origin')
            else:
                evs = fail_info or []
            if not isinstance(evs, list):
                evs = [str(evs)]
            origin_suffix = f' [{failure_origin}]' if failure_origin else ''
            parts.append(f"  HARD_FAIL({qf}): TRUE{origin_suffix} — {'; '.join((str(ev) for ev in evs[:3]))}")
        parts.append('  → Deterministic retry edges added for direct retries (if any)')
    parts.append('')
    parts.append(f'## Current Think Block (turn {t})')
    parts.append(current_think if current_think else '(no think block)')
    parts.append('')
    parts.append(f'## New Queries (turn {t}) — PARALLEL (no edges among them)')
    for qid in new_qids:
        parts.append(f'''  {qid}: "{qid_to_text.get(qid, '?')}"''')
    parts.append('')
    parts.append('## Hint Charts (per new query)')
    for qid in new_qids:
        q_text = qid_to_text.get(qid, '?')
        payload = per_query_payload.get(qid, {})
        token_status = payload.get('token_status', {})
        unsupported_new = payload.get('unsupported_new_tokens', [])
        think_support = payload.get('think_support', {})
        seen_latest_qid = payload.get('seen_latest_qid', {})
        seen_first_qid = payload.get('seen_first_qid', {})
        candidates_by_token = payload.get('candidates_by_token', {})
        used_signals = payload.get('used_signals', set())
        parts.append(f'\n### Target {qid}: "{q_text}"')
        parts.append('Tokens:')
        for tok, st in token_status.items():
            extra = ''
            if st in ('SEEN', 'Q0_SEEN'):
                fq = seen_first_qid.get(tok)
                lq = seen_latest_qid.get(tok)
                if fq:
                    extra += f', first_query={fq}'
                if lq and lq != fq:
                    extra += f', latest_query={lq}'
            parts.append(f'  - {tok}: {{status: {st}{extra}}}')
        if unsupported_new:
            parts.append(f'UNSUPPORTED_NEW (cannot justify edges): {unsupported_new}')
        parts.append(f"Used_signals U(q): {(sorted(used_signals) if used_signals else '(empty)')}")
        parts.append('Think_support (sentence-anchored; only sentences containing the token):')
        if not think_support:
            parts.append('  (none)')
        else:
            for tok, hits in think_support.items():
                if not hits:
                    parts.append(f'  {tok}: (none)')
                    continue
                parts.append(f'  {tok}:')
                for h in hits[:10]:
                    parts.append(f"    - [s{h['sent_idx']}] {h['sent_text']}")
        parts.append('Source_provenance (grouped by source, deduplicated):')
        if not candidates_by_token:
            parts.append('  (none)')
        else:
            from collections import OrderedDict
            source_meta: dict = {}
            window_registry: OrderedDict = OrderedDict()
            for tok, sources in candidates_by_token.items():
                if not sources:
                    continue
                for s in sources:
                    sqid = s['source_qid']
                    if sqid not in source_meta:
                        sq = (s.get('source_query') or '').replace('\n', ' ')
                        if len(sq) > 180:
                            sq = sq[:180] + '…'
                        bst = s.get('best_source_type', 'snippet')
                        source_meta[sqid] = {'query': sq, 'turn': s.get('source_turn'), 'type_tag': '[VISIT]' if bst == 'visit' else '[snippet]', 'fp': s.get('is_first_provenance', False), 'hit_counts': {}}
                    source_meta[sqid]['hit_counts'][tok] = s.get('hit_count', 0)
                    if s.get('is_first_provenance'):
                        source_meta[sqid]['fp'] = True
                    for p in s.get('provenance', []):
                        wkey = (sqid, p.get('chunk_label'), p.get('window_lo'), p.get('window_hi'))
                        if wkey not in window_registry:
                            win = (p.get('window_text') or '').strip().replace('\n', ' ')
                            if len(win) > 900:
                                win = win[:900] + '…'
                            ptype = p.get('source_type', 'snippet')
                            window_registry[wkey] = {'text': win, 'ptype_tag': '[VISIT]' if ptype == 'visit' else '[snippet]', 'tokens': set()}
                        window_registry[wkey]['tokens'].add(tok)
            from itertools import groupby as _groupby
            merged_windows: dict = {}
            for (sqid, chunk, wlo, whi), winfo in window_registry.items():
                key = (sqid, chunk)
                merged_windows.setdefault(key, []).append({'lo': wlo, 'hi': whi, 'text': winfo['text'], 'ptype_tag': winfo['ptype_tag'], 'tokens': set(winfo['tokens'])})
            for key in merged_windows:
                wins = sorted(merged_windows[key], key=lambda w: (w['lo'] or 0, w['hi'] or 0))
                merged = []
                for w in wins:
                    if merged and w['lo'] is not None and (merged[-1]['hi'] is not None) and (w['lo'] <= merged[-1]['hi']):
                        prev = merged[-1]
                        prev['tokens'] |= w['tokens']
                        if (w['hi'] or 0) > (prev['hi'] or 0):
                            prev['hi'] = w['hi']
                            prev['text'] = w['text'] if len(w['text']) > len(prev['text']) else prev['text']
                    else:
                        merged.append(w)
                merged_windows[key] = merged
            for sqid, meta in source_meta.items():
                tok_hits = ', '.join((f'{t}={c}' for t, c in meta['hit_counts'].items()))
                fp_tag = ' [first-prov]' if meta['fp'] else ''
                parts.append(f'''  source={sqid} (T{meta['turn']}) {meta['type_tag']}{fp_tag} query="{meta['query']}" token_hits={{{tok_hits}}}''')
                for (sq2, chunk), wins in merged_windows.items():
                    if sq2 != sqid:
                        continue
                    for w in wins:
                        covers = ','.join(sorted(w['tokens']))
                        parts.append(f"    prov {w['ptype_tag']} chunk={chunk} window=[{w['lo']}..{w['hi']}] covers=[{covers}] :: {w['text']}")
    parts.append('\nOutput JSON only:')
    return '\n'.join(parts)

def phase1_hybrid_classify(question: str, messages: List[Dict], queries: List[Dict], turn_to_qids: Dict[int, List[str]], top_k: int, sent_window: int, max_snips: int, result_counts: Dict[str, int], qid_search_docs: Dict[str, str], turn_docs: Dict[str, str], visit_summary_hard_failures: Dict[str, List[Dict[str, Any]]], query_url_metadata: Dict[str, Dict[str, List[str]]], turn_thinks: Dict[int, str], turn_contexts: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[str]]:
    from .evidence import compute_token_statuses, denoise_provenance, first_prior_query_with_token, latest_prior_query_with_token, select_top_k_sources_from_hits, sentence_hits_in_text, token_provenance_in_prior_docs
    from .failures import build_deterministic_hard_failure_edges_for_turn, detect_hard_failures, detect_soft_failure, soft_failure_anchor_is_compatible
    from .llm import strict_edge_policy
    from .parent_selection import apply_mpsc, compute_parent_coverage, compute_used_signals
    from .parsing import tokenize, tokenize_set
    qid_to_text = {q['id']: q['text'] for q in queries}
    qid_to_turn = {q['id']: q['turn'] for q in queries}
    q0_token_set = tokenize_set(question)
    all_edges: List[Dict] = []
    per_turn_raw: List[Dict] = []
    explicitly_no_source: List[str] = []
    max_turn = max(turn_to_qids.keys()) if turn_to_qids else 0
    anchors: Dict[int, str] = {}
    for t_idx in range(1, max_turn + 1):
        qids_in_turn = turn_to_qids.get(t_idx, [])
        if qids_in_turn:
            anchors[t_idx] = qids_in_turn[0]
    soft_fail_by_turn: Dict[int, Tuple[bool, List[str]]] = {}
    for t_idx in range(1, max_turn + 1):
        think_text = turn_thinks.get(t_idx, '') or ''
        sf, sf_ev = detect_soft_failure(think_text, turn_idx=t_idx)
        soft_fail_by_turn[t_idx] = (sf, sf_ev)
    hard_fail_by_turn: Dict[int, Dict[str, List[str]]] = {}
    for t_idx in range(1, max_turn + 1):
        prev_qids = turn_to_qids.get(t_idx - 1, []) if t_idx > 1 else []
        if prev_qids:
            context = turn_contexts[t_idx]
            hard_fail_by_turn[t_idx] = detect_hard_failures(prev_qids, context['result_counts'], context['search_docs'], visit_summary_hard_failures=context['visit_failures'])
        else:
            hard_fail_by_turn[t_idx] = {}
    for t in tqdm(range(1, max_turn + 1), desc='  Phase 1 (hybrid)'):
        new_qids = turn_to_qids.get(t, [])
        if not new_qids:
            continue
        think_text = turn_thinks.get(t, '') or ''
        anchor_curr = anchors.get(t)
        anchor_prev = anchors.get(t - 1) if t > 1 else None
        soft_fail_think, soft_fail_evidence = soft_fail_by_turn.get(t, (False, []))
        soft_fail_anchor_rejection = ''
        if strict_edge_policy() and soft_fail_think and anchor_prev and anchor_curr and (not soft_failure_anchor_is_compatible(qid_to_text.get(anchor_prev, ''), qid_to_text.get(anchor_curr, ''), soft_fail_evidence)):
            soft_fail_anchor_rejection = 'conflicting_year_branch_without_plural_scope'
            soft_fail_think = False
            print(f'    ℹ Turn {t}: rejected deterministic soft-failure anchor {anchor_prev}->{anchor_curr} ({soft_fail_anchor_rejection})')
        hard_fails_prev = hard_fail_by_turn.get(t, {})
        if hard_fails_prev:
            all_edges.extend(build_deterministic_hard_failure_edges_for_turn(t, new_qids, turn_to_qids, anchors, qid_to_text, hard_fails_prev, turn_contexts[t]['urls']))
        per_query_payload: Dict[str, Dict[str, Any]] = {}
        per_query_used_signals: Dict[str, Set[str]] = {}
        per_query_parent_coverage: Dict[str, Dict[str, Set[str]]] = {}
        prior_qs_turn = [q for q in queries if q['turn'] < t]
        prior_qid_to_text_turn = {q['id']: q['text'] for q in prior_qs_turn}
        for qid in new_qids:
            q_text = qid_to_text.get(qid, '')
            q_turn = qid_to_turn.get(qid, t)
            q_tokens = tokenize(q_text)
            prior_qs = prior_qs_turn
            prior_qid_to_text = prior_qid_to_text_turn
            token_status = compute_token_statuses(q_tokens=q_tokens, q0_token_set=q0_token_set, prior_queries=prior_qs)
            think_support: Dict[str, List[Dict[str, Any]]] = {}
            unsupported_new: List[str] = []
            unsupported_q0_seen: List[str] = []
            seen_latest_qid: Dict[str, Optional[str]] = {}
            seen_first_qid: Dict[str, Optional[str]] = {}
            for tok, st in token_status.items():
                if st == 'Q0_ONLY':
                    continue
                hits = sentence_hits_in_text(tok, think_text)
                if hits:
                    think_support[tok] = hits
                if st == 'NEW' and (not hits):
                    unsupported_new.append(tok)
                if st in ('SEEN', 'Q0_SEEN'):
                    lq = latest_prior_query_with_token(tok, prior_qs)
                    fq = first_prior_query_with_token(tok, prior_qs)
                    seen_latest_qid[tok] = lq
                    seen_first_qid[tok] = fq
            candidates_by_token: Dict[str, List[Dict[str, Any]]] = {}
            for tok, st in token_status.items():
                if st == 'Q0_ONLY':
                    continue
                raw_hits = token_provenance_in_prior_docs(token=tok, prior_qs=prior_qs, qid_doc=turn_contexts[t]['docs'], sent_window=sent_window, max_hits_total=max(80, top_k * max_snips * 10))
                top_sources = select_top_k_sources_from_hits(hits=raw_hits, top_k_sources=top_k, max_hits_per_source=max_snips)
                for s in top_sources:
                    s['source_query'] = prior_qid_to_text.get(s['source_qid'], '')
                candidates_by_token[tok] = top_sources
            candidates_by_token = denoise_provenance(candidates_by_token, min_tokens_per_source=2, rare_doc_freq=2)
            for tok, st in token_status.items():
                if st != 'Q0_SEEN':
                    continue
                has_think = bool(think_support.get(tok))
                has_prov = bool(candidates_by_token.get(tok))
                if not has_think and (not has_prov):
                    unsupported_q0_seen.append(tok)
            used_signals = compute_used_signals(token_status=token_status, think_support=think_support, candidates_by_token=candidates_by_token)
            parent_coverage = compute_parent_coverage(used_signals=used_signals, candidates_by_token=candidates_by_token, think_support=think_support, token_status=token_status)
            per_query_used_signals[qid] = used_signals
            per_query_parent_coverage[qid] = parent_coverage
            per_query_payload[qid] = {'token_status': token_status, 'unsupported_new_tokens': unsupported_new, 'unsupported_q0_seen_tokens': unsupported_q0_seen, 'think_support': think_support, 'seen_latest_qid': seen_latest_qid, 'seen_first_qid': seen_first_qid, 'candidates_by_token': candidates_by_token, 'used_signals': used_signals}
        soft_fail_detected = False
        prompt = build_user_prompt_hybrid(t=t, question=question, qid_to_text=qid_to_text, turn_to_qids=turn_to_qids, result_counts=turn_contexts[t]['result_counts'], current_think=think_text, new_qids=new_qids, per_query_payload=per_query_payload, anchor_prev=anchor_prev, anchor_curr=anchor_curr, soft_fail_prev=soft_fail_think, soft_fail_evidence=soft_fail_evidence if t > 1 else None, hard_fails_prev=hard_fails_prev if t > 1 else None)
        result = call_llm_json(settings.PHASE1_SYSTEM_HYBRID, prompt, max_tokens=3200, call_name='query_edge_attribution')
        from ..safety import AttributionUnavailable
        if not isinstance(result, dict) or not isinstance(result.get('edges'), list) or not isinstance(result.get('no_source_found'), list):
            raise AttributionUnavailable('Missing or invalid query-attribution result; no graph was finalized')
        if any(not isinstance(edge, dict) for edge in result['edges']) or any(not isinstance(qid, str) for qid in result['no_source_found']):
            raise AttributionUnavailable('Invalid query-attribution schema')
        raw_edges = []
        no_src = []
        if result:
            raw_edges = result.get('edges', [])
            no_src = result.get('no_source_found', [])
        mpsc_debug = {}
        for qid in new_qids:
            us = per_query_used_signals.get(qid, set())
            pc = per_query_parent_coverage.get(qid, {})
            mpsc_debug[qid] = {'used_signals': sorted(us), 'parent_coverage': {p: sorted(s) for p, s in pc.items()}}
        per_turn_raw.append({'turn': t, 'soft_fail_anchor_rejection': soft_fail_anchor_rejection, 'raw_edges': raw_edges, 'no_source_found': no_src, 'mpsc_debug': mpsc_debug, 'prompt': prompt, 'per_query_payload': {qid: {'token_status': per_query_payload.get(qid, {}).get('token_status', {}), 'unsupported_new_tokens': per_query_payload.get(qid, {}).get('unsupported_new_tokens', []), 'unsupported_q0_seen_tokens': per_query_payload.get(qid, {}).get('unsupported_q0_seen_tokens', []), 'think_support': per_query_payload.get(qid, {}).get('think_support', {}), 'seen_latest_qid': per_query_payload.get(qid, {}).get('seen_latest_qid', {}), 'seen_first_qid': per_query_payload.get(qid, {}).get('seen_first_qid', {}), 'candidates_by_token': per_query_payload.get(qid, {}).get('candidates_by_token', {}), 'used_signals': sorted(per_query_payload.get(qid, {}).get('used_signals', []))} for qid in new_qids}, 'accepted_edges': [], 'rejected_edges': []})
        valid_ns: List[str] = []
        if no_src:
            valid_ns = [qid for qid in no_src if qid in set(new_qids)]
            explicitly_no_source.extend(valid_ns)
            if valid_ns:
                print(f'    ℹ Turn {t}: LLM no_source_found = {valid_ns}')
        valid_ns_set = set(valid_ns)
        valid_sources = {'Prior_knowledge'}
        for prev_t in range(1, t):
            for pq in turn_to_qids.get(prev_t, []):
                valid_sources.add(pq)
        valid_targets = set(new_qids)
        llm_edges_by_target: Dict[str, List[Dict]] = defaultdict(list)
        for e in raw_edges:
            src = e.get('source', '')
            tgt = e.get('target', '')
            if src == 'Q0':
                continue
            if src not in valid_sources or tgt not in valid_targets:
                continue
            edge_kind = e.get('edge_kind', 'evidence_derived')
            if edge_kind not in settings.EDGE_KINDS:
                edge_kind = 'evidence_derived'
            if src == 'Prior_knowledge':
                edge_kind = 'prior_knowledge_derived'
            if strict_edge_policy():
                payload = per_query_payload.get(tgt, {})
                actual_coverage = set(per_query_parent_coverage.get(tgt, {}).get(src, set()))
                actual_coverage &= set(per_query_used_signals.get(tgt, set()))
                rejection_reason = ''
                evidence_text = str(e.get('evidence', ''))
                has_sentence_anchor = bool(re.search('\\[s\\d+\\]', evidence_text, re.I))
                has_provenance_anchor = bool(re.search('\\[(?:visit|snippet)\\]', evidence_text, re.I))
                if edge_kind == 'failure_derived':
                    if not soft_fail_think:
                        rejection_reason = 'no_explicit_soft_failure_in_current_think'
                    elif not has_sentence_anchor:
                        rejection_reason = 'soft_failure_missing_sentence_anchor'
                elif src == 'Prior_knowledge':
                    statuses = payload.get('token_status', {})
                    think_support = payload.get('think_support', {})
                    candidates = payload.get('candidates_by_token', {})
                    actual_coverage = {sig for sig in actual_coverage if statuses.get(sig) == 'NEW' and bool(think_support.get(sig)) and (not bool(candidates.get(sig)))}
                    if not actual_coverage:
                        rejection_reason = 'pk_signal_not_new_think_anchored_and_provenance_free'
                    elif not has_sentence_anchor:
                        rejection_reason = 'pk_edge_missing_think_sentence_anchor'
                elif edge_kind != 'evidence_derived':
                    rejection_reason = 'query_source_has_incompatible_edge_kind'
                elif not actual_coverage:
                    rejection_reason = 'source_has_no_admissible_provenance_coverage'
                elif not re.search(f'\\b{re.escape(src)}\\b', evidence_text, re.I):
                    rejection_reason = 'query_edge_evidence_does_not_name_source'
                elif not (has_sentence_anchor or has_provenance_anchor):
                    normalized = normalize_provenance_citations(evidence_text, src, payload, actual_coverage)
                    if normalized == evidence_text:
                        rejection_reason = 'query_edge_missing_sentence_or_provenance_anchor'
                    else:
                        e = dict(e, evidence=normalized)
                        per_turn_raw[-1].setdefault('citation_format_repairs', []).append({
                            'source': src, 'target': tgt,
                            'original_evidence': evidence_text, 'normalized_evidence': normalized,
                        })
                if rejection_reason:
                    per_turn_raw[-1]['rejected_edges'].append({'source': src, 'target': tgt, 'edge_kind': edge_kind, 'reason': rejection_reason, 'raw_edge': e})
                    continue
                if edge_kind != 'failure_derived':
                    e = dict(e)
                    e['covered_signals'] = sorted(actual_coverage)
            conf = e.get('confidence', 'high')
            if conf not in ('high', 'uncertain'):
                conf = 'uncertain'
            edge_dict = {'source': src, 'target': tgt, 'type': 'lead_to', 'edge_kind': edge_kind, 'evidence': e.get('evidence', ''), 'covered_signals': e.get('covered_signals', []), 'confidence': conf, 'phase': 'phase1_llm', 'metadata': {'edge_policy': {'policy': settings.EDGE_POLICY_VERSION, 'gate': 'validated_llm_local_attribution', 'reason': e.get('validation_reason', '')}} if strict_edge_policy() else {}}
            if edge_kind == 'failure_derived':
                fs = e.get('failure_subtype', 'soft')
                edge_dict['failure_subtype'] = fs if fs in ('soft', 'hard') else 'soft'
            llm_edges_by_target[tgt].append(edge_dict)
        accepted = 0
        accepted_support_edges = 0
        silent_miss_candidates: List[str] = []
        silent_miss_recovered: List[str] = []
        silent_miss_unresolved: List[str] = []
        for qid in new_qids:
            target_llm_edges = llm_edges_by_target.get(qid, [])
            used_sigs = per_query_used_signals.get(qid, set())
            parent_cov = per_query_parent_coverage.get(qid, {})
            unsupported_q0_seen = per_query_payload.get(qid, {}).get('unsupported_q0_seen_tokens', [])
            payload = per_query_payload.get(qid, {})
            maybe_silent_miss = not target_llm_edges and qid not in valid_ns_set and bool(parent_cov or used_sigs or payload.get('candidates_by_token') or payload.get('think_support'))
            if maybe_silent_miss:
                silent_miss_candidates.append(qid)
            filtered = []
            if target_llm_edges or parent_cov:
                filtered = apply_mpsc(candidate_edges=target_llm_edges, target_qid=qid, parent_coverage=parent_cov, used_signals=used_sigs, qid_to_turn=qid_to_turn)
            for edge in filtered:
                all_edges.append(edge)
                accepted += 1
                if edge.get('edge_kind') in {'evidence_derived', 'prior_knowledge_derived'}:
                    accepted_support_edges += 1
                per_turn_raw[-1]['accepted_edges'].append(edge)
            if maybe_silent_miss:
                if filtered:
                    silent_miss_recovered.append(qid)
                else:
                    silent_miss_unresolved.append(qid)
            if unsupported_q0_seen:
                all_edges.append({'source': 'Q0', 'target': qid, 'type': 'lead_to', 'edge_kind': 'constraint_use', 'evidence': f"Q0 fallback for unsupported Q0_SEEN tokens: {', '.join(sorted(set(unsupported_q0_seen)))}", 'confidence': 'high', 'phase': 'deterministic_q0_seen_fallback', 'metadata': {'fallback_q0_seen_tokens': sorted(set(unsupported_q0_seen)), 'reason': 'Q0_SEEN token had no think support and no provenance support'}})
            print(f'    ✦ Q0 fallback {qid}: unsupported Q0_SEEN tokens = {sorted(set(unsupported_q0_seen))}')
        if t > 1:
            soft_fail_detected = bool(soft_fail_think)
            if (soft_fail_detected and anchor_prev and anchor_curr
                    and len(turn_to_qids.get(t - 1, [])) == 1 and len(new_qids) == 1):
                sf_ev_text = '; '.join(soft_fail_evidence[:3])
                if len(sf_ev_text) > 300:
                    sf_ev_text = sf_ev_text[:300] + '…'
                progress_note = f'This turn also produced {accepted_support_edges} support edge(s). ' if accepted_support_edges > 0 else 'No support edges were produced in this turn. '
                if strict_edge_policy():
                    all_edges[:] = [edge for edge in all_edges if not (edge.get('source') == anchor_prev and edge.get('target') == anchor_curr and (edge.get('edge_kind') == 'evidence_derived'))]
                    per_turn_raw[-1]['accepted_edges'] = [edge for edge in per_turn_raw[-1].get('accepted_edges', []) if not (edge.get('source') == anchor_prev and edge.get('target') == anchor_curr and (edge.get('edge_kind') == 'evidence_derived'))]
                all_edges.append({'source': anchor_prev, 'target': anchor_curr, 'type': 'lead_to', 'edge_kind': 'failure_derived', 'failure_subtype': 'soft', 'evidence': f'Soft failure: think block indicates unresolved insufficiency. {progress_note}{sf_ev_text}', 'confidence': 'high', 'phase': 'deterministic_soft_fail', 'metadata': {'failure_reason': 'insufficiency', 'scope': 'turn-level', 'criteria': 'think_declared_insufficiency', 'accepted_support_edges': accepted_support_edges, 'missing_fields': sf_ev_text, 'edge_policy': {'policy': settings.EDGE_POLICY_VERSION, 'gate': 'explicit_soft_failure_detector', 'pair_precedence': 'failure_over_evidence'} if strict_edge_policy() else None}})
        if accepted == 0 and (not no_src):
            print(f'    ⚠ Turn {t}: LLM returned {len(raw_edges)} edges, 0 accepted (after validation + MPSC).')
        elif accepted < len(raw_edges):
            print(f'    ℹ Turn {t}: MPSC reduced {len(raw_edges)} LLM edges to {accepted}')
        if silent_miss_recovered:
            per_turn_raw[-1]['silent_miss_recovered'] = silent_miss_recovered
            print(f'    ℹ Turn {t}: silent-miss fallback recovered {silent_miss_recovered}')
        if silent_miss_unresolved:
            per_turn_raw[-1]['silent_miss_unresolved'] = silent_miss_unresolved
            print(f'    ⚠ Turn {t}: unresolved silent miss {silent_miss_unresolved}')
    return (all_edges, per_turn_raw, explicitly_no_source)
