"""Decompose the agent answer and identify retrieved support."""
from __future__ import annotations
import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from . import settings

def extract_final_answer(messages: List[Dict], task: Dict) -> str:
    """Read the agent response, never the benchmark's answer/gold field."""
    last = next((m.get('content') or '' for m in reversed(messages)
                 if m.get('role') == 'assistant'), '')
    match = re.search('<answer>(.*?)</answer>', last, re.DOTALL)
    if match:
        return match.group(1).strip()
    prediction = task.get('prediction')
    if isinstance(prediction, str) and prediction.strip():
        return prediction.strip()
    # An untagged terminal response is valid, but an unfinished tool action or
    # a think-only message is not a final answer.
    if '<tool_call>' in last or '<use_mcp_tool>' in last:
        return ''
    return re.sub('<think>.*?</think>', '', last, flags=re.DOTALL).strip()

def extract_final_answer_support_text(messages: List[Dict], answer_text: str) -> str:
    """
    Extract supporting analysis for q→Answer matching.
    Prefer the final assistant message's <think> block(s). If unavailable,
    fall back to the same message with <answer> removed.
    """
    if not messages:
        return ''
    answer_text = (answer_text or '').strip()
    for msg in reversed(messages):
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content', '') or ''
        if answer_text and answer_text not in content and ('<answer>' not in content):
            continue
        thinks = [m.group(1).strip() for m in re.finditer('<think>(.*?)</think>', content, re.DOTALL) if m.group(1).strip()]
        if thinks:
            return '\n---\n'.join(thinks)
        content_wo_answer = re.sub('<answer>.*?</answer>', '', content, flags=re.DOTALL).strip()
        if content_wo_answer:
            return content_wo_answer
    return ''

def extract_minimal_answer(question: str, answer_text: str) -> str:
    """
    Trim a long final answer down to the minimal directly responsive answer.

    This is used only to reduce answer-unit decomposition / support matching
    cost. The original answer text is preserved in the final output graph.
    """
    from .attribution import call_llm_json
    answer_text = (answer_text or '').strip()
    question = (question or '').strip()
    if not answer_text:
        return ''
    if len(answer_text) <= 280:
        return answer_text
    prompt = f'Question:\n{question}\n\nFinal Answer:\n{answer_text}\n\nExtract the minimal direct answer (JSON only).'
    result = call_llm_json(settings.MINIMAL_ANSWER_EXTRACT_SYSTEM, prompt, max_tokens=600, retries=2, call_name='minimal_answer_extract')
    if result and isinstance(result.get('minimal_answer'), str):
        minimal = result['minimal_answer'].strip()
        if minimal and _answer_claim_is_anchored(minimal, answer_text):
            print(f'    Minimal answer extracted: {len(answer_text)} -> {len(minimal)} chars')
            return minimal
    return answer_text

def _answer_claim_is_anchored(claim: str, answer_text: str) -> bool:
    claim = (claim or '').strip()
    answer_text = (answer_text or '').strip()
    if not claim or not answer_text:
        return False
    if claim.lower() in answer_text.lower():
        return True
    claim_norm = _answer_signal_norm(claim)
    answer_norm = _answer_signal_norm(answer_text)
    return bool(claim_norm and answer_norm and (claim_norm in answer_norm))

def is_abstention_answer(answer_text: str) -> bool:
    """Return true only for a complete, explicit no-answer response."""
    normalized = re.sub('[*_`#]', '', answer_text or '').strip().lower()
    return bool(re.fullmatch('(?:no answer(?: was)? found|no answer|unable to (?:find|determine|identify)(?: the answer)?|could not (?:find|determine|identify)(?: the answer)?|insufficient information)(?:[.!])?', normalized))

def _infer_answer_unit_type(claim: str) -> str:
    claim = (claim or '').strip()
    if not claim:
        return 'attribute'
    if re.fullmatch('(?:1[0-9]{3}|20[0-9]{2})', claim):
        return 'date'
    if re.fullmatch('[0-9]+(?:\\.[0-9]+)?', claim):
        return 'number'
    if re.search('\\d', claim):
        return 'number'
    return 'entity_name'

def _is_low_information_answer_unit(claim: str) -> bool:
    claim = (claim or '').strip()
    if not claim:
        return True
    norm = _answer_signal_norm(claim)
    if not norm:
        return True
    if norm in settings.LOW_INFORMATION_ANSWER_UNITS:
        return True
    words = [w.lower() for w in _ordered_answer_words(claim)]
    if not words:
        return True
    if len(words) == 1 and (words[0] in settings.LOW_INFORMATION_ANSWER_UNITS or words[0] in settings.STOPWORDS):
        return True
    if all((w in settings.STOPWORDS or w in settings.LOW_INFORMATION_ANSWER_UNITS for w in words)):
        return True
    return False

def _deterministic_answer_units(answer_text: str) -> List[Dict[str, str]]:
    """
    Deterministically extract stable answer units from the minimal answer text.
    These units must be literal spans from the answer text itself.
    """
    answer_text = (answer_text or '').strip()
    if not answer_text:
        return []
    units: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add_unit(claim: str, unit_type: Optional[str]=None) -> None:
        claim = (claim or '').strip().strip(',.;:()[]{}')
        if not claim:
            return
        if _is_low_information_answer_unit(claim):
            return
        if not _answer_claim_is_anchored(claim, answer_text):
            return
        norm = _answer_signal_norm(claim)
        if not norm or norm in seen:
            return
        seen.add(norm)
        units.append({'unit_id': f'a{len(units) + 1}', 'claim': claim, 'unit_type': unit_type or _infer_answer_unit_type(claim)})
    for m in re.finditer('"([^"]+)"|“([^”]+)”|\\\'([^\\\']+)\\\'', answer_text):
        quoted = next((g for g in m.groups() if g), '')
        add_unit(quoted, 'entity_name')
    for m in re.finditer('\\b(?:1[0-9]{3}|20[0-9]{2})\\b', answer_text):
        add_unit(m.group(0), 'date')
    for m in re.finditer('\\b[A-Z]{2,}(?:-[A-Z0-9]{2,})*\\b', answer_text):
        add_unit(m.group(0), 'entity_name')
    title_pat = re.compile("\\b(?:[A-Z][A-Za-z0-9'&.-]*)(?:\\s+(?:[A-Z][A-Za-z0-9'&.-]*|of|the|and|for|in|on|de|la)){0,4}\\b")
    for m in title_pat.finditer(answer_text):
        phrase = m.group(0).strip()
        if len(_ordered_answer_words(phrase)) < 2 and (not re.fullmatch("[A-Z][A-Za-z0-9'&.-]*", phrase)):
            continue
        add_unit(phrase, 'entity_name')
    if not units:
        add_unit(answer_text, 'description')
    return units

def _stabilize_answer_units(answer_text: str, llm_units: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Keep only LLM units that are literal / normalized spans from the answer text,
    then backfill missing stable units deterministically from the answer text.
    """
    stable: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for u in llm_units:
        claim = (u.get('claim') or '').strip()
        if not claim or not _answer_claim_is_anchored(claim, answer_text):
            continue
        if _is_low_information_answer_unit(claim):
            continue
        norm = _answer_signal_norm(claim)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        stable.append({'unit_id': f'a{len(stable) + 1}', 'claim': claim, 'unit_type': u.get('unit_type') or _infer_answer_unit_type(claim)})
    for u in _deterministic_answer_units(answer_text):
        norm = _answer_signal_norm(u['claim'])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        stable.append({'unit_id': f'a{len(stable) + 1}', 'claim': u['claim'], 'unit_type': u['unit_type']})
    return stable

def decompose_answer_units(question: str, answer_text: str) -> List[Dict[str, str]]:
    """
    LLM call to decompose the final answer into atomic answer units.
    Returns [{unit_id, claim, unit_type}, ...].
    """
    from .attribution import call_llm_json
    from .llm import _answer_decompose_budget
    prompt = f'Question:\n{question}\n\nFinal Answer:\n{answer_text}\n\nDecompose the answer into atomic answer units (JSON only):'
    decompose_tokens = _answer_decompose_budget(answer_text)
    result = call_llm_json(settings.ANSWER_DECOMPOSE_SYSTEM, prompt, max_tokens=decompose_tokens, retries=2, call_name='answer_decompose')
    if result and 'answer_units' in result:
        units = result['answer_units']
        valid = []
        for u in units:
            if isinstance(u, dict) and u.get('unit_id') and u.get('claim'):
                u.setdefault('unit_type', 'attribute')
                valid.append(u)
        stable = _stabilize_answer_units(answer_text, valid)
        if stable:
            print(f'    Answer decomposition: {len(stable)} units')
            for u in stable:
                print(f'''      {u['unit_id']} ({u['unit_type']}): "{u['claim']}"''')
            return stable
    stable = _deterministic_answer_units(answer_text)
    print('    ⚠ Answer decomposition LLM failed or produced unanchored claims, using stable text-anchored units')
    print(f'    Answer decomposition: {len(stable)} units')
    for u in stable:
        print(f'''      {u['unit_id']} ({u['unit_type']}): "{u['claim']}"''')
    return stable

def _best_claim_window(doc: str, claim: str, sent_window: int=3) -> Tuple[str, str, float]:
    """
    Pick the most relevant sentence window in a query document for a claim.
    Returns (hit_sentence, window_text, score).
    """
    from .evidence import split_sentences
    from .parsing import tokenize_set
    doc = doc or ''
    if not doc:
        return ('', '', 0.0)
    claim_lower = (claim or '').lower().strip()
    claim_norm = _answer_signal_norm(claim)
    claim_toks = tokenize_set(claim)
    sents = split_sentences(doc)
    if not sents:
        return ('', doc[:600], 0.0)
    best = ('', '', 0.0)
    for si, sent in enumerate(sents):
        sent_lower = sent.lower()
        sent_norm = _answer_signal_norm(sent)
        score = 0.0
        if claim_lower and claim_lower in sent_lower or (claim_norm and claim_norm in sent_norm):
            score += 10.0
        if claim_toks:
            sent_toks = tokenize_set(sent)
            overlap = claim_toks & sent_toks
            score += len(overlap)
            if claim_toks:
                score += len(overlap) / max(len(claim_toks), 1)
        if score > best[2]:
            lo = max(0, si - sent_window)
            hi = min(len(sents) - 1, si + sent_window)
            best = (sent.strip(), ' '.join(sents[lo:hi + 1]).strip(), score)
    if best[2] > 0:
        return best
    return ('', ' '.join(sents[:min(len(sents), sent_window * 2 + 1)]).strip(), 0.0)

def _llm_answer_support_verdicts(question: str, answer_support_text: str, candidate_pairs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Batch LLM verdicts for candidate answer-unit ↔ query-evidence pairs.
    Returns {pair_id: {supports, confidence, reason, evidence_sentence}}.
    """
    from .attribution import call_llm_json
    from ..safety import AttributionUnavailable
    if not candidate_pairs:
        return {}
    parts = [f'## Question\n{question[:800]}']
    if answer_support_text:
        parts.append(f'## Answer Analysis\n{answer_support_text[:1800]}')
    parts.append('## Candidate Pairs')
    for pair in candidate_pairs:
        parts.append(f"pair_id={pair['pair_id']}\nunit_id={pair['unit_id']} claim={json.dumps(pair['claim'], ensure_ascii=False)}\nquery={pair['qid']} text={json.dumps(pair['query_text'], ensure_ascii=False)}\nevidence_excerpt={json.dumps(pair['window_text'][:1000], ensure_ascii=False)}\n")
    parts.append('Output JSON only.')
    result = call_llm_json(settings.ANSWER_SUPPORT_MATCH_SYSTEM, '\n\n'.join(parts), max_tokens=2200, retries=2, call_name='answer_support_match')
    verdict_map: Dict[str, Dict[str, Any]] = {}
    if result and isinstance(result.get('verdicts'), list):
        for item in result['verdicts']:
            if not isinstance(item, dict) or not item.get('pair_id'):
                continue
            if not isinstance(item.get('supports'), bool):
                raise AttributionUnavailable('Answer support verdict must contain a JSON boolean')
            if item['pair_id'] in verdict_map:
                raise AttributionUnavailable('Duplicate answer support verdict')
            verdict_map[item['pair_id']] = {'supports': item['supports'], 'confidence': item.get('confidence', 'uncertain') if item.get('confidence') in ('high', 'uncertain') else 'uncertain', 'reason': item.get('reason', ''), 'evidence_sentence': item.get('evidence_sentence', '')}
    expected = {pair['pair_id'] for pair in candidate_pairs}
    if set(verdict_map) != expected:
        raise AttributionUnavailable('Missing or invalid answer-support verdicts; no graph was finalized')
    return verdict_map

def _anchored_evidence_sentence(candidate: str, window_text: str, fallback: str) -> str:
    """Keep model-selected evidence only when it is literally present in the tool window."""
    candidate_norm = _answer_signal_norm(candidate)
    window_norm = _answer_signal_norm(window_text)
    if candidate_norm and window_norm and (candidate_norm in window_norm):
        return candidate.strip()
    return (fallback or '').strip()

def _query_parent_map(graph_edges: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    parent_map: Dict[str, Set[str]] = defaultdict(set)
    for edge in graph_edges or []:
        src = edge.get('source', '')
        tgt = edge.get('target', '')
        if not (src.startswith('q') and tgt.startswith('q')):
            continue
        parent_map[tgt].add(src)
    return parent_map

def _expand_query_ancestors(seed_qids: Set[str], graph_edges: List[Dict[str, Any]]) -> Set[str]:
    parent_map = _query_parent_map(graph_edges)
    seen = set(seed_qids)
    stack = list(seed_qids)
    while stack:
        qid = stack.pop()
        for parent in parent_map.get(qid, set()):
            if parent in seen:
                continue
            seen.add(parent)
            stack.append(parent)
    return seen

def _llm_answer_distributed_support_verdict(question: str, answer_support_text: str, answer_unit: Dict[str, str], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    from .attribution import call_llm_json
    if not candidates:
        return {'selected_qids': [], 'confidence': 'uncertain', 'reason': ''}
    lines = [f'Question:\n{question}', '', f"Answer unit:\n{answer_unit.get('unit_id', '')}: {answer_unit.get('claim', '')}", '']
    if answer_support_text:
        lines.append(f'Final answer reasoning context:\n{answer_support_text[:1800]}')
        lines.append('')
    lines.append('Candidate query evidence:')
    for cand in candidates:
        lines.append(f'''- {cand['qid']}: query="{cand['query_text']}"\n  snippet="{cand['window_text'][:500]}"\n  relation={cand['relation_to_anchor']}\n  lexical={cand['lexical_summary']}''')
    result = call_llm_json(settings.ANSWER_DISTRIBUTED_SUPPORT_SYSTEM, '\n'.join(lines), max_tokens=900, retries=2, call_name='answer_distributed_support')
    if not result:
        return {'selected_qids': [], 'confidence': 'uncertain', 'reason': ''}
    selected = result.get('selected_qids', [])
    if not isinstance(selected, list):
        selected = []
    valid_qids = {cand['qid'] for cand in candidates}
    selected = [qid for qid in selected if isinstance(qid, str) and qid in valid_qids]
    return {'selected_qids': selected, 'confidence': result.get('confidence', 'uncertain') if result.get('confidence') in {'high', 'uncertain'} else 'uncertain', 'reason': result.get('reason', '')}

def answer_provenance_matching(answer_units: List[Dict[str, str]], queries: List[Dict], qid_doc: Dict[str, str], question: str='', answer_support_text: str='', sent_window: int=3) -> Dict[str, Dict[str, Any]]:
    """
    For each answer unit, find which queries' tool outputs contain supporting evidence.

    Returns: {
      qid: {
        "supports_units": [unit_ids],
        "provenance": [{unit_id, claim, hit_sentence, window_text}, ...]
      }
    }
    """
    from .parsing import tokenize_set
    q_coverage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'supports_units': [], 'provenance': []})
    for au in answer_units:
        claim = au['claim'].strip()
        uid = au['unit_id']
        claim_lower = claim.lower()
        claim_toks = tokenize_set(claim)
        candidates: List[Dict[str, Any]] = []
        for q in queries:
            qid = q['id']
            doc = qid_doc.get(qid, '')
            if not doc:
                continue
            hit_sent, window_text, score = _best_claim_window(doc, claim, sent_window=sent_window)
            doc_lower = doc.lower()
            direct = bool(claim_lower and claim_lower in doc_lower)
            doc_toks = tokenize_set(doc)
            overlap = claim_toks & doc_toks if claim_toks else set()
            overlap_ratio = len(overlap) / max(len(claim_toks), 1) if claim_toks else 0.0
            prefilter_score = score
            if direct:
                prefilter_score += 50
            if overlap_ratio >= 0.7:
                prefilter_score += 20
            elif overlap_ratio >= 0.4:
                prefilter_score += 10
            if prefilter_score > 0:
                candidates.append({'qid': qid, 'query_text': q.get('text', ''), 'hit_sentence': hit_sent, 'window_text': window_text, 'prefilter_score': prefilter_score, 'rule_direct': direct, 'rule_overlap_ratio': overlap_ratio})
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x['prefilter_score'], x['qid']), reverse=True)
        shortlisted = candidates[:min(4, len(candidates))]
        pair_payload = []
        for idx, cand in enumerate(shortlisted):
            pair_payload.append({'pair_id': f"{uid}_{cand['qid']}_{idx}", 'unit_id': uid, 'claim': claim, 'qid': cand['qid'], 'query_text': cand['query_text'], 'window_text': cand['window_text'], 'hit_sentence': cand['hit_sentence'], 'rule_direct': cand['rule_direct'], 'rule_overlap_ratio': cand['rule_overlap_ratio']})
        verdicts = _llm_answer_support_verdicts(question, answer_support_text, pair_payload)
        for pair in pair_payload:
            verdict = verdicts.get(pair['pair_id'])
            supports = False
            confidence = 'uncertain'
            reason = ''
            evidence_sentence = pair['hit_sentence']
            if verdict:
                supports = bool(verdict.get('supports', False))
                confidence = verdict.get('confidence', 'uncertain')
                reason = verdict.get('reason', '')
                evidence_sentence = verdict.get('evidence_sentence') or pair['hit_sentence']
            else:
                supports = bool(pair['rule_direct'] or pair['rule_overlap_ratio'] >= 0.85)
                confidence = 'uncertain'
                reason = 'LLM unavailable; falling back to strong lexical match'
            if not supports:
                continue
            entry = q_coverage[pair['qid']]
            if uid not in entry['supports_units']:
                entry['supports_units'].append(uid)
                entry['provenance'].append({'unit_id': uid, 'claim': claim, 'hit_sentence': (evidence_sentence or '')[:300], 'window_text': (pair['window_text'] or '')[:600], 'match_reason': reason, 'match_confidence': confidence})
    return dict(q_coverage)

def augment_answer_provenance_with_distributed_support(answer_units: List[Dict[str, str]], q_coverage: Dict[str, Dict[str, Any]], queries: List[Dict], qid_doc: Dict[str, str], question: str='', answer_support_text: str='', graph_edges: Optional[List[Dict[str, Any]]]=None) -> Dict[str, Dict[str, Any]]:
    """
    Add low-confidence distributed query support for answer units that are not
    directly supported by any single query document.
    """
    from .evidence import tool_evidence_only
    from .llm import strict_edge_policy
    from .parsing import tokenize_set
    if not answer_units:
        return dict(q_coverage)
    covered_units = {uid for info in q_coverage.values() for uid in info.get('supports_units', [])}
    unsupported_units = [au for au in answer_units if au['unit_id'] not in covered_units]
    if not unsupported_units:
        return dict(q_coverage)
    coverage = defaultdict(lambda: {'supports_units': [], 'provenance': []})
    for qid, info in q_coverage.items():
        coverage[qid]['supports_units'] = list(info.get('supports_units', []))
        coverage[qid]['provenance'] = list(info.get('provenance', []))
    anchor_qids = set(q_coverage.keys())
    if anchor_qids:
        anchor_scope = _expand_query_ancestors(anchor_qids, graph_edges or [])
    else:
        recent_queries = sorted(queries, key=lambda q: (q.get('turn', 0), int(q['id'][1:]) if q.get('id', '').startswith('q') and q['id'][1:].isdigit() else 0), reverse=True)
        anchor_scope = {q['id'] for q in recent_queries[:6]}
    answer_support_toks = tokenize_set(answer_support_text)
    recent_rank = {q['id']: idx for idx, q in enumerate(sorted(queries, key=lambda q: (q.get('turn', 0), int(q['id'][1:]) if q.get('id', '').startswith('q') and q['id'][1:].isdigit() else 0), reverse=True))}
    for au in unsupported_units:
        claim = au['claim'].strip()
        uid = au['unit_id']
        claim_lower = claim.lower()
        claim_toks = tokenize_set(claim)
        candidates: List[Dict[str, Any]] = []
        for q in queries:
            qid = q['id']
            qtext = q.get('text', '') or ''
            raw_doc = qid_doc.get(qid, '') or ''
            doc = tool_evidence_only(raw_doc) if strict_edge_policy() else raw_doc
            qtext_lower = qtext.lower()
            qtext_toks = tokenize_set(qtext)
            doc_toks = tokenize_set(doc)
            query_text_direct = bool(claim_lower and claim_lower in qtext_lower or _answer_normalized_contains(qtext, claim))
            query_text_overlap = len(claim_toks & qtext_toks) / max(len(claim_toks), 1) if claim_toks else 0.0
            doc_overlap = len(claim_toks & doc_toks) / max(len(claim_toks), 1) if claim_toks else 0.0
            support_overlap = len((qtext_toks | doc_toks) & answer_support_toks) if answer_support_toks else 0
            anchor_bonus = 8 if qid in anchor_scope else 0
            recency_bonus = max(0, 6 - recent_rank.get(qid, 99))
            score = 0.0
            if query_text_direct:
                score += 30
            if query_text_overlap >= 0.6:
                score += 18
            elif query_text_overlap >= 0.35:
                score += 10
            if doc_overlap >= 0.5:
                score += 14
            elif doc_overlap >= 0.25:
                score += 8
            if support_overlap >= 6:
                score += 10
            elif support_overlap >= 3:
                score += 5
            score += anchor_bonus + recency_bonus
            if score <= 0:
                continue
            hit_sent, window_text, _ = _best_claim_window(doc, claim, sent_window=3)
            if not window_text:
                window_text = qtext or doc[:600]
            relation = 'anchor_scope' if qid in anchor_scope else 'recent'
            candidates.append({'qid': qid, 'query_text': qtext, 'window_text': window_text, 'hit_sentence': hit_sent, 'score': score, 'relation_to_anchor': relation, 'lexical_summary': f'query_text_direct={query_text_direct}, query_text_overlap={query_text_overlap:.2f}, doc_overlap={doc_overlap:.2f}, answer_support_overlap={support_overlap}'})
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x['score'], x['qid']), reverse=True)
        shortlisted = candidates[:min(6, len(candidates))]
        verdict = _llm_answer_distributed_support_verdict(question=question, answer_support_text=answer_support_text, answer_unit=au, candidates=shortlisted)
        selected_qids = verdict.get('selected_qids', [])
        if not selected_qids:
            continue
        selected_lookup = {cand['qid']: cand for cand in shortlisted if cand['qid'] in selected_qids}
        for qid in selected_qids:
            cand = selected_lookup.get(qid)
            if not cand:
                continue
            entry = coverage[qid]
            if uid in entry['supports_units']:
                continue
            entry['supports_units'].append(uid)
            entry['provenance'].append({'unit_id': uid, 'claim': claim, 'hit_sentence': (cand.get('hit_sentence') or '')[:300], 'window_text': (cand.get('window_text') or '')[:600], 'match_reason': verdict.get('reason', '') or 'Distributed multi-query answer support', 'match_confidence': verdict.get('confidence', 'uncertain'), 'support_mode': 'distributed'})
    return dict(coverage)

def _ordered_answer_words(text: str) -> List[str]:
    text = unicodedata.normalize('NFKC', text or '')
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace('—', '-').replace('–', '-')
    return re.findall('[A-Za-z0-9]+', text)

def _answer_signal_norm(text: str) -> str:
    words = _ordered_answer_words(text)
    if not words:
        text = unicodedata.normalize('NFKC', text or '').strip().lower()
        return re.sub('\\s+', ' ', text)
    return ' '.join((w.lower() for w in words))

def _answer_normalized_contains(haystack: str, needle: str) -> bool:
    hay_norm = _answer_signal_norm(haystack)
    needle_norm = _answer_signal_norm(needle)
    if not hay_norm or not needle_norm:
        return False
    return needle_norm in hay_norm

def build_answer_signals(answer_text: str, answer_units: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Build multi-granularity answer signals:
      - whole answer phrase
      - stable unit claims
      - high-value tokens from those unit claims
    """
    signal_map: Dict[str, Dict[str, Any]] = {}

    def add_signal(text: str, granularity: str, unit_id: Optional[str]=None) -> None:
        text = (text or '').strip()
        if not text:
            return
        norm = _answer_signal_norm(text)
        if not norm:
            return
        if granularity == 'token':
            if len(norm) < 3 and (not any((ch.isdigit() for ch in norm))):
                return
        elif len(norm) < 4:
            return
        existing = signal_map.get(norm)
        if existing is None:
            signal_id = f'a_sig_{len(signal_map) + 1}'
            signal_map[norm] = {'signal_id': signal_id, 'text': text, 'granularity': granularity, 'unit_ids': [unit_id] if unit_id else []}
            return
        if unit_id and unit_id not in existing['unit_ids']:
            existing['unit_ids'].append(unit_id)
        if settings.ANSWER_SIGNAL_GRANULARITY_RANK[granularity] < settings.ANSWER_SIGNAL_GRANULARITY_RANK[existing['granularity']]:
            existing['text'] = text
            existing['granularity'] = granularity
    answer_text = (answer_text or '').strip()
    if answer_text and len(answer_text) <= 160:
        add_signal(answer_text, 'whole_answer')
    for au in answer_units:
        uid = au.get('unit_id')
        claim = (au.get('claim') or '').strip()
        if not claim:
            continue
        add_signal(claim, 'unit_claim', uid)
        seen_tok = set()
        words = _ordered_answer_words(claim)
        for tok in words:
            if tok in seen_tok:
                continue
            seen_tok.add(tok)
            tok_lower = tok.lower()
            is_high_value = bool(re.fullmatch('(?:1[0-9]{3}|20[0-9]{2})', tok)) or bool(re.fullmatch('[0-9]+(?:\\.[0-9]+)?', tok)) or bool(re.fullmatch('[A-Z]{2,}(?:-[A-Z0-9]{2,})*', tok)) or (tok[:1].isupper() and tok_lower not in settings.STOPWORDS)
            if not is_high_value:
                continue
            add_signal(tok, 'token', uid)
    signals = list(signal_map.values())
    signals.sort(key=lambda s: (settings.ANSWER_SIGNAL_GRANULARITY_RANK.get(s['granularity'], 99), -len(s.get('text', '')), s['signal_id']))
    return signals

def has_direct_whole_answer_provenance(answer_signals: List[Dict[str, Any]], q_coverage: Dict[str, Dict[str, Any]]) -> bool:
    from .llm import strict_edge_policy
    whole_answer_signal_ids = {sig['signal_id'] for sig in answer_signals if sig.get('granularity') == 'whole_answer'}
    if not whole_answer_signal_ids:
        return False
    for info in q_coverage.values():
        for prov in info.get('provenance', []):
            if prov.get('signal_id') in whole_answer_signal_ids:
                if strict_edge_policy() and (not (prov.get('match_confidence') == 'high' and prov.get('document_support') is True)):
                    continue
                return True
    return False

def _is_answer_targeted_query_text(query_text: str) -> bool:
    qnorm = _answer_signal_norm(query_text)
    if not qnorm:
        return False
    for pat in settings.ANSWER_TARGETED_QUERY_PATTERNS:
        if '.*' in pat:
            if re.search(pat, qnorm):
                return True
        elif pat in qnorm:
            return True
    return False

def _is_high_value_unit_claim_signal(signal: Optional[Dict[str, Any]]) -> bool:
    if not signal:
        return False
    if signal.get('granularity') != 'unit_claim':
        return False
    text = (signal.get('text') or '').strip()
    words = [w for w in _ordered_answer_words(text) if w.lower() not in settings.STOPWORDS]
    if not words:
        return False
    if re.search('\\d', text):
        return True
    if len(words) >= 1 and any((w[:1].isupper() for w in words)):
        return True
    return len(' '.join(words)) >= 5

def _extract_urls_from_text(text: str) -> Set[str]:
    urls = set()
    for url in re.findall('https?://[^\\s\\])>\\"]+', text or ''):
        urls.add(url.rstrip('.,);]'))
    return urls

def _answer_support_duplicate_signature(info: Dict[str, Any]) -> Dict[str, Any]:
    supports = tuple(sorted(info.get('supports_signals', []) or info.get('supports_units', [])))
    urls: Set[str] = set()
    hit_sentences: Set[str] = set()
    window_snips: Set[str] = set()
    for prov in info.get('provenance', []):
        urls |= _extract_urls_from_text(prov.get('window_text', ''))
        hit = _answer_signal_norm(prov.get('hit_sentence', ''))
        if hit:
            hit_sentences.add(hit)
        elif prov.get('window_text'):
            window_snips.add(_answer_signal_norm((prov.get('window_text') or '')[:220]))
    return {'supports': supports, 'urls': tuple(sorted(urls)), 'hit_sentences': tuple(sorted(hit_sentences)), 'window_snips': tuple(sorted(window_snips))}

def _answer_supports_are_true_duplicates(info_a: Dict[str, Any], info_b: Dict[str, Any]) -> bool:
    sig_a = _answer_support_duplicate_signature(info_a)
    sig_b = _answer_support_duplicate_signature(info_b)
    if sig_a['supports'] != sig_b['supports']:
        return False
    if sig_a['urls'] and sig_b['urls'] and (sig_a['urls'] != sig_b['urls']):
        return False
    if sig_a['hit_sentences'] and sig_b['hit_sentences'] and (sig_a['hit_sentences'] != sig_b['hit_sentences']):
        return False
    if sig_a['urls'] and sig_b['urls']:
        return True
    if sig_a['hit_sentences'] and sig_b['hit_sentences']:
        return True
    return bool(sig_a['window_snips'] and sig_a['window_snips'] == sig_b['window_snips'])

def _answer_supports_same_cluster(info_a: Dict[str, Any], info_b: Dict[str, Any]) -> bool:
    sig_a = _answer_support_duplicate_signature(info_a)
    sig_b = _answer_support_duplicate_signature(info_b)
    if not set(sig_a['supports']) & set(sig_b['supports']):
        return False
    if set(sig_a['urls']) & set(sig_b['urls']):
        return True
    if set(sig_a['hit_sentences']) & set(sig_b['hit_sentences']):
        return True
    if set(sig_a['window_snips']) & set(sig_b['window_snips']):
        return True
    return False

def _build_query_ancestor_map(q_q_edges: Optional[List[Dict[str, Any]]]) -> Dict[str, Set[str]]:

    def _canon_qid(node_id: str) -> str:
        nid = (node_id or '').strip()
        if nid in {'A0', 'Answer'}:
            return 'Answer'
        return nid
    parents: Dict[str, Set[str]] = defaultdict(set)
    for e in q_q_edges or []:
        src = _canon_qid(e.get('source', e.get('src', '')))
        tgt = _canon_qid(e.get('target', e.get('tgt', '')))
        if not (src.startswith('q') and tgt.startswith('q')):
            continue
        parents[tgt].add(src)
    ancestor_map: Dict[str, Set[str]] = {}
    for qid in set(parents.keys()) | {p for ps in parents.values() for p in ps}:
        seen: Set[str] = set()
        stack = list(parents.get(qid, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(parents.get(cur, []))
        ancestor_map[qid] = seen
    return ancestor_map

def _answer_parent_preference_key(qid: str, info: Dict[str, Any], signal_by_id: Dict[str, Dict[str, Any]], qid_to_turn: Dict[str, int], qid_to_text: Dict[str, str]) -> Tuple[int, int, int, int, int, int]:
    prov = info.get('provenance', [])
    whole_hits = sum((1 for p in prov if p.get('granularity') == 'whole_answer' and p.get('support_mode') == 'direct'))
    unit_claim_hits = sum((1 for p in prov if p.get('granularity') == 'unit_claim' and p.get('support_mode') == 'direct' and _is_high_value_unit_claim_signal(signal_by_id.get(p.get('signal_id', '')))))
    targeted = 1 if _is_answer_targeted_query_text(qid_to_text.get(qid, '')) else 0
    signal_count = len(set(info.get('supports_signals', []) or info.get('supports_units', [])))
    high_conf_hits = sum((1 for p in prov if p.get('match_confidence') == 'high'))
    turn = qid_to_turn.get(qid, 0)
    qnum = int(qid[1:]) if qid.startswith('q') and qid[1:].isdigit() else -1
    return (whole_hits, unit_claim_hits, targeted, signal_count, high_conf_hits, turn * 1000 + qnum)

def answer_signal_provenance_matching(answer_signals: List[Dict[str, Any]], queries: List[Dict], qid_doc: Dict[str, str], question: str='', answer_support_text: str='', sent_window: int=3) -> Dict[str, Dict[str, Any]]:
    """
    Find provenance for multi-granularity answer signals. Queries jointly support
    the answer when their covered signal sets union to the answer signal set.
    """
    from .evidence import tool_evidence_only
    from .llm import strict_edge_policy
    from .parsing import tokenize_set
    q_coverage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'supports_signals': [], 'provenance': []})
    for sig in answer_signals:
        signal_text = (sig.get('text') or '').strip()
        signal_id = sig.get('signal_id', '')
        granularity = sig.get('granularity', 'unit_claim')
        if granularity == 'token':
            continue
        if not signal_text or not signal_id:
            continue
        signal_lower = signal_text.lower()
        signal_toks = tokenize_set(signal_text)
        candidates: List[Dict[str, Any]] = []
        for q in queries:
            qid = q['id']
            query_text = q.get('text', '') or ''
            query_lower = query_text.lower()
            query_toks = tokenize_set(query_text)
            raw_doc = qid_doc.get(qid, '') or ''
            doc = tool_evidence_only(raw_doc) if strict_edge_policy() else raw_doc
            doc_lower = doc.lower()
            doc_toks = tokenize_set(doc)
            direct_in_query = bool(signal_lower and signal_lower in query_lower or _answer_normalized_contains(query_text, signal_text))
            direct_in_doc = bool(signal_lower and signal_lower in doc_lower or _answer_normalized_contains(doc, signal_text))
            if granularity == 'token':
                direct_in_query = signal_lower in query_toks
                direct_in_doc = signal_lower in doc_toks
                prefilter_score = 0.0
                if direct_in_doc:
                    prefilter_score += 30.0
                elif direct_in_query and (not strict_edge_policy()):
                    prefilter_score += 8.0
                if not prefilter_score:
                    continue
                hit_sent = ''
                window_text = doc[:600] if direct_in_doc else query_text
            else:
                evidence_doc = doc if strict_edge_policy() else doc or query_text
                if not evidence_doc:
                    continue
                hit_sent, window_text, score = _best_claim_window(evidence_doc if strict_edge_policy() else doc or query_text, signal_text, sent_window=sent_window)
                overlap_doc = len(signal_toks & doc_toks) / max(len(signal_toks), 1) if signal_toks else 0.0
                overlap_query = len(signal_toks & query_toks) / max(len(signal_toks), 1) if signal_toks else 0.0
                prefilter_score = score
                if direct_in_doc:
                    prefilter_score += 35.0
                if direct_in_query and (not strict_edge_policy()):
                    prefilter_score += 18.0
                if overlap_doc >= 0.7:
                    prefilter_score += 15.0
                elif overlap_doc >= 0.4:
                    prefilter_score += 8.0
                if overlap_query >= 0.7 and (not strict_edge_policy()):
                    prefilter_score += 12.0
                elif overlap_query >= 0.4 and (not strict_edge_policy()):
                    prefilter_score += 6.0
                if granularity == 'whole_answer':
                    prefilter_score += 5.0
                elif granularity == 'unit_claim':
                    prefilter_score += 3.0
                if prefilter_score <= 0:
                    continue
            candidates.append({'qid': qid, 'query_text': query_text, 'hit_sentence': hit_sent, 'window_text': window_text or query_text[:600] or doc[:600], 'prefilter_score': prefilter_score, 'rule_direct': bool(direct_in_doc or (direct_in_query and (not strict_edge_policy()))), 'direct_in_doc': direct_in_doc, 'direct_in_query': direct_in_query, 'rule_overlap_ratio': max(len(signal_toks & doc_toks) / max(len(signal_toks), 1) if signal_toks else 0.0, len(signal_toks & query_toks) / max(len(signal_toks), 1) if signal_toks and (not strict_edge_policy()) else 0.0) if granularity != 'token' else 1.0 if direct_in_doc or (direct_in_query and (not strict_edge_policy())) else 0.0})
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x['prefilter_score'], x['qid']), reverse=True)
        shortlist_limit = 8 if strict_edge_policy() else 4
        shortlisted = candidates[:min(shortlist_limit, len(candidates))]
        if granularity == 'token':
            for cand in shortlisted:
                if not cand.get('direct_in_doc') and (strict_edge_policy() or not cand.get('direct_in_query')):
                    continue
                entry = q_coverage[cand['qid']]
                if signal_id in entry['supports_signals']:
                    continue
                entry['supports_signals'].append(signal_id)
                entry['provenance'].append({'signal_id': signal_id, 'signal_text': signal_text, 'granularity': granularity, 'unit_ids': sig.get('unit_ids', []), 'hit_sentence': (cand.get('hit_sentence') or '')[:300], 'window_text': (cand.get('window_text') or '')[:600], 'match_reason': 'Token-level direct document hit' if cand.get('direct_in_doc') else 'Token-level query lexical hint', 'match_confidence': 'uncertain' if not cand.get('direct_in_doc') else 'high', 'support_mode': 'direct', 'document_support': bool(cand.get('direct_in_doc'))})
            continue
        pair_payload = []
        for idx, cand in enumerate(shortlisted):
            pair_payload.append({'pair_id': f"{signal_id}_{cand['qid']}_{idx}", 'unit_id': signal_id, 'claim': signal_text, 'qid': cand['qid'], 'query_text': cand['query_text'], 'window_text': cand['window_text'], 'hit_sentence': cand['hit_sentence'], 'rule_direct': cand['rule_direct'], 'rule_overlap_ratio': cand['rule_overlap_ratio']})
        verdicts = _llm_answer_support_verdicts(question, answer_support_text, pair_payload)
        for pair in pair_payload:
            verdict = verdicts.get(pair['pair_id'])
            supports = False
            confidence = 'uncertain'
            reason = ''
            evidence_sentence = pair['hit_sentence']
            if verdict:
                supports = bool(verdict.get('supports', False))
                confidence = verdict.get('confidence', 'uncertain')
                reason = verdict.get('reason', '')
                evidence_sentence = verdict.get('evidence_sentence') or pair['hit_sentence']
            else:
                supports = bool(not strict_edge_policy() and (pair['rule_direct'] or pair['rule_overlap_ratio'] >= 0.85))
                confidence = 'uncertain'
                reason = 'LLM unavailable; falling back to strong lexical match'
            if not supports:
                continue
            if strict_edge_policy() and confidence != 'high':
                continue
            if not _answer_claim_is_anchored(evidence_sentence, pair['window_text']):
                # Do not replace fabricated citations with a convenient lexical hit.
                continue
            entry = q_coverage[pair['qid']]
            if signal_id in entry['supports_signals']:
                continue
            entry['supports_signals'].append(signal_id)
            entry['provenance'].append({'signal_id': signal_id, 'signal_text': signal_text, 'granularity': granularity, 'unit_ids': sig.get('unit_ids', []), 'hit_sentence': _anchored_evidence_sentence(evidence_sentence, pair['window_text'], pair['hit_sentence'])[:300], 'window_text': (pair['window_text'] or '')[:600], 'match_reason': reason, 'match_confidence': confidence, 'support_mode': 'direct', 'document_support': True})
    return dict(q_coverage)

def _validated_answer_support_signals(info: Dict[str, Any], signal_by_id: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Only verified factual claims support an answer; token hits are candidates."""
    return {
        prov['signal_id'] for prov in info.get('provenance', [])
        if prov.get('signal_id') in signal_by_id
        and prov.get('granularity') in {'whole_answer', 'unit_claim'}
        and prov.get('match_confidence') == 'high'
        and prov.get('document_support') is True
    }

def answer_mpsc(answer_signals: List[Dict[str, Any]], q_coverage: Dict[str, Dict[str, Any]], queries: Optional[List[Dict[str, Any]]]=None, q_q_edges: Optional[List[Dict[str, Any]]]=None) -> Tuple[List[Dict], List[str]]:
    """
    Answer-side parent selection:
      1) keep all strong support candidates
      2) add extra queries only if needed to cover supported signals
      3) apply light de-duplication rather than strict minimal set cover

    Returns:
      edges: list of q→A edge dicts
      unsupported_signals: list of answer signal_ids not covered by any query
    """
    from .llm import strict_edge_policy
    all_signal_ids = {sig['signal_id'] for sig in answer_signals}
    signal_by_id = {sig['signal_id']: sig for sig in answer_signals}
    qid_to_turn = {q['id']: q.get('turn', 0) for q in queries or []}
    qid_to_text = {q['id']: q.get('text', '') for q in queries or []}
    ancestor_map = _build_query_ancestor_map(q_q_edges)
    direct_whole_signal_ids: Set[str] = set()
    for qid, info in q_coverage.items():
        for p in info.get('provenance', []):
            is_direct_whole = p.get('granularity') == 'whole_answer' and p.get('support_mode') == 'direct'
            if strict_edge_policy():
                is_direct_whole = bool(is_direct_whole and p.get('match_confidence') == 'high' and (p.get('document_support') is True))
            if is_direct_whole:
                if p.get('signal_id'):
                    direct_whole_signal_ids.add(p['signal_id'])
    dominated_signal_ids: Set[str] = set()
    if direct_whole_signal_ids:
        whole_norms = {_answer_signal_norm(signal_by_id[sid].get('text', '')) for sid in direct_whole_signal_ids if sid in signal_by_id}
        for sig in answer_signals:
            sid = sig['signal_id']
            granularity = sig.get('granularity')
            if sid in direct_whole_signal_ids:
                continue
            if granularity == 'whole_answer':
                continue
            sig_norm = _answer_signal_norm(sig.get('text', ''))
            if not sig_norm:
                continue
            if any((sig_norm in whole_norm for whole_norm in whole_norms)):
                dominated_signal_ids.add(sid)
    cov_map: Dict[str, Set[str]] = {}
    for qid, info in q_coverage.items():
        if strict_edge_policy():
            su = _validated_answer_support_signals(info, signal_by_id)
            if su:
                cov_map[qid] = su
            continue
        if not any((p.get('match_confidence') == 'high' for p in info.get('provenance', []))):
            continue
        su = set(info.get('supports_signals', []) or info.get('supports_units', []))
        if su:
            cov_map[qid] = su
    coverable = set().union(*cov_map.values()) if cov_map else set()
    effective_coverable = coverable - dominated_signal_ids if direct_whole_signal_ids else coverable
    selected: List[str] = []
    covered: Set[str] = set()
    for qid, info in q_coverage.items():
        if qid not in cov_map:
            continue
        if strict_edge_policy():
            selected.append(qid)
            covered |= cov_map.get(qid, set())
            continue
        prov = info.get('provenance', [])
        has_direct_whole = any((p.get('granularity') == 'whole_answer' and p.get('support_mode') == 'direct' for p in prov))
        has_high_value_unit_claim = any((p.get('granularity') == 'unit_claim' and p.get('support_mode') == 'direct' and _is_high_value_unit_claim_signal(signal_by_id.get(p.get('signal_id', ''))) for p in prov))
        is_answer_targeted = _is_answer_targeted_query_text(qid_to_text.get(qid, ''))
        supported_signals = cov_map.get(qid, set())
        only_dominated_support = bool(direct_whole_signal_ids) and bool(supported_signals) and supported_signals.issubset(dominated_signal_ids)
        if has_direct_whole or ((has_high_value_unit_claim or is_answer_targeted) and (not only_dominated_support)):
            selected.append(qid)
            covered |= cov_map.get(qid, set())
    remaining = {qid: units for qid, units in cov_map.items() if qid not in selected}
    while covered & effective_coverable != effective_coverable and remaining:
        best_qid = None
        best_gain = 0
        best_turn = -1
        best_num = -1
        for qid, units in remaining.items():
            gain = len((units & effective_coverable) - covered)
            turn = qid_to_turn.get(qid, 0)
            num = int(qid[1:]) if qid.startswith('q') and qid[1:].isdigit() else -1
            if gain > best_gain or (gain == best_gain and (turn > best_turn or (turn == best_turn and num > best_num))):
                best_gain = gain
                best_qid = qid
                best_turn = turn
                best_num = num
        if best_qid is None or best_gain == 0:
            break
        selected.append(best_qid)
        covered |= remaining[best_qid]
        del remaining[best_qid]
    pruned_by_frontier: Set[str] = set()
    for anc in selected:
        anc_support = cov_map.get(anc, set())
        anc_pref = _answer_parent_preference_key(anc, q_coverage[anc], signal_by_id, qid_to_turn, qid_to_text)
        for desc in selected:
            if anc == desc:
                continue
            if anc not in ancestor_map.get(desc, set()):
                continue
            desc_support = cov_map.get(desc, set())
            desc_pref = _answer_parent_preference_key(desc, q_coverage[desc], signal_by_id, qid_to_turn, qid_to_text)
            if not anc_support.issubset(desc_support):
                continue
            if not _answer_supports_same_cluster(q_coverage[anc], q_coverage[desc]):
                continue
            if desc_pref < anc_pref:
                continue
            pruned_by_frontier.add(anc)
            break
    selected = [qid for qid in selected if qid not in pruned_by_frontier]
    deduped: List[str] = []
    for qid in sorted(selected, key=lambda q: _answer_parent_preference_key(q, q_coverage[q], signal_by_id, qid_to_turn, qid_to_text), reverse=True):
        if any((_answer_supports_are_true_duplicates(q_coverage[qid], q_coverage[kept]) for kept in deduped)):
            continue
        deduped.append(qid)
    selected = sorted(deduped, key=lambda q: (qid_to_turn.get(q, 0), int(q[1:]) if q.startswith('q') and q[1:].isdigit() else -1))
    covered = set()
    for qid in selected:
        covered |= cov_map.get(qid, set())
    unsupported = sorted(effective_coverable - covered)
    edges: List[Dict] = []
    for qid in selected:
        info = q_coverage[qid]
        signals_covered = sorted(cov_map.get(qid, set()))
        provenance = list(info['provenance'])
        if strict_edge_policy():
            provenance = [p for p in provenance if p.get('signal_id') in cov_map.get(qid, set()) and p.get('match_confidence') == 'high' and (p.get('document_support') is True)]
        ev_parts = []
        edge_confidence = 'high'
        for p in provenance:
            s = p.get('hit_sentence', '') or p.get('window_text', '')
            if s:
                granularity = p.get('granularity', '?')
                ev_parts.append(f'''{p.get('signal_id', '?')} ({granularity}): "{p.get('signal_text', '')}" supported by {qid}''')
            if p.get('match_confidence') == 'uncertain':
                edge_confidence = 'uncertain'
        ev_text = '; '.join(ev_parts[:5])
        if len(ev_text) > 400:
            ev_text = ev_text[:400] + '…'
        edges.append({'source': qid, 'target': 'Answer', 'type': 'lead_to', 'edge_kind': 'evidence_derived', 'evidence': ev_text, 'confidence': edge_confidence, 'phase': 'answer_mpsc', 'metadata': {'supports_signals': signals_covered, 'provenance': provenance, 'selection_policy': 'claim_or_complete_token_plus_light_dedup' if strict_edge_policy() else 'high_confidence_support_plus_light_dedup', 'answer_targeted_query': _is_answer_targeted_query_text(qid_to_text.get(qid, '')), 'edge_policy': {'policy': settings.EDGE_POLICY_VERSION, 'gate': 'document_claim_or_complete_token'} if strict_edge_policy() else None}})
    return (edges, unsupported)

def unsupported_answer_unit_ids(answer_units: List[Dict[str, str]], answer_signals: List[Dict[str, Any]], answer_edges: List[Dict[str, Any]]) -> List[str]:
    """Coverage requires verified claims, not a union of lexical token hits."""
    all_units = {unit['unit_id'] for unit in answer_units if unit.get('unit_id')}
    covered = set()
    for edge in answer_edges:
        if not edge.get('source', '').startswith('q') or edge.get('target') != 'Answer':
            continue
        for prov in (edge.get('metadata') or {}).get('provenance', []):
            if prov.get('match_confidence') != 'high' or prov.get('document_support') is not True:
                continue
            if prov.get('granularity') == 'whole_answer':
                return []
            if prov.get('granularity') == 'unit_claim':
                covered.update(prov.get('unit_ids') or [])
    return sorted(all_units - covered)

def build_prior_knowledge_answer_edge(answer_text: str, answer_units: List[Dict[str, str]], unsupported_units: List[str], answer_signals: Optional[List[Dict[str, Any]]]=None, mode: str='orphan_fallback') -> Dict[str, Any]:
    """
    Add a Prior_knowledge -> Answer edge in two cases:
      - orphan_fallback: no q->Answer support exists at all
      - synthesis_additive: q->Answer support exists, but one or more atomic
        answer units still have no retained query/tool provenance
    """
    from .llm import strict_edge_policy
    if not unsupported_units or not set(unsupported_units) <= {u['unit_id'] for u in answer_units}:
        raise ValueError('PK answer edges require explicit unsupported atomic answer units')
    unit_label = ', '.join(unsupported_units[:8]) if unsupported_units else 'all answer content'
    if len(unsupported_units) > 8:
        unit_label += ', ...'
    if mode == 'synthesis_additive':
        evidence = f'Additive synthesis: retained query evidence does not support every atomic answer unit; assigning Prior_knowledge coverage to {unit_label}.'
        fallback_reason = 'unsupported_atomic_answer_units'
        phase = 'answer_synthesis_additive'
    else:
        evidence = f'Fallback: final answer has no supporting query/tool provenance; assigning Prior_knowledge coverage to {unit_label}.'
        fallback_reason = 'no_query_answer_provenance'
        phase = 'answer_orphan_fallback'
    if len(evidence) > 400:
        evidence = evidence[:400] + '…'
    unsupported_set = set(unsupported_units)
    if mode == 'orphan_fallback' and (not unsupported_set):
        unsupported_set = {au['unit_id'] for au in answer_units}
    supported_signal_ids = []
    for sig in answer_signals or []:
        unit_ids = set(sig.get('unit_ids', []) or [])
        if mode == 'orphan_fallback' or unit_ids & unsupported_set:
            supported_signal_ids.append(sig['signal_id'])
    return {'source': 'Prior_knowledge', 'target': 'Answer', 'type': 'lead_to', 'edge_kind': 'prior_knowledge_derived', 'evidence': evidence, 'confidence': 'uncertain', 'phase': phase, 'metadata': {'answer_text_preview': (answer_text or '')[:300], 'supports_units': sorted(unsupported_set), 'supports_signals': supported_signal_ids, 'unsupported_units': sorted(unsupported_set), 'fallback_reason': fallback_reason, 'pk_answer_mode': mode, 'edge_policy': {'policy': settings.EDGE_POLICY_VERSION, 'gate': 'unsupported_atomic_answer_units_only'} if strict_edge_policy() else None}}
