"""Attribute retrieved evidence to its query and availability time."""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from urllib.parse import urlparse, unquote
from . import settings

def _load_single_json_object(content: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

def sanitize_tool_response_for_provenance(content: str) -> str:
    """
    Remove search-topic header lines from a tool response before indexing it as
    provenance evidence.

    Keep real snippets / visited-page text intact, and keep the raw content
    elsewhere for result-count extraction.
    """
    if not content:
        return ''
    sanitized = settings.SEARCH_RESULT_HEADER_RE.sub('', content)
    sanitized = settings.VISIT_RESULT_HEADER_RE.sub('', sanitized)
    sanitized = re.sub('(?im)^[ \\t]*(date published|published|publication date)\\s*:\\s*.*$', '', sanitized)
    sanitized = re.sub('\\n{3,}', '\n\n', sanitized)
    return sanitized.strip()

def split_search_tool_response_blocks(content: str) -> List[Dict[str, Any]]:
    """
    Split a combined search tool response into query-specific blocks.

    Each block starts with a header like:
      A Google search for '...' found N results:
    and continues until the next such header or the end of the response.
    """
    if not content:
        return []
    matches = list(settings.SEARCH_RESULT_HEADER_RE.finditer(content))
    if not matches:
        return []
    blocks: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        # A combined reply may alternate search and visit blocks.
        other_header = settings.VISIT_RESULT_HEADER_RE.search(content, match.end())
        if other_header is not None:
            end = min(end, other_header.start())
        raw_block = content[start:end].strip()
        blocks.append({'query_text': match.group(1).strip(), 'result_count': int(match.group(2)), 'raw_block': raw_block, 'sanitized_block': sanitize_tool_response_for_provenance(raw_block)})
    return blocks

def split_visit_tool_response_blocks(content: str) -> List[Dict[str, Any]]:
    """
    Split a combined visit tool response into goal-specific blocks.

    Each block starts with a header like:
      The useful information in <url> for user goal <goal> as follows:
    and continues until the next visit header or the end of the response.
    """
    if not content:
        return []
    matches = list(settings.VISIT_RESULT_HEADER_RE.finditer(content))
    if not matches:
        return []
    blocks: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        # A combined reply may alternate search and visit blocks.
        other_header = settings.SEARCH_RESULT_HEADER_RE.search(content, match.end())
        if other_header is not None:
            end = min(end, other_header.start())
        raw_block = content[start:end].strip()
        blocks.append({'url': match.group(1).strip(), 'goal_text': match.group(2).strip(), 'raw_block': raw_block, 'raw_source': raw_block, 'sanitized_block': sanitize_tool_response_for_provenance(raw_block)})
    return blocks

def _shorten_visit_failure_text(text: str, max_len: int=240) -> str:
    collapsed = re.sub('\\s+', ' ', text or '').strip()
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[:max_len - 1] + '…'

def collect_visit_summary_hard_failures(raw_source: str, sanitized_block: str='') -> List[str]:
    """
    Detect failures that come from the visit-summary tool itself, not from the
    main model's think block.
    """
    evidences: List[str] = []
    seen: Set[str] = set()
    obj = _load_single_json_object(raw_source or '')
    candidate_text = raw_source or sanitized_block or ''
    if isinstance(obj, dict):
        err = str(obj.get('error') or '').strip()
        if err:
            ev = f'visit summary tool error: {_shorten_visit_failure_text(err)}'
            evidences.append(ev)
            seen.add(ev)
        elif obj.get('success') is False:
            ev = 'visit summary tool returned success=false'
            evidences.append(ev)
            seen.add(ev)
        candidate_text = '\n'.join((part for part in [str(obj.get('extracted_info') or ''), err] if part))
    for label, pat in settings.VISIT_SUMMARY_HARD_FAIL_PATTERNS:
        match = pat.search(candidate_text or '')
        if not match:
            continue
        excerpt = _shorten_visit_failure_text((candidate_text or '')[max(0, match.start() - 80):match.end() + 160])
        ev = f'{label}: {excerpt}'
        if ev not in seen:
            evidences.append(ev)
            seen.add(ev)
    return evidences

def normalize_url_for_match(url: str) -> str:
    """
    Light normalization for exact URL ownership checks.
    """
    url = (url or '').strip()
    if not url:
        return ''
    url = re.sub('^https?://', '', url, flags=re.I)
    url = url.rstrip('/')
    return url.lower()

def extract_url_site(url: str) -> str:
    raw = (url or '').strip()
    if not raw:
        return ''
    if not re.match('^[a-z]+://', raw, flags=re.I):
        raw = 'https://' + raw.lstrip('/')
    parsed = urlparse(raw)
    host = (parsed.netloc or parsed.path or '').strip().lower()
    host = host.split('@')[-1].split(':')[0]
    if host.startswith('www.'):
        host = host[4:]
    return host

def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract raw URLs from a search-result block.
    """
    if not text:
        return []
    urls = re.findall('https?://[^\\s)\\]}>\\"\']+', text, flags=re.I)
    out = []
    seen = set()
    for u in urls:
        nu = normalize_url_for_match(u)
        if nu and nu not in seen:
            seen.add(nu)
            out.append(nu)
    return out

def qid_sort_key(qid: str) -> Tuple[int, str]:
    if qid.startswith('q') and qid[1:].isdigit():
        return (int(qid[1:]), qid)
    return (-1, qid)

def empty_query_url_metadata() -> Dict[str, Set[str]]:
    return {'search_urls': set(), 'search_sites': set(), 'search_url_tokens': set(), 'visit_urls': set(), 'visit_sites': set(), 'visit_url_tokens': set()}

def register_query_url(query_url_metadata: Dict[str, Dict[str, Set[str]]], qid: str, url: str, source: str) -> None:
    if not qid or not url or qid == 'Prior_knowledge':
        return
    meta = query_url_metadata[qid]
    norm_url = normalize_url_for_match(url)
    site = extract_url_site(url)
    url_tokens, _ = extract_url_match_tokens(url)
    prefix = 'visit' if source == 'visit' else 'search'
    if norm_url:
        meta[f'{prefix}_urls'].add(norm_url)
    if site:
        meta[f'{prefix}_sites'].add(site)
    for tok in url_tokens:
        meta[f'{prefix}_url_tokens'].add(tok)

def append_visit_failure_record(visit_failure_records: Dict[str, List[Dict[str, Any]]], owner_qid: str, *, turn: int, visit_url: str, action: str, evidences: List[str]) -> None:
    if not owner_qid or owner_qid == 'Prior_knowledge' or (not evidences):
        return
    url_tokens, url_phrase = extract_url_match_tokens(visit_url)
    visit_failure_records[owner_qid].append({'turn': turn, 'visit_url': visit_url, 'normalized_visit_url': normalize_url_for_match(visit_url), 'visit_site': extract_url_site(visit_url), 'url_tokens': list(url_tokens), 'url_phrase': url_phrase, 'action': action, 'evidences': list(dict.fromkeys(evidences))})

def extract_url_match_tokens(url: str) -> Tuple[List[str], str]:
    """
    Extract entity-like matching signals from the URL alone.
    Used only for visit-owner recovery after exact URL ownership fails.
    """
    raw = (url or '').strip()
    if not raw:
        return ([], '')
    parsed = urlparse(raw)
    path = unquote(parsed.path or '')
    segments = [seg for seg in path.split('/') if seg]
    slug = segments[-1] if segments else ''
    slug = re.sub('\\.(html?|php|aspx?)$', '', slug, flags=re.I)
    slug = slug.strip().lower()
    raw_tokens = re.findall('[a-z0-9]+', slug.replace('_', ' ').replace('-', ' '))
    stop = {'wiki', 'wikipedia', 'index', 'php', 'html', 'htm'}
    tokens = [tok for tok in raw_tokens if len(tok) >= 2 and tok not in stop]
    phrase = ' '.join(tokens).strip()
    return (tokens, phrase)

def best_query_owner_from_url_tokens(candidate_qids: List[str], qid_doc: Dict[str, str], url_tokens: List[str], url_phrase: str) -> Optional[str]:
    """
    Rank candidate queries by URL-token evidence in the query's accumulated
    doc/snippet/think content, then recency.

    Priority:
      1) full URL phrase match in qid_doc
      2) number of URL-token hits in qid_doc
      3) recency (later qid wins)
    """
    from .parsing import tokenize_set
    if not candidate_qids or not url_tokens:
        return None
    best_qid = None
    best_key: Optional[Tuple[int, int, int]] = None
    min_required_hits = 2 if len(url_tokens) >= 2 else 1
    for qid in candidate_qids:
        q_doc_norm = ' '.join((qid_doc.get(qid, '') or '').split()).strip().lower()
        if not q_doc_norm:
            continue
        q_toks = tokenize_set(q_doc_norm)
        hit_count = len(set(url_tokens) & q_toks)
        if hit_count < min_required_hits:
            continue
        phrase_match = 1 if url_phrase and url_phrase in q_doc_norm else 0
        qid_num = int(qid[1:]) if qid.startswith('q') and qid[1:].isdigit() else -1
        key = (phrase_match, hit_count, qid_num)
        if best_key is None or key > best_key:
            best_key = key
            best_qid = qid
    return best_qid

def match_tool_response_query_to_qid(query_text: str, candidate_qids: List[str], qid_to_text: Dict[str, str]) -> Optional[str]:
    """Match only an unambiguous query in the outstanding request batch."""
    normalized = ' '.join((query_text or '').split()).casefold()
    if not normalized:
        return None
    matches = [qid for qid in candidate_qids
               if ' '.join((qid_to_text.get(qid) or '').split()).casefold() == normalized]
    return matches[0] if len(matches) == 1 else None

def extract_query_docs(messages: List[Dict], turn_to_qids: Dict[int, List[str]], qid_to_text: Dict[str, str]) -> Tuple:
    """
    Build query docs and immutable pre-query contexts (last return value).

        Returns:
      qid_doc[qid]        : evidence text for that query (think before the search + tool responses after it)
      turn_think_only[t]  : think blocks text for turn t
      result_counts       : qid -> result count (best-effort)
      qid_search_doc[qid] : search-tool-only evidence text for that query
      turn_url_owners[t]  : normalized_url -> [qid owners from structured search blocks in that turn]
      visit_summary_hard_failures[qid]: structured visit-summary failure records tied to that query
      query_url_metadata[qid]: search/visit URL + site signals for retry matching
    """
    from .parsing import assistant_has_search_turn
    from ..safety import evidence_snapshot, unresolved_response
    result_counts: Dict[str, int] = {}
    turn_contexts = {}
    pending_search_qids = []
    turn = 0
    current_thinks: List[str] = []
    current_turn_qids: List[str] = []
    pending_qids_for_next_tool_responses: List[str] = []
    first_search_seen = False
    turn_think_only: Dict[int, str] = defaultdict(str)
    qid_doc: Dict[str, str] = defaultdict(str)
    qid_search_doc: Dict[str, str] = defaultdict(str)
    turn_url_owners: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    visit_summary_hard_failures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    query_url_metadata: Dict[str, Dict[str, Set[str]]] = defaultdict(empty_query_url_metadata)
    visit_match_log: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '') or ''
        if role == 'assistant':
            if '<tool_call>' in content or '<use_mcp_tool>' in content:
                pending_search_qids = []
            for tm in re.finditer('<think>(.*?)</think>', content, re.DOTALL):
                think = tm.group(1).strip()
                if think:
                    current_thinks.append(think)
            if assistant_has_search_turn(messages, idx):
                turn += 1
                turn_contexts[turn] = evidence_snapshot(qid_doc, result_counts, qid_search_doc, visit_summary_hard_failures, query_url_metadata)
                combined_think = '\n---\n'.join(current_thinks).strip()
                turn_think_only[turn] = combined_think
                current_turn_qids = turn_to_qids.get(turn, [])
                pending_qids_for_next_tool_responses = list(current_turn_qids)
                pending_search_qids = list(current_turn_qids)
                if combined_think:
                    if first_search_seen:
                        for qid in current_turn_qids:
                            qid_doc[qid] += f'[Turn {turn} Think]\n{combined_think}\n'
                    else:
                        print(f'[THINK_INDEX] Skip indexing pre-search think for turn {turn}')
                current_thinks = []
                first_search_seen = True
        elif role == 'user' and '<tool_response>' in content:
            search_blocks = split_search_tool_response_blocks(content)
            if search_blocks:
                unmatched_pending_qids = list(pending_search_qids)
                for block in search_blocks:
                    q_text_in_response = block['query_text']
                    n = block['result_count']
                    matched_qid = match_tool_response_query_to_qid(q_text_in_response, unmatched_pending_qids, qid_to_text)
                    if matched_qid is None:
                        visit_match_log.append(unresolved_response(turn, 'search header has no unique outstanding query', q_text_in_response))
                        continue
                    if matched_qid is not None and matched_qid not in result_counts:
                        result_counts[matched_qid] = n
                    if matched_qid in unmatched_pending_qids:
                        unmatched_pending_qids.remove(matched_qid)
                    sanitized_block = block.get('sanitized_block', '')
                    raw_block = block.get('raw_block', '') or ''
                    if matched_qid and sanitized_block:
                        qid_doc[matched_qid] += f'[Tool_Response]\n{sanitized_block}\n'
                        qid_search_doc[matched_qid] += f'[Tool_Response]\n{sanitized_block}\n'
                        block_urls = extract_urls_from_text(raw_block)
                        for u in block_urls:
                            owners = turn_url_owners[turn][u]
                            if matched_qid not in owners:
                                owners.append(matched_qid)
                            register_query_url(query_url_metadata, matched_qid, u, source='search')
                pending_search_qids = unmatched_pending_qids
            visit_blocks = split_visit_tool_response_blocks(content)
            if visit_blocks:
                for block in visit_blocks:
                    visit_url = block.get('url', '')
                    raw_source = block.get('raw_source', '') or block.get('raw_block', '')
                    sanitized_block = block.get('sanitized_block', '')
                    visit_failure_evidence = collect_visit_summary_hard_failures(raw_source, sanitized_block)
                    norm_visit_url = normalize_url_for_match(visit_url)
                    visit_url_tokens, visit_url_phrase = extract_url_match_tokens(visit_url)
                    visit_site = extract_url_site(visit_url)
                    print('\n' + '=' * 80)
                    print(f'[VISIT_MATCH] turn={turn}')
                    print(f'[VISIT_MATCH] pending_qids={pending_qids_for_next_tool_responses}')
                    print(f'[VISIT_MATCH] extracted visit_url={visit_url}')
                    visit_event: Dict[str, Any] = {'turn': turn, 'visit_url': visit_url, 'normalized_visit_url': norm_visit_url, 'pending_qids': list(pending_qids_for_next_tool_responses), 'same_turn_matches': [], 'prior_matches': [], 'searched_prior_turns': 0, 'action': '', 'owner_qid': None, 'hard_failure_evidence': list(visit_failure_evidence), 'sanitized_excerpt': sanitized_block[:600], 'visit_site': visit_site, 'url_tokens': list(visit_url_tokens), 'url_phrase': visit_url_phrase}
                    if not sanitized_block:
                        print('[VISIT_MATCH] action=SKIP_EMPTY_BLOCK')
                        visit_event['action'] = 'SKIP_EMPTY_BLOCK'
                        visit_match_log.append(visit_event)
                        continue
                    same_turn_matches = list(turn_url_owners.get(turn, {}).get(norm_visit_url, []))
                    same_turn_matches = sorted(set(same_turn_matches), key=qid_sort_key)
                    visit_event['same_turn_matches'] = list(same_turn_matches)
                    print(f'[VISIT_MATCH] same_turn_matches={same_turn_matches}')
                    if same_turn_matches:
                        owner_qid = same_turn_matches[-1]
                        print(f'[VISIT_MATCH] action=ATTACH_TO_SAME_TURN owner={owner_qid}')
                        visit_event['action'] = 'ATTACH_TO_SAME_TURN'
                        visit_event['owner_qid'] = owner_qid
                        qid_doc[owner_qid] += f'[Tool_Response]\n{sanitized_block}\n'
                        register_query_url(query_url_metadata, owner_qid, visit_url, source='visit')
                        append_visit_failure_record(visit_summary_hard_failures, owner_qid, turn=turn, visit_url=visit_url, action='ATTACH_TO_SAME_TURN', evidences=visit_failure_evidence)
                        visit_match_log.append(visit_event)
                        continue
                    prior_matches: List[str] = []
                    searched_prior_turns = 0
                    for prev_turn in sorted(turn_url_owners.keys()):
                        if prev_turn >= turn:
                            continue
                        searched_prior_turns += 1
                        owners = turn_url_owners.get(prev_turn, {}).get(norm_visit_url, [])
                        for qid in owners:
                            prior_matches.append(qid)
                    prior_matches = sorted(set(prior_matches), key=qid_sort_key)
                    visit_event['prior_matches'] = list(prior_matches)
                    visit_event['searched_prior_turns'] = searched_prior_turns
                    print(f'[VISIT_MATCH] searching_prior_pool_size={searched_prior_turns}')
                    print(f'[VISIT_MATCH] prior_matches={prior_matches}')
                    if prior_matches:
                        owner_qid = prior_matches[-1]
                        print(f'[VISIT_MATCH] action=ATTACH_TO_LATEST_PRIOR owner={owner_qid}')
                        visit_event['action'] = 'ATTACH_TO_LATEST_PRIOR'
                        visit_event['owner_qid'] = owner_qid
                        qid_doc[owner_qid] += f'[Tool_Response]\n{sanitized_block}\n'
                        register_query_url(query_url_metadata, owner_qid, visit_url, source='visit')
                        append_visit_failure_record(visit_summary_hard_failures, owner_qid, turn=turn, visit_url=visit_url, action='ATTACH_TO_LATEST_PRIOR', evidences=visit_failure_evidence)
                        visit_match_log.append(visit_event)
                        continue
                    url_tokens, url_phrase = extract_url_match_tokens(visit_url)
                    visit_event['url_tokens'] = list(url_tokens)
                    visit_event['url_phrase'] = url_phrase
                    print(f'[VISIT_MATCH] url_tokens={url_tokens}')
                    owner_qid = best_query_owner_from_url_tokens(pending_qids_for_next_tool_responses, qid_doc, url_tokens, url_phrase)
                    if owner_qid:
                        print(f'[VISIT_MATCH] action=ATTACH_TO_SAME_TURN_URL_TOKEN_MATCH owner={owner_qid}')
                        visit_event['action'] = 'ATTACH_TO_SAME_TURN_URL_TOKEN_MATCH'
                        visit_event['owner_qid'] = owner_qid
                        qid_doc[owner_qid] += f'[Tool_Response]\n{sanitized_block}\n'
                        register_query_url(query_url_metadata, owner_qid, visit_url, source='visit')
                        append_visit_failure_record(visit_summary_hard_failures, owner_qid, turn=turn, visit_url=visit_url, action='ATTACH_TO_SAME_TURN_URL_TOKEN_MATCH', evidences=visit_failure_evidence)
                        visit_match_log.append(visit_event)
                        continue
                    prior_candidate_qids: List[str] = []
                    for prev_turn in sorted(turn_to_qids.keys()):
                        if prev_turn >= turn:
                            continue
                        prior_candidate_qids.extend(turn_to_qids.get(prev_turn, []))
                    owner_qid = best_query_owner_from_url_tokens(prior_candidate_qids, qid_doc, url_tokens, url_phrase)
                    if owner_qid:
                        print(f'[VISIT_MATCH] action=ATTACH_TO_LATEST_PRIOR_URL_TOKEN_MATCH owner={owner_qid}')
                        visit_event['action'] = 'ATTACH_TO_LATEST_PRIOR_URL_TOKEN_MATCH'
                        visit_event['owner_qid'] = owner_qid
                        qid_doc[owner_qid] += f'[Tool_Response]\n{sanitized_block}\n'
                        register_query_url(query_url_metadata, owner_qid, visit_url, source='visit')
                        append_visit_failure_record(visit_summary_hard_failures, owner_qid, turn=turn, visit_url=visit_url, action='ATTACH_TO_LATEST_PRIOR_URL_TOKEN_MATCH', evidences=visit_failure_evidence)
                        visit_match_log.append(visit_event)
                        continue
                    print('[VISIT_MATCH] action=ATTACH_TO_PRIOR_KNOWLEDGE owner=Prior_knowledge')
                    visit_event['action'] = 'ATTACH_TO_PRIOR_KNOWLEDGE'
                    visit_event['owner_qid'] = 'Prior_knowledge'
                    qid_doc['Prior_knowledge'] += f'[Tool_Response]\n{sanitized_block}\n'
                    visit_match_log.append(visit_event)
                continue
            if not search_blocks and not visit_blocks:
                sanitized_content = sanitize_tool_response_for_provenance(content)
                if not sanitized_content:
                    continue
                if len(pending_search_qids) == 1:
                    # A single outstanding explicit search identifies its reply
                    # without guessing an owner or broadcasting a batch result.
                    qid = pending_search_qids.pop()
                    qid_doc[qid] += f'[Tool_Response]\n{sanitized_content}\n'
                else:
                    visit_match_log.append(unresolved_response(turn, 'unparsed response has no unique outstanding search'))
    turn_url_owners_out = {t: {u: list(qids) for u, qids in url_map.items()} for t, url_map in turn_url_owners.items()}
    visit_summary_hard_failures_out = {qid: list(evs) for qid, evs in visit_summary_hard_failures.items()}
    query_url_metadata_out = {qid: {key: sorted(values) for key, values in meta.items()} for qid, meta in query_url_metadata.items()}
    return (dict(qid_doc), dict(turn_think_only), result_counts, dict(qid_search_doc), turn_url_owners_out, visit_summary_hard_failures_out, query_url_metadata_out, visit_match_log, turn_contexts)

def build_constructed_url_pk_edges(visit_match_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For unmatched visit URLs that were recovered via URL-token matching, add an
    explicit Prior_knowledge edge to the recovered owner query. The visit still
    belongs to the recovered query; this extra edge records the model skill of
    constructing an authoritative URL (e.g., Wikipedia / Scholar / site:-style
    targeting) from prior knowledge.
    """
    edges: List[Dict[str, Any]] = []
    seen_targets: Set[str] = set()
    for event in visit_match_log:
        action = event.get('action', '')
        owner_qid = event.get('owner_qid')
        if action not in {'ATTACH_TO_SAME_TURN_URL_TOKEN_MATCH', 'ATTACH_TO_LATEST_PRIOR_URL_TOKEN_MATCH'}:
            continue
        if not owner_qid or not str(owner_qid).startswith('q'):
            continue
        if owner_qid in seen_targets:
            continue
        seen_targets.add(owner_qid)
        url_tokens = event.get('url_tokens', []) or []
        url_phrase = event.get('url_phrase', '') or ''
        visit_url = event.get('visit_url', '') or ''
        token_desc = url_phrase if url_phrase else ', '.join(url_tokens)
        evidence = f"Constructed URL likely used prior knowledge: no exact URL owner match; visited URL '{visit_url}' was recovered via URL token match ({(token_desc if token_desc else 'no slug phrase')})."
        edges.append({'source': 'Prior_knowledge', 'target': owner_qid, 'type': 'lead_to', 'edge_kind': 'prior_knowledge_derived', 'evidence': evidence, 'confidence': 'high', 'phase': 'deterministic_constructed_url_pk', 'metadata': {'create_url': True, 'visit_url': visit_url, 'url_tokens': list(url_tokens), 'url_phrase': url_phrase, 'visit_recovery_action': action, 'reason': 'constructed_authoritative_url_from_prior_knowledge'}})
    return edges

def split_sentences(text: str) -> List[str]:
    text = (text or '').strip()
    if not text:
        return []
    return re.split('(?<=[.!?])\\s+', text)

def sentence_hits_in_text(token: str, text: str) -> List[Dict[str, Any]]:
    """Return sentence-anchored hits: [{sent_idx, sent_text}, ...]"""
    sents = split_sentences(text or '')
    if not sents:
        return []
    pat = re.compile(f'\\b{re.escape(token)}\\b', re.IGNORECASE)
    out = []
    for i, s in enumerate(sents):
        if pat.search(s):
            out.append({'sent_idx': i, 'sent_text': s.strip()})
    return out

def split_query_doc_into_chunks(qdoc: str) -> List[Tuple[str, str]]:
    """
    Split qid_doc[qid] into labeled chunks:
      - [Turn X Think]
      - [Tool_Response #k]
    """
    text = qdoc or ''
    parts = re.split('(\\[Turn \\d+ Think\\]|\\[Tool_Response\\])', text)
    chunks: List[Tuple[str, str]] = []
    cur_label = 'UNKNOWN'
    cur_buf: List[str] = []
    tool_i = 0

    def flush():
        nonlocal cur_label, cur_buf
        if cur_buf:
            chunks.append((cur_label, ''.join(cur_buf).strip()))
            cur_buf = []
    for p in parts:
        if p.startswith('[Turn ') and 'Think]' in p:
            flush()
            cur_label = p.strip()
        elif p.strip() == '[Tool_Response]':
            flush()
            tool_i += 1
            cur_label = f'[Tool_Response #{tool_i}]'
        else:
            cur_buf.append(p)
    flush()
    return [(lab, txt) for lab, txt in chunks if txt]

def classify_chunk_source_type(chunk_label: str, chunk_text: str) -> str:
    """
    Classify a provenance chunk as 'visit', 'snippet', or 'think'.

    Visited-page content is treated as *stronger* provenance than search
    snippets because it represents information the agent actively navigated
    to and read in full, making it more credible and verifiable.

    Detection rules (applied to the first ~300 chars of the chunk):
      visit   — tool_response starting with "The useful information in <URL>"
      snippet — tool_response starting with "A Google search for '...' found"
      think   — chunk label contains "Think"
    """
    if 'Think' in chunk_label:
        return 'think'
    head = (chunk_text or '').strip()[:300]
    if 'The useful information in ' in head:
        return 'visit'
    return 'snippet'

def tool_evidence_only(qdoc: str) -> str:
    """Return attributable Tool_Response text, excluding model think blocks."""
    chunks = split_query_doc_into_chunks(qdoc)
    return '\n'.join((text for label, text in chunks if label.startswith('[Tool_Response #')))

def token_provenance_in_prior_docs(token: str, prior_qs: List[Dict[str, Any]], qid_doc: Dict[str, str], sent_window: int, max_hits_total: int) -> List[Dict[str, Any]]:
    """
    Keyword search over ALL prior query docs (qid_doc).
    Capture context + query id + chunk label + sentence indices.
    Prefer recency (higher turn first).
    """
    from .llm import strict_edge_policy
    pat = re.compile(f'\\b{re.escape(token)}\\b', re.IGNORECASE)
    hits: List[Dict[str, Any]] = []
    prior_sorted = sorted(prior_qs, key=lambda q: (q['turn'], int(q['id'][1:])), reverse=True)
    for pq in prior_sorted:
        src_qid = pq['id']
        qdoc = qid_doc.get(src_qid, '')
        if not qdoc:
            continue
        if not pat.search(qdoc):
            continue
        chunks = split_query_doc_into_chunks(qdoc)
        for chunk_label, chunk_text in chunks:
            if not chunk_text or not pat.search(chunk_text):
                continue
            if strict_edge_policy() and (not chunk_label.startswith('[Tool_Response #')):
                continue
            sents = split_sentences(chunk_text)
            if not sents:
                continue
            src_type = classify_chunk_source_type(chunk_label, chunk_text)
            for si, sent in enumerate(sents):
                if pat.search(sent):
                    lo = max(0, si - sent_window)
                    hi = min(len(sents) - 1, si + sent_window)
                    window_text = ' '.join(sents[lo:hi + 1]).strip()
                    hits.append({'token': token, 'source_qid': src_qid, 'source_turn': pq['turn'], 'chunk_label': chunk_label, 'source_type': src_type, 'hit_sentence_idx': si, 'window_lo': lo, 'window_hi': hi, 'hit_sentence': sent.strip(), 'window_text': window_text})
                    if len(hits) >= max_hits_total:
                        return hits
    return hits

def select_top_k_sources_from_hits(hits: List[Dict[str, Any]], top_k_sources: int, max_hits_per_source: int) -> List[Dict[str, Any]]:
    """
    Robust top-k selection with visit-page priority:
      - group by source_qid
      - rank sources by (has_visit desc, source_turn desc, hit_count desc)
        so that a source containing visited-page provenance is always
        preferred over a snippet-only source, even from a later turn.
      - within each source, sort visit hits before snippet hits
      - keep up to max_hits_per_source provenance entries per source
    """
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    turn_of: Dict[str, int] = {}
    best_type: Dict[str, int] = {}
    for h in hits:
        src = h['source_qid']
        by_src[src].append(h)
        turn_of[src] = max(turn_of.get(src, -1), h.get('source_turn', -1))
        tp = settings._SOURCE_TYPE_PRIORITY.get(h.get('source_type', 'snippet'), 1)
        best_type[src] = max(best_type.get(src, 0), tp)
    scored = []
    for src, hs in by_src.items():
        scored.append((src, best_type.get(src, 0), turn_of.get(src, -1), len(hs)))
    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

    def _build_entry(src: str, bt: int, t: int, n: int, *, is_first_prov: bool=False) -> Dict[str, Any]:
        hs = by_src[src]
        hs.sort(key=lambda h: (settings._SOURCE_TYPE_PRIORITY.get(h.get('source_type', 'snippet'), 1), h.get('source_turn', -1), h.get('chunk_label', ''), h.get('hit_sentence_idx', -1)), reverse=True)
        return {'source_qid': src, 'source_turn': t, 'hit_count': n, 'best_source_type': {v: k for k, v in settings._SOURCE_TYPE_PRIORITY.items()}.get(bt, 'snippet'), 'is_first_provenance': is_first_prov, 'provenance': hs[:max_hits_per_source]}
    selected = scored[:top_k_sources]
    selected_srcs = {s[0] for s in selected}
    out = [_build_entry(src, bt, t, n) for src, bt, t, n in selected]
    if scored:
        first_prov = min(scored, key=lambda x: (x[2], x[0]))
        if first_prov[0] not in selected_srcs:
            fp_src, fp_bt, fp_t, fp_n = first_prov
            out.append(_build_entry(fp_src, fp_bt, fp_t, fp_n, is_first_prov=True))
    return out

def denoise_provenance(candidates_by_token: Dict[str, List[Dict[str, Any]]], think_support: Optional[Dict[str, List[Dict[str, Any]]]]=None, min_tokens_per_source: int=2, rare_doc_freq: int=2) -> Dict[str, List[Dict[str, Any]]]:
    """
    P1 Provenance Denoising — multi-token corroboration + rare-token protection.

    Problem: pure keyword provenance produces many coincidental single-token
    matches (e.g., "born" matching Andy Ruiz Jr., "las" matching Las Vegas
    hotels). A flat ≥2-token filter would also kill genuinely relevant
    single-token matches for rare/specific entities.

    Solution — keep a source S for token t if EITHER:
      (a) S covers ≥ min_tokens_per_source distinct query tokens (multi-token
          corroboration — the strongest signal), OR
      (b) t is a RARE token: appears in ≤ rare_doc_freq distinct source docs
          across the entire provenance set. Rare tokens are specific enough
          that even a single-token match is informative (e.g., "riverside"
          appearing in only 1-2 source docs).

    Fallback: if ALL sources for a token are removed, keep the single best
    original source (first-provenance preferred) to avoid total signal loss.
    """
    if not candidates_by_token:
        return candidates_by_token
    source_token_count: Dict[str, Set[str]] = defaultdict(set)
    for tok, sources in candidates_by_token.items():
        for src in sources:
            source_token_count[src['source_qid']].add(tok)
    token_doc_freq: Dict[str, int] = {}
    for tok, sources in candidates_by_token.items():
        token_doc_freq[tok] = len({s['source_qid'] for s in sources})
    multi_token_sources = {sqid for sqid, toks in source_token_count.items() if len(toks) >= min_tokens_per_source}
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for tok, sources in candidates_by_token.items():
        is_rare = token_doc_freq.get(tok, 0) <= rare_doc_freq
        kept = []
        for s in sources:
            sqid = s['source_qid']
            if sqid in multi_token_sources:
                kept.append(s)
            elif is_rare:
                kept.append(s)
        if kept:
            filtered[tok] = kept
        elif sources:
            best = max(sources, key=lambda s: (s.get('is_first_provenance', False), s.get('hit_count', 0)))
            filtered[tok] = [best]
    return filtered

def compute_token_statuses(q_tokens: List[str], q0_token_set: Set[str], prior_queries: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Token statuses:
      Q0_ONLY:  token in Q0, NOT in any prior query → truly irrelevant for q→q
      Q0_SEEN:  token in Q0 AND in some prior query → participates in q→q MPSC
      SEEN:     token in prior queries, not in Q0
      NEW:      token not in Q0 and not in prior queries
    """
    from .parsing import tokenize_set
    seen_tokens = set()
    for pq in prior_queries:
        seen_tokens |= tokenize_set(pq.get('text', ''))
    status = {}
    for tok in q_tokens:
        if tok in q0_token_set:
            if tok in seen_tokens:
                status[tok] = 'Q0_SEEN'
            else:
                status[tok] = 'Q0_ONLY'
        elif tok in seen_tokens:
            status[tok] = 'SEEN'
        else:
            status[tok] = 'NEW'
    return status

def latest_prior_query_with_token(token: str, prior_queries: List[Dict[str, Any]]) -> Optional[str]:
    """LATEST prior query whose QUERY TEXT contains token."""
    from .parsing import tokenize_set
    best_qid = None
    best_turn = -1
    best_num = -1
    for pq in prior_queries:
        if token in tokenize_set(pq.get('text', '')):
            t = pq['turn']
            num = int(pq['id'][1:]) if pq['id'].startswith('q') and pq['id'][1:].isdigit() else -1
            if t > best_turn or (t == best_turn and num > best_num):
                best_turn = t
                best_num = num
                best_qid = pq['id']
    return best_qid

def first_prior_query_with_token(token: str, prior_queries: List[Dict[str, Any]]) -> Optional[str]:
    """EARLIEST prior query whose QUERY TEXT contains token (first provenance default)."""
    from .parsing import tokenize_set
    best_qid = None
    best_turn = float('inf')
    best_num = float('inf')
    for pq in prior_queries:
        if token in tokenize_set(pq.get('text', '')):
            t = pq['turn']
            num = int(pq['id'][1:]) if pq['id'].startswith('q') and pq['id'][1:].isdigit() else float('inf')
            if t < best_turn or (t == best_turn and num < best_num):
                best_turn = t
                best_num = num
                best_qid = pq['id']
    return best_qid
