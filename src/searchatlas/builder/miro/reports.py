"""Write optional human-readable graph construction reports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from . import settings

def print_dag_summary(graph: Dict, queries: List[Dict]):
    incoming: Dict[str, List[Dict]] = defaultdict(list)
    for e in graph['edges']:
        incoming[e['target']].append(e)
    print('\n  === DAG Edge Summary ===')
    for q in queries:
        qid, t, text = (q['id'], q['turn'], q['text'][:80])
        edges_in = incoming.get(qid, [])
        if edges_in:
            src_strs = []
            for e in edges_in:
                ek = e.get('edge_kind', '?')
                fs = e.get('failure_subtype', '')
                src_strs.append(f"{e['source']}({ek}/{fs})" if fs else f"{e['source']}({ek})")
            print(f'  {qid} (T{t:>2}): "{text}"')
            print(f"          <- {', '.join(src_strs)}")
        else:
            print(f'  {qid} (T{t:>2}): "{text}"')
            print('          <- ⚠ ORPHAN (no_source_found — expected under contribution-only)')
    answer_edges = incoming.get('Answer', [])
    if answer_edges:
        src_strs = [f"{e['source']}({e.get('edge_kind', '?')})" for e in answer_edges]
        print(f'  Answer:')
        print(f"          <- {', '.join(src_strs)}")

def print_edges_human_friendly(graph: Dict, queries: List[Dict]):
    qid_to_text = {q['id']: q['text'] for q in queries}
    print('\n  === EDGES (human-friendly) ===')
    for e in graph['edges']:
        src, tgt = (e['source'], e['target'])
        kind = e.get('edge_kind', '?')
        fs = e.get('failure_subtype', '')
        if fs:
            kind = f'{kind}/{fs}'
        conf = e.get('confidence', '')
        phase = e.get('phase', '')
        if src == 'Q0':
            src_label = 'Q0'
        elif src == 'Prior_knowledge':
            src_label = 'Prior_knowledge'
        else:
            src_label = f'''{src}:"{qid_to_text.get(src, '')[:60]}"'''
        if tgt == 'Answer':
            tgt_label = 'Answer'
        else:
            tgt_label = f'''{tgt}:"{qid_to_text.get(tgt, '')[:60]}"'''
        phase_tag = f' [{phase}]' if phase else ''
        print(f'  {src_label}  ->  {tgt_label}   [{kind}, {conf}{phase_tag}]')
        ev = (e.get('evidence') or '').strip()
        if ev:
            if len(ev) > 220:
                ev = ev[:220] + '…'
            print(f'     evidence: {ev}')
        cs = e.get('covered_signals', [])
        if cs:
            print(f'     covered_signals: {cs}')

def _qid_num_for_sort(qid: str) -> int:
    return int(qid[1:]) if qid.startswith('q') and qid[1:].isdigit() else -1

def _format_json_block(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=list)

def _summarize_qid_doc_chunks(qdoc: str) -> str:
    from .evidence import split_query_doc_into_chunks
    chunks = split_query_doc_into_chunks(qdoc)
    lines: List[str] = []
    if not chunks:
        return '(no doc text)'
    for idx, (label, text) in enumerate(chunks, start=1):
        preview = text.strip()
        if len(preview) > 1000:
            preview = preview[:1000] + '\n... [truncated]'
        lines.append(f'[{idx}] {label}\n{preview}')
    return '\n\n'.join(lines)

def write_human_debug_report(path: Path, *, task_id: str, question: str, queries: List[Dict[str, Any]], turn_to_qids: Dict[int, List[str]], qid_doc: Dict[str, str], turn_thinks: Dict[int, str], result_counts: Dict[str, int], turn_url_owners: Dict[int, Dict[str, List[str]]], visit_match_log: List[Dict[str, Any]], q0_edges: List[Dict[str, Any]], q0_units: List[Dict[str, Any]], unit_first_use: Dict[str, str], per_turn_raw: List[Dict[str, Any]], answer_text: str, answer_support_text: str, answer_units: List[Dict[str, Any]], answer_signals: List[Dict[str, Any]], answer_coverage: Dict[str, Any], answer_edges: List[Dict[str, Any]], graph: Dict[str, Any], issues: List[str]) -> None:
    from .parsing import tokenize
    lines: List[str] = []
    by_qid = {q['id']: q for q in queries}
    lines.append('=' * 100)
    lines.append(f'TASK DEBUG REPORT: {task_id}')
    lines.append('=' * 100)
    lines.append('')
    lines.append('QUESTION')
    lines.append('-' * 100)
    lines.append(question)
    lines.append('')
    lines.append('QUERY INDEX')
    lines.append('-' * 100)
    for q in queries:
        rc = result_counts.get(q['id'])
        rc_str = 'unknown'
        if rc is not None:
            rc_str = str(rc)
        lines.append(f'''{q['id']}  turn={q['turn']}  results={rc_str}  text="{q['text']}"''')
        lines.append(f"  tokens: {tokenize(q['text'])}")
    lines.append('')
    lines.append('Q0 EDGES')
    lines.append('-' * 100)
    if q0_units:
        lines.append('Q0 units:')
        lines.append(_format_json_block(q0_units))
        lines.append('')
        lines.append('First-use map:')
        lines.append(_format_json_block(unit_first_use))
        lines.append('')
    lines.append('Q0 edges:')
    lines.append(_format_json_block(q0_edges))
    lines.append('')
    lines.append('TURN THINK BLOCKS')
    lines.append('-' * 100)
    for turn in sorted(turn_to_qids):
        lines.append(f'Turn {turn}')
        lines.append(turn_thinks.get(turn, '(no think block)') or '(no think block)')
        lines.append('')
    lines.append('VISITED URL MATCHING')
    lines.append('-' * 100)
    lines.append('Structured URL owners by turn:')
    lines.append(_format_json_block(turn_url_owners))
    lines.append('')
    if visit_match_log:
        for idx, event in enumerate(visit_match_log, start=1):
            lines.append(f'[Visit Match {idx}]')
            lines.append(f"  turn: {event.get('turn')}")
            lines.append(f"  url: {event.get('visit_url')}")
            lines.append(f"  normalized_url: {event.get('normalized_visit_url')}")
            lines.append(f"  pending_qids: {event.get('pending_qids')}")
            lines.append(f"  same_turn_matches: {event.get('same_turn_matches')}")
            lines.append(f"  prior_matches: {event.get('prior_matches')}")
            lines.append(f"  searched_prior_turns: {event.get('searched_prior_turns')}")
            lines.append(f"  action: {event.get('action')}")
            lines.append(f"  owner_qid: {event.get('owner_qid')}")
            excerpt = event.get('sanitized_excerpt') or ''
            if excerpt:
                lines.append('  excerpt:')
                lines.append('    ' + excerpt.replace('\n', '\n    '))
            lines.append('')
    else:
        lines.append('(no visit-match events)')
        lines.append('')
    lines.append('QUERY DOCS / EVIDENCE TEXT')
    lines.append('-' * 100)
    for q in queries:
        qid = q['id']
        lines.append(f'''{qid}  turn={q['turn']}  text="{q['text']}"''')
        lines.append(_summarize_qid_doc_chunks(qid_doc.get(qid, '')))
        lines.append('')
    if qid_doc.get('Prior_knowledge'):
        lines.append('Prior_knowledge')
        lines.append(_summarize_qid_doc_chunks(qid_doc.get('Prior_knowledge', '')))
        lines.append('')
    lines.append('PER-TURN LLM DEBUG')
    lines.append('-' * 100)
    for turn_debug in per_turn_raw:
        turn = turn_debug['turn']
        lines.append(f'TURN {turn}')
        lines.append(f'New qids: {turn_to_qids.get(turn, [])}')
        lines.append(f"Visible tool-result messages: {turn_debug.get('visible_tool_message_indices', 'ALL')} (keep_tool_result={turn_debug.get('keep_tool_result', '?')})")
        lines.append('')
        lines.append('Prompt sent to LLM:')
        lines.append(turn_debug.get('prompt', '(missing prompt)'))
        lines.append('')
        lines.append('Per-query payload:')
        for qid in turn_to_qids.get(turn, []):
            payload = turn_debug.get('per_query_payload', {}).get(qid, {})
            lines.append(f'  {qid}:')
            lines.append('    token_status:')
            lines.append('    ' + _format_json_block(payload.get('token_status', {})).replace('\n', '\n    '))
            lines.append(f"    unsupported_new_tokens: {payload.get('unsupported_new_tokens', [])}")
            lines.append(f"    unsupported_q0_seen_tokens: {payload.get('unsupported_q0_seen_tokens', [])}")
            lines.append('    think_support:')
            lines.append('    ' + _format_json_block(payload.get('think_support', {})).replace('\n', '\n    '))
            lines.append('    seen_first_qid:')
            lines.append('    ' + _format_json_block(payload.get('seen_first_qid', {})).replace('\n', '\n    '))
            lines.append('    seen_latest_qid:')
            lines.append('    ' + _format_json_block(payload.get('seen_latest_qid', {})).replace('\n', '\n    '))
            lines.append(f"    used_signals: {payload.get('used_signals', [])}")
            lines.append('    candidates_by_token:')
            lines.append('    ' + _format_json_block(payload.get('candidates_by_token', {})).replace('\n', '\n    '))
            lines.append('')
        lines.append('Raw LLM edges:')
        lines.append(_format_json_block(turn_debug.get('raw_edges', [])))
        lines.append('')
        lines.append('Rejected edges after edge-policy validation:')
        lines.append(_format_json_block(turn_debug.get('rejected_edges', [])))
        lines.append('')
        lines.append('Accepted edges after validation/MPSC:')
        lines.append(_format_json_block(turn_debug.get('accepted_edges', [])))
        lines.append('')
        lines.append('MPSC debug:')
        lines.append(_format_json_block(turn_debug.get('mpsc_debug', {})))
        lines.append('')
        lines.append(f"no_source_found: {turn_debug.get('no_source_found', [])}")
        lines.append('=' * 100)
        lines.append('')
    lines.append('ANSWER PIPELINE')
    lines.append('-' * 100)
    lines.append(f'Answer text: {answer_text}')
    lines.append('')
    lines.append('Answer support text:')
    lines.append(answer_support_text or '(none)')
    lines.append('')
    lines.append('Answer units:')
    lines.append(_format_json_block(answer_units))
    lines.append('')
    lines.append('Answer signals:')
    lines.append(_format_json_block(answer_signals))
    lines.append('')
    lines.append('Answer coverage by query:')
    lines.append(_format_json_block(answer_coverage))
    lines.append('')
    lines.append('Answer edges:')
    lines.append(_format_json_block(answer_edges))
    lines.append('')
    lines.append('FINAL GRAPH')
    lines.append('-' * 100)
    lines.append(_format_json_block(graph))
    lines.append('')
    lines.append('VALIDATION ISSUES')
    lines.append('-' * 100)
    if issues:
        lines.extend((f'- {issue}' for issue in issues))
    else:
        lines.append('(none)')
    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')
