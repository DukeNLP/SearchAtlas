"""Assemble and validate the evidence-dependency DAG."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
from . import settings

def assemble_dag(question: str, queries: List[Dict], edges: List[Dict], answer_text: str='', answer_units: Optional[List[Dict[str, str]]]=None) -> Dict:
    nodes: List[Dict] = []
    gid = 1
    nodes.append({'id': 'Q0', 'global_id': gid, 'type': 'Question', 'text': question})
    gid += 1
    for q in queries:
        nodes.append({'id': q['id'], 'global_id': gid, 'type': 'Query', 'text': q['text'], 'turn': q['turn']})
        gid += 1
    nodes.append({'id': 'Prior_knowledge', 'global_id': gid, 'type': 'Prior_knowledge', 'text': "Model's general knowledge and reasoning not directly tied to a specific source document."})
    gid += 1
    if answer_text:
        nodes.append({'id': 'Answer', 'global_id': gid, 'type': 'Answer', 'text': answer_text, 'normalized_units': answer_units or []})
    qid_to_turn_map = {q['id']: q['turn'] for q in queries}
    seen: Set[Tuple[str, str, str]] = set()
    unique: List[Dict] = []
    same_turn_removed = 0
    for e in edges:
        src, tgt = (e['source'], e['target'])
        kind = e.get('edge_kind', '?')
        failure_subtype = e.get('failure_subtype', '') if kind == 'failure_derived' else ''
        key = (src, tgt, kind, failure_subtype)
        if key in seen:
            continue
        if src.startswith('q') and tgt.startswith('q') and (qid_to_turn_map.get(src) == qid_to_turn_map.get(tgt)) and (qid_to_turn_map.get(src) is not None):
            same_turn_removed += 1
            continue
        seen.add(key)
        unique.append(e)
    if same_turn_removed:
        print(f'  ⚠ Sanitizer: removed {same_turn_removed} same-turn Q→Q edge(s)')
    return {'nodes': nodes, 'edges': unique}

def validate_dag(graph: Dict) -> List[str]:
    issues: List[str] = []
    node_ids = {n['id'] for n in graph['nodes']}
    query_ids = {n['id'] for n in graph['nodes'] if n['type'] == 'Query'}
    id_to_turn: Dict[str, int] = {}
    for n in graph['nodes']:
        if n['type'] == 'Query':
            id_to_turn[n['id']] = n['turn']
    id_to_turn['Q0'] = 0
    id_to_turn['Prior_knowledge'] = 0
    id_to_turn['Answer'] = float('inf')
    targets = {e['target'] for e in graph['edges']}
    for qid in sorted(query_ids):
        if qid not in targets:
            issues.append(f'ORPHAN: {qid}')
    for e in graph['edges']:
        src_t = id_to_turn.get(e['source'], -1)
        tgt_t = id_to_turn.get(e['target'], -1)
        if e['target'] == 'Answer':
            pass
        elif src_t > tgt_t and src_t >= 0 and (tgt_t >= 0):
            issues.append(f"BACKWARD: {e['source']}(t={src_t})->{e['target']}(t={tgt_t})")
        if src_t == tgt_t and src_t > 0 and e['source'].startswith('q') and e['target'].startswith('q'):
            issues.append(f"SAME_TURN: {e['source']}->{e['target']} (both turn {src_t})")
        if e['source'] not in node_ids:
            issues.append(f"INVALID_SRC: {e['source']}")
        if e['target'] not in node_ids:
            issues.append(f"INVALID_TGT: {e['target']}")
        ek = e.get('edge_kind', '')
        if ek and ek not in settings.EDGE_KINDS:
            issues.append(f"UNKNOWN_EDGE_KIND: {ek} on {e['source']}->{e['target']}")
    return issues
