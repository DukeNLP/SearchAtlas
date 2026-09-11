"""Prune query parents while preserving grounded signal coverage."""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
from . import settings

def compute_used_signals(token_status: Dict[str, str], think_support: Dict[str, List[Dict[str, Any]]], candidates_by_token: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    """Exclude Q0-only carryover and unsupported query-text-only guesses."""
    return {tok for tok, status in token_status.items()
            if status != 'Q0_ONLY' and (think_support.get(tok) or candidates_by_token.get(tok))}

def compute_parent_coverage(used_signals: Set[str], candidates_by_token: Dict[str, List[Dict[str, Any]]], think_support: Dict[str, List[Dict[str, Any]]], token_status: Dict[str, str]) -> Dict[str, Set[str]]:
    """PK requires a new signal introduced in reasoning without prior retrieval."""
    coverage = defaultdict(set)
    for token in used_signals:
        sources = candidates_by_token.get(token, [])
        if sources:
            for source in sources:
                if source.get('source_qid'):
                    coverage[source['source_qid']].add(token)
        elif token_status.get(token) == 'NEW' and think_support.get(token):
            coverage['Prior_knowledge'].add(token)
    return dict(coverage)

def find_first_provenance_parent(token: str, candidates_by_token: Dict[str, List[Dict[str, Any]]], think_text: str) -> Optional[str]:
    """
    For a token with multiple provenance hits, return the first-provenance parent
    (earliest turn, lowest qid number) UNLESS think explicitly says
    'previous/last search found X' => use latest.
    """
    sources = candidates_by_token.get(token, [])
    if not sources:
        return None
    use_latest = False
    if think_text:
        for pat in settings.LATEST_REF_PATTERNS:
            for m in pat.finditer(think_text):
                context = think_text[max(0, m.start() - 150):m.end() + 150]
                if re.search(f'\\b{re.escape(token)}\\b', context, re.I):
                    use_latest = True
                    break
            if use_latest:
                break
    if use_latest:
        sources_sorted = sorted(sources, key=lambda s: (s.get('source_turn', 0), s.get('source_qid', '')), reverse=True)
    else:
        sources_sorted = sorted(sources, key=lambda s: (s.get('source_turn', 0), s.get('source_qid', '')))
    return sources_sorted[0].get('source_qid') if sources_sorted else None

def apply_mpsc(candidate_edges: List[Dict], target_qid: str, parent_coverage: Dict[str, Set[str]], used_signals: Set[str], qid_to_turn: Optional[Dict[str, int]]=None) -> List[Dict]:
    """Prune validated candidate edges; never invent parents to fill a gap.

    Greedy cover prefers the most recent query at equal coverage and gives PK
    the lowest tie priority. Failure-response edges are not content coverage.
    """
    other = [edge for edge in candidate_edges if edge.get('target') != target_qid]
    target = [edge for edge in candidate_edges if edge.get('target') == target_qid]
    failures = [edge for edge in target if edge.get('edge_kind') == 'failure_derived']
    support = [edge for edge in target if edge.get('edge_kind') != 'failure_derived']
    remaining = {edge['source']: set(parent_coverage.get(edge['source'], set())) & used_signals
                 for edge in support}
    selected, covered = set(), set()
    while remaining and covered != used_signals:
        def key(parent):
            qnum = int(parent[1:]) if parent.startswith('q') and parent[1:].isdigit() else 0
            return (-len(remaining[parent] - covered), parent == 'Prior_knowledge',
                    -(qid_to_turn or {}).get(parent, 0), -qnum)
        best = min(remaining, key=key)
        if not remaining[best] - covered:
            break
        selected.add(best)
        covered.update(remaining.pop(best))
    kept = []
    for edge in support:
        if edge['source'] in selected:
            edge = dict(edge)
            metadata = dict(edge.get('metadata') or {})
            metadata['edge_policy'] = {
                'policy': settings.EDGE_POLICY_VERSION,
                'mpsc_gate': 'validated_candidate_with_parent_coverage',
            }
            edge['metadata'] = metadata
            kept.append(edge)
    return other + failures + kept
