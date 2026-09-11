"""Derive named metrics from graph structure and explicit constraint deployments."""

from collections import defaultdict

from .graphs import is_query, topological_order, validate_graph


def _memberships(annotation, query_ids):
    if annotation is None:
        return None, None
    units = annotation.get('constraint_units')
    mapping = annotation.get('query_unit_ids')
    if not isinstance(units, list) or not isinstance(mapping, dict):
        raise ValueError('Constraint annotations need constraint_units and query_unit_ids')
    unit_ids = []
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get('unit_id'), str) or not unit['unit_id']:
            raise ValueError('Each constraint unit needs a nonempty string unit_id')
        unit_ids.append(unit['unit_id'])
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError('Duplicate constraint unit_id')
    if set(mapping) != set(query_ids):
        raise ValueError('query_unit_ids must list every query, including queries with no deployments')
    result = {}
    for query, deployed in mapping.items():
        if not isinstance(deployed, list) or any(not isinstance(unit, str) for unit in deployed):
            raise ValueError('Each deployment must be a list of unit IDs')
        if not set(deployed) <= set(unit_ids):
            raise ValueError('Deployment references an unknown constraint unit')
        if len(deployed) != len(set(deployed)):
            raise ValueError('Duplicate unit deployment at a query')
        result[query] = set(deployed)
    return result, set(unit_ids)


def graph_diagnostics(payload, annotation=None):
    """No correctness labels, saved risks, or case-specific overrides are read."""
    graph = validate_graph(payload)
    queries = {node['id'] for node in graph['nodes'] if is_query(node)}
    edges = graph['edges']
    evidence = {(e['source'], e['target']) for e in edges if e['edge_kind'] == 'evidence_derived'}
    qq = {(u, v) for u, v in evidence if u in queries and v in queries}
    frontier = {u for u, v in evidence if u in queries and v == 'Answer'}
    parents, children = defaultdict(set), defaultdict(set)
    for u, v in qq:
        parents[v].add(u)
        children[u].add(v)
    reached = set(frontier)
    stack = list(frontier)
    while stack:
        for parent in parents[stack.pop()]:
            if parent not in reached:
                reached.add(parent)
                stack.append(parent)
    answer_edges = {(u, v) for u, v in qq if u in reached and v in reached}
    ordered = topological_order(reached, answer_edges)
    forward = dict.fromkeys(reached, 0)
    for u in ordered:
        for v in children[u] & reached:
            forward[v] = max(forward[v], forward[u] + 1)
    depth = max((forward[u] for u in frontier), default=0)
    backward = {u: 0 if u in frontier else -1 for u in reached}
    for u in reversed(ordered):
        for v in children[u] & reached:
            if backward[v] >= 0:
                backward[u] = max(backward[u], backward[v] + 1)
    backbone = {(u, v) for u, v in answer_edges if forward[u] + 1 + backward[v] == depth}
    primary = {u for edge in backbone for u in edge} or frontier
    units, unit_ids = _memberships(annotation, queries)

    def coverage(selected):
        if units is None:
            return None
        covered = set().union(*(units[q] for q in selected))
        return len(covered) / len(unit_ids) if unit_ids else 0.0

    deployments = sum(len(units[q]) for q in reached) if units is not None else None
    backbone_grounding = None if units is None else (
        sum(len(units[q]) for q in primary) / deployments if deployments else 0.0)
    pk_targets = {e['target'] for e in edges
                  if e['source'] == 'Prior_knowledge' and e['edge_kind'] == 'prior_knowledge_derived'}
    pk_answer = int('Answer' in pk_targets)
    metrics = {
        'answer_backbone_concentration': len(backbone) / len(qq) if qq else 0.0,
        'answer_backbone_concentration_within_answer_graph': len(backbone) / len(answer_edges) if answer_edges else 0.0,
        'answer_support_directness': len(frontier) / (len(frontier) + len(answer_edges)) if frontier else 0.0,
        'backbone_constraint_share': backbone_grounding,
        'primary_path_coverage': coverage(primary),
        'direct_frontier_coverage': coverage(frontier),
        'answer_graph_coverage': coverage(reached),
        'trajectory_coverage': coverage(queries),
        'pk_answer_edge': pk_answer,
        'answer_graph_pk_density': (len(pk_targets & reached) + pk_answer) / len(reached) if reached else None,
        'direct_frontier_pk_density': (len(pk_targets & frontier) + pk_answer) / len(frontier) if frontier else None,
        'query_efficiency': len(reached) / len(queries) if queries else 0.0,
    }
    return {
        'metrics': metrics,
        'counts': {
            'nodes': len(graph['nodes']), 'query_nodes': len(queries),
            'edge_records': len(edges), 'edge_pairs': len({(e['source'], e['target']) for e in edges}),
            'query_evidence_edges': len(qq), 'answer_reaching_queries': len(reached),
            'direct_answer_parents': len(frontier), 'answer_backbone_edges': len(backbone),
            'off_answer_queries': len(queries - reached), 'answer_query_depth': depth,
            'constraint_units': None if unit_ids is None else len(unit_ids),
        },
        'support_sets': {
            'answer_frontier': sorted(frontier), 'answer_reaching_queries': sorted(reached),
            'primary_path_queries': sorted(primary),
            'backbone_edges': [list(edge) for edge in sorted(backbone)],
        },
    }
