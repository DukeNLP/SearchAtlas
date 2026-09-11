"""Exact edge-pair reconstruction scores, macro-averaged across paired DAGs."""

from collections import defaultdict
from statistics import mean

from .graphs import EDGE_KINDS, edge_sets, is_query, validate_graph


def precision_recall_f1(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    if not predicted and not reference:
        return dict(precision=1.0, recall=1.0, f1=1.0)
    hits = len(predicted & reference)
    return dict(precision=hits / len(predicted) if predicted else 0.0,
                recall=hits / len(reference) if reference else 0.0,
                f1=2 * hits / (len(predicted) + len(reference)))


def _check_pair(prediction, reference):
    pred, ref = validate_graph(prediction), validate_graph(reference)
    pnodes = {n['id']: n for n in pred['nodes'] if is_query(n)}
    rnodes = {n['id']: n for n in ref['nodes'] if is_query(n)}
    if pnodes.keys() != rnodes.keys():
        raise ValueError('Prediction/reference query IDs differ; align occurrences before evaluation')
    for qid in pnodes:
        p, r = pnodes[qid], rnodes[qid]
        if p.get('text') and r.get('text') and ' '.join(p['text'].split()) != ' '.join(r['text'].split()):
            raise ValueError('Prediction/reference query texts disagree for the same node ID')
        if p.get('turn') is not None and r.get('turn') is not None and p['turn'] != r['turn']:
            raise ValueError('Prediction/reference query turns disagree')


def _macro(rows):
    return {'n': len(rows), **{key: mean(r[key] for r in rows) if rows else None
                             for key in ('precision', 'recall', 'f1')}}


def reconstruction_results(predictions, references):
    if not predictions or predictions.keys() != references.keys():
        raise ValueError('Prediction and reference case IDs must be nonempty and match exactly')
    detail, typed = [], defaultdict(list)
    for case_id, pred in predictions.items():
        ref = references[case_id]
        _check_pair(pred, ref)
        ps, pk, pc = edge_sets(pred)
        rs, rk, rc = edge_sets(ref)
        row = {'case_id': case_id, **precision_recall_f1(ps, rs),
               'prediction_edges': len(ps), 'reference_edges': len(rs),
               'prediction_kind_conflicts': pc, 'reference_kind_conflicts': rc}
        detail.append(row)
        for kind in EDGE_KINDS:
            typed[kind].append({'case_id': case_id, **precision_recall_f1(pk[kind], rk[kind])})
    return {'macro': _macro(detail), 'per_case': detail,
            'edge_types': {kind: {'macro': _macro(rows), 'per_case': rows} for kind, rows in typed.items()},
            'protocol': {'matching': 'exact_source_target', 'aggregation': 'per_dag_macro',
                         'duplicate_pair_kind': 'first', 'both_empty_score': 1.0}}
