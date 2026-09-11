"""Evaluate DAG diagnostics, reconstruction, and outcome association offline."""

import argparse
import json
import math
from pathlib import Path

from .constraints import prepare_annotation
from .diagnostics import graph_diagnostics
from .outcomes import outcome_results
from .reconstruction import reconstruction_results


def read_records(path):
    text = path.read_text(encoding='utf-8')
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if path.suffix.lower() != '.jsonl':
            raise
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict) and 'cases' in payload:
        payload = payload['cases']
    records = payload if isinstance(payload, list) else [payload]
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Each record must be an object')
        key = record.get('case_id', record.get('task_id'))
        if not isinstance(key, str) or not key:
            raise ValueError('Each record needs a nonempty string case_id or task_id')
        if key in result:
            raise ValueError('Duplicate case ID; use unique case_id values across agents or runs')
        result[key] = record
    if not result:
        raise ValueError('Input has no records')
    return result


def _diagnostics(args):
    records = read_records(args.input)
    annotations = read_records(args.annotations) if args.annotations else {}
    if not annotations.keys() <= records.keys():
        raise ValueError('Constraint annotations contain unknown case IDs')
    cases = []
    for case_id, payload in records.items():
        annotation = prepare_annotation(annotations.get(case_id), payload, args.match_rule)
        cases.append({'case_id': case_id, **graph_diagnostics(payload, annotation)})
    return {'cases': cases, 'protocol': {'pk_source': 'graph_edges',
                                       'constraint_matching': args.match_rule or 'provided_memberships'}}


def _outcomes(args):
    records, labels = read_records(args.input), read_records(args.labels)
    if records.keys() != labels.keys():
        raise ValueError('Diagnostic and label case IDs must match exactly')
    rows = []
    for case_id, record in records.items():
        label = labels[case_id]
        if args.regime and label.get('regime') != args.regime:
            continue
        if 'correct' not in label:
            raise ValueError('Each label record needs correct (0 or 1)')
        row = {name: label.get(name) for name in ('question_id', 'dataset', 'regime', 'model', 'correct')}
        row['case_id'] = case_id
        for target, source in [('topology', args.topology), ('grounding', args.grounding), ('pk_risk', args.pk_risk)]:
            value = record.get('metrics', {}).get(source)
            if value is None or isinstance(value, (str, bool)) or not math.isfinite(float(value)):
                raise ValueError(f'Selected metric {source} is missing or undefined for a case; no automatic imputation is applied')
            row[target] = round(float(value), args.metric_decimals)
        rows.append(row)
    result = outcome_results(rows, args.folds, args.ap_method)
    result['protocol'].update({'metrics': {'topology': args.topology, 'grounding': args.grounding, 'pk_risk': args.pk_risk},
                               'metric_decimals': args.metric_decimals})
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    diag = commands.add_parser('diagnostics', help='Compute named metrics from DAGs and constraint memberships')
    diag.add_argument('--input', required=True, type=Path)
    diag.add_argument('--annotations', type=Path)
    diag.add_argument('--match-rule', choices=['distinctive_terms', 'token_coverage'],
                      help='Required only when constraint units have no supplied query memberships')
    diag.add_argument('--output', required=True, type=Path)
    recon = commands.add_parser('reconstruction', help='Compute exact edge-pair P/R/F1 against reference DAGs')
    recon.add_argument('--predictions', required=True, type=Path)
    recon.add_argument('--references', required=True, type=Path)
    recon.add_argument('--output', required=True, type=Path)
    outcomes = commands.add_parser('outcomes', help='Compute within-agent held-out AUC/AP and auxiliary threshold results')
    outcomes.add_argument('--input', required=True, type=Path, help='Output from diagnostics')
    outcomes.add_argument('--labels', required=True, type=Path)
    outcomes.add_argument('--regime', choices=['sequential', 'parallel'], help='Optionally select one regime')
    outcomes.add_argument('--topology', required=True, help='Named topology metric to use')
    outcomes.add_argument('--grounding', required=True, help='Named grounding metric to use')
    outcomes.add_argument('--pk-risk', required=True, help='Named PK metric to subtract')
    outcomes.add_argument('--folds', type=int, default=5)
    outcomes.add_argument('--ap-method', choices=['rank', 'grouped'], default='rank')
    outcomes.add_argument('--metric-decimals', type=int, choices=range(0, 16), default=6)
    outcomes.add_argument('--output', required=True, type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == 'diagnostics':
            result = _diagnostics(args)
        elif args.command == 'reconstruction':
            result = reconstruction_results(read_records(args.predictions), read_records(args.references))
        else:
            result = _outcomes(args)
        encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
