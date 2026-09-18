"""One isolated job process; its stdout/stderr belong to a private log."""

import argparse
import os
from pathlib import Path
import sys

from .io import load_records, read_json, write_json


def evaluate(input_path, annotations_path=None, match_rule=None, reference_path=None):
    from searchatlas.evaluation.constraints import prepare_annotation
    from searchatlas.evaluation.diagnostics import graph_diagnostics
    from searchatlas.evaluation.reconstruction import reconstruction_results
    records = load_records(input_path)
    annotations = load_records(annotations_path) if annotations_path else {}
    if not annotations.keys() <= records.keys():
        raise ValueError('Constraint annotations contain unknown case IDs')
    cases = []
    for case_id, payload in records.items():
        annotation = prepare_annotation(annotations.get(case_id), payload, match_rule)
        cases.append({'case_id': case_id, **graph_diagnostics(payload, annotation)})
    result = {'cases': cases, 'protocol': {
        'pk_source': 'graph_edges', 'constraint_matching': match_rule or 'provided_memberships',
        'missing_annotations': 'grounding_metrics_undefined',
    }}
    if reference_path:
        result['reconstruction'] = reconstruction_results(records, load_records(reference_path))
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    directory = args.config.parent
    if config['kind'] == 'analysis':
        from searchatlas.builder.__main__ import main as build
        argv = ['searchatlas-build', '--agent', config['agent'],
                '--input', str(directory / 'input.json'), '--output', str(directory / 'graph.json'),
                '--task-ids', *config['task_ids'], '--backend', config['backend'],
                '--llm-timeout', str(config['llm_timeout']),
                '--q0-mode', config['q0_mode'],
                '--keep-tool-result', str(config['keep_tool_result']), '--execute']
        if config.get('model'):
            argv.extend(['--model', config['model']])
        # Each worker is a fresh process. The existing builder may use global settings.
        sys.argv = argv
        build()
        graph_path = directory / 'graph.json'
    else:
        graph_path = directory / 'input.json'
    if not graph_path.is_file():
        raise RuntimeError('Builder did not produce a graph artifact')
    records = load_records(graph_path)
    if config['kind'] == 'analysis' and set(records) != set(config['task_ids']):
        raise ValueError('Builder output does not contain every requested task')
    result = evaluate(graph_path,
                      directory / 'annotations.json' if config['has_annotations'] else None,
                      config.get('match_rule'),
                      directory / 'reference.json' if config['has_reference'] else None)
    write_json(directory / 'result.json', result)


if __name__ == '__main__':
    main()
