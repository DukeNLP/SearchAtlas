"""Write fictional fixtures demonstrating the offline evaluation commands."""

import argparse
from copy import deepcopy
import json
from pathlib import Path


def make_demo():
    graphs, references, annotations, labels = [], [], [], []
    for agent in ('ExampleAgent-A', 'ExampleAgent-B'):
        for i in range(6):
            case_id = f'{agent}-Example-{i + 1}'
            nodes = [{'id': 'Q0'}, {'id': 'Prior_knowledge'}, {'id': 'Answer'}]
            nodes += [{'id': 'q1', 'type': 'Query', 'turn': 1, 'text': 'Example Hall opening year'},
                      {'id': 'q2', 'type': 'Query', 'turn': 2, 'text': 'Example Hall 1942 confirmation'}]
            edges = [{'source': 'Q0', 'target': 'q1', 'edge_kind': 'constraint_use'},
                     {'source': 'q1', 'target': 'q2', 'edge_kind': 'evidence_derived'},
                     {'source': 'q2', 'target': 'Answer', 'edge_kind': 'evidence_derived'}]
            reference = {'case_id': case_id, 'graph': {'nodes': nodes, 'edges': edges}}
            prediction = deepcopy(reference)
            # These labels and graphs are deliberately illustrative, not model outputs.
            if i % 2 == 0:
                prediction['graph']['edges'][-1] = {
                    'source': 'Prior_knowledge', 'target': 'Answer', 'edge_kind': 'prior_knowledge_derived'}
            graphs.append(prediction)
            references.append(reference)
            annotations.append({'case_id': case_id,
                                'constraint_units': [{'unit_id': 'u1', 'text': 'Example Hall'},
                                                     {'unit_id': 'u2', 'text': 'opening year'}],
                                'query_unit_ids': {'q1': ['u1'], 'q2': ['u2']}})
            labels.append({'case_id': case_id, 'question_id': f'Example-{i + 1}',
                           'dataset': 'FictionalDemo', 'regime': 'sequential',
                           'model': agent, 'correct': i % 2})
    return {'graphs.json': graphs, 'references.json': references,
            'constraints.json': annotations, 'labels.json': labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in make_demo().items():
        (args.output / name).write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'Fictional fixtures: {args.output}')


if __name__ == '__main__':
    main()
