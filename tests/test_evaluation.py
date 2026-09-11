"""Hand-calculated graph fixtures and offline evaluation regression tests."""

import contextlib
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from searchatlas.evaluation.__main__ import main, read_records
from searchatlas.evaluation.constraints import prepare_annotation, unit_matches
from searchatlas.evaluation.diagnostics import graph_diagnostics
from searchatlas.evaluation.graphs import edge_sets, validate_graph
from searchatlas.evaluation.outcomes import (
    STAGES, average_precision, best_threshold, binary_results, heldout_predictions,
    heldout_scores, normalized_scores, outcome_results, question_folds, roc_auc,
)
from searchatlas.evaluation.reconstruction import precision_recall_f1, reconstruction_results


def edge(source, target, kind='evidence_derived'):
    return dict(source=source, target=target, edge_kind=kind)


def fixture():
    graph = {'nodes': [{'id': 'Q0'}, {'id': 'Prior_knowledge'}, {'id': 'Answer'}]
             + [{'id': f'q{i}', 'type': 'Query', 'text': f'Example query {i}'} for i in range(1, 6)],
             'edges': [edge('q1', 'q2'), edge('q2', 'Answer'), edge('q3', 'Answer'),
                       edge('q4', 'q5'), edge('Prior_knowledge', 'q1', 'prior_knowledge_derived'),
                       edge('Prior_knowledge', 'Answer', 'prior_knowledge_derived')]}
    annotations = {'constraint_units': [{'unit_id': f'u{i}'} for i in range(1, 4)],
                   'query_unit_ids': {'q1': ['u1'], 'q2': ['u2'], 'q3': ['u2'], 'q4': ['u3'], 'q5': []}}
    return graph, annotations


def outcome_rows():
    return [dict(case_id=f'Example-{i}', question_id=f'Question-{i}', dataset='Example',
                 regime='sequential', model='ExampleAgent', correct=i % 2,
                 topology=float(i % 2), grounding=float(i % 2), pk_risk=float(1 - i % 2))
            for i in range(6)]


class DiagnosticsTests(unittest.TestCase):
    def test_named_scopes_match_hand_computation(self):
        graph, annotations = fixture()
        result = graph_diagnostics(graph, annotations)
        values = result['metrics']
        expected = {'answer_backbone_concentration': 0.5,
                    'answer_backbone_concentration_within_answer_graph': 1,
                    'answer_support_directness': 2 / 3, 'backbone_constraint_share': 2 / 3,
                    'primary_path_coverage': 2 / 3, 'direct_frontier_coverage': 1 / 3,
                    'trajectory_coverage': 1, 'answer_graph_coverage': 2 / 3,
                    'pk_answer_edge': 1, 'answer_graph_pk_density': 2 / 3,
                    'direct_frontier_pk_density': 0.5, 'query_efficiency': 3 / 5}
        for name, value in expected.items():
            self.assertAlmostEqual(values[name], value, msg=name)
        self.assertEqual(result['counts']['off_answer_queries'], 2)

    def test_node_names_do_not_determine_topological_order(self):
        graph, annotations = fixture()
        aliases = {'q1': 'q99', 'q2': 'q1'}
        for node in graph['nodes']:
            node['id'] = aliases.get(node['id'], node['id'])
        for item in graph['edges']:
            for key in ('source', 'target'):
                item[key] = aliases.get(item[key], item[key])
        annotations['query_unit_ids'] = {aliases.get(q, q): u for q, u in annotations['query_unit_ids'].items()}
        self.assertEqual(graph_diagnostics(graph, annotations)['metrics']['answer_backbone_concentration'], 0.5)

    def test_backbone_is_union_of_equal_longest_paths(self):
        graph, _ = fixture()
        graph['edges'].append(edge('q3', 'q2'))
        result = graph_diagnostics(graph)
        self.assertEqual(result['support_sets']['backbone_edges'], [['q1', 'q2'], ['q3', 'q2']])

    def test_duplicate_edges_do_not_inflate_metrics(self):
        graph, annotations = fixture()
        original = graph_diagnostics(graph, annotations)['metrics']
        graph['edges'] += deepcopy(graph['edges'])
        self.assertEqual(graph_diagnostics(graph, annotations)['metrics'], original)

    def test_pk_density_can_exceed_one(self):
        graph = {'nodes': [{'id': 'q1'}, {'id': 'Answer'}, {'id': 'Prior_knowledge'}],
                 'edges': [edge('q1', 'Answer'), edge('Prior_knowledge', 'q1', 'prior_knowledge_derived'),
                           edge('Prior_knowledge', 'Answer', 'prior_knowledge_derived')]}
        self.assertEqual(graph_diagnostics(graph)['metrics']['answer_graph_pk_density'], 2)

    def test_no_retrieval_denominator_is_undefined_not_zero_risk(self):
        graph = {'nodes': [{'id': 'Answer'}, {'id': 'Prior_knowledge'}],
                 'edges': [edge('Prior_knowledge', 'Answer', 'prior_knowledge_derived')]}
        result = graph_diagnostics(graph)['metrics']
        self.assertEqual(result['pk_answer_edge'], 1)
        self.assertIsNone(result['answer_graph_pk_density'])

    def test_missing_units_are_not_zero_coverage(self):
        graph, _ = fixture()
        self.assertIsNone(graph_diagnostics(graph)['metrics']['primary_path_coverage'])

    def test_bad_memberships_fail_loudly(self):
        graph, annotations = fixture()
        for mapping in ({'q1': ['u1']}, {**annotations['query_unit_ids'], 'q2': ['unknown']},
                        {**annotations['query_unit_ids'], 'q2': ['u2', 'u2']}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                graph_diagnostics(graph, {**annotations, 'query_unit_ids': mapping})

    def test_cycles_and_missing_endpoints_are_rejected(self):
        graph, _ = fixture()
        for extra in (edge('q2', 'q1'), edge('missing', 'q1')):
            invalid = deepcopy(graph)
            invalid['edges'].append(extra)
            with self.assertRaises(ValueError):
                validate_graph(invalid)

    def test_outcome_fields_do_not_affect_diagnostics(self):
        graph, annotations = fixture()
        expected = graph_diagnostics(graph, annotations)
        result = graph_diagnostics({'graph': graph, 'correct': 0, 'answer_context_risk': 0}, annotations)
        self.assertEqual(result, expected)
        with self.assertRaises(ValueError):
            prepare_annotation({**annotations, 'answer_context_risk': 0}, graph)

    def test_lexical_rule_requires_explicit_selection(self):
        graph = {'nodes': [{'id': 'q1', 'text': 'Example Hall opening year 1942'}], 'edges': []}
        annotation = {'constraint_units': [{'unit_id': 'u1', 'text': 'Example Hall'}]}
        with self.assertRaises(ValueError):
            prepare_annotation(annotation, graph)
        result = prepare_annotation(annotation, graph, 'distinctive_terms')
        self.assertEqual(result['query_unit_ids'], {'q1': ['u1']})
        self.assertFalse(unit_matches('Mexican restaurant in 1942', '1942', 'distinctive_terms'))
        self.assertTrue(unit_matches('Example Hall', 'Example Hall', 'token_coverage'))


class ReconstructionTests(unittest.TestCase):
    def test_empty_set_conventions(self):
        self.assertEqual(precision_recall_f1(set(), set())['f1'], 1)
        self.assertEqual(precision_recall_f1({'edge'}, set())['f1'], 0)
        self.assertEqual(precision_recall_f1(set(), {'edge'})['f1'], 0)

    def test_macro_f1_is_average_of_per_graph_f1(self):
        graph, _ = fixture()
        one = {**graph, 'edges': [edge('q1', 'q2')]}
        two = {**graph, 'edges': [edge('q1', 'q2'), edge('q2', 'Answer')]}
        result = reconstruction_results({'a': two, 'b': one}, {'a': one, 'b': two})
        self.assertEqual(result['macro']['precision'], 0.75)
        self.assertEqual(result['macro']['recall'], 0.75)
        self.assertAlmostEqual(result['macro']['f1'], 2 / 3)
        self.assertEqual(result['edge_types']['constraint_use']['macro']['f1'], 1)

    def test_duplicate_kind_policy_is_explicit(self):
        graph, _ = fixture()
        graph['edges'].append(edge('q1', 'q2', 'failure_derived'))
        pairs, kinds, conflicts = edge_sets(graph)
        self.assertEqual(conflicts, 1)
        self.assertIn(('q1', 'q2'), kinds['evidence_derived'])
        self.assertNotIn(('q1', 'q2'), kinds['failure_derived'])
        self.assertEqual(len(pairs), 6)

    def test_pairing_does_not_silently_remap_queries(self):
        graph, _ = fixture()
        reference = deepcopy(graph)
        reference['nodes'][3]['text'] = 'Different query occurrence'
        with self.assertRaises(ValueError):
            reconstruction_results({'a': graph}, {'a': reference})
        with self.assertRaises(ValueError):
            reconstruction_results({'a': graph}, {'b': graph})


class OutcomeTests(unittest.TestCase):
    def test_auc_matches_pairwise_definition(self):
        labels, scores = [1, 0, 1, 0], [0.8, 0.8, 0.5, 0.1]
        expected = sum((a > b) + 0.5 * (a == b) for a, y in zip(scores, labels) if y
                       for b, z in zip(scores, labels) if not z) / 4
        self.assertEqual(roc_auc(labels, scores), expected)

    def test_ap_tie_conventions(self):
        self.assertEqual(average_precision([1, 0], [1, 1], 'rank'), 1)
        self.assertEqual(average_precision([1, 0], [1, 1], 'grouped'), 0.5)
        self.assertEqual(average_precision([0, 1], [1, 1], 'rank'), 0.5)

    def test_undefined_and_invalid_ranking_inputs(self):
        self.assertIsNone(roc_auc([1, 1], [1, 2]))
        self.assertIsNone(average_precision([0, 0], [1, 2]))
        for labels, scores in (([], []), ([1], []), ([2], [1]), ([1], [math.nan])):
            with self.assertRaises(ValueError):
                roc_auc(labels, scores)

    def test_population_sd_and_train_only_normalization(self):
        self.assertEqual(normalized_scores([{'x': 0}, {'x': 2}], [{'x': 3}], [('x', 1)]), [2])

    def test_question_grouping_and_label_independent_scores(self):
        rows = outcome_rows()
        rows.append({**rows[0], 'case_id': 'Example-repeat'})
        folds = question_folds(rows)
        self.assertEqual(len(folds), 6)
        scores = heldout_scores(rows, STAGES['+ PK reliance'])
        relabeled = [{**row, 'correct': 1 - row['correct']} for row in rows]
        self.assertEqual(heldout_scores(relabeled, STAGES['+ PK reliance']), scores)

    def test_threshold_uses_only_training_question_labels(self):
        rows = outcome_rows()
        before = {r['case_id']: r for r in heldout_predictions(rows)}
        rows[2]['correct'] = 1 - rows[2]['correct']
        after = {r['case_id']: r for r in heldout_predictions(rows)}
        self.assertEqual(before[rows[2]['case_id']]['threshold'], after[rows[2]['case_id']]['threshold'])
        self.assertEqual(before[rows[2]['case_id']]['prediction'], after[rows[2]['case_id']]['prediction'])
        self.assertIsNone(best_threshold([1, 1], [0, 1]))

    def test_within_agent_and_macro_outputs(self):
        first = outcome_rows()
        second = [{**row, 'case_id': row['case_id'] + '-second', 'model': 'OtherAgent'} for row in first]
        result = outcome_results(first + second)
        final = next(r for r in result['macro'] if r['stage'] == '+ PK reliance')
        self.assertEqual(final['models'], 2)
        self.assertEqual(final['auc'], 1)
        self.assertEqual(len(result['thresholds']), 2)
        self.assertTrue(all(r['n_evaluated'] == 6 for r in result['thresholds']))
        self.assertTrue(all(r['metrics']['f1'] == 1 for r in result['thresholds']))

    def test_binary_results_hand_computation(self):
        result = binary_results([1, 1, 0, 0], [1, 0, 1, 0])
        for key in ('accuracy', 'precision', 'recall', 'f1', 'auc', 'ap', 'balanced_accuracy'):
            self.assertEqual(result[key], 0.5)


class EvaluationCliTests(unittest.TestCase):
    def test_builder_shaped_payload_and_reconstruction_cli(self):
        graph, annotation = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inp, ann, output, recon = (root / name for name in ('graphs.json', 'units.json', 'metrics.json', 'recon.json'))
            inp.write_text(json.dumps([{'task_id': 'Example', 'graph': graph}]))
            ann.write_text(json.dumps([{'case_id': 'Example', **annotation}]))
            with patch.object(sys, 'argv', ['evaluate', 'diagnostics', '--input', str(inp), '--annotations', str(ann), '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(read_records(output)['Example']['metrics']['primary_path_coverage'], 2 / 3)
            with patch.object(sys, 'argv', ['evaluate', 'reconstruction', '--predictions', str(inp), '--references', str(inp), '--output', str(recon)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(json.loads(recon.read_text())['macro']['f1'], 1)

    def test_outcomes_cli(self):
        rows = outcome_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inp, labels, output = (root / name for name in ('metrics.json', 'labels.json', 'outcomes.json'))
            inp.write_text(json.dumps({'cases': [{'case_id': r['case_id'], 'metrics': {k: r[k] for k in ('topology', 'grounding', 'pk_risk')}} for r in rows]}))
            labels.write_text(json.dumps(rows))
            argv = ['evaluate', 'outcomes', '--input', str(inp), '--labels', str(labels), '--topology', 'topology', '--grounding', 'grounding', '--pk-risk', 'pk_risk', '--output', str(output)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(json.loads(output.read_text())['protocol']['metric_decimals'], 6)

    def test_duplicate_cases_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.json'
            path.write_text(json.dumps([{'case_id': 'Example'}, {'case_id': 'Example'}]))
            with self.assertRaises(ValueError):
                read_records(path)


if __name__ == '__main__':
    unittest.main()
