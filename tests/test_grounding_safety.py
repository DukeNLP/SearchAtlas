"""Synthetic regression tests for provenance semantics; no external API calls."""
import contextlib
from collections import Counter
import importlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from searchatlas.builder.safety import AttributionUnavailable, validate_tasks


def tool(name, arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


def search(*queries):
    return {'role': 'assistant', 'content': tool('search', {'query': list(queries)})}


def reply(text):
    return {'role': 'user', 'content': '<tool_response>\n' + text + '\n</tool_response>'}


def results(query, text, count=1):
    return f"A Google search for '{query}' found {count} results:\n{text}"


def visit(url, text):
    return f'The useful information in {url} for user goal check as follows:\n{text}'


class GroundingSafetyTests(unittest.TestCase):
    adapters = ('standard', 'miro')

    def module(self, adapter, name):
        return importlib.import_module(f'searchatlas.builder.{adapter}.{name}')

    def extract(self, adapter, messages):
        queries, turns = self.module(adapter, 'parsing').extract_queries(messages)
        with contextlib.redirect_stdout(io.StringIO()):
            extracted = self.module(adapter, 'evidence').extract_query_docs(
                messages, turns, {q['id']: q['text'] for q in queries})
        return queries, extracted

    def test_visit_search_shaped_reply_is_not_a_query(self):
        messages = [search('alpha'), reply(results('alpha', 'First result.')),
                    {'role': 'assistant', 'content': tool('visit', {'url': ['https://example.org/alpha']})},
                    reply(results('fabricated-query', 'Visit response.'))]
        for adapter in self.adapters:
            with self.subTest(adapter=adapter):
                queries, extracted = self.extract(adapter, messages)
                self.assertEqual([q['text'] for q in queries], ['alpha'])
                self.assertNotIn('Visit response.', extracted[0].get('q1', ''))

    def test_miro_xml_visit_json_reply_is_not_a_query(self):
        content = ('<use_mcp_tool><tool_name>scrape_and_extract_info</tool_name>'
                   '<arguments>{"url":"https://example.org/source"}</arguments></use_mcp_tool>')
        messages = [{'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': json.dumps({'searchParameters': {'q': 'not issued'}, 'organic': []})}]
        queries, _ = self.module('miro', 'parsing').extract_queries(messages)
        self.assertEqual(queries, [])

    def test_multiple_calls_in_one_message_share_one_turn(self):
        messages = [{'role': 'assistant', 'content': tool('search', {'query': 'alpha'}) + tool('search', {'query': 'beta'})},
                    reply(results('alpha', 'A.') + '\n' + results('beta', 'B.'))]
        for adapter in self.adapters:
            queries, extracted = self.extract(adapter, messages)
            self.assertEqual([q['turn'] for q in queries], [1, 1])
            self.assertEqual(set(extracted[-1]), {1})

    def test_mixed_tool_call_order_and_repeated_queries_preserved(self):
        mcp = '<use_mcp_tool><tool_name>google_search</tool_name><arguments>{"q":"beta"}</arguments></use_mcp_tool>'
        messages = [{'role': 'assistant', 'content': mcp + tool('search', {'query': ['alpha', 'alpha']})}]
        for adapter in self.adapters:
            qs, _ = self.module(adapter, 'parsing').extract_queries(messages)
            self.assertEqual([q['text'] for q in qs], ['beta', 'alpha', 'alpha'])
            self.assertEqual([q['turn'] for q in qs], [1, 1, 1])

    def test_unmatched_header_not_assigned_by_position_or_to_future_query(self):
        for header in ('unrelated', 'beta'):
            messages = [search('alpha'), reply(results(header, 'UNASSIGNED_FACT')), search('beta')]
            for adapter in self.adapters:
                _, extracted = self.extract(adapter, messages)
                self.assertNotIn('UNASSIGNED_FACT', '\n'.join(extracted[0].values()))

    def test_ambiguous_batch_result_not_broadcast(self):
        messages = [search('alpha', 'beta'), reply('UNKNOWN_OWNER_FACT')]
        for adapter in self.adapters:
            _, extracted = self.extract(adapter, messages)
            self.assertNotIn('UNKNOWN_OWNER_FACT', '\n'.join(extracted[0].values()))

    def test_single_explicit_search_can_own_headerless_reply(self):
        for adapter in self.adapters:
            _, extracted = self.extract(adapter, [search('alpha'), reply('SINGLE_OWNER_FACT')])
            self.assertIn('SINGLE_OWNER_FACT', extracted[0]['q1'])

    def test_exact_query_match_only_and_duplicate_headers_ambiguous(self):
        for adapter in self.adapters:
            match = self.module(adapter, 'evidence').match_tool_response_query_to_qid
            self.assertIsNone(match('alpha beta', ['q1'], {'q1': 'alpha'}))
            self.assertIsNone(match('alpha', ['q1', 'q2'], {'q1': 'alpha', 'q2': 'alpha'}))
            self.assertEqual(match(' ALPHA  beta ', ['q1'], {'q1': 'alpha beta'}), 'q1')

    def test_search_and_visit_blocks_do_not_contaminate_each_other(self):
        content = results('alpha', 'SEARCH_ONLY') + '\n' + visit('https://example.org/v', 'VISIT_ONLY')
        for adapter in self.adapters:
            evidence = self.module(adapter, 'evidence')
            self.assertNotIn('VISIT_ONLY', evidence.split_search_tool_response_blocks(content)[0]['sanitized_block'])
            self.assertNotIn('SEARCH_ONLY', evidence.split_visit_tool_response_blocks(content)[0]['sanitized_block'])

    def test_later_visit_absent_from_earlier_attribution_and_failure_context(self):
        url = 'https://example.org/source'
        messages = [search('alpha'), reply(results('alpha', f'Early fact. {url}')),
                    search('beta'), reply(results('beta', 'Other fact.')),
                    {'role': 'assistant', 'content': tool('visit', {'url': [url]})},
                    reply(visit(url, 'LATER_FACT. The page returned HTTP 404.')),
                    search('gamma')]
        for adapter in self.adapters:
            with self.subTest(adapter=adapter):
                _, extracted = self.extract(adapter, messages)
                contexts = extracted[-1]
                self.assertNotIn('LATER_FACT', '\n'.join(contexts[2]['docs'].values()))
                self.assertIn('LATER_FACT', contexts[3]['docs']['q1'])
                self.assertFalse(contexts[2]['visit_failures'].get('q1'))
                self.assertTrue(contexts[3]['visit_failures'].get('q1'))

    def test_miro_all_history_excludes_future_results(self):
        parser = self.module('miro', 'parsing')
        self.assertEqual(parser.compute_visible_tool_message_indices(3, -1), {1, 2, 3})
        self.assertEqual(parser.compute_visible_tool_message_indices(3, 2), {2, 3})
        self.assertEqual(parser.compute_visible_tool_message_indices(0, -1), set())
        with self.assertRaises(ValueError):
            parser.compute_visible_tool_message_indices(3, -2)
        evidence = self.module('miro', 'evidence')
        hits = evidence.token_provenance_in_prior_docs(
            'futurefact', [{'id': 'q1', 'turn': 1}],
            {'q1': [{'kind': 'tool', 'label': '[Tool_Response]', 'text': 'futurefact', 'tool_message_idx': 5}]},
            parser.compute_visible_tool_message_indices(3, -1), 1, 10)
        self.assertEqual(hits, [])

    def test_gold_answer_is_never_used_as_fallback(self):
        for adapter in self.adapters:
            extract = self.module(adapter, 'answers').extract_final_answer
            self.assertEqual(extract([], {'answer': 'SECRET_GOLD'}), '')
            self.assertEqual(extract([search('alpha')], {'answer': 'SECRET_GOLD'}), '')
            self.assertEqual(extract([{'role': 'assistant', 'content': '<think>guess</think>'}], {'answer': 'SECRET_GOLD'}), '')
            self.assertEqual(extract([{'role': 'assistant', 'content': 'Agent choice'}], {'answer': 'SECRET_GOLD'}), 'Agent choice')
            self.assertEqual(extract([], {'prediction': 'Agent choice', 'answer': 'SECRET_GOLD'}), 'Agent choice')

    def test_minimal_answer_cannot_introduce_new_content(self):
        original = 'This is a long explanation. ' * 20 + 'The answer is Orion.'
        for adapter in self.adapters:
            with patch.object(self.module(adapter, 'attribution'), 'call_llm_json', return_value={'minimal_answer': 'Invented answer'}):
                self.assertEqual(self.module(adapter, 'answers').extract_minimal_answer('Question', original), original)

    def test_token_only_negative_evidence_cannot_create_answer_edge(self):
        signals = [{'signal_id': 't1', 'text': 'Orion', 'granularity': 'token', 'unit_ids': ['a1']}]
        for adapter in self.adapters:
            answers = self.module(adapter, 'answers')
            with patch.object(answers, '_llm_answer_support_verdicts', side_effect=AssertionError('Token must not make a support call')):
                coverage = answers.answer_signal_provenance_matching(signals, [{'id': 'q1', 'text': 'candidate'}],
                    {'q1': '[Tool_Response]\nOrion is NOT the correct candidate.'})
            edges, _ = answers.answer_mpsc(signals, coverage)
            self.assertEqual(edges, [])

    def test_negative_claim_verdict_not_overridden_by_token_hit(self):
        signals = [{'signal_id': 'a1', 'text': 'Orion', 'granularity': 'unit_claim', 'unit_ids': ['a1']},
                   {'signal_id': 't1', 'text': 'Orion', 'granularity': 'token', 'unit_ids': ['a1']}]
        def reject(_q, _analysis, pairs):
            return {p['pair_id']: {'supports': False, 'confidence': 'high', 'evidence_sentence': p['window_text']} for p in pairs}
        for adapter in self.adapters:
            answers = self.module(adapter, 'answers')
            with patch.object(answers, '_llm_answer_support_verdicts', side_effect=reject):
                coverage = answers.answer_signal_provenance_matching(signals, [{'id': 'q1', 'text': 'candidate'}],
                    {'q1': '[Tool_Response]\nOrion is NOT the correct candidate.'})
            self.assertEqual(answers.answer_mpsc(signals, coverage)[0], [])

    def test_fabricated_evidence_quote_cannot_support_answer(self):
        def fabricate(_q, _analysis, pairs):
            return {p['pair_id']: {'supports': True, 'confidence': 'high', 'evidence_sentence': 'Invented quotation.'} for p in pairs}
        for adapter in self.adapters:
            answers = self.module(adapter, 'answers')
            with patch.object(answers, '_llm_answer_support_verdicts', side_effect=fabricate):
                coverage = answers.answer_signal_provenance_matching(
                    [{'signal_id': 'a1', 'text': 'Orion', 'granularity': 'unit_claim', 'unit_ids': ['a1']}],
                    [{'id': 'q1', 'text': 'candidate'}], {'q1': '[Tool_Response]\nOrion is a candidate.'})
            self.assertEqual(coverage, {})

    def test_distributed_atomic_support_does_not_require_whole_phrase_hit(self):
        units = [{'unit_id': 'a1', 'claim': 'Orion'}, {'unit_id': 'a2', 'claim': '1942'}]
        signals = [{'signal_id': u['unit_id'], 'text': u['claim'], 'granularity': 'unit_claim', 'unit_ids': [u['unit_id']]} for u in units]
        coverage = {f'q{i}': {'supports_signals': [s['signal_id']], 'provenance': [dict(s,
                    match_confidence='high', document_support=True, support_mode='direct',
                    hit_sentence=s['text'], window_text=s['text'])]} for i, s in enumerate(signals, 1)}
        for adapter in self.adapters:
            answers = self.module(adapter, 'answers')
            edges, _ = answers.answer_mpsc(signals, coverage)
            self.assertEqual(len(edges), 2)
            self.assertFalse(answers.has_direct_whole_answer_provenance(signals, coverage))
            self.assertEqual(answers.unsupported_answer_unit_ids(units, signals, edges), [])
            with self.assertRaises(ValueError):
                answers.build_prior_knowledge_answer_edge('Orion, 1942', units, [], signals, 'synthesis_additive')

    def test_missing_atomic_unit_gets_explicit_pk_support(self):
        units = [{'unit_id': 'a1', 'claim': 'Orion'}, {'unit_id': 'a2', 'claim': '1942'}]
        for adapter in self.adapters:
            edge = self.module(adapter, 'answers').build_prior_knowledge_answer_edge('Orion, 1942', units, ['a2'], mode='synthesis_additive')
            self.assertEqual(edge['metadata']['unsupported_units'], ['a2'])

    def test_failed_support_api_or_malformed_boolean_raises_not_pk(self):
        pairs = [{'pair_id': 'p1', 'unit_id': 'a1', 'claim': 'Orion', 'qid': 'q1', 'query_text': 'candidate', 'window_text': 'Orion.'}]
        bad = [None, {}, {'verdicts': []}, {'verdicts': [{'pair_id': 'p1', 'supports': 'false'}]}]
        for adapter in self.adapters:
            for response in bad:
                with patch.object(self.module(adapter, 'attribution'), 'call_llm_json', return_value=response):
                    with self.assertRaises(AttributionUnavailable):
                        self.module(adapter, 'answers')._llm_answer_support_verdicts('Question', '', pairs)

    def test_hard_failure_never_uses_nonfailed_batch_anchor(self):
        failures = {f'q{i}': {'evidences': ['zero results'], 'failure_origin': 'search'} for i in range(2, 6)}
        texts = {'q1': 'successful alpha', 'q2': 'beta candidate', 'q3': 'gamma', 'q4': 'delta', 'q5': 'epsilon', 'q6': 'beta candidate'}
        for adapter in self.adapters:
            with contextlib.redirect_stdout(io.StringIO()):
                edges = self.module(adapter, 'failures').build_deterministic_hard_failure_edges_for_turn(
                    2, ['q6'], {1: ['q1', 'q2', 'q3', 'q4', 'q5'], 2: ['q6']}, {1: 'q1', 2: 'q6'}, texts, failures, {})
            self.assertEqual([(e['source'], e['target']) for e in edges], [('q2', 'q6')])

    def test_same_domain_alone_is_not_a_retry(self):
        metadata = {'q1': {'search_sites': ['example.org']}, 'q2': {'search_sites': ['example.org']}}
        for adapter in self.adapters:
            match, _ = self.module(adapter, 'failures').select_localized_hard_retry(
                'q1', {'failure_origin': 'search'}, ['q2'], {'q1': 'alpha', 'q2': 'beta'}, metadata)
            self.assertIsNone(match)

    def test_strict_policy_cannot_be_switched_to_unsafe_compatibility(self):
        for adapter in self.adapters:
            with patch.object(self.module(adapter, 'settings'), 'EDGE_POLICY', 'permissive'):
                self.assertTrue(self.module(adapter, 'llm').strict_edge_policy())

    def test_end_to_end_fixed_model_responses(self):
        source = Path(__file__).resolve().parents[1] / 'examples/synthetic_trace.json'
        sentence = 'The fictional Example Hall opened in 1942.'
        def model(_system, prompt, **kwargs):
            call_name = kwargs.get('call_name')
            if call_name == 'query_edge_attribution':
                return {'edges': [], 'no_source_found': ['q1']}
            if call_name == 'answer_decompose':
                return {'answer_units': [{'unit_id': 'a1', 'claim': '1942', 'unit_type': 'date'}]}
            if call_name == 'answer_support_match':
                return {'verdicts': [{'pair_id': p, 'supports': True, 'confidence': 'high',
                                     'evidence_sentence': sentence} for p in re.findall(r'pair_id=([^\n]+)', prompt)]}
            raise AssertionError(f'Unexpected model call: {call_name}')
        for adapter, reports_enabled in ((a, enabled) for a in self.adapters for enabled in (False, True)):
            with self.subTest(adapter=adapter, reports=reports_enabled), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'out' / 'graph.json'
                argv = ['builder', '--input', str(source), '--output', str(output), '--task-ids', 'Example-1']
                if reports_enabled:
                    argv += ['--debug-report-dir', str(Path(tmp) / 'reports')]
                if adapter == 'miro':
                    argv += ['--keep-tool-result', '-1']
                with patch.object(sys, 'argv', argv), patch.object(self.module(adapter, 'settings'), 'API_KEY', 'offline-test'), \
                        patch.object(self.module(adapter, 'settings'), 'LLM_EXTRA_BODY', {'client_secret': 'fictional-secret', 'limit': 128}), \
                        patch.object(self.module(adapter, 'attribution'), 'call_llm_json', side_effect=model), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.module(adapter, 'pipeline').main()
                result = json.loads(output.read_text())[0]
                pairs = {(e['source'], e['target']) for e in result['graph']['edges']}
                self.assertIn(('q1', 'Answer'), pairs)
                self.assertNotIn(('Prior_knowledge', 'Answer'), pairs)
                self.assertEqual(result['stats']['num_queries'], 1)
                self.assertEqual(result['settings']['llm_extra_body'], {'client_secret': '[REDACTED]', 'limit': 128})
                self.assertEqual(len(list(Path(tmp).rglob('*_debug_report.txt'))), int(reports_enabled))

    def test_native_tool_schema_requires_explicit_adapter(self):
        task = {'task_id': 'Example', 'question': 'Question', 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'search'}}]}]}
        with self.assertRaises(ValueError):
            validate_tasks([task], ['Example'])

    def test_edge_kind_stats_match_final_graph_after_deduplication(self):
        source = Path(__file__).resolve().parents[1] / 'examples/synthetic_trace.json'
        constraint = {'source': 'Q0', 'target': 'q1', 'edge_kind': 'constraint_use'}
        prior = {'source': 'Prior_knowledge', 'target': 'q1', 'edge_kind': 'prior_knowledge_derived'}
        cases = [([], {}),
                 ([constraint, prior], {'constraint_use': 1, 'prior_knowledge_derived': 1}),
                 ([constraint, constraint, constraint, prior, prior],
                  {'constraint_use': 1, 'prior_knowledge_derived': 1})]
        for adapter in self.adapters:
            for edges, expected in cases:
                with self.subTest(adapter=adapter, input_edges=len(edges)), tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / 'graph.json'
                    argv = ['builder', '--input', str(source), '--output', str(output), '--task-ids', 'Example-1']
                    if adapter == 'miro':
                        argv += ['--keep-tool-result', '-1']
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(sys, 'argv', argv))
                        stack.enter_context(patch.object(self.module(adapter, 'settings'), 'API_KEY', 'offline-test'))
                        stack.enter_context(patch.object(self.module(adapter, 'constraints'), 'q0_edges_from_tokens_firstuse', return_value=[]))
                        stack.enter_context(patch.object(self.module(adapter, 'attribution'), 'phase1_hybrid_classify', return_value=(edges, [], [])))
                        stack.enter_context(patch.object(self.module(adapter, 'answers'), 'extract_final_answer', return_value=''))
                        stack.enter_context(patch.object(self.module(adapter, 'attribution'), 'call_llm_json', side_effect=AssertionError('Offline test')))
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        self.module(adapter, 'pipeline').main()
                    result = json.loads(output.read_text())[0]
                    final_edges = result['graph']['edges']
                    self.assertEqual(result['stats']['edge_kinds'], expected)
                    self.assertEqual(result['stats']['edge_kinds'], dict(Counter(e['edge_kind'] for e in final_edges)))
                    self.assertEqual(sum(result['stats']['edge_kinds'].values()), result['stats']['num_edges'])
                    self.assertEqual(result['stats']['num_edges'], len(final_edges))

    def test_requested_task_must_exist(self):
        with self.assertRaises(ValueError):
            validate_tasks([{'task_id': 'Example'}], ['Absent'])

    def test_api_exception_redacts_key(self):
        for adapter in self.adapters:
            with patch.object(self.module(adapter, 'settings'), 'API_KEY', 'synthetic-private-key'):
                text = self.module(adapter, 'llm')._format_llm_exception(RuntimeError('synthetic-private-key rejected'))
            self.assertNotIn('synthetic-private-key', text)
            self.assertIn('[REDACTED]', text)

    def test_pipeline_api_failure_does_not_finalize_graph(self):
        source = Path(__file__).resolve().parents[1] / 'examples/synthetic_trace.json'
        for adapter in self.adapters:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'graph.json'
                argv = ['builder', '--input', str(source), '--output', str(output), '--task-ids', 'Example-1']
                with patch.object(sys, 'argv', argv), patch.object(self.module(adapter, 'settings'), 'API_KEY', 'offline-test'), \
                        patch.object(self.module(adapter, 'attribution'), 'call_llm_json', return_value=None), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(AttributionUnavailable):
                        self.module(adapter, 'pipeline').main()
                self.assertFalse(output.exists())

    def test_parent_cover_does_not_invent_unproposed_edges(self):
        for adapter in self.adapters:
            kept = self.module(adapter, 'parent_selection').apply_mpsc([], 'q2', {'q1': {'alpha'}}, {'alpha'}, {'q1': 1})
            self.assertEqual(kept, [])


if __name__ == '__main__':
    unittest.main()
