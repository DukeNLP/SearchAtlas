"""Real MCP worker -> fake CLI subprocess -> builder -> public evaluation.

Only executable selection is replaced. No model, parser, construction, or
evaluation function is mocked, and every input is the published synthetic case.
"""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from searchatlas.evaluation.constraints import prepare_annotation
from searchatlas.evaluation.diagnostics import graph_diagnostics
from searchatlas.mcp.io import write_json
from searchatlas.mcp.service import AnalysisService


PACKAGE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'posix', 'CLI backends require macOS, Linux, or WSL')
class CLIEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def wait_done(self, service, job_id):
        for _ in range(400):
            status = service.get_analysis_status(job_id)
            if status['status'] not in {'queued', 'running'}:
                return status
            await asyncio.sleep(.05)
        self.fail('Offline CLI integration job did not finish')

    async def run_example(self, backend, agent, *, fail=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fake = root / 'fake_cli.py'
            shutil.copyfile(PACKAGE / 'tests/fixtures/fake_attribution_cli.py', fake)
            fake.chmod(0o700)
            shutil.copyfile(PACKAGE / 'examples/synthetic_trace.json', root / 'trace.json')
            annotation = {'case_id': 'Example-1', 'constraint_units': [
                {'unit_id': 'u1', 'text': 'Example Hall'}]}
            write_json(root / 'annotations.json', annotation)
            reference = {'case_id': 'Example-1', 'graph': {
                'nodes': [{'id': 'Q0'}, {'id': 'Prior_knowledge'}, {'id': 'Answer'},
                          {'id': 'q1', 'type': 'Query', 'text': 'Example Hall opening year', 'turn': 1}],
                'edges': [{'source': 'Q0', 'target': 'q1', 'edge_kind': 'constraint_use'},
                          {'source': 'q1', 'target': 'Answer', 'edge_kind': 'evidence_derived'}]}}
            write_json(root / 'reference.json', reference)
            # The worker imports the source tree; the fake CLI itself uses only stdlib.
            environment = {'PYTHONPATH': str(PACKAGE / 'src'), 'OPENAI_API_KEY': '',
                           'SEARCHATLAS_CODEX_BIN': str(fake), 'SEARCHATLAS_CLAUDE_BIN': str(fake),
                           'LLM_EXTRA_BODY_JSON': '', 'LLM_USAGE_JSONL': '',
                           'SEARCHATLAS_CLI_WORKER': '', 'SEARCHATLAS_CLI_MODEL': ''}
            with patch.dict(os.environ, environment):
                service = AnalysisService(root, backend=backend, q0_mode='llm',
                                          model='simulate-failure' if fail else 'offline-fixture',
                                          llm_timeout=5, job_timeout=20, keep_tool_result=-1)
                try:
                    initial = await service.start_analysis(
                        'trace.json', agent, ['Example-1'], 'annotations.json',
                        'token_coverage', 'reference.json')
                    job_id = initial['job_id']
                    final = await self.wait_done(service, job_id)
                    directory = service.output_dir / job_id
                    if fail:
                        self.assertEqual(final['status'], 'failed')
                        self.assertFalse((directory / 'graph.json').exists())
                        self.assertFalse((directory / 'result.json').exists())
                        self.assertFalse(service.get_analysis_result(job_id)['result_available'])
                        return
                    self.assertEqual(final['status'], 'completed',
                                     (directory / 'worker.log').read_text())
                    graph_record = json.loads((directory / 'graph.json').read_text())[0]
                    edges = graph_record['graph']['edges']
                    self.assertIn(('q1', 'Answer'), {(e['source'], e['target']) for e in edges})
                    self.assertNotIn(('Prior_knowledge', 'Answer'),
                                     {(e['source'], e['target']) for e in edges})
                    self.assertEqual(graph_record['stats']['num_queries'], 1)
                    self.assertEqual(graph_record['stats']['edge_kinds'],
                                     dict(Counter(e['edge_kind'] for e in edges)))
                    summary = service.get_analysis_result(job_id)
                    self.assertEqual(summary['reconstruction']['macro']['f1'], 1)
                    metrics = summary['items'][0]['metrics']
                    expected = graph_diagnostics(graph_record, prepare_annotation(
                        annotation, graph_record, 'token_coverage'))['metrics']
                    self.assertEqual(metrics, expected)
                    self.assertEqual(metrics['direct_frontier_coverage'], 1)
                    self.assertEqual(metrics['pk_answer_edge'], 0)
                    self.assertEqual(metrics['query_efficiency'], 1)
                    calls = [json.loads(line) for line in fake.with_suffix('.calls.jsonl').read_text().splitlines()]
                    self.assertTrue(all(item['backend'] == backend for item in calls))
                    self.assertTrue({'q0_decompose', 'q0_match_arbiter', 'query_edge_attribution',
                                     'answer_decompose', 'answer_support_match'} <= {item['call'] for item in calls})
                    # Evaluation-only tools must never launch another inference call.
                    write_json(root / 'built_graph.json', [graph_record])
                    evaluation = await service.evaluate_graph('built_graph.json', 'annotations.json',
                                                              'token_coverage', 'reference.json')
                    self.assertEqual((await self.wait_done(service, evaluation['job_id']))['status'], 'completed')
                    evaluated = service.get_analysis_result(evaluation['job_id'])
                    self.assertEqual(evaluated['items'][0]['metrics'], metrics)
                    self.assertEqual(evaluated['reconstruction']['macro']['f1'], 1)
                    self.assertEqual(len(fake.with_suffix('.calls.jsonl').read_text().splitlines()), len(calls))
                finally:
                    await service.close()

    async def test_codex_standard_build_and_evaluation(self):
        await self.run_example('codex_cli', 'tydp')

    async def test_codex_miro_build_and_evaluation(self):
        await self.run_example('codex_cli', 'mirothinker')

    async def test_claude_standard_build_and_evaluation(self):
        await self.run_example('claude_cli', 'tydp')

    async def test_claude_miro_build_and_evaluation(self):
        await self.run_example('claude_cli', 'mirothinker')

    async def test_failed_cli_does_not_produce_a_graph_or_report(self):
        await self.run_example('codex_cli', 'tydp', fail=True)


if __name__ == '__main__':
    unittest.main()
