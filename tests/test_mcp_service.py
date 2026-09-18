"""Offline MCP jobs, workspace boundaries, and public evaluator parity."""

import asyncio
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from searchatlas.evaluation.diagnostics import graph_diagnostics
from searchatlas.mcp.io import load_records, sanitized_tasks, workspace_input, write_json
from searchatlas.mcp.service import AnalysisService
from searchatlas.mcp.worker import evaluate


SOURCE = str(Path(__file__).resolve().parents[1] / 'src')


def fixture():
    return {'case_id': 'example', 'graph': {
        'nodes': [{'id': 'Q0'}, {'id': 'Answer'}, {'id': 'Prior_knowledge'},
                  {'id': 'q1', 'type': 'Query', 'text': 'Example Hall opening year', 'turn': 1},
                  {'id': 'q2', 'type': 'Query', 'text': 'Example Hall archive 1942', 'turn': 2}],
        'edges': [{'source': 'Q0', 'target': 'q1', 'edge_kind': 'constraint_use'},
                  {'source': 'q1', 'target': 'q2', 'edge_kind': 'evidence_derived'},
                  {'source': 'q2', 'target': 'Answer', 'edge_kind': 'evidence_derived'}],
    }, 'per_turn_raw': [{'prompt': 'PRIVATE PROMPT'}]}


def annotations():
    return {'case_id': 'example', 'constraint_units': [{'unit_id': 'u1', 'text': 'Example Hall'}],
            'query_unit_ids': {'q1': ['u1'], 'q2': ['u1']}}


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / '.searchatlas-mcp'
        write_json(self.root / 'trace.json', {'task_id': 'one', 'question': 'Question?',
                                            'messages': [{'role': 'assistant', 'content': '<answer>A</answer>'}]})

    def tearDown(self):
        self.temp.cleanup()

    def test_allowed_json_and_rejected_traversal(self):
        self.assertEqual(workspace_input(self.root, 'trace.json', self.output), self.root / 'trace.json')
        for name in ('../trace.json', '.env', '.git/config', 'credentials.json', 'api_key.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                workspace_input(self.root, name, self.output)

    def test_symlink_cannot_escape_or_target_hidden_file(self):
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / 'outside.json'
            write_json(target, {})
            (self.root / 'escape.json').symlink_to(target)
            with self.assertRaises(ValueError):
                workspace_input(self.root, 'escape.json', self.output)
        write_json(self.root / '.private.json', {})
        (self.root / 'alias.json').symlink_to(self.root / '.private.json')
        with self.assertRaises(ValueError):
            workspace_input(self.root, 'alias.json', self.output)

    def test_trace_snapshot_omits_metadata_and_gold_answers(self):
        record = load_records(self.root / 'trace.json')['one']
        record.update(api_key='private-value', gold_answer='DO NOT SEND', correct=True)
        record['messages'][0]['private_config'] = 'DO NOT SEND'
        clean = sanitized_tasks({'one': record}, ['one'])[0]
        self.assertEqual(set(clean), {'task_id', 'question', 'messages'})
        self.assertEqual(set(clean['messages'][0]), {'role', 'content'})

    def test_credential_like_trace_content_requires_user_sanitization(self):
        record = load_records(self.root / 'trace.json')['one']
        record['messages'][0]['content'] = 'Bearer sensitive-example-value'
        with self.assertRaisesRegex(ValueError, 'credential-like'):
            sanitized_tasks({'one': record}, ['one'])

    def test_task_ids_cannot_inject_cli_options(self):
        with self.assertRaises(ValueError):
            sanitized_tasks({}, ['--output'])

    def test_native_tool_calls_are_not_silently_dropped(self):
        record = load_records(self.root / 'trace.json')['one']
        record['messages'][0]['tool_calls'] = [{'id': 'tool'}]
        with self.assertRaisesRegex(ValueError, 'Unsupported message'):
            sanitized_tasks({'one': record}, ['one'])


class EvaluationParityTests(unittest.TestCase):
    def test_worker_uses_existing_evaluator_without_changing_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, annotation = fixture(), annotations()
            write_json(root / 'graph.json', [graph])
            write_json(root / 'annotations.json', [annotation])
            result = evaluate(root / 'graph.json', root / 'annotations.json', reference_path=root / 'graph.json')
            self.assertEqual(result['cases'][0]['metrics'], graph_diagnostics(graph, annotation)['metrics'])
            self.assertEqual(result['reconstruction']['macro']['f1'], 1)

    def test_missing_annotations_are_undefined(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            write_json(path, fixture())
            self.assertIsNone(evaluate(path)['cases'][0]['metrics']['backbone_constraint_share'])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        write_json(self.root / 'graph.json', fixture())
        write_json(self.root / 'annotations.json', annotations())
        write_json(self.root / 'trace.json', {'task_id': 'one', 'question': 'Question?',
                                            'messages': [{'role': 'assistant', 'content': '<answer>A</answer>'}]})
        self.environment = patch.dict(os.environ, {'PYTHONPATH': SOURCE})
        self.environment.start()
        self.service = AnalysisService(self.root)

    async def asyncTearDown(self):
        await self.service.close()
        self.environment.stop()
        self.temp.cleanup()

    async def wait_done(self, job_id):
        for _ in range(100):
            status = self.service.get_analysis_status(job_id)
            if status['status'] not in {'queued', 'running'}:
                return status
            await asyncio.sleep(.05)
        self.fail('Local test worker failed to terminate')

    async def test_offline_job_end_to_end_and_pagination(self):
        state = await self.service.evaluate_graph('graph.json', 'annotations.json', reference_path='graph.json')
        job_id = state['job_id']
        self.assertEqual((await self.wait_done(job_id))['status'], 'completed')
        summary = self.service.get_analysis_result(job_id)
        self.assertEqual(summary['reconstruction']['macro']['f1'], 1)
        self.assertEqual(summary['items'][0]['metrics'], graph_diagnostics(fixture(), annotations())['metrics'])
        nodes = self.service.get_analysis_result(job_id, 'nodes', 'example', limit=2)
        self.assertEqual(nodes['next_offset'], 2)
        self.assertEqual(len(nodes['items']), 2)
        second = self.service.get_analysis_result(job_id, 'nodes', 'example', offset=2, limit=2)
        self.assertNotEqual(nodes['items'], second['items'])
        directory = self.service.output_dir / job_id
        self.assertNotIn('PRIVATE PROMPT', (directory / 'input.json').read_text())
        self.assertEqual(stat.S_IMODE((directory / 'result.json').stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)

    async def test_analysis_snapshots_only_selected_trace(self):
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'import time; time.sleep(60)']
        state = await self.service.start_analysis('trace.json', 'tydp', ['one'])
        directory = self.service.output_dir / state['job_id']
        config = json.loads((directory / 'config.json').read_text())
        self.assertEqual(config['backend'], 'codex_cli')
        self.assertEqual(config['task_ids'], ['one'])
        self.assertNotIn('OPENAI_API_KEY', (directory / 'config.json').read_text())
        self.assertFalse(self.service.get_analysis_result(state['job_id'])['result_available'])
        await self.service.cancel_analysis(state['job_id'])

    async def test_failed_worker_never_becomes_completed(self):
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'raise SystemExit(7)']
        state = await self.service.evaluate_graph('graph.json')
        self.assertEqual((await self.wait_done(state['job_id']))['status'], 'failed')

    async def test_successful_exit_without_report_is_not_completion(self):
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'pass']
        state = await self.service.evaluate_graph('graph.json')
        self.assertEqual((await self.wait_done(state['job_id']))['status'], 'failed')

    async def test_analysis_result_without_graph_is_not_completion(self):
        program = "import json; from pathlib import Path; Path('result.json').write_text(json.dumps({'cases': [{'case_id': 'one'}]}))"
        self.service.worker_command = lambda directory: [sys.executable, '-c', program]
        state = await self.service.start_analysis('trace.json', 'tydp', ['one'])
        self.assertEqual((await self.wait_done(state['job_id']))['status'], 'failed')

    async def test_timeout_is_reported_and_worker_terminated(self):
        self.service.job_timeout = .05
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'import time; time.sleep(60)']
        state = await self.service.evaluate_graph('graph.json')
        result = await self.wait_done(state['job_id'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('timeout', result['error'])
        self.assertNotIn(state['job_id'], self.service.processes)

    async def test_cancel_running_job_terminates_worker(self):
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'import time; time.sleep(60)']
        state = await self.service.evaluate_graph('graph.json')
        for _ in range(100):
            if state['job_id'] in self.service.processes:
                break
            await asyncio.sleep(.01)
        process = self.service.processes[state['job_id']]
        cancelled = await self.service.cancel_analysis(state['job_id'])
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertIsNotNone(process.returncode)

    @unittest.skipUnless(os.name == 'posix', 'POSIX process groups')
    async def test_cancellation_reaches_nested_cli_process_group(self):
        child = "import os,time; from pathlib import Path; Path('child.pid').write_text(str(os.getpid())); time.sleep(60)"
        wrapper = ("import os,sys; from pathlib import Path; "
                   "from searchatlas.builder.cli_backend import _run_bounded; "
                   f"_run_bounded([sys.executable, '-c', {child!r}], prompt='', cwd=Path.cwd(), env=dict(os.environ), timeout=60)")
        self.service.worker_command = lambda directory: [sys.executable, '-c', wrapper]
        state = await self.service.evaluate_graph('graph.json')
        pidfile = self.service.output_dir / state['job_id'] / 'child.pid'
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(.01)
        self.assertTrue(pidfile.exists(), 'Nested CLI process was not started')
        child_pid = int(pidfile.read_text())
        await self.service.cancel_analysis(state['job_id'])
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    async def test_queue_limit_and_shutdown(self):
        self.service.max_jobs = 1
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'import time; time.sleep(60)']
        state = await self.service.evaluate_graph('graph.json')
        with self.assertRaisesRegex(ValueError, 'queue is full'):
            await self.service.evaluate_graph('graph.json')
        await self.service.close()
        self.assertEqual(self.service.get_analysis_status(state['job_id'])['status'], 'cancelled')

    async def test_restarted_pending_job_is_interrupted_not_resumed(self):
        job_id = 'a' * 32
        directory = self.service.output_dir / job_id
        directory.mkdir()
        write_json(directory / 'state.json', {'job_id': job_id, 'status': 'running'})
        recovered = AnalysisService(self.root)
        self.assertEqual(recovered.get_analysis_status(job_id)['status'], 'interrupted')
        await recovered.close()

    async def test_malformed_saved_state_does_not_prevent_startup(self):
        directory = self.service.output_dir / ('b' * 32)
        directory.mkdir()
        write_json(directory / 'state.json', [])
        recovered = AnalysisService(self.root)
        self.assertNotIn(directory.name, recovered.jobs)
        await recovered.close()

    async def test_operator_options_are_fixed_in_worker_config(self):
        self.service.q0_mode = 'llm'
        self.service.keep_tool_result = -1
        self.service.worker_command = lambda directory: [sys.executable, '-c', 'import time; time.sleep(60)']
        state = await self.service.start_analysis('trace.json', 'mirothinker', ['one'])
        config = json.loads((self.service.output_dir / state['job_id'] / 'config.json').read_text())
        self.assertEqual(config['q0_mode'], 'llm')
        self.assertEqual(config['keep_tool_result'], -1)
        await self.service.cancel_analysis(state['job_id'])

    async def test_protected_outputs_and_bad_pagination(self):
        with self.assertRaises(ValueError):
            AnalysisService(self.root, self.root)
        with self.assertRaises(ValueError):
            AnalysisService(self.root, self.root.parent / 'outside')
        with self.assertRaises(ValueError):
            self.service.get_analysis_status('../escape')
        state = await self.service.evaluate_graph('graph.json')
        await self.wait_done(state['job_id'])
        with self.assertRaises(ValueError):
            self.service.get_analysis_result(state['job_id'], limit=101)

    async def test_references_and_units_require_matching_case_ids(self):
        other = fixture()
        other['case_id'] = 'wrong'
        write_json(self.root / 'other.json', other)
        with self.assertRaisesRegex(ValueError, 'Reference case IDs'):
            await self.service.evaluate_graph('graph.json', reference_path='other.json')
        unit = annotations()
        del unit['query_unit_ids']
        write_json(self.root / 'units.json', unit)
        with self.assertRaisesRegex(ValueError, 'match_rule'):
            await self.service.evaluate_graph('graph.json', annotations_path='units.json')


if __name__ == '__main__':
    unittest.main()
