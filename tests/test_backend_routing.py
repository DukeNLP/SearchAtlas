"""Exercise both input adapters without provider SDKs or model calls."""

import contextlib
import importlib
import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from searchatlas.builder.__main__ import main
from searchatlas.builder.runtime import configure_backend


class BackendRoutingTests(unittest.TestCase):
    args = ['builder', '--agent', 'tydp', '--input', 'missing.json',
            '--output', 'graph.json', '--task-ids', 'Example-1']

    def test_cli_dry_run_does_not_inherit_api_model(self):
        for backend in ('codex_cli', 'claude_cli'):
            output = io.StringIO()
            with self.subTest(backend=backend), patch.object(sys, 'argv', self.args + ['--backend', backend]), \
                    patch.dict(os.environ, {'OPENAI_MODEL': 'api-only-model'}, clear=True), \
                    contextlib.redirect_stdout(output):
                main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['backend'], backend)
            self.assertIsNone(plan['model'])

    def test_cli_execution_dispatches_without_api_key(self):
        with patch.object(sys, 'argv', self.args + ['--backend', 'claude_cli', '--model', 'sonnet', '--execute']), \
                patch.dict(os.environ, {}, clear=True), \
                patch('searchatlas.builder.__main__.importlib.import_module') as module:
            main()
        module.return_value.main.assert_called_once()

    def test_nonfinite_and_nonpositive_timeouts_rejected(self):
        for value in ('nan', 'inf', '0', '-2'):
            with self.subTest(value=value), patch.object(sys, 'argv', self.args + ['--llm-timeout', value]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_runtime_configuration_is_backend_specific(self):
        settings = SimpleNamespace(API_KEY='', LLM_TIMEOUT_SEC=180)
        with patch.dict(os.environ, {}, clear=True):
            configure_backend(settings, SimpleNamespace(backend='claude_cli', model=None, llm_timeout=90))
            self.assertIsNone(settings.MODEL)
            self.assertEqual(settings.LLM_TIMEOUT_SEC, 90)
            with self.assertRaises(SystemExit):
                configure_backend(settings, SimpleNamespace(backend='api', model=None, llm_timeout=None))

    def test_both_adapters_route_calls_and_preserve_api_transport(self):
        for adapter in ('standard', 'miro'):
            llm = importlib.import_module(f'searchatlas.builder.{adapter}.llm')
            with self.subTest(adapter=adapter), patch.object(llm.settings, 'LLM_BACKEND', 'codex_cli'), \
                    patch('searchatlas.builder.cli_backend.create_cli_completion') as cli, \
                    patch.object(llm.settings, 'client', new=MagicMock()) as api:
                llm._create_chat_completion({'messages': []}, call_name='answer_decompose')
                cli.assert_called_once_with('codex_cli', {'messages': []}, call_name='answer_decompose')
                api.chat.completions.create.assert_not_called()
            with patch.object(llm.settings, 'LLM_BACKEND', 'api'), \
                    patch.object(llm.settings, 'client', new=MagicMock()) as api:
                llm._create_chat_completion({'model': 'example'}, call_name='answer_decompose')
                api.chat.completions.create.assert_called_once_with(model='example')

    def test_cli_errors_cannot_become_absent_evidence(self):
        for adapter in ('standard', 'miro'):
            llm = importlib.import_module(f'searchatlas.builder.{adapter}.llm')
            attribution = importlib.import_module(f'searchatlas.builder.{adapter}.attribution')
            constraints = importlib.import_module(f'searchatlas.builder.{adapter}.constraints')
            error = RuntimeError('Synthetic model transport failure')
            with self.subTest(adapter=adapter), patch.object(llm.settings, 'LLM_BACKEND', 'codex_cli'), \
                    patch.object(llm, '_create_chat_completion', side_effect=error) as completion:
                with self.assertRaisesRegex(RuntimeError, 'Synthetic'):
                    attribution.call_llm_json('system', 'data', call_name='query_edge_attribution')
                completion.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, 'Synthetic'):
                    constraints.decompose_q0_units('Which candidate?')
                with self.assertRaisesRegex(RuntimeError, 'Synthetic'):
                    constraints._llm_arbiter_q0_matches('Question', [{
                        'pair_id': 'p1', 'unit_id': 'u1', 'unit_type': 'attribute',
                        'q0_span': 'candidate', 'qid': 'q1', 'query_text': 'candidate example'}])

    def test_cli_q0_units_must_be_nonempty_and_unique(self):
        unit = {'unit_id': 'u1', 'unit_type': 'entity', 'q0_span': 'Example Hall'}
        for adapter in ('standard', 'miro'):
            constraints = importlib.import_module(f'searchatlas.builder.{adapter}.constraints')
            attribution = importlib.import_module(f'searchatlas.builder.{adapter}.attribution')
            for units in ([], [unit, unit], [{**unit, 'unit_id': ' '}], [{**unit, 'q0_span': ' '} ]):
                with self.subTest(adapter=adapter, units=units), \
                        patch.object(constraints.settings, 'LLM_BACKEND', 'codex_cli'), \
                        patch.object(attribution, 'call_llm_json', return_value={'units': units}), \
                        self.assertRaises(ValueError):
                    constraints.decompose_q0_units('Example Hall?')

    def test_cli_q0_arbiter_requires_all_requested_pairs(self):
        pair = {'pair_id': 'p1', 'unit_id': 'u1', 'unit_type': 'entity',
                'q0_span': 'Example Hall', 'qid': 'q1', 'query_text': 'Example'}
        for adapter in ('standard', 'miro'):
            constraints = importlib.import_module(f'searchatlas.builder.{adapter}.constraints')
            llm = importlib.import_module(f'searchatlas.builder.{adapter}.llm')
            verdict = {'pair_id': 'p1', 'match': True, 'reason': 'Matching entity'}
            for verdicts in ([], [verdict, verdict], [{**verdict, 'pair_id': 'p2'}]):
                response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(verdicts)))])
                with self.subTest(adapter=adapter, verdicts=verdicts), \
                        patch.object(constraints.settings, 'LLM_BACKEND', 'codex_cli'), \
                        patch.object(llm, '_create_chat_completion', return_value=response), \
                        self.assertRaises(ValueError):
                    constraints._llm_arbiter_q0_matches('Example Hall?', [pair])


if __name__ == '__main__':
    unittest.main()
