"""Offline checks for CLI behavior, credential redaction, and report paths."""

import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from searchatlas.builder.__main__ import build_parser, main
from searchatlas.builder.llm_compat import record_llm_event
from searchatlas.builder.privacy import redact_metadata, redact_text
from searchatlas.builder.safety import task_report_filename


class PublicInterfaceTests(unittest.TestCase):
    args = ['builder', '--agent', 'tydp', '--input', 'input.json',
            '--output', 'output.json', '--task-ids', 'Example']

    def test_dry_run_needs_no_key_or_input_file(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', self.args), patch.dict(os.environ, {}, clear=True), \
                patch('searchatlas.builder.__main__.importlib.import_module') as importer, \
                contextlib.redirect_stdout(output):
            main()
        self.assertFalse(json.loads(output.getvalue())['execute'])
        importer.assert_not_called()

    def test_execution_without_key_stops_before_loading_adapter(self):
        with patch.object(sys, 'argv', self.args + ['--execute']), \
                patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()), \
                patch('searchatlas.builder.__main__.importlib.import_module') as importer:
            with self.assertRaises(SystemExit):
                main()
        importer.assert_not_called()

    def test_help_contains_public_options_only(self):
        help_text = build_parser().format_help()
        for option in ('--q0-mode', '--max-snips', '--debug-report-dir', '--keep-tool-result'):
            self.assertIn(option, help_text)
        for removed in ('--skip-phase2', '--llm-io-mode', '--edge-policy'):
            self.assertNotIn(removed, help_text)

    def test_invalid_budget_and_obsolete_flags_are_rejected(self):
        for extra in (['--keep-tool-result', '-2'], ['--top-k', '0'], ['--skip-phase2'], ['--llm-io-mode', 'baseline']):
            with self.subTest(extra=extra), patch.object(sys, 'argv', self.args + extra), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()

    def test_metadata_redaction_preserves_request_configuration(self):
        original = {'headers': {'Authorization': 'Bearer example-token'},
                    'client_secret': 'fictional-secret', 'options': [{'api_key': 'fictional-key'}],
                    'completion_tokens': 256}
        result = redact_metadata(original)
        self.assertEqual(result['headers']['Authorization'], '[REDACTED]')
        self.assertEqual(result['options'][0]['api_key'], '[REDACTED]')
        self.assertEqual(result['client_secret'], '[REDACTED]')
        self.assertEqual(result['completion_tokens'], 256)
        self.assertEqual(original['client_secret'], 'fictional-secret')
        self.assertEqual(original['headers']['Authorization'], 'Bearer example-token')

    def test_error_redaction_masks_keys_and_url_passwords(self):
        token = 'sk-' + 'x' * 32
        text = 'failed: ' + token + ' https://user:password@example.org/api Bearer example-token'
        redacted = redact_text(text)
        for secret in (token, 'user:password', 'example-token'):
            self.assertNotIn(secret, redacted)
        self.assertIn('example.org/api', redacted)

    def test_usage_log_redacts_metadata_and_errors(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'OPENAI_API_KEY': 'fictional-runtime-key'}):
            path = Path(directory) / 'usage.jsonl'
            record_llm_event(str(path), event='request_error', call_name='test', model='test',
                             prompt_chars=1, max_tokens=1, attempt=1,
                             error='fictional-runtime-key rejected',
                             metadata={'api_key': 'fictional-config-key'})
            content = path.read_text()
        self.assertNotIn('fictional-runtime-key', content)
        self.assertNotIn('fictional-config-key', content)
        self.assertEqual(json.loads(content)['prompt_chars'], 1)

    def test_report_filenames_cannot_escape_output_directory(self):
        self.assertEqual(task_report_filename('Example-1'), 'Example-1_debug_report.txt')
        for task_id in ('../../Example', '/absolute/Example', 'folder\\Example', '.', 'x' * 300):
            with self.subTest(task_id=task_id):
                name = task_report_filename(task_id)
                self.assertEqual(Path(name).name, name)
                self.assertNotIn('/', name)
                self.assertNotIn('\\', name)
                self.assertLess(len(name), 120)
        self.assertNotEqual(task_report_filename('a/b'), task_report_filename('a?b'))

    def test_only_supported_io_mode_is_exposed(self):
        for adapter in ('standard', 'miro'):
            settings = importlib.import_module(f'searchatlas.builder.{adapter}.settings')
            self.assertEqual(settings.LLM_IO_MODE, 'hardened')
            self.assertFalse(hasattr(settings, 'MPSC_DEBUG_FILE'))


if __name__ == '__main__':
    unittest.main()
