"""Regression for observed, non-fatal official CLI startup notices."""
import json
import unittest

from searchatlas.builder.cli_backend import CLIBackendError, _codex_result


UNSTABLE = 'Under-development features enabled: skip_host_skill_discovery. Under-development features are incomplete.'
DISABLED = ('Code Mode is unavailable because code-mode host is disabled. '
            'Code mode will fail closed; enable `features.code_mode_host` '
            'and install `codex-code-mode-host`.')


def notice(message):
    return {'type': 'item.completed', 'item': {'id': 'notice', 'type': 'error', 'message': message}}


def events(*prefix):
    return '\n'.join(json.dumps(event) for event in [
        {'type': 'thread.started'}, *prefix, {'type': 'turn.started'},
        {'type': 'item.completed', 'item': {'id': 'answer', 'type': 'agent_message',
                                          'text': '{"edges":[],"no_source_found":["q1"]}'}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 8}},
    ])


class CodexStartupNoticeTests(unittest.TestCase):
    def test_recognized_notices_before_turn_preserve_valid_result_and_usage(self):
        with self.assertLogs('searchatlas.builder.cli_backend', level='WARNING') as captured:
            result, usage, model = _codex_result(events(notice(UNSTABLE), notice(DISABLED)))
        self.assertEqual(result, {'edges': [], 'no_source_found': ['q1']})
        self.assertEqual(usage['input_tokens'], 12)
        self.assertEqual(len(captured.output), 2)

    def test_unrecognized_or_inference_errors_still_fail(self):
        for item in (notice('Authentication failed'), notice('Under-development features enabled: another_feature.'),
                     {'type': 'turn.failed'}, {'type': 'error'}):
            with self.subTest(item=item), self.assertRaises(CLIBackendError):
                _codex_result(events(item))

    def test_known_notice_after_turn_started_is_not_ignored(self):
        with self.assertRaises(CLIBackendError):
            _codex_result(events({'type': 'turn.started'}, notice(DISABLED)))

    def test_notices_do_not_make_an_incomplete_turn_valid(self):
        with self.assertLogs('searchatlas.builder.cli_backend', level='WARNING'), self.assertRaises(CLIBackendError):
            _codex_result(json.dumps(notice(DISABLED)))

    def test_tool_action_still_rejected_after_known_notice(self):
        tool = {'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'not executed'}}
        with self.assertLogs('searchatlas.builder.cli_backend', level='WARNING'), self.assertRaisesRegex(CLIBackendError, 'tool action'):
            _codex_result(events(notice(DISABLED), tool))


if __name__ == '__main__':
    unittest.main()
