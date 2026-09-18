"""CLI workers tested with fake transports and local Python, never an LLM."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from searchatlas.builder import cli_backend as cli
from searchatlas.builder.response_schemas import response_schema, validate_response


def codex_events(payload, *, item_type="agent_message"):
    return "\n".join(json.dumps(item) for item in [
        {"type": "thread.started", "thread_id": "fake"},
        {"type": "item.completed", "item": {"type": item_type,
                                                 "text": json.dumps(payload)}},
        {"type": "turn.completed", "usage": {"input_tokens": 40, "output_tokens": 8}},
    ])


def claude_result(payload):
    return json.dumps({"type": "result", "subtype": "success", "is_error": False,
                       "structured_output": payload, "usage": {"input_tokens": 10,
                       "cache_read_input_tokens": 20, "output_tokens": 5}})


class CLIBackendTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = {"messages": [{"role": "system", "content": "Use evidence only"},
                                   {"role": "user", "content": "private prompt text"}],
                       "timeout": 3, "model": "chosen-model"}

    def invoke(self, backend, payload, call="minimal_answer_extract"):
        raw = codex_events(payload) if backend == "codex_cli" else claude_result(payload)
        with patch.object(cli.shutil, "which", return_value="/fake/official-cli"), \
                patch.object(cli, "_preflight", return_value=frozenset(cli._CODEX_DISABLED_FEATURES)), \
                patch.object(cli, "_run_bounded", return_value=raw) as run:
            response = cli.create_cli_completion(backend, self.kwargs, call_name=call)
        return response, run.call_args

    def test_codex_contract_and_isolation(self):
        response, call = self.invoke("codex_cli", {"minimal_answer": "Example"})
        self.assertEqual(json.loads(response.choices[0].message.content), {"minimal_answer": "Example"})
        self.assertEqual(response.usage["total_tokens"], 48)
        self.assertEqual(response.model, "chosen-model")
        argv = call.args[0]
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ephemeral", argv)
        self.assertIn("features.shell_tool=false", argv)
        self.assertIn("features.hooks=false", argv)
        self.assertIn("features.skip_host_skill_discovery=true", argv)
        self.assertIn('web_search="disabled"', argv)
        self.assertIn("mcp_servers={}", argv)
        self.assertEqual(argv[-1], "-")
        self.assertFalse(any("private prompt text" in arg for arg in argv))
        self.assertIn("private prompt text", call.kwargs["prompt"])
        self.assertFalse(call.kwargs["cwd"].exists())

    def test_claude_contract_preserves_subscription_login(self):
        response, call = self.invoke("claude_cli", {"minimal_answer": "Example"})
        self.assertEqual(response.usage["prompt_tokens"], 30)
        argv = call.args[0]
        self.assertIn("--safe-mode", argv)
        self.assertNotIn("--bare", argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", argv)
        self.assertIn('{"mcpServers":{}}', argv)
        self.assertIn('{"disableAllHooks":true}', argv)
        self.assertIn("--no-session-persistence", argv)
        self.assertIn("--system-prompt-file", argv)
        self.assertFalse(any("private prompt text" in arg for arg in argv))

    def test_q0_arbiter_envelope_unwrapped(self):
        verdicts = [{"pair_id": "p1", "match": False, "reason": "No support"}]
        for backend in ("codex_cli", "claude_cli"):
            response, call = self.invoke(backend, {"verdicts": verdicts}, "q0_match_arbiter")
            self.assertEqual(json.loads(response.choices[0].message.content), verdicts)
            self.assertIn('wrap the requested verdict array', call.kwargs["prompt"])

    def test_cli_schema_rejects_wrong_boolean_and_missing_fields(self):
        for bad in ({"verdicts": [{"pair_id": "p1", "match": "false", "reason": "x"}]},
                    {"verdicts": [{"pair_id": "p1", "match": False}]}):
            with self.assertRaises(cli.CLIBackendError):
                self.invoke("codex_cli", bad, "q0_match_arbiter")

    def test_unknown_task_fails_before_subprocess(self):
        with patch.object(cli, "_run_bounded") as run:
            with self.assertRaises(cli.CLIBackendError):
                cli.create_cli_completion("codex_cli", self.kwargs, call_name="unknown")
            run.assert_not_called()

    def test_worker_environment_drops_api_auth_and_startup_injection(self):
        env = {"HOME": "/home/test", "PATH": "/usr/bin", "OPENAI_API_KEY": "secret",
               "ANTHROPIC_API_KEY": "secret", "CODEX_API_KEY": "secret",
               "ANTHROPIC_BASE_URL": "https://bad.test", "NODE_OPTIONS": "malicious",
               "CLAUDECODE": "outer-session", "CODEX_THREAD_ID": "outer-id"}
        with patch.dict(os.environ, env, clear=True):
            child = cli._child_environment()
        self.assertEqual(set(child) - {"HOME", "PATH"},
                         {"NO_COLOR", "TERM", "SEARCHATLAS_CLI_WORKER"})

    def test_recursive_worker_is_rejected(self):
        with patch.dict(os.environ, {"SEARCHATLAS_CLI_WORKER": "1"}):
            with self.assertRaisesRegex(cli.CLIBackendError, "Recursive"):
                cli.create_cli_completion("codex_cli", self.kwargs)

    def test_nonfinite_timeout_and_nontext_input_rejected(self):
        for timeout in (float("nan"), float("inf"), -1, 0, "bad"):
            with self.assertRaises(cli.CLIBackendError):
                cli.create_cli_completion("codex_cli", {**self.kwargs, "timeout": timeout},
                                          call_name="minimal_answer_extract")
        with self.assertRaises(cli.CLIBackendError):
            cli._prompt([{"role": "user", "content": [{"type": "image"}]}], "x")

    def test_codex_tool_calls_and_failed_turns_rejected(self):
        with self.assertRaisesRegex(cli.CLIBackendError, "tool action"):
            cli._codex_result(codex_events({"minimal_answer": "x"}, item_type="mcp_tool_call"))
        with self.assertRaises(cli.CLIBackendError):
            cli._codex_result('{"type":"turn.failed","error":"SECRET"}')

    def test_claude_failed_result_and_missing_structured_output_rejected(self):
        for obj in ({"type": "result", "subtype": "error_max_turns", "is_error": True},
                    {"type": "result", "subtype": "success", "is_error": False,
                     "result": '{"minimal_answer":"looks valid"}'}):
            with self.assertRaises(cli.CLIBackendError):
                cli._claude_result(json.dumps(obj))

    def test_duplicate_keys_and_nan_rejected_without_echo(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', 'SECRET not json'):
            with self.assertRaises(cli.CLIBackendError) as caught:
                cli._json_loads(raw)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_old_cli_refused_before_model_request(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(cli, "_run_bounded", return_value="old cli help") as run:
            with self.assertRaisesRegex(cli.CLIBackendError, "upgrade"):
                cli._preflight("claude_cli", sys.executable, cwd=Path(temp), env={})
            self.assertEqual(run.call_count, 1)

    def test_codex_missing_optional_features_is_supported(self):
        help_text = ' '.join(cli._CODEX_REQUIRED_FLAGS)
        feature_text = '\n'.join(f'{name} stable true' for name in cli._CODEX_REQUIRED_FEATURES)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(cli, '_PREFLIGHT_CACHE', {}), \
                patch.object(cli, '_run_bounded', side_effect=[help_text, feature_text]):
            features = cli._preflight('codex_cli', sys.executable, cwd=Path(temp), env={})
            self.assertEqual(features, cli._CODEX_REQUIRED_FEATURES)

    def test_codex_missing_essential_skill_isolation_is_refused(self):
        help_text = ' '.join(cli._CODEX_REQUIRED_FLAGS)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(cli, '_PREFLIGHT_CACHE', {}), \
                patch.object(cli, '_run_bounded', side_effect=[help_text, 'shell_tool stable true']):
            with self.assertRaisesRegex(cli.CLIBackendError, 'isolation'):
                cli._preflight('codex_cli', sys.executable, cwd=Path(temp), env={})

    def test_only_advertised_optional_features_are_disabled(self):
        available = {'shell_tool', 'skip_host_skill_discovery', 'js_repl', 'codex_hooks'}
        with patch.object(cli.shutil, 'which', return_value='/fake/official-cli'), \
                patch.object(cli, '_preflight', return_value=frozenset(available)), \
                patch.object(cli, '_run_bounded', return_value=codex_events({'minimal_answer': 'x'})) as run:
            cli.create_cli_completion('codex_cli', self.kwargs, call_name='minimal_answer_extract')
        argv = run.call_args.args[0]
        self.assertIn('features.js_repl=false', argv)
        self.assertIn('features.codex_hooks=false', argv)
        self.assertNotIn('features.browser_use=false', argv)

    def test_known_schemas_are_closed_and_independent(self):
        for name in ("minimal_answer_extract", "q0_decompose", "answer_decompose",
                     "q0_match_arbiter", "answer_support_match",
                     "answer_distributed_support", "query_edge_attribution"):
            schema = response_schema(name)
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
        schema = response_schema("minimal_answer_extract")
        schema["properties"].clear()
        self.assertIn("minimal_answer", response_schema("minimal_answer_extract")["properties"])

    def test_nullable_failure_subtype_and_unknown_keys(self):
        payload = {"edges": [{"source": "q1", "target": "q2", "edge_kind": "evidence_derived",
                              "failure_subtype": None, "covered_signals": ["example"],
                              "evidence": "[snippet] q1", "validation_reason": "retrieved_clue_reused",
                              "confidence": "high"}], "no_source_found": []}
        validate_response(payload, response_schema("query_edge_attribution"))
        payload["unsafe_extra"] = "x"
        with self.assertRaises(ValueError):
            validate_response(payload, response_schema("query_edge_attribution"))

    def test_real_subprocess_stdin_and_safe_exit_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            result = cli._run_bounded([sys.executable, "-c", "import sys;print(sys.stdin.read())"],
                                      prompt="hello", cwd=Path(temp), env=cli._child_environment(), timeout=2)
            self.assertEqual(result.strip(), "hello")
            with self.assertRaises(cli.CLIBackendError) as caught:
                cli._run_bounded([sys.executable, "-c", "import sys;sys.stderr.write('SECRET');sys.exit(4)"],
                                 prompt="", cwd=Path(temp), env=cli._child_environment(), timeout=2)
            self.assertNotIn("SECRET", str(caught.exception))

    def test_real_subprocess_timeout_restores_signal_handlers(self):
        before = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as temp:
            start = time.monotonic()
            with self.assertRaisesRegex(cli.CLIBackendError, "timed out"):
                cli._run_bounded([sys.executable, "-c", "import time;time.sleep(20)"],
                                 prompt="", cwd=Path(temp), env=cli._child_environment(), timeout=0.1)
            self.assertLess(time.monotonic() - start, 3)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_output_overflow_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cli, "MAX_OUTPUT_BYTES", 100):
            with self.assertRaisesRegex(cli.CLIBackendError, "safety limit"):
                cli._run_bounded([sys.executable, "-c", "print('x'*20000)"], prompt="",
                                 cwd=Path(temp), env=cli._child_environment(), timeout=2)

    @unittest.skipUnless(os.name == "posix", "POSIX signal propagation")
    def test_worker_sigterm_terminates_nested_cli_group(self):
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as temp:
            timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
            try:
                with patch.object(cli, "_stop_process_group", wraps=cli._stop_process_group) as stop:
                    with self.assertRaises(SystemExit) as caught:
                        cli._run_bounded([sys.executable, "-c", "import time;time.sleep(20)"],
                                         prompt="", cwd=Path(temp), env=cli._child_environment(), timeout=3)
                    self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)
                    self.assertGreaterEqual(stop.call_count, 1)
                    self.assertIsNotNone(stop.call_args.args[0].returncode)
            finally:
                timer.cancel()
                timer.join()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_descendant_holding_pipes_open_is_terminated(self):
        with tempfile.TemporaryDirectory() as temp:
            script = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])"
            start = time.monotonic()
            with self.assertRaisesRegex(cli.CLIBackendError, "descendants"):
                cli._run_bounded([sys.executable, "-c", script], prompt="", cwd=Path(temp),
                                 env=cli._child_environment(), timeout=0.3)
            self.assertLess(time.monotonic() - start, 3)


if __name__ == "__main__":
    unittest.main()
