# Local MCP and CLI inference

SearchAtlas can run as a local STDIO MCP server in Codex, Claude Code, or another
compatible desktop/CLI client. The server reuses the public builder and offline
evaluator. An MCP client and an inference backend are independent choices: for
example, a Codex client can use a SearchAtlas server configured with `claude_cli`.

## Install and authenticate

Use Python 3.10+ on macOS, Linux, or WSL:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[mcp]'
```

Install and sign into the official CLI you want to use, following the
[Codex CLI guide](https://developers.openai.com/codex/cli/) or
[Claude Code setup guide](https://code.claude.com/docs/en/setup).
SearchAtlas does not read, copy, or exchange authentication tokens. The CLI handles
its own saved login. No `OPENAI_API_KEY` is required for a CLI backend. Inherited
API keys and provider-routing environment variables are not passed to CLI workers.
Use `--backend api` and install `.[builder,mcp]` for explicit API-key operation.

These are fresh non-interactive inference sessions, not calls to the current
conversation. A subscription does not imply unlimited or free inference: model
availability, quotas, and charges follow your CLI account and provider terms.
Each trajectory can require many local attribution requests. SearchAtlas does not
fall back to a paid API if CLI authentication, model access, or quotas fail.

## Connect a client

Register the server once, replacing both absolute paths. Choose a narrow workspace
containing only the inputs you want the client to analyze:

```sh
codex mcp add searchatlas -- /absolute/path/to/venv/bin/python -m searchatlas.mcp --workspace /absolute/path/to/workspace --backend codex_cli
```

Or in Claude Code:

```sh
claude mcp add --transport stdio searchatlas -- /absolute/path/to/venv/bin/python -m searchatlas.mcp --workspace /absolute/path/to/workspace --backend claude_cli
```

Restart/reconnect the client if needed. Standard output is reserved for MCP
messages. Do not add a shell banner or redirect build logs to the server's stdout.
See the official [Codex MCP configuration](https://developers.openai.com/codex/mcp/)
and [Claude Code MCP configuration](https://code.claude.com/docs/en/mcp).

Server options:

| Option | Meaning |
| --- | --- |
| `--workspace PATH` | Required root for user-selected JSON/JSONL files. |
| `--backend codex_cli` | Default; alternatives: `claude_cli`, `api`. |
| `--model NAME` | Optional model override; otherwise the selected backend's default. |
| `--llm-timeout 180` | Maximum seconds for an inference call, apart from initial CLI checks. |
| `--job-timeout 3600` | Maximum seconds for a running construction/evaluation job. |
| `--q0-mode rule` | Builder constraint-use edges: `rule` or LLM-assisted `llm`. |
| `--keep-tool-result 5` | Miro visibility budget; use `-1` to retain all earlier tool results. |
| `--output-dir PATH` | Private job directory within workspace; default `.searchatlas-mcp`. |

The server fixes the backend/model at startup; MCP tool arguments cannot change
them. `SEARCHATLAS_CODEX_BIN` or `SEARCHATLAS_CLAUDE_BIN` can select an installed
executable if it is not on `PATH`. `SEARCHATLAS_CLI_MODEL` supplies an optional CLI
model default. These are operator settings, never values extracted from a trace.

## Analyze a trace

Input uses the existing [tagged search-log format](../examples/synthetic_trace.json),
not a native Codex/Claude conversation export. `agent` selects the input adapter,
not the model doing attribution. Supported values are `tydp`, `tydp_gpt5`,
`tydp_qwen3`, `websailor`, and `mirothinker`.

For example, ask your MCP client:

> Analyze `examples/synthetic_trace.json`, task `Example-1`, with the `tydp`
> adapter. Use `constraints.json` for constraint annotations. Start the job,
> poll its status, and summarize its diagnostics and answer-supporting edges.

The corresponding `start_analysis` arguments are:

```json
{
  "input_path": "examples/synthetic_trace.json",
  "agent": "tydp",
  "task_ids": ["Example-1"],
  "annotations_path": "constraints.json"
}
```

`start_analysis` returns a `job_id` immediately. Use `get_analysis_status(job_id)`
until completed, then `get_analysis_result(job_id)`. To inspect edges, use
`get_analysis_result(job_id, view="edges", case_id="Example-1", limit=20)`.
Continue with `next_offset` until it is null. Other views are `nodes` and
`support_sets`; the default `summary` contains metrics and graph counts.

`cancel_analysis(job_id)` stops a queued or running job and its inference process.
The server runs one job at a time and accepts at most ten active/queued jobs.
Normal server shutdown cancels unfinished jobs. Persisted unfinished states are
marked `interrupted` on restart; they are not resumed automatically.

## Supply grounding annotations and optional reference DAGs

Without constraint annotations, topology and PK diagnostics are available but
constraint-grounding metrics are `null`. They are not silently treated as zero.
For the fictional one-query example, `constraints.json` can contain:

```json
[
  {
    "case_id": "Example-1",
    "constraint_units": [
      {"unit_id": "u1", "text": "Example Hall"},
      {"unit_id": "u2", "text": "opening year"}
    ],
    "query_unit_ids": {"q1": ["u1", "u2"]}
  }
]
```

Provide memberships for every query, including empty lists. Alternatively omit
`query_unit_ids` and explicitly choose `match_rule="distinctive_terms"` or
`"token_coverage"` to compute lexical deployments from the supplied units.
Builder Q0 first-use edges are not complete unit deployments; `--q0-mode llm`
alone does not supply evaluator annotations.

Add `reference_path` to compute exact source-target edge precision, recall, and
F1 against supplied reference DAGs. Reference case IDs must match the selected
cases, and query IDs/text/turns must align. These are reconstruction metrics, not
an answer-correctness score. A single unlabeled trace does not produce AUC/AP or a
calibrated composite score. Use the batch [outcome evaluator](evaluation.md) with
labels for that analysis.

`evaluate_graph(input_path, annotations_path, match_rule, reference_path)` queues
the same evaluation for an existing DAG file without making any model calls.
Poll and retrieve its result using the same job tools.

## Inference isolation and local artifacts

The [Codex non-interactive interface](https://developers.openai.com/codex/noninteractive/)
and [Claude Code headless interface](https://code.claude.com/docs/en/headless)
return structured JSON. Each request uses a fresh temporary directory and schema.
User/project integrations, tools, and nested MCP access are disabled. Unsupported
CLI versions fail an initial feature check; they are not run with weaker settings.
The existing source, chronology, and evidence validators still decide which edges
are retained. CLI output errors abort the job rather than becoming missing evidence.

Claude uses `--safe-mode`, not `--bare`, to retain saved-login support. Managed
administrator policies/hooks can still apply. Use a trusted CLI installation;
these controls are not a sandbox against a compromised executable or its admin
configuration. CLI backends bound time and output size, but do not enforce the
API's exact completion-token limits or reproduce its sampling behavior.

Inputs are limited to JSON/JSONL files inside the workspace. Hidden, credential,
and configuration paths are rejected, including symlink escapes. Only selected
trace fields enter the builder; reference DAGs and annotations are used by the
offline evaluator. Selected evidence is sent to the configured model service, so
obtain permission and sanitize sensitive traces before starting construction.

Job directories are private to the local user. They contain input snapshots,
graphs, results, state, and `worker.log`. Graphs and logs can include sensitive
trace excerpts: review them before sharing and remove jobs when no longer needed.
The MCP result tools expose bounded, redacted views, not raw prompts or logs.

## Offline tests

```sh
python -m unittest discover -s tests -v
```

Tests use fictional inputs, fixed responses, fake CLI executables, and local MCP
sessions. They do not consume API or subscription inference. A live provider test
is a separate, explicitly authorized check after installation/login.

Codex CLI has also passed a live build-and-evaluate check. Claude Code support is
covered by offline adapter tests; live-provider validation is pending.
