# SearchAtlas

Code for [SearchAtlas: Analyzing Agentic Search Strategies via Evidential Query Graphs](https://arxiv.org/abs/2609.10901),
accepted to **Findings of EMNLP 2026**.

Build evidence-dependency query DAGs and evaluate interpretable search-process diagnostics.
The package includes a DAG builder, an offline evaluator, examples, and tests.

## Installation

Python 3.10 or newer is required.

```sh
python -m pip install -e .             # Offline evaluation
python -m pip install -e '.[builder]' # Also install model API support
```

## Build a DAG

Set `OPENAI_API_KEY` in your environment. `OPENAI_MODEL` selects the attribution
model, and `OPENAI_BASE_URL` optionally selects a compatible endpoint.

```sh
python -m searchatlas.builder --agent tydp --input examples/synthetic_trace.json --output local-output/graph.json --task-ids Example-1
```

This previews the command. Add `--execute` to make model calls and save the graph.
`--agent` selects the input adapter, not the attribution model.

Inputs contain `task_id`, `question`, and `messages` (or `trajectory`). Supported
logs encode search/visit actions using `<tool_call>` JSON or `<use_mcp_tool>` XML,
with tool results as user messages. See [the fictional example](examples/synthetic_trace.json).
Each issued search becomes a query node; page visits supply evidence to queries.

Useful options:

- `--q0-mode rule` uses lexical constraint rules; `--q0-mode llm` enables LLM-assisted units.
- `--keep-tool-result K` sets Miro tool-result visibility; the default is 5, and `-1` retains all earlier results.
- `--debug-report-dir DIR` enables detailed text reports; they are off by default.
- `--help` lists all options.

The builder uses spaCy lemmas if `en_core_web_sm` is installed, otherwise built-in
normalization. Keep this dependency choice fixed when comparing runs.

## Evaluate DAGs

The evaluator needs no API key or model calls. Generate fictional fixtures to try it:

```sh
python examples/make_evaluation_demo.py --output local-output/demo
python -m searchatlas.evaluation diagnostics --input local-output/demo/graphs.json --annotations local-output/demo/constraints.json --output local-output/demo/metrics.json
python -m searchatlas.evaluation reconstruction --predictions local-output/demo/graphs.json --references local-output/demo/references.json --output local-output/demo/reconstruction.json
python -m searchatlas.evaluation outcomes --input local-output/demo/metrics.json --labels local-output/demo/labels.json --regime sequential --topology answer_backbone_concentration --grounding backbone_constraint_share --pk-risk pk_answer_edge --output local-output/demo/outcomes.json
```

- `diagnostics`: topology, constraint coverage, PK reliance, and support-node sets.
- `reconstruction`: per-DAG and macro P/R/F1, including edge-type results.
- `outcomes`: within-agent held-out AUC/AP, module ablations, and auxiliary threshold accuracy/F1.

The evaluator accepts builder outputs and caller-provided annotations. Primary-path
and direct-frontier coverage are named separately; select the intended metric scope
when computing outcome associations. See [input schemas and metric definitions](docs/evaluation.md).
The demo is hand-designed synthetic data, not an experimental result.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests run offline using synthetic inputs and fixed model responses.

## Release and privacy

This is an updated implementation; benchmark data, research DAGs, and human labels
are not bundled. See [evaluation scope](docs/evaluation.md#release-scope) for numerical reproduction requirements.

Keys are read from environment variables; `.env.example` is a blank template, not
an automatically loaded configuration. Graphs, terminal output, and enabled debug
reports can contain input text and URLs—review generated files before sharing.

## License

[Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies.

## Citation

If you use SearchAtlas in your research, please cite our paper:

```bibtex
@misc{sang2026searchatlas,
  title         = {{SearchAtlas}: Analyzing Agentic Search Strategies via Evidential Query Graphs},
  author        = {Sang, Jiacheng and Li, Mengyuan and Chen, Sanxing and Huang, Yukun and Feng, Yu and Dhingra, Bhuwan},
  year          = {2026},
  eprint        = {2609.10901},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {Accepted to Findings of EMNLP 2026},
  url           = {https://arxiv.org/abs/2609.10901}
}
```
