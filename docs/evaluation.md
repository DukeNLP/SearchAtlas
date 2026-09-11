# Evaluation

The evaluator runs offline with Python's standard library. It accepts builder
outputs or equivalent graphs, optional constraint annotations, reference DAGs,
and correctness labels. No benchmark-specific directory layout is required.

## Inputs

Files may contain a JSON object, a JSON list, or JSONL records. Each record needs
a unique `case_id`; `task_id` is accepted for builder output. When combining
multiple agents or runs, assign a distinct case ID to each trajectory.

Graph records use `{"case_id": "example", "graph": {"nodes": [...], "edges": [...]}}`.
Nodes need unique string IDs. Query nodes use `type: "Query"` or IDs such as
`q1`; special nodes are `Q0`, `Prior_knowledge`, and `Answer`. Edges need `source`,
`target`, and `edge_kind`. Node texts and turns are optional for structural
metrics. Cycles, unknown edge kinds, and missing endpoints are rejected.

Constraint annotations use this schema:

```json
{
  "case_id": "example",
  "constraint_units": [{"unit_id": "u1", "text": "Example Hall"}],
  "query_unit_ids": {"q1": ["u1"], "q2": []}
}
```

List every query in `query_unit_ids`, including empty memberships. Alternatively,
omit memberships and select a lexical matcher with `--match-rule`:

- `distinctive_terms`: singularized, stopword-filtered terms; multi-term units
  require multiple informative matches or one long distinctive term.
- `token_coverage`: filtered token overlap; one match for units with at most two
  tokens, otherwise at least two matches and half the unit's tokens.

Lexical matching requires unit and query text. Supplied memberships are used
unchanged. Grounding metrics are `null` when no constraint annotation is provided.
Q0 edges alone are not treated as a complete constraint decomposition.

## Graph diagnostics

All answer-support paths below use evidence-derived edges. Let the *frontier* be
the queries with direct evidence edges to `Answer`, and the *answer graph* include
those queries and all their evidence-derived ancestors. The *backbone* is the
union of query-to-query edges on every longest answer-reaching path. The
*primary-path queries* are the endpoints of those backbone edges; when there
are no such edges, this set is the frontier.

| Output metric | Computation |
| --- | --- |
| `answer_backbone_concentration` | Backbone query-to-query edges / all evidence-derived query-to-query edges. |
| `answer_backbone_concentration_within_answer_graph` | Same numerator / answer-graph query-to-query edges. |
| `answer_support_directness` | Direct query-to-answer edges / (direct edges + answer-graph query-to-query edges). |
| `backbone_constraint_share` | Unit deployments on primary-path queries / deployments in the answer graph. A unit deployed at two queries counts twice. |
| `primary_path_coverage` | Distinct units deployed on primary-path queries / all question units. |
| `direct_frontier_coverage` | Distinct units deployed at direct answer parents / all question units. |
| `answer_graph_coverage` | Distinct units deployed anywhere in the answer graph / all question units. |
| `trajectory_coverage` | Distinct units deployed by any query / all question units. |
| `pk_answer_edge` | 1 if a prior-knowledge-derived `Prior_knowledge → Answer` edge exists; otherwise 0. |
| `answer_graph_pk_density` | (Distinct answer-reaching queries with PK parents + the PK-to-answer flag) / answer-reaching queries. |
| `direct_frontier_pk_density` | Same calculation restricted to direct answer parents. |
| `query_efficiency` | Answer-reaching queries / all queries. |

Duplicate edge records do not increase these metrics. PK density can exceed 1;
it is `null` when its query denominator is zero. Other structural ratios with
empty denominators use 0. Outputs include support-node sets and counts so the
calculation scope can be inspected directly.

Primary-path coverage and direct-frontier coverage are distinct outputs. The
sequential graph-based PK flag also differs from checking answer strings in
pre-answer context. This implementation reads PK from graph edges, not saved
risk observations or external edge-selection masks. Use the metric names that
match the analysis being conducted; changing scope can change numerical results.

## Reconstruction against human reference DAGs

`reconstruction` pairs records by case ID. Query IDs must describe the same
occurrences in prediction and reference; provided query texts and turns are
checked for agreement. The evaluator never renumbers nodes automatically.

Precision, recall, and F1 compare exact source–target edge pairs and are computed
separately for every DAG. Each metric is then macro-averaged independently.
Macro F1 is not recomputed from macro precision and macro recall. When both edge
sets are empty, P/R/F1 are 1; when exactly one is empty, they are 0.

Per-edge-type scores use the same procedure. A duplicated pair is assigned its
first recorded edge kind; the output reports how many pairs have conflicting
kinds. Reconstruction scores ignore edge evidence text and model confidence.

## Outcome association

Correctness labels are supplied separately:

```json
{
  "case_id": "example",
  "question_id": "Example-1",
  "dataset": "MyBenchmark",
  "regime": "sequential",
  "model": "MyAgent",
  "correct": 1
}
```

Use `--topology`, `--grounding`, and `--pk-risk` to select named diagnostic
outputs. For a sequential analysis, one explicit combination is
`answer_backbone_concentration`, `backbone_constraint_share`, and `pk_answer_edge`.
For parallel analysis, select `answer_support_directness`, the desired coverage
scope, and the desired PK density scope. `--regime` selects one question regime.
Undefined selected metrics are rejected, not silently dropped or imputed.

Evaluation is performed within each dataset, regime, and agent. Question IDs are
sorted lexicographically and assigned round-robin to five folds by default;
repeated trajectories of one question stay in the same fold. Metric values are
rounded to six decimals before normalization (`--metric-decimals` controls this).
Means and population standard deviations are fitted only on training questions.
Zero standard deviation uses a scale of 1. Topology and grounding enter with a
positive sign, and PK risk with a negative sign.

Reports include single-module, cumulative, and drop-one AUC/AP. Cross-agent
summaries macro-average the within-agent scores and report how many agents have
defined metrics. AUC gives half credit to ties and is undefined for a one-class
cell. Default `--ap-method rank` averages precision at each positive rank and
preserves input order for ties. `--ap-method grouped` treats tied scores as one
threshold; these AP conventions can produce different values.

Auxiliary threshold evaluation fits normalization and a balanced-accuracy
threshold on the training fold, then applies both to held-out questions. No
held-out labels affect that threshold. Reports include accuracy, precision,
recall, F1, confusion counts, and evaluation coverage. A fold whose training
labels contain only one class has no fitted threshold and is reported explicitly.

## Release scope

This is an updated builder and graph-based evaluator. Exact reproduction of a
published table additionally requires its matching graph artifacts, annotations,
and calculation definitions; those datasets are not bundled with this code release.
