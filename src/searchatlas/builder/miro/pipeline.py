"""Run the query-DAG construction workflow."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
from ..llm_compat import LLM_IO_COMPAT_VERSION
from . import settings

def main():
    from .answers import answer_mpsc, answer_signal_provenance_matching, build_answer_signals, build_prior_knowledge_answer_edge, decompose_answer_units, extract_final_answer, extract_final_answer_support_text, extract_minimal_answer, has_direct_whole_answer_provenance, is_abstention_answer, unsupported_answer_unit_ids
    from .attribution import phase1_hybrid_classify
    from .constraints import build_q0_unit_edges, decompose_q0_units, q0_edges_from_tokens_firstuse
    from .evidence import build_constructed_url_pk_edges, build_visible_qid_doc_map, extract_query_docs, validate_turn_visibility_alignment
    from .graph import assemble_dag, validate_dag
    from .llm import _answer_decompose_budget, strict_edge_policy
    from .parsing import extract_queries, tokenize
    from .reports import write_human_debug_report
    from ..safety import task_report_filename, validate_tasks
    from ..privacy import redact_metadata
    parser = argparse.ArgumentParser(description='Build a trace-supported query DAG from tagged search logs')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='query_dag_hybrid_output.json')
    parser.add_argument('--debug-report-dir', type=str, default='', help='Write detailed per-task reports to this directory (disabled by default; includes input text).')
    parser.add_argument('--task-ids', '-t', nargs='+', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=5, help='Top-K sources per token')
    parser.add_argument('--max-snips', type=int, default=2, help='Max provenance windows per source per token')
    parser.add_argument('--sent-window', type=int, default=5, help='Sentence window +/- around token hit')
    parser.add_argument('--keep-tool-result', type=int, default=5, help='Exact Miro tool-result visibility budget. -1 keeps all; K keeps only the most recent K merged tool-result messages.')
    parser.add_argument('--q0-mode', type=str, default='rule', choices=['rule', 'llm'], help="Q0 edge mode: 'rule' = token overlap + first-use (deterministic, no LLM); 'llm' = LLM unit decomposition + first-use matching (1 extra API call)")
    args = parser.parse_args()
    if not settings.API_KEY:
        raise SystemExit('ERROR: OPENAI_API_KEY is not set. Export it before running.')
    print(f'Edge policy: {settings.EDGE_POLICY} ({settings.EDGE_POLICY_VERSION})')
    print(f'LLM I/O: {LLM_IO_COMPAT_VERSION}; mode={settings.LLM_IO_MODE}; json_mode={settings.LLM_JSON_MODE}; temperature={settings.LLM_TEMPERATURE}; min_completion_tokens={settings.LLM_MIN_COMPLETION_TOKENS}')
    print(f'Loading: {args.input}')
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    output_path = Path(args.output)
    validate_tasks(data, args.task_ids)
    debug_report_dir = Path(args.debug_report_dir) if args.debug_report_dir else None
    if debug_report_dir is not None:
        debug_report_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in data if t.get('task_id') in args.task_ids]
    if not tasks:
        print(f'No tasks with IDs: {args.task_ids}')
        return
    print(f'Processing {len(tasks)} task(s)...\n')
    results = []
    for task in tasks:
        tid = task.get('task_id', '?')
        question = task.get('question', '')
        messages = task.get('messages', task.get('trajectory', []))
        print(f"{'=' * 60}")
        print(f'Task: {tid}')
        print(f'Question: {question[:140]}...')
        queries, turn_to_qids = extract_queries(messages)
        qid_to_text = {q['id']: q['text'] for q in queries}
        max_turn = max(turn_to_qids.keys()) if turn_to_qids else 0
        print(f'  Queries: {len(queries)} across {max_turn} turns')
        qid_doc, qid_chunks, turn_thinks, result_counts, qid_search_doc, turn_url_owners, visit_summary_hard_failures, query_url_metadata, visit_match_log, turn_tool_message_cutoff, total_tool_messages, turn_contexts = extract_query_docs(messages, turn_to_qids, qid_to_text)
        print(f'  Query docs: {len(qid_doc)} qids with doc text')
        print(f'  Result counts: {len(result_counts)}/{len(queries)} (best-effort)')
        visibility_validation = validate_turn_visibility_alignment(messages=messages, turn_tool_message_cutoff=turn_tool_message_cutoff, keep_tool_result=args.keep_tool_result)
        if not visibility_validation['ok']:
            raise RuntimeError(f"Tool-result visibility mismatch for {tid}: {visibility_validation['mismatches'][:3]}")
        turn_visible_tool_message_indices = visibility_validation['turn_visible_tool_message_indices']
        final_visible_tool_message_indices = visibility_validation['final_visible_tool_message_indices']
        visible_final_qid_doc = build_visible_qid_doc_map(qid_chunks, visible_tool_message_indices=final_visible_tool_message_indices)
        final_visible_count = 'ALL' if final_visible_tool_message_indices is None else len(final_visible_tool_message_indices)
        print(f'  Tool-result visibility: validated against message history (keep_tool_result={args.keep_tool_result}, total_tool_messages={total_tool_messages}, final_visible={final_visible_count})')
        print(f'\n  === Q0 EDGES (mode={args.q0_mode}) ===')
        q0_units: List[Dict[str, str]] = []
        unit_first_use: Dict[str, str] = {}
        if args.q0_mode == 'llm':
            q0_units = decompose_q0_units(question)
            q0_edges, unit_first_use = build_q0_unit_edges(q0_units, queries, question=question)
            print(f'  Q0 edges (LLM unit tiered match): {len(q0_edges)} edges for {len(q0_units)} units')
            for uid, qid in unit_first_use.items():
                print(f'    {uid} -> {qid}')
            unmatched_units = [u['unit_id'] for u in q0_units if u['unit_id'] not in unit_first_use]
            if unmatched_units:
                print(f'    ⚠ Unmatched Q0 units (no query operationalizes them): {unmatched_units}')
        else:
            q0_edges = q0_edges_from_tokens_firstuse(question, queries)
            print(f'  Q0 edges (rule-based first-use + reuse): {len(q0_edges)}')
            for e in q0_edges:
                print(f"    Q0 -> {e['target']}: {e['evidence'][:100]}")
        print('\n  === TOKENIZATION CHECK ===')
        q0_toks = tokenize(question)
        print(f'  Q0 tokens (sample up to 40): {q0_toks[:40]}')
        for q in queries:
            print(f"  {q['id']} tokens: {tokenize(q['text'])}")
        print(f'\n  Query attribution: evidence, prior knowledge, and failure recovery, model={settings.MODEL}')
        qq_edges_list, per_turn_raw, explicitly_no_source = phase1_hybrid_classify(question=question, messages=messages, queries=queries, turn_to_qids=turn_to_qids, top_k=args.top_k, sent_window=args.sent_window, max_snips=args.max_snips, result_counts=result_counts, qid_search_docs=qid_search_doc, turn_docs=qid_doc, visit_summary_hard_failures=visit_summary_hard_failures, query_url_metadata=query_url_metadata, turn_doc_chunks=qid_chunks, turn_thinks=turn_thinks, turn_visible_tool_message_indices=turn_visible_tool_message_indices, keep_tool_result=args.keep_tool_result, turn_contexts=turn_contexts)
        all_edges = list(q0_edges) + list(qq_edges_list)
        constructed_url_pk_edges = build_constructed_url_pk_edges(visit_match_log)
        if constructed_url_pk_edges:
            all_edges.extend(constructed_url_pk_edges)
            print(f'  Constructed-URL PK edges: {len(constructed_url_pk_edges)}')
            for e in constructed_url_pk_edges:
                print(f"    Prior_knowledge -> {e['target']}: create_url for {e['metadata'].get('visit_url', '')}")
        print('\n  === ANSWER PIPELINE ===')
        answer_text = extract_final_answer(messages, task)
        print(f"  Final answer: {answer_text[:200]}{('…' if len(answer_text) > 200 else '')}")
        answer_units_list: List[Dict[str, str]] = []
        answer_signals_list: List[Dict[str, Any]] = []
        answer_edges: List[Dict] = []
        unsupported_answer_signals: List[str] = []
        unsupported_answer_units: List[str] = []
        answer_support_text = ''
        answer_text_for_units = answer_text
        q_coverage: Dict[str, Any] = {}
        direct_whole_answer_hit = False
        if answer_text:
            answer_support_text = extract_final_answer_support_text(messages, answer_text)
            if answer_support_text:
                print(f'  Final answer support text: {len(answer_support_text)} chars')
            answer_text_for_units = extract_minimal_answer(question, answer_text)
            if answer_text_for_units != answer_text:
                print(f"  Minimal answer: {answer_text_for_units[:200]}{('…' if len(answer_text_for_units) > 200 else '')}")
            if is_abstention_answer(answer_text_for_units):
                print('  Explicit no-answer response: skipping Answer support and PK fallback')
                answer_units_list = []
            else:
                answer_units_list = decompose_answer_units(question, answer_text_for_units)
            if answer_units_list:
                answer_signals_list = build_answer_signals(answer_text=answer_text_for_units, answer_units=answer_units_list)
                print(f'  Answer signals: {len(answer_signals_list)}')
                q_coverage = answer_signal_provenance_matching(answer_signals=answer_signals_list, queries=queries, qid_doc=visible_final_qid_doc, question=question, answer_support_text=answer_support_text, sent_window=3)
                print(f'  Answer provenance: {len(q_coverage)} queries have supporting evidence')
                for qid, info in sorted(q_coverage.items()):
                    supported = info.get('supports_signals', info.get('supports_units', []))
                    granularities = sorted({p.get('granularity', '?') for p in info.get('provenance', [])})
                    print(f'    {qid}: supports {supported}  granularities={granularities}')
                direct_whole_answer_hit = has_direct_whole_answer_provenance(answer_signals=answer_signals_list, q_coverage=q_coverage)
                print(f'  Whole-answer direct provenance hit: {direct_whole_answer_hit}')
                answer_edges, unsupported_answer_signals = answer_mpsc(answer_signals=answer_signals_list, q_coverage=q_coverage, queries=queries, q_q_edges=qq_edges_list)
                unsupported_answer_units = unsupported_answer_unit_ids(answer_units=answer_units_list, answer_signals=answer_signals_list, answer_edges=answer_edges)
                print(f'  Answer-MPSC: {len(answer_edges)} q→A edges')
                if unsupported_answer_signals:
                    print(f'    ⚠ Unsupported answer signals (no provenance): {unsupported_answer_signals}')
                if not answer_edges:
                    if q_coverage and (not strict_edge_policy()):
                        print('    ↳ Weak query provenance remains unresolved; leaving Answer orphan instead of forcing PK support')
                    else:
                        pk_answer_edge = build_prior_knowledge_answer_edge(answer_text=answer_text, answer_units=answer_units_list, unsupported_units=[u['unit_id'] for u in answer_units_list], answer_signals=answer_signals_list, mode='orphan_fallback')
                        answer_edges = [pk_answer_edge]
                        print('    ↳ Added Prior_knowledge -> Answer fallback (no query provenance)')
                elif strict_edge_policy() and unsupported_answer_units:
                    pk_answer_edge = build_prior_knowledge_answer_edge(answer_text=answer_text, answer_units=answer_units_list, unsupported_units=unsupported_answer_units, answer_signals=answer_signals_list, mode='synthesis_additive')
                    answer_edges.append(pk_answer_edge)
                    print(f'    ↳ Added Prior_knowledge -> Answer for unsupported answer units: {unsupported_answer_units}')
                all_edges.extend(answer_edges)
        else:
            print('  ⚠ No final answer found — skipping q→A edges')
        kind_counts = defaultdict(int)
        for e in all_edges:
            kind_counts[e.get('edge_kind', '?')] += 1
        print(f'\n  Total edges before assembly: {len(all_edges)}')
        for k, v in sorted(kind_counts.items()):
            print(f'    {k}: {v}')
        if explicitly_no_source:
            print(f'    no_source_found (orphans expected under contribution-only): {explicitly_no_source}')
        graph = assemble_dag(question, queries, all_edges, answer_text=answer_text, answer_units=answer_units_list)
        issues = validate_dag(graph)
        fatal = [issue for issue in issues if not issue.startswith('ORPHAN:')]
        if fatal:
            raise ValueError(f'Invalid graph for {tid}: {fatal}')
        q0_e = sum((1 for e in graph['edges'] if e['source'] == 'Q0'))
        pk_e = sum((1 for e in graph['edges'] if e['source'] == 'Prior_knowledge'))
        qq_e = sum((1 for e in graph['edges'] if e['source'].startswith('q') and e['target'].startswith('q')))
        qa_e = sum((1 for e in graph['edges'] if e['target'] == 'Answer'))
        sf_e = sum((1 for e in graph['edges'] if e.get('edge_kind') == 'failure_derived' and e.get('failure_subtype') == 'soft'))
        hf_e = sum((1 for e in graph['edges'] if e.get('edge_kind') == 'failure_derived' and e.get('failure_subtype') == 'hard'))
        edge_rejection_counts = Counter((rejected.get('reason', 'unknown') for turn_debug in per_turn_raw for rejected in turn_debug.get('rejected_edges', [])))
        print(f'\n  FINAL (after assembly):')
        print(f"    Nodes: {len(graph['nodes'])}")
        print(f"    Edges: {len(graph['edges'])}")
        print(f'      Q0->Q:             {q0_e}')
        print(f'      Prior_knowledge:   {pk_e}')
        print(f'      Q->Q (total):      {qq_e}')
        print(f'        soft_fail:       {sf_e}')
        print(f'        hard_fail/retry: {hf_e}')
        print(f'      Q->Answer:         {qa_e}')
        if issues:
            print(f'    Issues ({len(issues)}):')
            for iss in issues[:30]:
                print(f'      - {iss}')
            if len(issues) > 30:
                print(f'      ... ({len(issues) - 30} more)')
        else:
            print('    ✓ No validation issues')
        if debug_report_dir is not None:
            debug_report_path = debug_report_dir / task_report_filename(tid)
            write_human_debug_report(debug_report_path, task_id=tid, question=question, queries=queries, turn_to_qids=turn_to_qids, qid_doc=qid_doc, turn_thinks=turn_thinks, result_counts=result_counts, turn_url_owners=turn_url_owners, visit_match_log=visit_match_log, q0_edges=q0_edges, q0_units=q0_units, unit_first_use=unit_first_use, per_turn_raw=per_turn_raw, answer_text=answer_text, answer_support_text=answer_support_text, answer_units=answer_units_list, answer_signals=answer_signals_list, answer_coverage=q_coverage, answer_edges=answer_edges, graph=graph, issues=issues)
            print(f'    Debug report: {debug_report_path}')
        results.append({'task_id': tid, 'question': question, 'answer_text': answer_text, 'graph': graph, 'stats': {'num_queries': len(queries), 'num_turns': max_turn, 'num_edges': len(graph['edges']), 'total_tool_messages': total_tool_messages, 'final_visible_tool_messages': -1 if final_visible_tool_message_indices is None else len(final_visible_tool_message_indices), 'q0_edges': q0_e, 'q0_units': len(q0_units), 'q0_units_matched': len(unit_first_use), 'prior_edges': pk_e, 'qq_edges': qq_e, 'qa_edges': qa_e, 'soft_fail_edges': sf_e, 'hard_fail_edges': hf_e, 'edge_kinds': dict(kind_counts), 'phase1_no_source': explicitly_no_source, 'answer_units': len(answer_units_list), 'answer_signals': len(answer_signals_list), 'answer_signals_unsupported': unsupported_answer_signals, 'answer_units_unsupported': unsupported_answer_units, 'edge_policy_rejections': dict(edge_rejection_counts), 'issues': issues}, 'q0_units': q0_units, 'unit_first_use': unit_first_use, 'answer_units': answer_units_list, 'answer_signals': answer_signals_list, 'per_turn_raw': per_turn_raw, 'settings': {'top_k': args.top_k, 'max_snips': args.max_snips, 'sent_window': args.sent_window, 'keep_tool_result': args.keep_tool_result, 'tool_visibility_validated': visibility_validation['ok'], 'model': settings.MODEL, 'edge_policy': settings.EDGE_POLICY, 'edge_policy_version': settings.EDGE_POLICY_VERSION, 'llm_io_compat_version': LLM_IO_COMPAT_VERSION, 'llm_io_mode': settings.LLM_IO_MODE, 'llm_temperature': settings.LLM_TEMPERATURE, 'llm_json_mode': settings.LLM_JSON_MODE, 'llm_min_completion_tokens': settings.LLM_MIN_COMPLETION_TOKENS, 'llm_extra_body': redact_metadata(settings.LLM_EXTRA_BODY, secrets=(settings.API_KEY,)), 'answer_decompose_base_tokens': settings.ANSWER_DECOMPOSE_BASE_TOKENS, 'answer_decompose_max_tokens': settings.ANSWER_DECOMPOSE_MAX_TOKENS, 'answer_decompose_effective_tokens': _answer_decompose_budget(answer_text_for_units)}})
    print(f'\nSaving to: {args.output}')
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Done!')
