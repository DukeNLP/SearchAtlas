#!/usr/bin/env python3
"""Offline CLI fixture for Example Hall only. Never invokes a model or network."""
import json
from pathlib import Path
import re
import sys


FLAGS = '''--ignore-user-config --ephemeral --output-schema --json --sandbox
--skip-git-repo-check --safe-mode --tools --disallowedTools --strict-mcp-config
--mcp-config --settings --setting-sources --json-schema --no-session-persistence
--output-format --max-turns --permission-mode --system-prompt-file
--disable-slash-commands'''
FEATURES = '''shell_tool unified_exec multi_agent apps plugins hooks memories
skill_search skill_mcp_dependency_install code_mode code_mode_host browser_use
browser_use_external computer_use image_generation view_image in_app_browser
in_app_local_automation skip_host_skill_discovery'''
SENTENCE = 'The fictional Example Hall opened in 1942.'


def main():
    args = sys.argv[1:]
    if '--help' in args:
        print(FLAGS)
        return
    if args == ['features', 'list']:
        print('\n'.join(f'{name} stable true' for name in FEATURES.split()))
        return
    backend = 'codex_cli' if args and args[0] == 'exec' else 'claude_cli'
    if '--model' in args and args[args.index('--model') + 1] == 'simulate-failure':
        raise SystemExit(9)
    if backend == 'codex_cli':
        schema = json.loads(Path(args[args.index('--output-schema') + 1]).read_text())
    else:
        schema = json.loads(args[args.index('--json-schema') + 1])
    raw = sys.stdin.read()
    messages = json.loads(raw.split('Task messages:\n', 1)[1])
    prompt = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
    # Make accidental use as a real attribution backend fail visibly.
    if 'Example Hall' not in prompt:
        raise SystemExit('This fixture supports only the synthetic Example Hall task')
    properties = schema['properties']
    if 'units' in properties:
        call = 'q0_decompose'
        payload = {'units': [{'unit_id': 'u1', 'unit_type': 'target_type',
                              'q0_span': 'fictional Example Hall'}]}
    elif 'minimal_answer' in properties:
        call = 'minimal_answer_extract'
        payload = {'minimal_answer': '1942'}
    elif 'answer_units' in properties:
        call = 'answer_decompose'
        payload = {'answer_units': [{'unit_id': 'a1', 'claim': '1942', 'unit_type': 'date'}]}
    elif 'edges' in properties:
        call = 'query_edge_attribution'
        payload = {'edges': [], 'no_source_found': ['q1']}
    elif 'selected_qids' in properties:
        call = 'answer_distributed_support'
        payload = {'selected_qids': ['q1'], 'confidence': 'high', 'reason': 'The excerpt gives the opening date.'}
    elif 'match' in properties['verdicts']['items']['properties']:
        call = 'q0_match_arbiter'
        payload = {'verdicts': [{'pair_id': pair, 'match': True, 'reason': 'The query targets Example Hall.'}
                                for pair in re.findall(r'pair_id=([^\s]+)', prompt)]}
    else:
        call = 'answer_support_match'
        payload = {'verdicts': [{'pair_id': pair, 'supports': True, 'confidence': 'high',
                                 'reason': 'The excerpt states the opening year.', 'evidence_sentence': SENTENCE}
                                for pair in re.findall(r'pair_id=([^\n]+)', prompt)]}
    with Path(__file__).with_suffix('.calls.jsonl').open('a') as handle:
        handle.write(json.dumps({'backend': backend, 'call': call}) + '\n')
    if backend == 'codex_cli':
        print(json.dumps({'type': 'thread.started', 'thread_id': 'offline-fixture'}))
        print(json.dumps({'type': 'item.completed', 'item': {
            'type': 'agent_message', 'text': json.dumps(payload)}}))
        print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 30, 'output_tokens': 10}}))
    else:
        print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                          'structured_output': payload, 'usage': {'input_tokens': 30, 'output_tokens': 10}}))


if __name__ == '__main__':
    main()
