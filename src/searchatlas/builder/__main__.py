"""Build an evidence-dependency DAG; model calls require --execute."""

import argparse
import importlib
import json
import os
import sys

from searchatlas import __version__


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--agent', required=True,
                        choices=['tydp', 'tydp_gpt5', 'tydp_qwen3', 'websailor', 'mirothinker'],
                        help='Input-format adapter, independent of the attribution model')
    parser.add_argument('--input', required=True, help='JSON task file')
    parser.add_argument('--output', required=True, help='Output DAG JSON file')
    parser.add_argument('--task-ids', nargs='+', required=True, help='Task IDs to process')
    parser.add_argument('--execute', action='store_true',
                        help='Make model calls; otherwise print the command plan only')
    parser.add_argument('--keep-tool-result', type=int, default=5,
                        help='Miro: prior tool results visible per query; -1 keeps all, 0 keeps none')
    parser.add_argument('--q0-mode', choices=['rule', 'llm'], default='rule',
                        help='Question constraints: lexical rules or LLM-assisted units')
    parser.add_argument('--top-k', type=positive_int, default=5,
                        help='Maximum candidate sources per token')
    parser.add_argument('--max-snips', type=positive_int, default=2,
                        help='Maximum provenance windows per source/token')
    parser.add_argument('--sent-window', type=positive_int, default=5,
                        help='Context sentences around a token match')
    parser.add_argument('--debug-report-dir', default='',
                        help='Opt in to detailed text reports (may contain sensitive input)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.keep_tool_result < -1:
        parser.error('--keep-tool-result must be -1 or nonnegative')
    adapter = 'miro' if args.agent == 'mirothinker' else 'standard'
    argv = [
        '--input', args.input, '--output', args.output,
        '--task-ids', *args.task_ids, '--q0-mode', args.q0_mode,
        '--top-k', str(args.top_k), '--max-snips', str(args.max_snips),
        '--sent-window', str(args.sent_window),
    ]
    if adapter == 'miro':
        argv.extend(['--keep-tool-result', str(args.keep_tool_result)])
    if args.debug_report_dir:
        argv.extend(['--debug-report-dir', args.debug_report_dir])
    if not args.execute:
        print(json.dumps({
            'agent': args.agent, 'adapter': adapter, 'arguments': argv,
            'model': os.getenv('OPENAI_MODEL', 'gpt-5.2'), 'execute': False,
        }, indent=2))
        return
    if not os.getenv('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY before --execute')
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        importlib.import_module(f'.{adapter}.pipeline', __package__).main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
