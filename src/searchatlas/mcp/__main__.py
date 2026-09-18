"""Serve SearchAtlas as a local STDIO MCP server."""

import argparse
import sys

from .server import create_server, run_stdio
from .service import AnalysisService


def positive_number(value):
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('Must be a finite positive number')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, help='Explicit root of user-selected JSON inputs')
    parser.add_argument('--output-dir', help='Private job subdirectory inside workspace; default .searchatlas-mcp')
    parser.add_argument('--backend', choices=['api', 'codex_cli', 'claude_cli'], default='codex_cli')
    parser.add_argument('--model', help='Optional model override; fixed for this server session')
    parser.add_argument('--llm-timeout', type=positive_number, default=180)
    parser.add_argument('--job-timeout', type=positive_number, default=3600)
    parser.add_argument('--q0-mode', choices=['rule', 'llm'], default='rule')
    parser.add_argument('--keep-tool-result', type=int, default=5,
                        help='Miro evidence visibility: -1 all results; nonnegative N keeps the latest N')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        service = AnalysisService(args.workspace, args.output_dir, backend=args.backend,
                                  model=args.model, llm_timeout=args.llm_timeout, job_timeout=args.job_timeout,
                                  q0_mode=args.q0_mode, keep_tool_result=args.keep_tool_result)
        server = create_server(service)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print('SearchAtlas MCP ready (stdio). Analysis uses the configured model; evaluation is offline.', file=sys.stderr)
    import anyio
    anyio.run(run_stdio, server, service)


if __name__ == '__main__':
    main()
