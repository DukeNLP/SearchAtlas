"""Select an inference transport without changing graph-construction rules."""

import argparse
import math
import os


BACKENDS = ('api', 'codex_cli', 'claude_cli')


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be a finite positive number of seconds')
    return number


def default_model(backend):
    if backend == 'api':
        return os.getenv('OPENAI_MODEL', 'gpt-5.2')
    # CLI model names and entitlements belong to the user's CLI installation.
    return os.getenv('SEARCHATLAS_CLI_MODEL') or None


def add_backend_arguments(parser):
    parser.add_argument('--backend', choices=BACKENDS,
                        default=os.getenv('SEARCHATLAS_LLM_BACKEND', 'api'),
                        help='Inference transport; CLI backends use saved CLI login')
    parser.add_argument('--model', default=None,
                        help='Attribution model; otherwise API/CLI defaults apply')
    parser.add_argument('--llm-timeout', type=positive_seconds, default=None,
                        help='Timeout in seconds for each inference call')


def configure_backend(settings, args):
    if args.backend not in BACKENDS:
        raise SystemExit('Unknown inference backend; choose api, codex_cli, or claude_cli')
    settings.LLM_BACKEND = args.backend
    settings.MODEL = args.model or default_model(args.backend)
    if args.llm_timeout is not None:
        settings.LLM_TIMEOUT_SEC = args.llm_timeout
    if not math.isfinite(settings.LLM_TIMEOUT_SEC) or settings.LLM_TIMEOUT_SEC <= 0:
        raise SystemExit('LLM timeout must be finite and positive')
    if args.backend == 'api' and not settings.API_KEY:
        raise SystemExit('ERROR: OPENAI_API_KEY is not set. Export it before running the API backend.')
