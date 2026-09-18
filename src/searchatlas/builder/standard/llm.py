"""Configure model requests, JSON parsing, and error handling."""
from __future__ import annotations
from typing import Dict, Any
from ..llm_compat import answer_decomposition_token_budget, completion_token_budget, parse_llm_json_text
from . import settings
from ..privacy import redact_text

def strict_edge_policy() -> bool:
    return True


def _parse_model_json(text: str, expected: type) -> Any:
    return parse_llm_json_text(text, expected=expected)

def _answer_decompose_budget(answer_text: str) -> int:
    return answer_decomposition_token_budget(answer_text, base_tokens=settings.ANSWER_DECOMPOSE_BASE_TOKENS, max_tokens=settings.ANSWER_DECOMPOSE_MAX_TOKENS)

def _completion_length_kwargs(max_tokens: int) -> Dict[str, Any]:
    max_tokens = completion_token_budget(max_tokens, settings.LLM_MIN_COMPLETION_TOKENS)
    if 'api.openai.com' in (settings.BASE_URL or ''):
        return {'max_completion_tokens': max_tokens}
    return {'max_tokens': max_tokens}

def _apply_llm_io_options(kwargs: Dict[str, Any], *, json_object: bool) -> None:
    if json_object and settings.LLM_JSON_MODE:
        kwargs['response_format'] = {'type': 'json_object'}
    if settings.LLM_EXTRA_BODY:
        kwargs['extra_body'] = settings.LLM_EXTRA_BODY

def _response_format_is_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return 'response_format' in text and any((marker in text for marker in ('unsupported', 'unknown', 'invalid', 'not support')))

def _create_chat_completion(kwargs: Dict[str, Any], *, call_name: str = '') -> Any:
    if settings.LLM_BACKEND != 'api':
        from ..cli_backend import create_cli_completion
        return create_cli_completion(settings.LLM_BACKEND, kwargs, call_name=call_name)
    try:
        return settings.client.chat.completions.create(**kwargs)
    except Exception as exc:
        if 'response_format' not in kwargs or not _response_format_is_unsupported(exc):
            raise
        fallback = dict(kwargs)
        fallback.pop('response_format', None)
        print('    JSON mode unsupported by endpoint; retrying without response_format')
        return settings.client.chat.completions.create(**fallback)

def _is_non_retryable_llm_error(e: Exception) -> bool:
    if getattr(e, 'non_retryable', False):
        return True
    status_code = getattr(e, 'status_code', None)
    response = getattr(e, 'response', None)
    response_status = getattr(response, 'status_code', None) if response is not None else None
    code = status_code if status_code is not None else response_status
    if code in {400, 401, 403, 404, 422}:
        return True
    err_text = str(e).lower()
    return 'unsupported parameter' in err_text or 'invalid_request_error' in err_text or 'authentication' in err_text or ('incorrect api key' in err_text) or ('model_not_found' in err_text)

def _is_connection_like_llm_error(e: Exception) -> bool:
    status_code = getattr(e, 'status_code', None)
    response = getattr(e, 'response', None)
    response_status = getattr(response, 'status_code', None) if response is not None else None
    code = status_code if status_code is not None else response_status
    if code in {408, 429, 500, 502, 503, 504}:
        return True
    err_text = str(e).lower()
    markers = ('connection error', 'apiconnectionerror', 'timeout', 'timed out', 'read timeout', 'gateway', 'temporarily unavailable', 'rate limit', 'server error')
    return any((m in err_text for m in markers))

def _format_llm_exception(e: Exception) -> str:
    parts = [f'{type(e).__name__}: {e!s}', f'repr={e!r}']
    if getattr(e, '__cause__', None) is not None:
        parts.append(f'cause={type(e.__cause__).__name__}: {e.__cause__!r}')
    if getattr(e, '__context__', None) is not None:
        parts.append(f'context={type(e.__context__).__name__}: {e.__context__!r}')
    text = ' | '.join(parts)
    return redact_text(text, secrets=(settings.API_KEY,))
