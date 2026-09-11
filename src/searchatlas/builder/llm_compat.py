#!/usr/bin/env python3
"""Model response parsing, token budgets, and optional usage logging."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .privacy import redact_metadata


LLM_IO_COMPAT_VERSION = "multi-model-compat-v1-20260805"


def env_flag(value: str, default: bool = False) -> bool:
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_json_object(value: str, *, label: str = "JSON object") -> Dict[str, Any]:
    value = (value or "").strip()
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must decode to an object")
    return parsed


def _clean_json_response(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned


def _extract_json_candidate(cleaned: str, opener: str, start: int) -> Optional[str]:
    if not cleaned or start < 0 or start >= len(cleaned) or cleaned[start] != opener:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return cleaned[start:]


def _escape_unquoted_inner_quotes(blob: str) -> str:
    output = []
    in_string = False
    escaped = False
    length = len(blob)
    for index, char in enumerate(blob):
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            output.append(char)
            escaped = False
        elif char == "\\":
            output.append(char)
            escaped = True
        elif char != '"':
            output.append(char)
        else:
            lookahead = index + 1
            while lookahead < length and blob[lookahead].isspace():
                lookahead += 1
            next_char = blob[lookahead] if lookahead < length else ""
            if not next_char or next_char in ':,}]"':
                output.append(char)
                in_string = False
            else:
                output.append('\\"')
    return "".join(output)


def _repair_common_json_breaks(blob: str) -> str:
    repaired = blob.strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)
    repaired = _escape_unquoted_inner_quotes(repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"}\s*{", "},{", repaired)
    repaired = re.sub(
        r'([}\]"]|(?:\btrue\b|\bfalse\b|\bnull\b)|-?\d+(?:\.\d+)?)\s+'
        r'("[A-Za-z_][A-Za-z0-9_ -]*"\s*:)',
        r"\1,\2",
        repaired,
    )
    return repaired


def parse_llm_json_text(text: str, expected: type = dict) -> Any:
    """Parse one balanced JSON value and repair only common syntax slips."""
    if expected not in {dict, list}:
        raise TypeError("expected must be dict or list")
    opener = "{" if expected is dict else "["
    cleaned = _clean_json_response(text)
    last_error: Optional[Exception] = None
    for start, char in enumerate(cleaned):
        if char != opener:
            continue
        blob = _extract_json_candidate(cleaned, opener, start)
        if not blob:
            continue
        try:
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                parsed = json.loads(_repair_common_json_breaks(blob))
            if not isinstance(parsed, expected):
                raise TypeError(f"Expected {expected.__name__}, got {type(parsed).__name__}")
            return parsed
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON candidate found", text or "", 0)


def completion_token_budget(requested: int, minimum: int = 0) -> int:
    return max(1, int(requested), int(minimum))


def answer_decomposition_token_budget(
    answer_text: str,
    *,
    base_tokens: int = 1500,
    max_tokens: int = 3000,
) -> int:
    """Scale JSON output space for long answers while retaining a hard cap."""
    base_tokens = max(1, int(base_tokens))
    max_tokens = max(base_tokens, int(max_tokens))
    estimated = 600 + math.ceil(len(answer_text or "") / 3)
    return min(max_tokens, max(base_tokens, estimated))


def _usage_as_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def _finish_reason(response: Any) -> str:
    try:
        return str(response.choices[0].finish_reason or "")
    except Exception:
        return ""


def record_llm_event(
    path: str,
    *,
    event: str,
    call_name: str,
    model: str,
    prompt_chars: int,
    max_tokens: int,
    attempt: int,
    response: Any = None,
    response_chars: int = 0,
    error: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a compact request/parse event without logging prompt contents."""
    if not path:
        return
    usage = _usage_as_dict(getattr(response, "usage", None)) if response is not None else {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    row = {
        "ts": time.time(),
        "llm_io_compat_version": LLM_IO_COMPAT_VERSION,
        "event": event,
        "call": call_name,
        "case_id": os.getenv("LLM_USAGE_CASE", ""),
        "model": model,
        "attempt": int(attempt),
        "prompt_chars": int(prompt_chars),
        "response_chars": int(response_chars),
        "max_tokens": int(max_tokens),
        "finish_reason": _finish_reason(response),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens) or 0,
        "error": (error or "")[:1000],
    }
    if metadata:
        row["metadata"] = metadata
    row = redact_metadata(row, secrets=(os.getenv("OPENAI_API_KEY", ""),))
    try:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"    WARNING: failed to record LLM event: {exc}")
