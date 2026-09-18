"""Closed response schemas shared by the CLI attribution backends.

These constrain representation, not evidence validity. The builder's existing
source, time, coverage, and citation gates remain authoritative.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _object(**properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


_TEXT = {"type": "string"}
_BOOL = {"type": "boolean"}
_CONFIDENCE = {"type": "string", "enum": ["high", "uncertain"]}
_SCHEMAS = {
    "minimal_answer_extract": _object(minimal_answer=_TEXT),
    "q0_decompose": _object(units=_array(_object(
        unit_id=_TEXT, unit_type=_TEXT, q0_span=_TEXT))),
    "answer_decompose": _object(answer_units=_array(_object(
        unit_id=_TEXT, claim=_TEXT, unit_type={"type": "string", "enum": [
            "entity_name", "date", "number", "attribute", "description"]}))),
    # Structured-output APIs require an object root; unwrap to the array expected
    # by the existing arbiter only after validating this envelope.
    "q0_match_arbiter": _object(verdicts=_array(_object(
        pair_id=_TEXT, match=_BOOL, reason=_TEXT))),
    "answer_support_match": _object(verdicts=_array(_object(
        pair_id=_TEXT, supports=_BOOL, confidence=_CONFIDENCE,
        reason=_TEXT, evidence_sentence=_TEXT))),
    "answer_distributed_support": _object(
        selected_qids=_array(_TEXT), confidence=_CONFIDENCE, reason=_TEXT),
    "query_edge_attribution": _object(
        edges=_array(_object(
            source=_TEXT, target=_TEXT,
            edge_kind={"type": "string", "enum": ["evidence_derived",
                       "prior_knowledge_derived", "failure_derived"]},
            failure_subtype={"type": ["string", "null"], "enum": ["soft", None]},
            covered_signals=_array(_TEXT), evidence=_TEXT,
            validation_reason={"type": "string", "enum": [
                "retrieved_clue_reused", "unsupported_signal_introduced",
                "explicit_failure_repair"]}, confidence=_CONFIDENCE)),
        no_source_found=_array(_TEXT)),
}


def response_schema(call_name: str) -> dict[str, Any]:
    """Return a fresh schema; unknown tasks must not use unconstrained output."""
    if call_name not in _SCHEMAS:
        raise ValueError("No CLI response schema is registered for this task")
    return deepcopy(_SCHEMAS[call_name])


def validate_response(value: Any, schema: dict[str, Any]) -> None:
    """Validate the small JSON Schema subset generated above, without extras.

    Error messages deliberately omit model-controlled content and field values.
    """
    expected = schema["type"]
    accepted = expected if isinstance(expected, list) else [expected]
    kinds = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "null": value is None}
    if not any(kinds.get(kind, False) for kind in accepted):
        raise ValueError("CLI response has an invalid field type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("CLI response contains an unsupported enum value")
    if isinstance(value, dict):
        properties = schema["properties"]
        if set(value) != set(properties):
            raise ValueError("CLI response has missing or unexpected fields")
        for key, child_schema in properties.items():
            validate_response(value[key], child_schema)
    elif isinstance(value, list):
        for item in value:
            validate_response(item, schema["items"])
