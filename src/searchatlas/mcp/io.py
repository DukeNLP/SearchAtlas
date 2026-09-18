"""Workspace-scoped JSON inputs and private, atomic job artifacts."""

import json
import os
from pathlib import Path
import re

from searchatlas.builder.privacy import redact_text
from searchatlas.builder.safety import validate_tasks


MAX_INPUT_BYTES = 64 * 1024 * 1024
_SENSITIVE_PART = re.compile(r"(?:^|[._-])(?:credentials?|secrets?|tokens?|api[._-]?keys?|auth|config)(?:$|[._-])", re.I)


def private_directory(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Job directory must be a real directory')
    path.chmod(0o700)
    return path


def write_json(path, value):
    """Replace a JSON artifact without exposing partially written results."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def workspace_input(workspace, value, output_dir):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Provide a JSON or JSONL path inside the configured workspace')
    supplied = Path(value).expanduser()
    lexical = supplied if supplied.is_absolute() else workspace / supplied
    try:
        relative = lexical.relative_to(workspace)
    except ValueError:
        raise ValueError('Input path is outside the configured workspace') from None
    if any(part in {'..', '.'} or part.startswith('.') or _SENSITIVE_PART.search(part)
           for part in relative.parts):
        raise ValueError('Hidden, credential, and configuration paths are not permitted')
    try:
        resolved = lexical.resolve(strict=True)
    except OSError:
        raise ValueError('Input file is unavailable') from None
    if not resolved.is_relative_to(workspace) or resolved.is_relative_to(output_dir):
        raise ValueError('Input must be a user-provided workspace file outside the job directory')
    resolved_relative = resolved.relative_to(workspace)
    if any(part.startswith('.') or _SENSITIVE_PART.search(part) for part in resolved_relative.parts):
        raise ValueError('Symlink resolves to a protected path')
    if resolved.suffix.lower() not in {'.json', '.jsonl'} or not resolved.is_file():
        raise ValueError('Input must be a regular JSON or JSONL file')
    if resolved.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError('Input exceeds the 64 MiB file limit')
    return resolved


def load_records(path):
    """Load with the same case-ID rules as the public evaluator."""
    from searchatlas.evaluation.__main__ import read_records
    return read_records(path)


def sanitized_tasks(records, task_ids):
    if (not isinstance(task_ids, list) or not 1 <= len(task_ids) <= 100
            or any(not isinstance(tid, str) or not tid or tid.startswith('-') for tid in task_ids)
            or len(task_ids) != len(set(task_ids))):
        raise ValueError('task_ids must contain 1–100 unique nonempty IDs, not CLI options')
    selected = []
    for tid in task_ids:
        if tid not in records:
            raise ValueError('A requested task ID is missing from the input')
        item = records[tid]
        if item.get('task_id') != tid:
            raise ValueError('Builder input requires task_id to match the selected record ID')
        validate_tasks([item], [tid])
        # Pass only the documented trace schema, never gold answers or metadata.
        clean = {'task_id': tid, 'question': item['question'],
                 'messages': [{'role': message['role'], 'content': message['content']}
                              for message in item.get('messages', item.get('trajectory', []))]}
        if isinstance(item.get('prediction'), str):
            clean['prediction'] = item['prediction']
        encoded = json.dumps(clean, ensure_ascii=False)
        if redact_text(encoded) != encoded:
            raise ValueError('Trace contains credential-like text; sanitize it before model processing')
        selected.append(clean)
    return selected


def scrub_graph_record(case_id, payload):
    """Retain evaluation fields; omit raw prompts, tool payloads, and metadata."""
    from searchatlas.evaluation.graphs import validate_graph
    graph = validate_graph(payload)
    node_fields = ('id', 'type', 'text', 'turn')
    edge_fields = ('source', 'target', 'edge_kind', 'failure_subtype', 'confidence', 'evidence', 'covered_signals')
    return {'case_id': case_id, 'graph': {
        'nodes': [{key: node[key] for key in node_fields if key in node} for node in graph['nodes']],
        'edges': [{key: edge[key] for key in edge_fields if key in edge} for edge in graph['edges']],
    }}
