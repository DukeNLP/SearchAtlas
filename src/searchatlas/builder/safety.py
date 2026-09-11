"""Validate input schemas, output filenames, and evidence availability."""
from copy import deepcopy
import hashlib
import re


def task_report_filename(task_id):
    """Keep user-supplied task IDs inside the selected report directory."""
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', task_id)[:80]
    if safe != task_id:
        safe += '_' + hashlib.sha256(task_id.encode('utf-8')).hexdigest()[:12]
    return f'{safe}_debug_report.txt'


class AttributionUnavailable(RuntimeError):
    """A failed attribution request must not be mistaken for absent evidence."""


def validate_tasks(data, requested_ids):
    """Reject unsupported log formats instead of silently treating them as no search."""
    if not isinstance(data, list) or any(not isinstance(task, dict) for task in data):
        raise ValueError('Input must be a task object or a list of task objects')
    ids = [task.get('task_id') for task in data]
    if any(not isinstance(tid, str) or not tid for tid in ids) or len(ids) != len(set(ids)):
        raise ValueError('Each input task needs a unique nonempty string task_id')
    missing = set(requested_ids) - set(ids)
    if missing:
        raise ValueError(f'Requested tasks not found: {sorted(missing)}')
    for task in data:
        if task['task_id'] not in requested_ids:
            continue
        if not isinstance(task.get('question'), str) or not task['question'].strip():
            raise ValueError('Each selected task needs question text')
        messages = task.get('messages', task.get('trajectory'))
        if not isinstance(messages, list):
            raise ValueError('Each selected task needs a messages list')
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get('role') not in {'system', 'user', 'assistant'}
                    or not isinstance(message.get('content'), str)
                    or message.get('tool_calls') or message.get('function_call')):
                raise ValueError('Unsupported message format: adapt native tool calls/results to the documented tagged-text schema')


def evidence_snapshot(docs, counts, search_docs, visit_failures, urls):
    """Freeze only information available before the next query is issued.

    Document strings are immutable. Mutable failure/URL records need independent
    copies so that later visits cannot change an earlier attribution context.
    """
    return {
        'docs': dict(docs),
        'result_counts': dict(counts),
        'search_docs': dict(search_docs),
        'visit_failures': deepcopy(dict(visit_failures)),
        'urls': deepcopy(dict(urls)),
    }


def unresolved_response(turn, reason, query_text=''):
    """Keep uncertainty visible without assigning an arbitrary owner."""
    return {'turn': turn, 'action': 'UNRESOLVED_RESPONSE', 'owner_qid': None,
            'visit_url': '', 'reason': reason, 'query_text': query_text}
