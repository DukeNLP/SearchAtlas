"""Bounded, workspace-scoped background jobs for a local MCP server."""

import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import uuid

from searchatlas.builder.privacy import redact_metadata

from .io import (load_records, private_directory, read_json, sanitized_tasks,
                 scrub_graph_record, workspace_input, write_json)


AGENTS = {'tydp', 'tydp_gpt5', 'tydp_qwen3', 'websailor', 'mirothinker'}
BACKENDS = {'api', 'codex_cli', 'claude_cli'}
MATCH_RULES = {None, 'distinctive_terms', 'token_coverage'}
TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def now():
    return datetime.now(timezone.utc).isoformat()


class AnalysisService:
    """Host code fixes the model backend; callers select only analysis inputs."""

    def __init__(self, workspace, output_dir=None, *, backend='codex_cli', model=None,
                 llm_timeout=180, job_timeout=3600, max_jobs=10,
                 q0_mode='rule', keep_tool_result=5):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError('workspace must be an existing directory')
        selected = Path(output_dir).expanduser() if output_dir else self.workspace / '.searchatlas-mcp'
        if not selected.is_absolute():
            selected = self.workspace / selected
        if selected.is_symlink():
            raise ValueError('output-dir cannot be a symlink')
        self.output_dir = selected.resolve()
        if self.output_dir == self.workspace or not self.output_dir.is_relative_to(self.workspace):
            raise ValueError('output-dir must be a subdirectory of workspace')
        private_directory(self.output_dir)
        if backend not in BACKENDS:
            raise ValueError('Unknown model backend')
        if model is not None and (not isinstance(model, str) or not model.strip() or model.startswith('-')):
            raise ValueError('model must be a nonempty model name, not a CLI option')
        if not all(math.isfinite(value) and value > 0 for value in (llm_timeout, job_timeout)) or max_jobs < 1:
            raise ValueError('Timeouts and max_jobs must be positive')
        if q0_mode not in {'rule', 'llm'} or not isinstance(keep_tool_result, int) or keep_tool_result < -1:
            raise ValueError('Use q0_mode rule|llm and keep_tool_result >= -1')
        self.backend, self.model = backend, model
        self.llm_timeout, self.job_timeout, self.max_jobs = llm_timeout, job_timeout, max_jobs
        self.q0_mode, self.keep_tool_result = q0_mode, keep_tool_result
        self.jobs, self.tasks, self.processes = {}, {}, {}
        self._slot = asyncio.Semaphore(1)
        self._closed = False
        self._recover_jobs()

    def _recover_jobs(self):
        for directory in self.output_dir.iterdir():
            if directory.is_symlink() or not re.fullmatch(r'[0-9a-f]{32}', directory.name):
                continue
            try:
                state = read_json(directory / 'state.json')
                if not isinstance(state, dict) or state.get('job_id') != directory.name or state.get('status') not in TERMINAL | {'queued', 'running'}:
                    continue
                if state['status'] not in TERMINAL:
                    state.update(status='interrupted', finished_at=now(),
                                 error='Server stopped before completion; start a new job to retry.')
                    write_json(directory / 'state.json', state)
                self.jobs[directory.name] = state
            except (ValueError, OSError, TypeError):
                continue

    def _directory(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r'[0-9a-f]{32}', job_id) or job_id not in self.jobs:
            raise ValueError('Unknown job_id')
        return self.output_dir / job_id

    def _update(self, job_id, **fields):
        self.jobs[job_id].update(fields, updated_at=now())
        write_json(self._directory(job_id) / 'state.json', self.jobs[job_id])

    def _input(self, value):
        return workspace_input(self.workspace, value, self.output_dir)

    def _new_job(self, kind, input_records, *, task_ids=None, agent=None,
                 annotations_path=None, match_rule=None, reference_path=None):
        if self._closed:
            raise ValueError('Server is shutting down')
        if sum(state['status'] not in TERMINAL for state in self.jobs.values()) >= self.max_jobs:
            raise ValueError('Job queue is full; wait for completion or cancel a job')
        if match_rule not in MATCH_RULES:
            raise ValueError('Unknown constraint matching rule')
        annotations = load_records(self._input(annotations_path)) if annotations_path else {}
        references = load_records(self._input(reference_path)) if reference_path else {}
        expected = set(task_ids) if task_ids else {r['case_id'] for r in input_records}
        if not set(annotations) <= expected:
            raise ValueError('Constraint annotations contain cases not selected for this job')
        if reference_path and set(references) != expected:
            raise ValueError('Reference case IDs must match the selected cases exactly')
        for record in annotations.values():
            if set(record) - {'case_id', 'task_id', 'constraint_units', 'query_unit_ids'}:
                raise ValueError('Constraint annotations accept only units and query memberships')
            if 'query_unit_ids' not in record and match_rule is None:
                raise ValueError('Select match_rule when annotations do not provide query memberships')
        cleaned_references = [scrub_graph_record(cid, value) for cid, value in references.items()]
        job_id = uuid.uuid4().hex
        directory = private_directory(self.output_dir / job_id)
        write_json(directory / 'input.json', input_records)
        if annotations_path:
            write_json(directory / 'annotations.json', list(annotations.values()))
        if reference_path:
            write_json(directory / 'reference.json', cleaned_references)
        config = {'kind': kind, 'agent': agent, 'task_ids': task_ids,
                  'backend': self.backend, 'model': self.model, 'llm_timeout': self.llm_timeout,
                  'q0_mode': self.q0_mode, 'keep_tool_result': self.keep_tool_result,
                  'match_rule': match_rule, 'has_annotations': bool(annotations_path),
                  'has_reference': bool(reference_path)}
        write_json(directory / 'config.json', config)
        self.jobs[job_id] = {'job_id': job_id, 'kind': kind, 'status': 'queued',
                             'case_count': len(input_records), 'created_at': now(),
                             'backend': self.backend if kind == 'analysis' else None,
                             'model': self.model if kind == 'analysis' else None}
        self._update(job_id)
        task = asyncio.create_task(self._run(job_id))
        self.tasks[job_id] = task
        task.add_done_callback(lambda done, key=job_id: self.tasks.pop(key, None))
        return self.get_analysis_status(job_id)

    async def start_analysis(self, input_path, agent, task_ids, annotations_path=None,
                             match_rule=None, reference_path=None):
        if agent not in AGENTS:
            raise ValueError('Unknown input adapter')
        records = sanitized_tasks(load_records(self._input(input_path)), task_ids)
        return self._new_job('analysis', records, task_ids=task_ids, agent=agent,
                             annotations_path=annotations_path, match_rule=match_rule,
                             reference_path=reference_path)

    async def evaluate_graph(self, input_path, annotations_path=None, match_rule=None, reference_path=None):
        """Queue offline evaluation only; this tool never calls a model."""
        records = load_records(self._input(input_path))
        cleaned = [scrub_graph_record(cid, record) for cid, record in records.items()]
        return self._new_job('evaluation', cleaned, annotations_path=annotations_path,
                             match_rule=match_rule, reference_path=reference_path)

    def worker_command(self, directory):
        return [sys.executable, '-m', 'searchatlas.mcp.worker', str(directory / 'config.json')]

    async def _stop_process(self, process):
        if process is None:
            return
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                pass
            # Also kill descendants that ignored TERM after the group leader exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.returncode is None:
            process.kill()
        await process.wait()

    async def _run(self, job_id):
        process = None
        try:
            async with self._slot:
                if self.jobs[job_id]['status'] == 'cancelled':
                    return
                directory = self._directory(job_id)
                log_fd = os.open(directory / 'worker.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(log_fd, 'wb') as log:
                    kwargs = {'stdout': log, 'stderr': log, 'cwd': str(directory)}
                    if os.name == 'posix':
                        kwargs['start_new_session'] = True
                    spawning = asyncio.create_task(asyncio.create_subprocess_exec(*self.worker_command(directory), **kwargs))
                    try:
                        process = await asyncio.shield(spawning)
                    except asyncio.CancelledError:
                        # Cancellation during spawn must not lose the child handle.
                        process = await spawning
                        raise
                    self.processes[job_id] = process
                    self._update(job_id, status='running', started_at=now())
                    code = await asyncio.wait_for(process.wait(), self.job_timeout)
                if code:
                    self._update(job_id, status='failed', finished_at=now(),
                                 error='Worker failed; inspect the private local worker.log. No complete result is available.')
                elif not (directory / 'result.json').is_file():
                    self._update(job_id, status='failed', finished_at=now(), error='Worker produced no evaluation result.')
                else:
                    graph_path = directory / ('graph.json' if self.jobs[job_id]['kind'] == 'analysis' else 'input.json')
                    if not graph_path.is_file():
                        raise ValueError('Worker produced no graph artifact')
                    graph_ids = set(load_records(graph_path))
                    result = read_json(directory / 'result.json')
                    if not isinstance(result.get('cases'), list) or len(result['cases']) != self.jobs[job_id]['case_count']:
                        raise ValueError('Worker result is incomplete')
                    if {row.get('case_id') for row in result['cases']} != graph_ids:
                        raise ValueError('Worker result case IDs do not match its graphs')
                    self._update(job_id, status='completed', finished_at=now())
        except asyncio.TimeoutError:
            await self._stop_process(process)
            self._update(job_id, status='failed', finished_at=now(), error='Job timeout; no complete result is available.')
        except asyncio.CancelledError:
            await self._stop_process(process)
            self._update(job_id, status='cancelled', finished_at=now())
        except Exception:
            await self._stop_process(process)
            self._update(job_id, status='failed', finished_at=now(),
                         error='Job could not be completed; inspect private local artifacts.')
        finally:
            self.processes.pop(job_id, None)

    def get_analysis_status(self, job_id):
        directory = self._directory(job_id)
        result = dict(self.jobs[job_id])
        result['artifact_directory'] = str(directory.relative_to(self.workspace))
        if result['status'] == 'completed':
            result['result_available'] = True
        return redact_metadata(result)

    def get_analysis_result(self, job_id, view='summary', case_id=None, offset=0, limit=20):
        directory = self._directory(job_id)
        if self.jobs[job_id]['status'] != 'completed':
            return {'job_id': job_id, 'status': self.jobs[job_id]['status'], 'result_available': False}
        if (not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
                or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100):
            raise ValueError('Use offset >= 0 and limit between 1 and 100')
        result = read_json(directory / 'result.json')
        extra = {}
        if view == 'summary':
            items = [{key: value for key, value in row.items() if key in {'case_id', 'metrics', 'counts'}}
                     for row in result['cases']]
            extra['protocol'] = result['protocol']
            if 'reconstruction' in result:
                reconstruction = result['reconstruction']
                extra['reconstruction'] = {'macro': reconstruction['macro'], 'protocol': reconstruction['protocol'],
                                            'edge_types': {kind: item['macro'] for kind, item in reconstruction['edge_types'].items()}}
        elif view in {'nodes', 'edges', 'support_sets'}:
            if not case_id:
                raise ValueError('case_id is required for graph detail views')
            if view == 'support_sets':
                row = next((row for row in result['cases'] if row['case_id'] == case_id), None)
                if row is None:
                    raise ValueError('Unknown case_id')
                items = [{'name': key, 'value': value} for key, value in row['support_sets'].items()]
            else:
                path = directory / ('graph.json' if self.jobs[job_id]['kind'] == 'analysis' else 'input.json')
                records = load_records(path)
                if case_id not in records:
                    raise ValueError('Unknown case_id')
                items = scrub_graph_record(case_id, records[case_id])['graph'][view]
        else:
            raise ValueError('view must be summary, nodes, edges, or support_sets')
        # Limit both record count and serialized size. Full artifacts stay local.
        page, used = [], 0
        for item in items[offset:offset + limit]:
            sanitized = redact_metadata(item)
            encoded = json.dumps(sanitized, ensure_ascii=False)
            if len(encoded) > 12000:
                sanitized = {'truncated': True, 'preview': encoded[:12000],
                             'note': 'Full record is available in the private local artifact.'}
                encoded = json.dumps(sanitized, ensure_ascii=False)
            if page and used + len(encoded) > 60000:
                break
            page.append(sanitized)
            used += len(encoded)
        next_offset = offset + len(page)
        return {'job_id': job_id, 'view': view, 'case_id': case_id, 'total': len(items),
                'offset': offset, 'next_offset': next_offset if next_offset < len(items) else None,
                'items': page, **extra}

    async def cancel_analysis(self, job_id):
        self._directory(job_id)
        if self.jobs[job_id]['status'] in TERMINAL:
            return self.get_analysis_status(job_id)
        self._update(job_id, status='cancelled', finished_at=now())
        task = self.tasks.get(job_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.get_analysis_status(job_id)

    async def close(self):
        self._closed = True
        for job_id in list(self.tasks):
            await self.cancel_analysis(job_id)
