"""Real STDIO MCP handshake and offline tool lifecycle; no model calls."""

import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    ClientSession = None

from searchatlas.mcp.io import write_json


SOURCE = str(Path(__file__).resolve().parents[1] / 'src')


def result_value(result):
    if result.isError:
        raise AssertionError(str(result.content))
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


@unittest.skipIf(ClientSession is None, 'Optional MCP SDK is not installed')
class StdioProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_list_evaluate_poll_and_read(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode='w+') as errors:
            root = Path(directory).resolve()
            write_json(root / 'graph.json', {'case_id': 'example', 'graph': {
                'nodes': [{'id': 'q1', 'type': 'Query', 'text': 'Example source'}, {'id': 'Answer'}],
                'edges': [{'source': 'q1', 'target': 'Answer', 'edge_kind': 'evidence_derived'}],
            }})
            params = StdioServerParameters(command=sys.executable,
                args=['-m', 'searchatlas.mcp', '--workspace', str(root)],
                env={**os.environ, 'PYTHONPATH': SOURCE})
            async with stdio_client(params, errlog=errors) as (read, write):
                async with ClientSession(read, write) as client:
                    initialized = await client.initialize()
                    self.assertEqual(initialized.serverInfo.name, 'SearchAtlas')
                    tools = await client.list_tools()
                    self.assertEqual({tool.name for tool in tools.tools}, {
                        'start_analysis', 'get_analysis_status', 'get_analysis_result',
                        'cancel_analysis', 'evaluate_graph'})
                    started = result_value(await client.call_tool('evaluate_graph', {'input_path': 'graph.json'}))
                    job_id = started['job_id']
                    for _ in range(100):
                        status = result_value(await client.call_tool('get_analysis_status', {'job_id': job_id}))
                        if status['status'] not in {'queued', 'running'}:
                            break
                        await asyncio.sleep(.02)
                    self.assertEqual(status['status'], 'completed')
                    result = result_value(await client.call_tool('get_analysis_result', {'job_id': job_id}))
                    self.assertEqual(result['items'][0]['metrics']['answer_support_directness'], 1)
                    self.assertIsNone(result['items'][0]['metrics']['primary_path_coverage'])
                    bad = await client.call_tool('evaluate_graph', {'input_path': '../private.json'})
                    self.assertTrue(bad.isError)
            # Protocol decoding above also verifies no builder/logger stdout polluted STDIO.

    @unittest.skipUnless(os.name == 'posix', 'POSIX signals')
    async def test_sigterm_gracefully_cancels_inflight_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            write_json(root / 'graph.json', {'case_id': 'example', 'graph': {'nodes': [], 'edges': []}})
            program = '''
import asyncio, sys, anyio
from searchatlas.mcp.service import AnalysisService
from searchatlas.mcp.server import run_stdio
service = AnalysisService(sys.argv[1])
service.worker_command = lambda directory: [sys.executable, '-c', 'import time; time.sleep(60)']
class Server:
    async def run_stdio_async(self):
        state = await service.evaluate_graph('graph.json')
        while state['job_id'] not in service.processes:
            await asyncio.sleep(.01)
        print(service.processes[state['job_id']].pid, flush=True)
        await asyncio.Event().wait()
anyio.run(run_stdio, Server(), service)
'''
            process = await asyncio.create_subprocess_exec(sys.executable, '-c', program, str(root),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': SOURCE})
            try:
                line = await asyncio.wait_for(process.stdout.readline(), 5)
                worker_pid = int(line.decode().strip())
                process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(process.wait(), 5)
                self.assertEqual(process.returncode, 0, (await process.stderr.read()).decode())
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker_pid, 0)
                states = list((root / '.searchatlas-mcp').glob('*/state.json'))
                self.assertEqual(json.loads(states[0].read_text())['status'], 'cancelled')
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()


if __name__ == '__main__':
    unittest.main()
