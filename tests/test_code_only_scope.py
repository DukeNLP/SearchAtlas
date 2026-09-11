"""Guard the distributable source tree against accidental research-data exports."""
import ast
import re
import unittest
from pathlib import Path


class CodeOnlyScopeTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_no_research_data_directories(self):
        for name in ('data', 'results', 'traces', 'decoded', 'experiments'):
            self.assertFalse((self.root / name).exists(), name)

    def test_sources_compile_without_execution(self):
        for path in (self.root / 'src').rglob('*.py'):
            with self.subTest(file=str(path.relative_to(self.root))):
                ast.parse(path.read_text(), filename=str(path))

    def test_no_embedded_credentials_or_private_paths(self):
        patterns = [
            r'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
            r'gh[pousr]_[A-Za-z0-9]{30,}',
            r'github_pat_[A-Za-z0-9_]{30,}',
            r'AIza[0-9A-Za-z_-]{30,}',
            r'\bAKIA[0-9A-Z]{16}\b',
            r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
            r'/Users/[^/\s]+/',
            r'/home/[^/\s]+/',
        ]
        # Test fixtures contain fake credential strings and regex patterns, so
        # scan distributed implementation and examples, not the scanner itself.
        paths = list((self.root / 'src').rglob('*.py'))
        paths += list((self.root / 'examples').rglob('*.json'))
        paths += list((self.root / 'examples').rglob('*.py'))
        paths += list((self.root / 'docs').rglob('*.md'))
        paths += list(self.root.glob('*.md')) + [self.root / '.env.example', self.root / 'pyproject.toml']
        for path in paths:
            content = path.read_text()
            for pattern in patterns:
                self.assertIsNone(re.search(pattern, content), f'{path.relative_to(self.root)} matches credential/path pattern')

    def test_api_key_is_environment_only(self):
        for adapter in ('standard', 'miro'):
            path = self.root / 'src/searchatlas/builder' / adapter / 'settings.py'
            tree = ast.parse(path.read_text())
            assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'API_KEY'
                                   for target in node.targets)]
            self.assertEqual(len(assignments), 1)
            value = assignments[0].value
            self.assertIsInstance(value, ast.Call)
            self.assertEqual(ast.unparse(value.func), 'os.getenv')
            self.assertEqual([arg.value for arg in value.args], ['OPENAI_API_KEY', ''])

    def test_no_local_environment_or_private_key_files(self):
        for path in self.root.rglob('*'):
            if not path.is_file() or any(part in {'build', '.venv', '__pycache__'} for part in path.parts):
                continue
            self.assertFalse(path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example', path.name)
            self.assertNotIn(path.suffix.lower(), {'.pem', '.key', '.p12', '.pfx'})

    def test_no_case_specific_result_overrides(self):
        pattern = re.compile(r'(?:BrowseComp-\d+|DeepSearchQA-\d+|WebWalkerQA-Hard-\d+|force_pk|manual_override)')
        for path in (self.root / 'src').rglob('*.py'):
            self.assertIsNone(pattern.search(path.read_text()), str(path.relative_to(self.root)))

    def test_evaluation_is_offline_and_has_no_fixed_result_files(self):
        for path in (self.root / 'src/searchatlas/evaluation').rglob('*.py'):
            text = path.read_text()
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.split('.')[0] in {'openai', 'requests', 'httpx'} for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0], {'openai', 'requests', 'httpx', 'builder'})
            for filename in ('expected_paper.json', 'frozen_metrics.jsonl', 'cases.jsonl', 'eligible_pk_edge_indices'):
                self.assertNotIn(filename, text)


if __name__ == '__main__':
    unittest.main()
