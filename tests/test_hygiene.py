"""Release hygiene: cheap checks that keep promises in the README true."""

import ast
import pathlib
import re

import agentparity

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "agentparity" / "core"


def test_version_is_consistent():
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    changelog = re.search(r"^## (\S+)", (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
    assert agentparity.__version__ == pyproject == changelog


def test_core_imports_only_the_standard_library():
    import sys

    allowed = set(sys.stdlib_module_names) | {"agentparity"}
    for path in CORE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for n in names:
                assert n.split(".")[0] in allowed, f"{path.name} imports {n}"


def test_every_source_file_is_reachable_from_a_test():
    """Each adapter ships with a fixture and a test."""
    adapters = {p.stem for p in (ROOT / "src/agentparity/adapters").glob("*.py")} - {"__init__", "function"}
    fixtures = {p.stem for p in (ROOT / "src/agentparity/conformance/fixtures").glob("*.py")} - {"__init__"}
    tests = {p.stem.removeprefix("test_") for p in (ROOT / "tests").glob("test_*.py")}
    assert adapters == fixtures, adapters ^ fixtures
    assert adapters <= tests, adapters - tests
