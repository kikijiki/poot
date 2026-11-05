"""The python runners that load a model once print the observer's two memory marks, in the line it parses."""

import ast
import contextlib
import io
import sys
import time
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import memory_phases as phases  # noqa: E402

RUNNERS = HARNESS_DIR.parent / "runners"


def runner_ast(name):
    return ast.parse((RUNNERS / name / "run.py").read_text())


def marked_events(node):
    """The events marked under `node`: the string argument of each `mark("...")` call."""
    return {call.args[0].value for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "mark"
            and call.args and isinstance(call.args[0], ast.Constant)}


def marks_by_function(tree):
    """`{function name: events it marks}` for the runner's top-level functions that mark anything."""
    found = {node.name: marked_events(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != "mark"}
    return {name: events for name, events in found.items() if events}


def mark_function(tree):
    """The runner's own `mark`, compiled alone (the scripts import torch or vllm, which a harness test must not)."""
    (definition,) = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "mark"]
    namespace = {"sys": sys, "time": time}
    exec(compile(ast.Module([definition], []), "run.py", "exec"), namespace)
    return namespace["mark"]


# Where each runner marks. The model is loaded once in `main` for a single-point run and in the curve function for a sweep (torch
# loads before it chooses the mode), and each path ends its first warmup generation.
EXPECTED = {
    "torch": {"main": {"load_end", "warmup_end"}, "run_decode_curve": {"warmup_end"}},
    "vllm": {"main": {"load_end", "warmup_end"}, "run_decode_curve": {"load_end", "warmup_end"}},
}


class RunnerMarksTest(unittest.TestCase):
    def test_a_python_runner_marks_the_load_and_the_first_warmup_on_every_path(self):
        for name, expected in EXPECTED.items():
            with self.subTest(runner=name):
                self.assertEqual(marks_by_function(runner_ast(name)), expected)

    def test_a_python_runner_prints_the_line_the_observer_parses(self):
        for name in EXPECTED:
            with self.subTest(runner=name):
                before = time.time()
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    mark_function(runner_ast(name))(phases.LOAD_END)
                event, unix_s = phases.parse_mark(err.getvalue())
                self.assertEqual(event, phases.LOAD_END)
                self.assertGreaterEqual(unix_s, before - 1e-3)
                self.assertLessEqual(unix_s, time.time() + 1e-3)


if __name__ == "__main__":
    unittest.main()
