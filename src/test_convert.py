#!/usr/bin/env python3
"""Lightweight tests for trace → prompt-completion conversion (no GPU)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow `python src/test_convert.py` from repo root or package dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_traces import (  # noqa: E402
    convert_traces,
    expand_trace_to_prompt_completion,
    load_jsonl,
)


class TestExpand(unittest.TestCase):
    def test_two_assistant_turns_make_two_examples(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "call tool"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "final answer"},
        ]
        ex = expand_trace_to_prompt_completion(messages)
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[0]["completion"][0]["content"], "call tool")
        self.assertEqual(ex[1]["completion"][0]["content"], "final answer")
        # Second prompt must include the tool result as context.
        roles = [m["role"] for m in ex[1]["prompt"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])

    def test_skips_empty_assistant(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "ok"},
        ]
        ex = expand_trace_to_prompt_completion(messages)
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["completion"][0]["content"], "ok")

    def test_only_assistant_is_completion(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        ex = expand_trace_to_prompt_completion(messages)
        self.assertEqual(ex[0]["completion"][0]["role"], "assistant")
        self.assertTrue(all(m["role"] != "assistant" or True for m in ex[0]["prompt"]))
        self.assertTrue(all(m["role"] != "assistant" for m in ex[0]["prompt"]))


class TestPipeline(unittest.TestCase):
    def test_example_traces_file(self) -> None:
        path = Path(__file__).resolve().parent.parent / "data" / "example_traces.jsonl"
        if not path.exists():
            self.skipTest("example_traces.jsonl missing")
        traces = load_jsonl(path)
        examples = convert_traces(traces)
        self.assertGreaterEqual(len(traces), 5)
        self.assertGreater(len(examples), len(traces))  # multi-turn expands
        for ex in examples:
            self.assertIn("prompt", ex)
            self.assertIn("completion", ex)
            self.assertEqual(ex["completion"][0]["role"], "assistant")

    def test_roundtrip_jsonl(self) -> None:
        traces = [
            {
                "trace_id": "t1",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        ]
        examples = convert_traces(traces)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "pc.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(
                        json.dumps(
                            {"prompt": ex["prompt"], "completion": ex["completion"]}
                        )
                        + "\n"
                    )
            loaded = load_jsonl(out)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["completion"][0]["content"], "a")


if __name__ == "__main__":
    unittest.main()
