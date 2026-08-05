#!/usr/bin/env python3
"""
Convert agent traces → TRL prompt-completion examples.

Why this step exists
--------------------
Hugging Face's "Training Agents" SFT workflow starts from *raw agent traces*
(multi-turn conversations that may include tool calls / tool results). TRL's
SFTTrainer expects either:

  1. language-modeling:   {"messages": [...]}  or {"text": "..."}
  2. prompt-completion:   {"prompt": [...], "completion": [...]}

We use **conversational prompt-completion** so TRL can compute loss on the
completion only (assistant tokens). That is the modern equivalent of
"assistant-only loss" / "mask the prompt".

  prompt     = system + user (+ tool turns) that *condition* the model
  completion = the assistant message we want the model to learn to produce

For multi-turn traces we expand *each assistant turn* into its own training
row (standard trajectory → SFT expansion). That way intermediate tool-calling
turns are supervised too, not just the final answer.

See:
  https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion
  https://huggingface.co/docs/trl/en/sft_trainer#train-on-completion-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file of agent traces (one JSON object per line)."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {e}") from e
    return rows


# ---------------------------------------------------------------------------
# Trace → prompt-completion expansion
# ---------------------------------------------------------------------------

def _normalize_message(msg: dict[str, Any]) -> dict[str, str]:
    """Keep only role/content; coerce content to string."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        # Tool payloads sometimes arrive as dict/list; keep them readable.
        content = json.dumps(content, ensure_ascii=False)
    return {"role": str(role), "content": content}


def expand_trace_to_prompt_completion(
    messages: list[dict[str, Any]],
    *,
    min_prompt_messages: int = 1,
) -> list[dict[str, Any]]:
    """
    Expand one multi-turn trace into N prompt-completion pairs.

    For each assistant message at index i:
      prompt     = messages[:i]   (everything the model saw before answering)
      completion = [messages[i]]  (the assistant reply we supervise)

    Non-assistant turns are never used as completions — we only train the
    model to *act as the agent*, not to emit user text or tool outputs.

    Example
    -------
    messages = [system, user, assistant_1, tool, assistant_2]
    → example A: prompt=[system, user],              completion=[assistant_1]
    → example B: prompt=[system, user, a1, tool],    completion=[assistant_2]
    """
    examples: list[dict[str, Any]] = []
    normalized = [_normalize_message(m) for m in messages]

    for i, msg in enumerate(normalized):
        if msg["role"] != "assistant":
            continue
        if i < min_prompt_messages:
            # Need at least some context before the first supervised turn.
            continue
        if not msg["content"].strip():
            continue

        prompt = normalized[:i]
        completion = [msg]

        # Skip degenerate rows (no user/system signal at all).
        if not any(m["role"] in ("user", "system") for m in prompt):
            continue

        examples.append({"prompt": prompt, "completion": completion})

    return examples


def convert_traces(
    traces: list[dict[str, Any]],
    *,
    messages_key: str = "messages",
) -> list[dict[str, Any]]:
    """Convert a list of traces into a flat list of prompt-completion dicts."""
    all_examples: list[dict[str, Any]] = []
    for t_idx, trace in enumerate(traces):
        messages = trace.get(messages_key)
        if not messages:
            raise ValueError(
                f"Trace index {t_idx} (id={trace.get('trace_id')!r}) "
                f"missing '{messages_key}' field."
            )
        expanded = expand_trace_to_prompt_completion(messages)
        # Optional bookkeeping for debugging / filtering later.
        for ex in expanded:
            ex["trace_id"] = trace.get("trace_id", f"trace-{t_idx}")
            ex["source"] = trace.get("source", "unknown")
        all_examples.extend(expanded)
    return all_examples


def iter_prompt_completion_preview(
    examples: list[dict[str, Any]],
    n: int = 2,
) -> Iterator[str]:
    """Human-readable preview of converted examples (for learning / debugging)."""
    for i, ex in enumerate(examples[:n]):
        prompt_roles = " → ".join(m["role"] for m in ex["prompt"])
        comp_preview = ex["completion"][0]["content"][:120].replace("\n", " ")
        yield (
            f"[example {i}] trace={ex.get('trace_id')} | "
            f"prompt roles: {prompt_roles} | "
            f"completion: {comp_preview!r}..."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert agent traces JSONL → prompt-completion JSONL for TRL SFT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "example_traces.jsonl",
        help="Path to agent traces JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "prompt_completion.jsonl",
        help="Where to write prompt-completion JSONL.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Print this many example summaries after conversion.",
    )
    args = parser.parse_args()

    traces = load_jsonl(args.input)
    examples = convert_traces(traces)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            # TRL only needs prompt + completion; drop metadata for the train file.
            row = {"prompt": ex["prompt"], "completion": ex["completion"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Loaded {len(traces)} traces → {len(examples)} prompt-completion examples")
    print(f"Wrote {args.output}")
    print("\n--- Preview (why completion-only loss matters) ---")
    print(
        "TRL will tokenize prompt+completion, then set labels=-100 on prompt tokens\n"
        "so the model only learns to produce the assistant completion.\n"
    )
    for line in iter_prompt_completion_preview(examples, n=args.preview):
        print(line)


if __name__ == "__main__":
    main()
