#!/usr/bin/env python3
"""
Educational demo: what "completion-only loss" / prompt masking looks like.

You do NOT need a GPU. This script tokenizes one prompt-completion example
with a chat template and shows which token positions would receive loss
(labels != -100) vs which are masked (labels == -100).

This is the same idea TRL applies automatically when you pass a
prompt-completion dataset with completion_only_loss=True.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# transformers is enough; no peft/trl/bitsandbytes required for this demo.
from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def show_masking(model_id: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    prompt = [
        {
            "role": "system",
            "content": "You are a helpful coding agent.",
        },
        {
            "role": "user",
            "content": "List files in /tmp.",
        },
    ]
    completion = [
        {
            "role": "assistant",
            "content": "invoke tool list_dir with path is /tmp",
        },
    ]

    # Full conversation (what the model sees as input_ids).
    full_messages = prompt + completion
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Prompt only, with generation prompt so it ends where the assistant starts.
    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Align: prompt tokens should be a prefix of full tokens for well-behaved templates.
    # (In rare chat templates they may not match exactly; TRL handles that more carefully.)
    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:
        print(
            "NOTE: prompt tokens are not an exact prefix of full tokens for this "
            "template. TRL uses a more robust completion mask; this demo is illustrative."
        )
        # Fall back to character-length heuristic for the demo display.
        n_prompt = min(n_prompt, len(full_ids) // 2)

    labels = [-100] * n_prompt + full_ids[n_prompt:]
    # Causal LM loss is computed on labels shifted by 1; we only illustrate the mask.

    print("=" * 60)
    print("Completion-only masking demo")
    print("=" * 60)
    print(f"model: {model_id}\n")
    print("--- prompt_text (MASKED from loss) ---")
    print(prompt_text)
    print("--- full_text ---")
    print(full_text)
    print()
    print(f"total tokens : {len(full_ids)}")
    print(f"prompt tokens: {n_prompt}  → labels = -100 (ignored)")
    print(f"completion   : {len(full_ids) - n_prompt}  → supervised")
    print()
    print("token | label   | piece")
    print("-" * 50)
    for i, (tid, lab) in enumerate(zip(full_ids, labels)):
        piece = tokenizer.decode([tid]).replace("\n", "\\n")
        tag = "MASK" if lab == -100 else "LOSS"
        if i < 8 or i >= n_prompt - 1:  # head + around the boundary + tail
            print(f"{i:5d} | {tag:4s} | {piece!r}")
        elif i == 8:
            print("  ... | .... | (middle of prompt omitted)")
    print()
    print("Key takeaway: the model still *conditions* on the prompt tokens;")
    print("it just does not pay loss for predicting them. Only assistant")
    print("completion tokens update the weights.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    show_masking(args.model)


if __name__ == "__main__":
    main()
