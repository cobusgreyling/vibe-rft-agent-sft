#!/usr/bin/env python3
"""
Interactive multi-turn chat with the LoRA-SFT agent (simulated tools).

Loads: base instruct model + LoRA adapter from training.
Parses:  invoke tool NAME with arg is value [with arg is value ...]
Simulates tool results so you can exercise multi-turn behavior without a real sandbox.

Usage (after training):
  python src/chat_agent.py --adapter outputs/lora-sft
  python src/chat_agent.py --adapter outputs/lora-sft --base-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a helpful coding agent. Use tools when needed. "
    "Respond with tool calls in this format:\n"
    "invoke tool tool_name with arg1 is value1\n"
    "When finished, give a clear final answer."
)

# invoke tool list_dir with path is ./data
# invoke tool write_file with path is x.py with content is print(1)
TOOL_RE = re.compile(
    r"invoke\s+tool\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+with\s+(?P<body>.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Return first tool call in assistant text, or None."""
    m = TOOL_RE.search(text)
    if not m:
        return None
    name = m.group("name")
    body = m.group("body").strip()
    # Split "with a is 1 with b is 2" → pairs (best-effort)
    parts = re.split(r"\s+with\s+", body, flags=re.IGNORECASE)
    args: dict[str, str] = {}
    for part in parts:
        if " is " in part:
            k, v = part.split(" is ", 1)
            args[k.strip()] = v.strip()
        elif part.strip():
            args["arg"] = part.strip()
    return {"name": name, "arguments": args, "raw": m.group(0)}


def simulate_tool(name: str, arguments: dict[str, str]) -> str:
    """Fake tool backend — enough to demo multi-turn agent loops."""
    name_l = name.lower()
    path = arguments.get("path", arguments.get("arg", "."))

    if name_l in ("list_dir", "ls", "list"):
        return "example_traces.jsonl\nprompt_completion.jsonl\nnotes.md"

    if name_l in ("read_file", "read", "cat"):
        if "requirements" in path:
            return "torch\ntransformers\ntrl\npeft\nbitsandbytes\n"
        if "readme" in path.lower():
            return "# Agent SFT Demo\n\nMinimal reproducible project for SFT on agent traces.\n"
        return f"[simulated contents of {path}]\nline 1\nline 2\n"

    if name_l in ("write_file", "write", "create_file"):
        content = arguments.get("content", "")
        return f"Wrote {len(content.splitlines()) or 1} lines to {path}"

    if name_l in ("run_shell", "shell", "bash", "run"):
        cmd = arguments.get("command", arguments.get("arg", ""))
        if "pytest" in cmd:
            return "5 passed in 0.42s"
        if "wc -l" in cmd or "find" in cmd:
            return "4"
        if "git status" in cmd:
            return " M src/train_sft.py\n?? data/new_traces.jsonl"
        return f"[simulated stdout for: {cmd}]"

    if name_l in ("grep", "search", "git_search"):
        return (
            "src/train_sft.py:48:        completion_only_loss=True,\n"
            "src/evaluate_format.py:12: # format checks\n"
        )

    return f"[simulated] tool={name!r} args={arguments!r} → ok"


def load_model(base: str, adapter: Path | None, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter is not None:
        from peft import PeftModel

        print(f"Loading adapter: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int = 256) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def agent_turn(
    model,
    tokenizer,
    user_text: str,
    *,
    max_tool_rounds: int = 4,
    max_new_tokens: int = 256,
    verbose: bool = True,
) -> str:
    """Run user message through model, executing simulated tools until final answer."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    final = ""
    for round_i in range(max_tool_rounds + 1):
        reply = generate(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        final = reply
        if verbose:
            print(f"\n--- assistant (round {round_i + 1}) ---\n{reply}")

        tool = parse_tool_call(reply)
        if tool is None:
            break  # final answer (no tool call)

        result = simulate_tool(tool["name"], tool["arguments"])
        if verbose:
            print(f"\n--- tool:{tool['name']} ---\n{result}")

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "tool", "content": result})

    return final


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with fine-tuned agent (simulated tools).")
    p.add_argument("--adapter", type=Path, default=ROOT / "outputs" / "lora-sft")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--once",
        type=str,
        default=None,
        help="Single prompt then exit (non-interactive).",
    )
    args = p.parse_args()

    adapter = None if args.base_only else args.adapter
    if adapter is not None and not adapter.exists():
        print(f"Adapter not found: {adapter}", file=sys.stderr)
        print("Train first, or pass --base-only.", file=sys.stderr)
        sys.exit(1)

    import torch

    model, tokenizer = load_model(
        args.base_model,
        adapter,
        use_4bit=not args.no_4bit and torch.cuda.is_available(),
    )
    print("Ready. Commands: empty line or 'quit' to exit.\n")

    if args.once:
        agent_turn(model, tokenizer, args.once, max_new_tokens=args.max_new_tokens)
        return

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"quit", "exit", "q"}:
            break
        agent_turn(model, tokenizer, user, max_new_tokens=args.max_new_tokens)
        print()


if __name__ == "__main__":
    main()
