#!/usr/bin/env python3
"""
Basic *format correctness* evaluation for an SFT'd agent model.

SFT on agent traces first teaches *shape*: tool-call syntax, final answers,
when to call tools vs answer directly. Before measuring task success you want
to know whether generations still look like valid agent turns.

What we score (cheap, deterministic, no external judge model)
-------------------------------------------------------------
1. non_empty          — model produced some text
2. no_user_role_leak  — did not echo chat role markers it shouldn't
3. tool_or_answer     — either a tool invoke OR a substantive final answer
4. tool_syntax_ok     — if a tool call is present, does it match our format?
5. final_answer_ok    — if no tool call, is the answer non-trivial?

These checks match the synthetic tool format used in example_traces.jsonl:

    invoke tool <name> with <arg> is <value> [with <arg> is <value> ...]

Swap the regexes if you train on OpenAI-style tool_calls JSON instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Heavy deps (torch / transformers / peft) are imported lazily in load/generate
# so `score_generation` can be unit-tested without a GPU stack installed.

# Matches: invoke tool list_dir with path is /tmp
#      or: invoke tool write_file with path is x.py with content is ...
TOOL_INVOKE_RE = re.compile(
    r"invoke\s+tool\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+with\s+.+\bis\b\s+.+",
    re.IGNORECASE | re.DOTALL,
)
TOOL_NAME_RE = re.compile(
    r"invoke\s+tool\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
# Role markers we never want the model to invent mid-completion.
ROLE_LEAK_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|^\s*user\s*:|^\s*system\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Format checks
# ---------------------------------------------------------------------------

def score_generation(text: str) -> dict[str, Any]:
    """Return per-check booleans + an overall pass flag."""
    text = (text or "").strip()
    has_tool = bool(TOOL_INVOKE_RE.search(text))
    tool_name_only = bool(TOOL_NAME_RE.search(text))
    # "Substantive" final answer: > 20 chars and not only a tool line.
    without_tool_lines = "\n".join(
        ln for ln in text.splitlines() if "invoke tool" not in ln.lower()
    ).strip()
    has_answer = len(without_tool_lines) >= 20

    checks = {
        "non_empty": len(text) > 0,
        "no_user_role_leak": not bool(ROLE_LEAK_RE.search(text)),
        "tool_or_answer": has_tool or has_answer,
        "tool_syntax_ok": (not tool_name_only) or has_tool,
        # If the model starts a tool call, require full "with ... is ..." shape.
        "final_answer_ok": has_tool or has_answer,
    }
    checks["pass"] = all(checks.values())
    checks["detected_tool"] = has_tool
    checks["preview"] = text[:200].replace("\n", "\\n")
    return checks


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "non_empty",
        "no_user_role_leak",
        "tool_or_answer",
        "tool_syntax_ok",
        "final_answer_ok",
        "pass",
    ]
    n = max(len(results), 1)
    return {k: sum(1 for r in results if r[k]) / n for k in keys}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful coding agent. Use tools when needed. "
    "Respond with tool calls in this format:\n"
    "invoke tool tool_name with arg1 is value1\n"
    "When finished, give a clear final answer."
)

# Held-out style prompts (not copied verbatim from training traces).
EVAL_PROMPTS: list[dict[str, str]] = [
    {
        "id": "list-dir",
        "user": "List the files under ./data and count them.",
        "expect_tool": True,
    },
    {
        "id": "read-file",
        "user": "Read the first few lines of requirements.txt and summarize dependencies.",
        "expect_tool": True,
    },
    {
        "id": "define-sft",
        "user": "In two sentences, explain completion-only loss in SFT.",
        "expect_tool": False,
    },
    {
        "id": "write-helper",
        "user": "Create a file called double.py with a function that doubles an integer.",
        "expect_tool": True,
    },
    {
        "id": "grep-usage",
        "user": "Search the repo for occurrences of completion_only_loss.",
        "expect_tool": True,
    },
    {
        "id": "no-tool-concept",
        "user": "What is LoRA in one short paragraph? Do not use tools.",
        "expect_tool": False,
    },
]


def load_model_and_tokenizer(
    base_model: str,
    adapter: Path | None,
    *,
    use_4bit: bool,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter) if adapter and (adapter / "tokenizer_config.json").exists() else base_model,
        trust_remote_code=True,
    )
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

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter is not None:
        print(f"Loading LoRA adapter from {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def generate_reply(
    model,
    tokenizer,
    user_text: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    # tokenize=False then encode keeps chat-template behavior explicit for learning.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens.
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate agent format correctness.")
    p.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "outputs" / "lora-sft",
        help="Path to trained LoRA adapter directory (run_card.json parent).",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model id (default: read from adapter run_card.json).",
    )
    p.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate the base model without loading a LoRA adapter.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "format_eval.json",
        help="Where to write the detailed score JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_model = args.base_model
    adapter: Path | None = None if args.base_only else args.adapter

    if adapter is not None and not adapter.exists():
        print(
            f"ERROR: adapter path {adapter} does not exist.\n"
            "Train first: python src/train_sft.py\n"
            "Or pass --base-only to score the unfine-tuned model.",
            file=sys.stderr,
        )
        sys.exit(1)

    if base_model is None:
        card_path = (adapter / "run_card.json") if adapter else None
        if card_path and card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            base_model = card.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        else:
            base_model = "Qwen/Qwen2.5-1.5B-Instruct"

    print("=" * 60)
    print("Format-correctness evaluation")
    print("=" * 60)
    print(f"base_model : {base_model}")
    print(f"adapter    : {adapter if adapter else '(none — base model only)'}")
    print()

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    model, tokenizer = load_model_and_tokenizer(
        base_model,
        adapter,
        use_4bit=not args.no_4bit and has_cuda,
    )

    per_prompt: list[dict[str, Any]] = []
    for item in EVAL_PROMPTS:
        print(f"→ {item['id']}: {item['user'][:60]}...")
        text = generate_reply(
            model,
            tokenizer,
            item["user"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        scores = score_generation(text)
        # Soft diagnostic: did we get a tool when the prompt implied one?
        scores["expect_tool"] = item["expect_tool"]
        scores["tool_when_expected"] = (
            (not item["expect_tool"]) or scores["detected_tool"]
        )
        row = {
            "id": item["id"],
            "user": item["user"],
            "generation": text,
            **scores,
        }
        per_prompt.append(row)
        status = "PASS" if scores["pass"] else "FAIL"
        print(f"  [{status}] tool={scores['detected_tool']} | {scores['preview'][:100]}")

    summary = aggregate(per_prompt)
    # Extra diagnostic rate (not part of strict pass).
    summary["tool_when_expected"] = sum(
        1 for r in per_prompt if r["tool_when_expected"]
    ) / max(len(per_prompt), 1)

    print("\n=== Summary (fraction of prompts) ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v:5.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model": base_model,
        "adapter": str(adapter) if adapter else None,
        "summary": summary,
        "per_prompt": per_prompt,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote detailed results → {args.output}")

    # Non-zero exit if strict pass rate is catastrophic (useful in CI later).
    if summary["pass"] < 0.34:
        print("WARNING: pass rate below 1/3 — format learning may have failed.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
