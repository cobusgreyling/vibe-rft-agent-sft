#!/usr/bin/env python3
"""
LoRA / QLoRA supervised fine-tuning on agent-trace prompt-completion data.

Target hardware: free Google Colab T4 (~15 GB VRAM).

Design choices (aligned with HF Training Agents SFT-on-traces):
  • Model: Qwen2.5-1.5B-Instruct (ungated, chat template ready, fits T4 with 4-bit)
           or google/gemma-2-2b-it (gated — set HF_TOKEN and --model accordingly)
  • Method: PEFT LoRA on attention/MLP projections
  • Quantization: bitsandbytes 4-bit (QLoRA) to stay under T4 memory
  • Data: conversational prompt-completion JSONL
  • Loss: completion_only_loss=True  →  assistant/completion tokens only

Memory recipe for T4
--------------------
  max_length=1024, batch_size=1, grad_accum=8, gradient_checkpointing=True, 4-bit
  → typically trains Qwen2.5-1.5B without OOM.

This script is intentionally small-step / demo-scale (not a production run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Defaults tuned for free Colab T4
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Alternative (requires accepting the Gemma license + HF login):
# DEFAULT_MODEL = "google/gemma-2-2b-it"

ROOT = Path(__file__).resolve().parent.parent


def load_prompt_completion_jsonl(path: Path) -> Dataset:
    """Load prompt-completion JSONL into a Hugging Face Dataset."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No examples found in {path}")
    # Datasets need nested structures; list-of-dicts with list fields is fine.
    return Dataset.from_list(rows)


def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization — the usual QLoRA setup for consumer GPUs."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16 tensor cores
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """
    LoRA on the main projection matrices.

    target_modules cover Qwen2 / Gemma2 attention + MLP. If a module name is
    missing on a given architecture, PEFT simply skips it when using
    `target_modules` as a list of substrings... actually PEFT expects exact
    module name fragments; these names work for both Qwen2.5 and Gemma-2.
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT on agent-trace prompt-completion data.")
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "prompt_completion.jsonl",
        help="Prompt-completion JSONL (from convert_traces.py).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SFT_MODEL", DEFAULT_MODEL),
        help=f"HF model id (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "lora-sft",
        help="Where to save LoRA adapters + trainer state.",
    )
    p.add_argument("--max-steps", type=int, default=30, help="Demo-scale step budget.")
    p.add_argument("--max-length", type=int, default=1024, help="Max sequence length.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate (~1e-4–2e-4).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit load (needs more VRAM; use for CPU smoke tests carefully).",
    )
    p.add_argument(
        "--eval-split",
        type=float,
        default=0.15,
        help="Fraction held out for trainer eval loss (0 to disable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        print(
            f"ERROR: {args.data} not found.\n"
            "Run:  python src/convert_traces.py\n"
            "first to build prompt_completion.jsonl from example_traces.jsonl.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("SFT on agent traces (TRL + LoRA)")
    print("=" * 60)
    print(f"model       : {args.model}")
    print(f"data        : {args.data}")
    print(f"output_dir  : {args.output_dir}")
    print(f"max_steps   : {args.max_steps}")
    print(f"max_length  : {args.max_length}")
    print(f"4-bit QLoRA : {not args.no_4bit}")
    print(f"device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram        : {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    print()

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    dataset = load_prompt_completion_jsonl(args.data)
    print(f"Loaded {len(dataset)} prompt-completion examples")
    print("Example[0] prompt roles :", [m["role"] for m in dataset[0]["prompt"]])
    print("Example[0] completion   :", dataset[0]["completion"][0]["content"][:100], "...")
    print()

    if 0.0 < args.eval_split < 1.0 and len(dataset) >= 4:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
        print(f"Train / eval split: {len(train_ds)} / {len(eval_ds)}")
    else:
        train_ds, eval_ds = dataset, None
        print("Eval split disabled (training on full set).")

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Causal LMs often have no pad token; reuse EOS so batching works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # 3. Model (optionally 4-bit)
    # ------------------------------------------------------------------
    model_kwargs: dict = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if not args.no_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = build_bnb_config()
        # dtype for non-quantized weights / compute path
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    print(f"Loading model {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    # Needed when using gradient checkpointing with LoRA.
    model.config.use_cache = False
    if not args.no_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config(r=args.lora_r)

    # ------------------------------------------------------------------
    # 4. SFTConfig — completion-only loss is the key masking switch
    # ------------------------------------------------------------------
    #
    # For prompt-completion datasets, TRL defaults to completion_only_loss=True:
    #   labels[prompt_tokens] = -100   → ignored by cross-entropy
    #   labels[completion_tokens] kept → model learns assistant behavior
    #
    # We set it explicitly so the educational intent is obvious in the code.
    #
    # Mixed precision note (Colab T4 + QLoRA):
    # Enabling fp16 turns on GradScaler. With 4-bit models, some grads can still
    # land in bfloat16 and crash with:
    #   NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda
    #   not implemented for 'BFloat16'
    # Safest demo default: no GradScaler (fp16=False, bf16=False). LoRA still
    # trains fine in float32 adapters on top of the 4-bit base.
    use_cuda = torch.cuda.is_available()
    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        save_total_limit=1,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=max(args.max_steps // 2, 1) if eval_ds is not None else None,
        max_length=args.max_length,
        completion_only_loss=True,  # ← mask prompt; train on completion only
        packing=False,  # packing + completion masks can be subtle; keep simple for learning
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit" if (not args.no_4bit and use_cuda) else "adamw_torch",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting training ...")
    train_result = trainer.train()
    metrics = train_result.metrics
    print("\nTrain metrics:", json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------
    # 5. Save adapters (small — only LoRA weights, not the full 1.5B)
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Persist a tiny run card for later evaluation scripts.
    card = {
        "base_model": args.model,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "completion_only_loss": True,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds is not None else 0,
        "metrics": metrics,
    }
    (args.output_dir / "run_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8"
    )
    print(f"\nSaved LoRA adapters + tokenizer → {args.output_dir}")
    print("Next: python src/evaluate_format.py --adapter", args.output_dir)


if __name__ == "__main__":
    main()
