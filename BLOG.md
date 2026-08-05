# Vibe Reinforcement Fine-Tuning

**SFT on agent traces first. Rewards later. Small open-weights models. An AI agent as your co-pilot.**

---

## The short version

**Vibe Reinforcement Fine-Tuning** is not a brand-new math formula. It is a *way of working*:

1. You describe what you want in plain language to an **AI agent harness** (Grok CLI, Cursor, Claude Code, etc.).
2. The agent writes data scripts, training configs, eval checks, and debugs the run.
3. You inspect loss, traces, and sample generations.
4. You refine and go again.

Under the hood the *technical* ladder is the same one Hugging Face’s Training Agents series pushes:

| Stage | What it is | What it teaches |
|-------|------------|-----------------|
| **1. SFT** (this post) | Imitate good agent traces | Style, tool format, multi-turn structure |
| **2. RFT / GRPO** (next) | Improve with a reward signal | Go *beyond* pure copy-paste of the teacher |

We stay on **small open-weights models** (about 0.5B–3B) with **LoRA / QLoRA** so the whole thing fits free Colab or a modest GPU.

This post is **rung one**: supervised fine-tuning on agent traces, with proper masking, a tiny hold-out set, and a clear picture of how the “training file” sits on the base model.

---

## Open source vs open weights (they are not the same)

People say “open model” and mean three different things. For this blog, the important split is:

```
┌─────────────────────────────────────────────────────────────┐
│                    "OPEN" IN AI                              │
├──────────────────────┬──────────────────────────────────────┤
│   OPEN SOURCE        │   OPEN WEIGHTS                       │
│                      │                                      │
│  Code + often data   │  The trained numbers (weights)       │
│  + license that lets │  are downloadable and usable.        │
│  you study/change    │  Training code/data may be closed    │
│  and redistribute.   │  or incomplete.                      │
│                      │                                      │
│  Example spirit:     │  Example spirit:                     │
│  full recipe book    │  "here is the finished cake,         │
│  + ingredients list  │   recipe optional"                   │
└──────────────────────┴──────────────────────────────────────┘
```

| Term | Means |
|------|--------|
| **Open source** | Software (and ideally data/docs) under a license that lets you use, study, modify, and share. Classic OSS definition. |
| **Open weights** | You can **download the model parameters** and run / fine-tune them. The original training run might still be proprietary. |
| **Closed** | You only get an API. No weights to put LoRA on. |

**What we train in this project:** an **open-weights** instruct model (e.g. `Qwen2.5-1.5B-Instruct`). We can pull it from Hugging Face, attach LoRA, and own the adapter. That is the point of the exercise—**you are not stuck behind a closed API** for the student model.

(Many open-weights models also have open or permissive licenses for the weights; always read the model card. “Open weights” ≠ automatic “do anything commercially.”)

### Can LoRA only be done on an open-weights model?

**LoRA** stands for **Lo**w-**R**ank **A**daptation: small trainable adapter matrices that sit on top of a base model so you do not rewrite every parameter.

**For the kind of LoRA we do here** (you train and save the adapter yourself): **yes — you need the weights.**

LoRA plugs **into** the model and updates those small matrices during training. That only works if you can:

1. **Load** the model parameters  
2. **Run** forward and backward passes on them  
3. **Save** the adapter folder  

A chat API that only returns text is not enough.

```
  OPEN WEIGHTS                         CLOSED API ONLY
  ────────────                         ────────────────
  Download Qwen / Llama / …            GPT / Claude via HTTPS
         │                                    │
         v                                    v
  Attach LoRA, train                   Send prompts, get text
  Save adapter folder                  No access to internals
         │                                    │
         v                                    v
  You own the student                  You rent the teacher
```

| Situation | DIY LoRA (this blog)? |
|-----------|------------------------|
| Open-weights model on your GPU / Colab | **Yes** |
| Weights on a private server you control | **Yes** (even if not public) |
| Chat / completions API only | **No** — you cannot attach LoRA to their model |
| Vendor “fine-tuning API” | Sometimes **they** run LoRA-like training for you; you usually do **not** get a portable PEFT folder the same way |

**Nuance so we stay accurate:** LoRA is not magically limited to “open source.” It is limited to **models whose weights you can load.** In practice that is almost always **open weights** (or private weights you already have).

**One line:** LoRA needs the model’s numbers on disk or GPU; an API that only returns text is not enough. That is why this whole vibe RFT story is built around an **open-weights student**.

---

## What “vibe” has to do with it

Classic fine-tuning blogs assume *you* hand-write every script. **Vibe** means the **agent harness** is in the loop:

```
  you (intent in English)
           │
           v
  ┌─────────────────────┐
  │  AI agent harness   │  ← Grok CLI / coding agent
  │  (the "vibe" part)  │
  └─────────┬───────────┘
            │ writes / runs / fixes
            v
  data prep → train → eval → chat demo
            │
            v
  you read metrics & samples → tweak intent → loop
```

So:

- **Vibe** = *how* the pipeline gets built and iterated (agent-assisted).
- **SFT / RFT** = *what* learning algorithm you run.

Without the vibe loop you can still SFT by hand. With it, the barrier drops: the harness scaffolds convert/train/eval while you stay on the science and the product intent.

---

## The big picture pipeline

Draw this on a whiteboard:

```
  [Stronger agent / teacher]
            │
            │  produces multi-turn runs
            v
  ┌───────────────────┐
  │  Agent TRACES     │  JSONL: system, user, assistant, tool, ...
  └─────────┬─────────┘
            │  expand each assistant turn
            v
  ┌───────────────────┐
  │  Prompt /         │  prompt = context
  │  Completion rows  │  completion = what we want the student to say
  └─────────┬─────────┘
            │  hold out ~10–20% for eval
            v
  ┌───────────────────┐
  │  SFT + LoRA       │  completion-only loss (mask the prompt)
  │  on open-weights  │
  │  small model      │
  └─────────┬─────────┘
            │
            v
  ┌───────────────────┐
  │  Format / task    │  did it still look like an agent?
  │  checks           │
  └─────────┬─────────┘
            │  later...
            v
  ┌───────────────────┐
  │  RFT / GRPO       │  rewards in the environment
  │  (beyond imitate) │
  └───────────────────┘
```

In our minimal Colab lab we used **synthetic** traces so anyone can run it. In a “real” distillation setup, those traces come from a **larger / stronger** model in a real harness.

---

## Is this model distillation?

**Yes—when the traces come from a stronger model.**

Not the old textbook kind where you match soft probability vectors (logits) token by token. This is **behavior distillation** (also called trajectory / trace distillation):

```
  TEACHER (big model + tools)          STUDENT (small open-weights)
  ─────────────────────────            ───────────────────────────
  Runs agent sessions                  Never saw the teacher's guts
  Writes tool calls + answers          Only sees the written trace
         │                                      ^
         │         SFT on completions           │
         +--------------------------------------+
                    "act like this"
```

| Distillation style | Teacher gives you… | This project |
|--------------------|--------------------|--------------|
| Classic KD | Soft logits | No |
| Sequence KD | Full answer text | Yes (completions) |
| Agent / trace KD | Multi-turn tool trajectories | Yes |

The student will not magically get the teacher’s full IQ. It gets **what showed up in the traces**: format, habits, structure, some planning style.

**One line for the fridge:**  
*SFT on teacher agent traces = distill the teacher’s behavior into an open-weights student.*

---

## The training data (what an “example” is)

### Raw form: a trace

One **trace** is a whole agent session—one task, several turns:

```
  TRACE (one coding task)
  ──────────────────────
  system:   "You are a coding agent. Tools look like: invoke tool ..."
  user:     "List files in /tmp and count them."
  assistant:"I'll list them.

             invoke tool list_dir with path is /tmp"
  tool:     "a.txt
             b.py
             notes.md"
  assistant:"There are 3 files: a.txt, b.py, notes.md."
```

In our demo file we had **12 traces**.

### Training form: prompt–completion rows

We do **not** shove one whole multi-turn blob in as a single “predict everything” example and hope for the best. We **expand** every **assistant** turn into its own supervised row:

```
  From the trace above you get TWO training rows:

  Row A
    prompt:     system + user
    completion: first assistant (tool call)

  Row B
    prompt:     system + user + assistant1 + tool result
    completion: second assistant (final answer)
```

```
  messages:  [sys][user][asst1][tool][asst2]
                │              │           │
                └──── Row A ───┘           │
                └────────── Row B ─────────┘
```

Why? So the model is trained on **intermediate** tool-calling behavior, not only the last “done” message.

In our lab: **12 traces → 24 prompt–completion examples**.

### Train vs hold-out (testing data)

You **must** keep some examples out of the weight updates. That is the **hold-out** (validation / eval split).

```
  All converted examples (24)
  ┌────────────────────────────────────────┐
  │  TRAIN (used to update LoRA)     ~80%  │  → 20 examples
  │  ████████████████████                  │
  │  HOLD-OUT / EVAL (only measure)  ~15%  │  →  4 examples
  │  ░░░░                                  │
  └────────────────────────────────────────┘
```

| Split | Used for | In our run |
|-------|----------|------------|
| **Train** | Compute loss → update LoRA | 20 |
| **Hold-out (eval)** | Measure loss *without* updating | 4 |

If train loss crashes toward zero but hold-out loss goes **up**, you are **memorizing** the 20 rows—not learning a general agent. That is exactly what a tiny demo set often does, and it is useful to *see*.

**How much hold-out?**  
Common default: **10–20%** if you have enough data. With only 24 rows, 15% is fine for a lab; for a real run you want a larger, cleaner hold-out (and a separate test set you almost never touch).

**Is 20 train examples enough?**  
Enough to **demo** the pipeline and sometimes shift **format**. Not enough to claim a production agent. Real SFT baselines use hundreds to tens of thousands of diverse traces.

---

## Loss and masking (the part everyone trips on)

### What is loss?

**Loss** is a single number: *how wrong was the model’s next-token guess compared to the labeled text?*

- High loss → model is surprised by the correct tokens → bad for that batch  
- Low loss → model assigns high probability to the correct tokens → better  

Training = **nudge the (LoRA) weights so loss goes down** on the tokens we care about.

### What is masking?

The model still **reads** the whole sequence. Masking decides **which positions count when we compute loss**.

We set those positions’ training labels to a special ignore index (in practice: **`-100`**). Cross-entropy skips them.

```
  Full sequence the model sees:

  [  PROMPT tokens  |  COMPLETION tokens  ]
    system, user,      assistant reply
    tool results       (tool call / answer)

  Labels used for loss:

  [ -100 -100 ... -100 |  t1  t2  t3  ... ]
       MASKED              SUPERVISED
       (no grade)          (the grade)
```

### What we mask, and how much

| Region | Mask? | Why |
|--------|-------|-----|
| System prompt | **Yes** | Do not train the model to regenerate your system text |
| User message | **Yes** | Do not train it to imitate the user |
| Tool results | **Yes** (in the prompt) | Tool output is environment truth, not “model speech” |
| **Assistant completion** | **No** | This is what we want the agent to say |

**How much to mask?**  
In the modern TRL setup: mask **100% of the prompt** side; supervise **100% of the completion** side. That is `completion_only_loss=True` on a prompt–completion dataset.

You are **not** masking “half the answer.” You are masking **the context**, and grading **the whole reply you labeled**.

### The quiz intuition (and why it is backwards)

People think: *to teach better answers, hide the answers.*

That is how an **exam** works. SFT is a **worked example**:

```
  EXAM (inference, after training)
    show:  prompt only
    hide:  answer
    model must generate

  WORKED EXAMPLE (SFT training)
    show:  prompt + correct answer
    grade: only the answer tokens
    model learns "when I see this context, these tokens should be likely"
```

If you masked the **completion** instead, you would be training the model to predict the **prompt**—the opposite of teaching better agent replies.

### Simple analogy

```
  Cooking class (SFT)
  ───────────────────
  Customer order slip  = prompt     → not graded if you recite it
  Dish you cook        = completion → graded

  You still READ the order slip.
  You only get scored on the dish.
```

---

## How the “training file” sits on the model (LoRA)

**LoRA** = **Lo**w-**R**ank **A**daptation. We do **not** usually rewrite all billions of base weights on a free T4. We attach a small **adapter**—and that only works because the **base is open weights** (or otherwise loadable), as covered above.

We attach a small **adapter**:

```
  BEFORE
  ──────
  ┌────────────────────────────────────┐
  │     Open-weights base model        │
  │     Qwen2.5-1.5B-Instruct          │
  │     (frozen or 4-bit quantized)    │
  └────────────────────────────────────┘


  AFTER SFT (LoRA / QLoRA)
  ────────────────────────
  ┌────────────────────────────────────┐
  │     Same base model                │
  │     (still mostly frozen)          │
  │                                    │
  │   ┌──────────────────────────┐     │
  │   │  LoRA adapter (small)    │ ←── this is what you save
  │   │  trained on your traces  │     (the "training file")
  │   └──────────────────────────┘     │
  └────────────────────────────────────┘

  At use time:
      output = base_model(x)  tweaked by  LoRA(x)
```

| Piece | Size / role |
|-------|-------------|
| **Base model** | Large; downloaded from Hub; open weights |
| **LoRA adapter** | Small folder (`outputs/lora-sft`); *your* fine-tune |
| **QLoRA** | Base loaded in **4-bit** to save VRAM; LoRA still trains in higher precision |

So when someone says “where is my fine-tuned model?” the honest answer is:

> **Base open-weights model + LoRA folder.**  
> Not a brand-new 1.5B from scratch.

You can merge LoRA into a full weight dump later if you want one folder; for Colab, keeping the adapter is enough.

---

## The process, step by step (clear and boring on purpose)

### 1. Collect or write traces

JSONL, one session per line (or equivalent). Roles: system / user / assistant / tool.

### 2. Convert → prompt–completion

For each assistant message: everything before it is **prompt**; that message is **completion**.

### 3. Split train / hold-out

Shuffle, hold out ~10–20%. Never tune weights on hold-out.

### 4. Load open-weights student

Example: Qwen2.5-1.5B-Instruct in 4-bit on a T4.

### 5. Attach LoRA

Train only low-rank matrices on attention/MLP projections.

### 6. SFT with completion-only loss

TRL `SFTTrainer`, `completion_only_loss=True` → mask prompt, grade completion.

### 7. Watch two curves

- **Train loss** ↓ — fitting the 20 rows  
- **Hold-out loss** — is it generalizing or memorizing?

### 8. Evaluate like an agent (not only loss)

Cheap **format** checks: tool syntax, no role leaks, tool-when-expected.  
Harder later: real tools, tests, sandbox success.

### 9. Use it

Load base + adapter; optional multi-turn loop that parses `invoke tool ...` and feeds tool results back.

### 10. (Next blog / next rung) RFT

Define a reward. Run GRPO-style reinforcement so the student can beat pure imitation.

---

## What we actually saw in the lab (honest numbers)

| Item | Value |
|------|--------|
| Traces | 12 (synthetic) |
| Prompt–completion rows | 24 |
| Train / hold-out | 20 / 4 |
| Student | Qwen2.5-1.5B-Instruct (open weights) + LoRA |
| Steps | 30 (many epochs on a tiny set) |
| Pattern | Train loss very low; hold-out loss rose → overfit risk |
| Format eval | Strong on tool-shaped replies (demo-scale) |

That is **success for a teaching baseline**, not a claim that 20 examples make a strong agent.

---

## How this becomes full “Vibe RFT”

```
  VIBE LOOP (always on)
  ─────────────────────
  intent → agent harness writes code → run → you read → refine

  LEARNING LADDER
  ───────────────
  Stage A  SFT on traces     ← imitation / behavior distillation
  Stage B  GRPO / RFT        ← reward in the loop, beyond cloning
```

| Stage | Question the model answers |
|-------|----------------------------|
| SFT | “What would a good teacher trace do here?” |
| RFT | “What gets a high reward in *my* environment?” |

SFT teaches **manners and format**.  
RFT raises the **ceiling** when imitation is not enough.

---

## ASCII cheat sheet (redraw these)

**1. Masking**

```
  INPUT IDS:   P P P P P P C C C C
  LABELS:     -1-1-1-1-1-1 C C C C
               mask......  loss...
```

**2. Distillation**

```
  [Big teacher agent] --traces--> [SFT] --LoRA--> [Small open-weights student]
```

**3. Files on disk**

```
  ~/.cache/.../Qwen2.5-1.5B-Instruct/     big base
  outputs/lora-sft/                       your small adapter
```

**4. Vibe**

```
  Human intent ⇄ Agent harness ⇄ Train/eval scripts ⇄ Metrics
```

---

## Takeaways

1. **Vibe RFT** = agentic workflow + SFT-then-reinforcement ladder on **small open-weights** models.  
2. **Open weights** ≠ **open source**; we fine-tune models whose **weights we can download**.  
3. **LoRA** needs loadable weights—**DIY adapters are not something you bolt onto a closed chat API.**  
4. Teacher traces + student SFT = **behavior distillation**.  
5. Training data = expanded **prompt / completion** rows from multi-turn traces.  
6. **Mask the prompt (all of it for loss); grade the completion (all of it).**  
7. **Hold out** data only to measure—do not train on it.  
8. The “fine-tune file” is usually a **LoRA adapter sitting on** a frozen/quantized base.  
9. The **vibe** part is the harness that builds and iterates this pipeline with you.

---

## What to do next

- Run the minimal lab: `demos/sft-agent-traces/` (Colab T4 notebook included).  
- Replace synthetic traces with **real teacher** agent logs when you care about distillation quality.  
- Scale data before you scale claims.  
- Then add **RFT / GRPO** with a reward you believe in (tests green, format valid, task success).

---

## Closing

Fine-tuning an open-weights model on agent traces is not mysterious: **show it good sessions, grade only the assistant’s lines, save a small adapter, measure on hold-out and with format checks.**  

The “vibe” is that you do not have to be a full-time ML engineer to drive the loop—an AI agent harness can carry the scaffolding while you stay responsible for the intent, the data quality, and whether the metrics actually mean anything.

**Imitate first. Reinforce second. Keep the weights open enough that you own the student.**

---

*Companion code: `demos/sft-agent-traces/` — convert → LoRA SFT → format eval → chat with simulated tools.*
