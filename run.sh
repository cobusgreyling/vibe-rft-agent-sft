#!/usr/bin/env bash
# End-to-end local run (GPU recommended for train/eval).
# Usage:  ./run.sh            # convert + optional train
#         ./run.sh convert    # conversion + unit tests only
#         ./run.sh all        # convert → train → eval

set -euo pipefail
cd "$(dirname "$0")"

step="${1:-convert}"

echo "==> [1/3] Convert agent traces → prompt-completion"
python src/convert_traces.py

echo "==> Unit tests (no GPU)"
python src/test_convert.py -v

if [[ "$step" == "convert" ]]; then
  echo
  echo "Conversion done. To train on a GPU (e.g. Colab T4):"
  echo "  pip install -r requirements.txt"
  echo "  python src/train_sft.py --max-steps 30"
  echo "  python src/evaluate_format.py"
  echo
  echo "Or open colab_sft_agent_traces.ipynb in Google Colab."
  exit 0
fi

if [[ "$step" == "all" || "$step" == "train" ]]; then
  echo "==> [2/3] LoRA SFT (needs GPU for a realistic run)"
  python src/train_sft.py --max-steps "${MAX_STEPS:-30}"
fi

if [[ "$step" == "all" || "$step" == "eval" ]]; then
  echo "==> [3/3] Format-correctness evaluation"
  python src/evaluate_format.py
fi

echo "Done."
