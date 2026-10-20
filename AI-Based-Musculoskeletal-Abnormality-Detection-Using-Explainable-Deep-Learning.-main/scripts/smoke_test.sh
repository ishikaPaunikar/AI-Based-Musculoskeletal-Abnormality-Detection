#!/usr/bin/env bash
# End-to-end check on SYNTHETIC data (metrics are meaningless). Writes to /tmp, not outputs/.
set -euo pipefail
cd "$(dirname "$0")/.."
export MURA_ROOT="$PWD/data/dummy/MURA-v1.1" MURA_OUTPUT="${TMPDIR:-/tmp}/mura_smoke"
python scripts/make_dummy_data.py --out "$MURA_ROOT"
for m in baseline efficientnet; do
  python -m src.train --model "$m" --run_name "$m" --weights none --epochs 2 --head_epochs 1 --batch_size 16 --unfreeze_layers 20
  python -m src.evaluate --run_name "$m" --n_boot 30
done
python -m src.compare
python -m src.gradcam --run_name efficientnet --n 2
python -m src.insights
echo "Smoke test finished. Outputs in $MURA_OUTPUT"
