#!/usr/bin/env bash
# 每次导出 = 一个独立部署目录（不覆盖历史）
#   bash scripts/export_weights.sh
#   bash scripts/export_weights.sh /path/to/model_5000.pt
#   WHEELDOG_WEIGHTS=/path bash scripts/export_weights.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUB="${WHEELDOG_WEIGHTS:-$ROOT/weights_hub}"
RUN_DIR="$ROOT/upstream_rl_training/logs/rsl_rl/wheeldog_lynx_flat"
EXPORT_PY="$ROOT/upstream_rl_training/scripts/tools/export_onnx_fast.py"
CONTRACT="$HUB/info/train_contract.yaml"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-himlab}"

if [[ $# -ge 1 ]]; then
  CKPT="$1"
else
  LATEST_RUN="$(ls -1dt "$RUN_DIR"/*/ 2>/dev/null | head -1)"
  CKPT="$(ls -1t "$LATEST_RUN"/model_*.pt | head -1)"
fi
[[ -f "$CKPT" ]] || { echo "missing ckpt: $CKPT" >&2; exit 1; }
ITER="$(basename "$CKPT" .pt | sed 's/model_//')"
RUN="$(basename "$(dirname "$CKPT")")"
DAY="$(date +%F)"
DEPLOY_ID="${DAY}_it${ITER}"
# 同日同 it 再导 → 加序号，绝不覆盖
DROP="$HUB/deploys/$DEPLOY_ID"
if [[ -e "$DROP" ]]; then
  n=2
  while [[ -e "$HUB/deploys/${DEPLOY_ID}_v${n}" ]]; do n=$((n + 1)); done
  DEPLOY_ID="${DEPLOY_ID}_v${n}"
  DROP="$HUB/deploys/$DEPLOY_ID"
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
conda activate "$CONDA_ENV"
unset PYTHONPATH
export PYTHONPATH="${ROOT}/upstream_rl_training/source/rl_training:${PYTHONPATH:-}"

TMP="/tmp/wheeldog_export_it${ITER}.onnx"
python -u "$EXPORT_PY" --checkpoint_path "$CKPT" --robot wheeldog --output_path "$TMP"

SINGLE="/tmp/wheeldog_lynx_it${ITER}_single.onnx"
python - <<PY
import onnx, onnxruntime as ort
from pathlib import Path
m = onnx.load("$TMP", load_external_data=True)
out = Path("$SINGLE")
onnx.save_model(m, str(out), save_as_external_data=False)
ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
print("[export] single", out, out.stat().st_size)
PY

mkdir -p "$DROP" "$HUB/info"
cp -f "$CKPT" "$DROP/model_${ITER}.pt"
cp -f "$SINGLE" "$DROP/wheeldog_lynx.onnx"
ln -sfn "model_${ITER}.pt" "$DROP/model.pt"
ln -sfn wheeldog_lynx.onnx "$DROP/policy.onnx"
[[ -f "$CONTRACT" ]] && cp -f "$CONTRACT" "$DROP/train_contract.yaml"
cat > "$DROP/META.yaml" <<EOF
task: Flat-Wheeldog-Lynx-v0
framework: Isaac Lab + RSL-RL (DeepRoboticsLab/rl_training)
run: $RUN
checkpoint: model_${ITER}.pt
exported_at: $(date -Iseconds)
obs_dim: 57
action_dim: 16
policy_hz: 50
onnx: wheeldog_lynx.onnx   # single-file, no .data
deploy_id: $DEPLOY_ID
mujoco_ref: $ROOT/mujoco_play/wheeldog_play_onnx.py
EOF

ln -sfn "$DEPLOY_ID" "$HUB/deploys/latest"

echo "[export] deploy=$DROP"
ls -lh "$DROP"
