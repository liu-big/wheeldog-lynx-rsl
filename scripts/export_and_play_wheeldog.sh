#!/usr/bin/env bash
# 一条龙：最新 ckpt → ONNX → MuJoCo 回放（盲策略）
#
#   bash scripts/export_and_play_wheeldog.sh
#   bash scripts/export_and_play_wheeldog.sh /path/to/model_500.pt
#   TASK=flat|rough SCENE=flat|stairs bash scripts/export_and_play_wheeldog.sh
#   EXPORT_ONLY=1 bash scripts/export_and_play_wheeldog.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream_rl_training"
HUB="${WHEELDOG_WEIGHTS:-$ROOT/weights_hub}"
EXPORT_PY="$UP/scripts/tools/export_onnx_fast.py"
MJ_PLAY="$ROOT/mujoco_play/wheeldog_play_onnx.py"
CONTRACT="$HUB/info/train_contract.yaml"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-himlab}"

TASK="${TASK:-rough}"   # flat | rough
SCENE="${SCENE:-stairs}" # flat | stairs
VX="${VX:-0.5}"
VY="${VY:-0.0}"
YAW="${YAW:-0.0}"
EXPORT_ONLY="${EXPORT_ONLY:-0}"

case "$TASK" in
  flat)  LOG_ROOT="$UP/logs/rsl_rl/wheeldog_lynx_flat" ;;
  rough) LOG_ROOT="$UP/logs/rsl_rl/wheeldog_lynx_rough" ;;
  *) echo "[err] TASK=flat|rough" >&2; exit 1 ;;
esac
case "$SCENE" in
  flat)   XML="${XML:-$ROOT/mujoco_play/wheeldog_flat_scene.xml}" ;;
  stairs) XML="${XML:-$ROOT/mujoco_play/wheeldog_stairs_scene.xml}" ;;
  *) echo "[err] SCENE=flat|stairs" >&2; exit 1 ;;
esac

if command -v systemctl >/dev/null; then
  eval "$(systemctl --user show-environment 2>/dev/null | grep -E '^(DISPLAY|XAUTHORITY|XDG_RUNTIME_DIR|DBUS_SESSION_BUS_ADDRESS)=' || true)"
fi
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
for xa in "${XAUTHORITY:-}" "/run/user/$(id -u)/gdm/Xauthority" "$HOME/.Xauthority"; do
  [[ -n "$xa" && -r "$xa" ]] && export XAUTHORITY="$xa" && break
done

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
conda activate "$CONDA_ENV"
unset PYTHONPATH
NROOT="$CONDA_ROOT/envs/$CONDA_ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$NROOT/nvidia/cu13/lib:$NROOT/nvidia/nccl/lib:$NROOT/torch/lib:${LD_LIBRARY_PATH:-}"
[[ -f "$NROOT/nvidia/nccl/lib/libnccl.so.2" ]] && export LD_PRELOAD="$NROOT/nvidia/nccl/lib/libnccl.so.2"
export PYTHONPATH="${UP}/source/rl_training:${PYTHONPATH:-}"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  CKPT="$1"
  [[ "$CKPT" != /* ]] && CKPT="$(realpath -m "$ROOT/$CKPT")"
else
  LATEST_RUN="$(ls -1dt "$LOG_ROOT"/*/ 2>/dev/null | head -1 || true)"
  [[ -n "$LATEST_RUN" ]] || { echo "[err] no runs in $LOG_ROOT" >&2; exit 1; }
  CKPT="$(ls -1t "$LATEST_RUN"/model_*.pt 2>/dev/null | head -1 || true)"
  [[ -n "$CKPT" ]] || { echo "[err] no model_*.pt in $LATEST_RUN" >&2; exit 1; }
fi
[[ -f "$CKPT" ]] || { echo "[err] missing ckpt: $CKPT" >&2; exit 1; }

ITER="$(basename "$CKPT" .pt | sed 's/model_//')"
RUN="$(basename "$(dirname "$CKPT")")"
DAY="$(date +%F)"
DEPLOY_ID="${DAY}_it${ITER}_${TASK}"
DROP="$HUB/deploys/$DEPLOY_ID"
if [[ -e "$DROP" ]]; then
  n=2
  while [[ -e "$HUB/deploys/${DEPLOY_ID}_v${n}" ]]; do n=$((n + 1)); done
  DEPLOY_ID="${DEPLOY_ID}_v${n}"
  DROP="$HUB/deploys/$DEPLOY_ID"
fi

TMP="/tmp/wheeldog_${TASK}_it${ITER}.onnx"
echo "[export] ckpt=$CKPT"
python -u "$EXPORT_PY" --checkpoint_path "$CKPT" --robot wheeldog --output_path "$TMP" --no_metadata

mkdir -p "$DROP" "$HUB/info"
python - <<PY
import onnx, onnxruntime as ort
from pathlib import Path
m = onnx.load("$TMP", load_external_data=True)
out = Path("$DROP/wheeldog_lynx.onnx")
onnx.save_model(m, str(out), save_as_external_data=False)
s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
print("[export] single", out, "in", s.get_inputs()[0].shape, out.stat().st_size)
PY
cp -f "$CKPT" "$DROP/model_${ITER}.pt"
ln -sfn "model_${ITER}.pt" "$DROP/model.pt"
ln -sfn wheeldog_lynx.onnx "$DROP/policy.onnx"
[[ -f "$CONTRACT" ]] && cp -f "$CONTRACT" "$DROP/train_contract.yaml"
cat > "$DROP/META.yaml" <<EOF
task: ${TASK^}-Wheeldog-Lynx-v0
run: $RUN
checkpoint: model_${ITER}.pt
deploy_id: $DEPLOY_ID
obs: history-stacked proprio (blind, no height scan)
exported_at: $(date -Iseconds)
EOF
ln -sfn "$DEPLOY_ID" "$HUB/deploys/latest"
ONNX="$DROP/wheeldog_lynx.onnx"
echo "[export] hub=$DROP"
echo "[export] latest -> $HUB/deploys/latest"

[[ "$EXPORT_ONLY" == "1" ]] && exit 0

echo "[mujoco] xml=$XML cmd=($VX,$VY,$YAW) DISPLAY=$DISPLAY"
exec python -u "$MJ_PLAY" --onnx "$ONNX" --xml "$XML" --vx "$VX" --vy "$VY" --yaw "$YAW"
