#!/usr/bin/env bash
# 一键：导出最新 M20 RSL checkpoint → ONNX → MuJoCo 可视化测试
#
# Usage:
#   bash scripts/export_and_play_m20.sh              # 自动找最新 model_*.pt
#   bash scripts/export_and_play_m20.sh /path/to/model_600.pt
#   VX=0.8 bash scripts/export_and_play_m20.sh       # 初始前进指令
#   EXPORT_ONLY=1 bash scripts/export_and_play_m20.sh  # 只导出不打开 viewer
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream_rl_training"
LOG_ROOT="$UP/logs/rsl_rl/deeprobotics_m20_rough"
EXPORT_DIR="$ROOT/exported"
MJ_PLAY="$ROOT/mujoco_play/m20_play_onnx.py"
EXPORT_PY="$UP/scripts/tools/export_onnx_fast.py"

VX="${VX:-0.5}"
VY="${VY:-0.0}"
YAW="${YAW:-0.0}"
EXPORT_ONLY="${EXPORT_ONLY:-0}"

# --- display for MuJoCo GUI ---
if command -v systemctl >/dev/null; then
  eval "$(systemctl --user show-environment 2>/dev/null | grep -E '^(DISPLAY|XAUTHORITY|XDG_RUNTIME_DIR|DBUS_SESSION_BUS_ADDRESS)=' || true)"
fi
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
for xa in "${XAUTHORITY:-}" "/run/user/$(id -u)/gdm/Xauthority" "$HOME/.Xauthority"; do
  [[ -n "$xa" && -r "$xa" ]] && export XAUTHORITY="$xa" && break
done

source "${CONDA_ROOT:-${HOME}/anaconda3}/etc/profile.d/conda.sh"
conda activate mujoco

# --- resolve checkpoint ---
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  CKPT="$1"
  if [[ "$CKPT" != /* ]]; then
    CKPT="$(realpath -m "$ROOT/$CKPT")"
  fi
else
  if [[ ! -d "$LOG_ROOT" ]]; then
    echo "[err] no run dir: $LOG_ROOT" >&2
    exit 1
  fi
  LATEST_RUN="$(ls -1dt "$LOG_ROOT"/*/ 2>/dev/null | head -1 || true)"
  if [[ -z "$LATEST_RUN" ]]; then
    echo "[err] no training runs under $LOG_ROOT" >&2
    exit 1
  fi
  CKPT="$(ls -1t "$LATEST_RUN"/model_*.pt 2>/dev/null | head -1 || true)"
  if [[ -z "$CKPT" ]]; then
    echo "[err] no model_*.pt in $LATEST_RUN" >&2
    exit 1
  fi
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[err] checkpoint not found: $CKPT" >&2
  exit 1
fi

ITER="$(basename "$CKPT" .pt | sed 's/model_//')"
RUN_NAME="$(basename "$(dirname "$CKPT")")"
mkdir -p "$EXPORT_DIR"
ONNX="$EXPORT_DIR/m20_${RUN_NAME}_it${ITER}.onnx"

echo "[export] ckpt=$CKPT"
echo "[export] onnx=$ONNX"
python -u "$EXPORT_PY" \
  --checkpoint_path "$CKPT" \
  --robot m20 \
  --output_path "$ONNX"

echo "[export] done → $ONNX"

if [[ "$EXPORT_ONLY" == "1" ]]; then
  exit 0
fi

echo "[mujoco] DISPLAY=$DISPLAY  cmd=($VX,$VY,$YAW)"
echo "[mujoco] keys: 8/2 vx±  4/6 vy±  7/9 yaw±  5/0 stop  R reset"
exec python -u "$MJ_PLAY" \
  --onnx "$ONNX" \
  --vx "$VX" --vy "$VY" --yaw "$YAW"
