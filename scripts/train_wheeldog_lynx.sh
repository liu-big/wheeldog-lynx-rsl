#!/usr/bin/env bash
# Wheeldog Lynx RSL train — migrate_lean（本机 himlab + Isaac Lab 5.1）
#
#   bash scripts/train_wheeldog_lynx.sh
#   GUI=0 NUM_ENVS=2048 TASK=Flat-Wheeldog-Lynx-v0 bash scripts/train_wheeldog_lynx.sh
#   GUI=0 NUM_ENVS=2048 TASK=Rough-Wheeldog-Lynx-v0 bash scripts/train_wheeldog_lynx.sh
#
# 日志：每次开训覆盖写入 logs/train_wheeldog_lynx.log（并刷新 logs/LATEST.log 软链）
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream_rl_training"
ILAB="${ISAACLAB_ROOT:-${ISAACLAB51_ROOT:-/home/hi/isaaclab_lyz/IsaacLab}}"
ISIM="${ISAACSIM_PATH:-/home/hi/isaacsim}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-himlab}"

# 默认 Rough 混合地形（楼梯+boxes+rough）；只要平地：TASK=Flat-Wheeldog-Lynx-v0
TASK="${TASK:-Rough-Wheeldog-Lynx-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
GUI="${GUI:-0}"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
conda activate "$CONDA_ENV"
export ZSH_VERSION=""
unset PYTHONPATH

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
unset CUDA_VISIBLE_DEVICES LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER || true

set +u
if [[ -f "$ILAB/_isaac_sim/setup_conda_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ILAB/_isaac_sim/setup_conda_env.sh"
else
  # shellcheck disable=SC1091
  source "$ISIM/setup_conda_env.sh"
fi
NROOT="$CONDA_ROOT/envs/$CONDA_ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$NROOT/nvidia/cu13/lib:$NROOT/nvidia/nccl/lib:$NROOT/torch/lib:${LD_LIBRARY_PATH:-}"
if [[ -z "${LD_PRELOAD:-}" && -f "$NROOT/nvidia/nccl/lib/libnccl.so.2" ]]; then
  export LD_PRELOAD="$NROOT/nvidia/nccl/lib/libnccl.so.2"
fi
set -euo pipefail

export ISAACLAB_ROOT="$ILAB"
export ISAACLAB51_ROOT="$ILAB"
# 强制用本包源码（覆盖 site-packages 里指向 portable 的 editable）
export PYTHONPATH="${UP}/source/rl_training:${PYTHONPATH:-}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v -i ros | paste -sd: -)"
fi

cd "$UP"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/train_wheeldog_lynx.log"
# 每次最新训练：覆盖写，不追加历史
: > "$LOG"
ln -sfn train_wheeldog_lynx.log "$ROOT/logs/LATEST.log"

ARGS=(--task="$TASK" --num_envs="$NUM_ENVS" --device=cuda:0)
if [[ "$GUI" != "1" ]]; then
  ARGS+=(--headless)
  unset DISPLAY XAUTHORITY || true
else
  if command -v systemctl >/dev/null; then
    eval "$(systemctl --user show-environment 2>/dev/null | grep -E '^(DISPLAY|XAUTHORITY|XDG_RUNTIME_DIR|DBUS_SESSION_BUS_ADDRESS)=' || true)"
  fi
  export DISPLAY="${DISPLAY:-:0}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  for xa in "${XAUTHORITY:-}" "/run/user/$(id -u)/gdm/Xauthority" "$HOME/.Xauthority"; do
    [[ -n "$xa" && -r "$xa" ]] && export XAUTHORITY="$xa" && break
  done
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

echo "[wheeldog-lynx] root=$ROOT IsaacLab=$ILAB env=$CONDA_ENV"
echo "[wheeldog-lynx] task=$TASK envs=$NUM_ENVS GUI=$GUI DISPLAY=${DISPLAY:-none}"
echo "[wheeldog-lynx] log=$LOG (overwrite each run)"
echo "[wheeldog-lynx] ${ARGS[*]}"
exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${ARGS[@]}" \
  2>&1 | tee "$LOG"
