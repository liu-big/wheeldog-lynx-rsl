#!/usr/bin/env bash
# Official Lynx/M20 reference train — validate new stack before Wheeldog port.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream_rl_training"
ILAB="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"

source "${CONDA_ROOT:-${HOME}/anaconda3}/etc/profile.d/conda.sh"
conda activate env_robotlab
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
unset CUDA_VISIBLE_DEVICES || true
set +u
source "$ILAB/_isaac_sim/setup_conda_env.sh"
NROOT="${CONDA_ROOT:-${HOME}/anaconda3}/envs/${CONDA_ENV:-env_robotlab}/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$NROOT/nvidia/cu13/lib:$NROOT/torch/lib:${LD_LIBRARY_PATH:-}"
set -euo pipefail

cd "$UP"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-8000}"
GUI="${GUI:-0}"
ARGS=(--task=Rough-Deeprobotics-M20-v0 --num_envs="$NUM_ENVS")
# rsl_rl train uses max_iterations from agent cfg; pass headless unless GUI=1
if [[ "$GUI" != "1" ]]; then
  ARGS+=(--headless)
else
  export DISPLAY="${DISPLAY:-:0}"
fi
echo "[m20-ref] ${ARGS[*]}"
exec python -u scripts/reinforcement_learning/rsl_rl/train.py "${ARGS[@]}"
