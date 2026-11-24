#!/usr/bin/env bash
# Install DeepRobotics rl_training into env_robotlab (Isaac Lab already present).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream_rl_training"
ILAB="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"

if [[ ! -d "$UP/source" ]]; then
  echo "[setup] missing $UP — clone DeepRoboticsLab/rl_training first" >&2
  exit 1
fi
if [[ ! -d "$UP/deep_robotics_model" ]] || [[ -z "$(ls -A "$UP/deep_robotics_model" 2>/dev/null || true)" ]]; then
  echo "[setup] missing deep_robotics_model assets" >&2
  exit 1
fi

source "${CONDA_ROOT:-${HOME}/anaconda3}/etc/profile.d/conda.sh"
conda activate env_robotlab
cd "$UP"
python -m pip install -e source/rl_training
# Isaac Lab requires numpy<2; rl_training may pull numpy 2.x — pin back.
python -m pip install 'numpy>=1.26,<2' 'packaging<24'
if [[ -f scripts/tools/setup_conda_runtime.py ]]; then
  python scripts/tools/setup_conda_runtime.py || true
fi
echo "[setup] OK. Reactivate conda then:"
echo "  bash $ROOT/scripts/train_m20_ref.sh"
echo "[setup] IsaacLab=$ILAB"
python -c "import numpy; print('[setup] numpy', numpy.__version__)"
