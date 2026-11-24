#!/usr/bin/env bash
# 一条龙：后台开训 + 覆盖最新日志 + 打印看日志 / MuJoCo 指令
#
#   bash scripts/oneclick_train.sh
#   TASK=Rough-Wheeldog-Lynx-v0 NUM_ENVS=2048 bash scripts/oneclick_train.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/train_wheeldog_lynx.log"
PID_FILE="$LOG_DIR/train.pid"
TASK="${TASK:-Rough-Wheeldog-Lynx-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
GUI="${GUI:-0}"

mkdir -p "$LOG_DIR"
# 停掉本包旧训练（不影响 HIMLoco 等其它进程）
if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "[oneclick] stop old pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
fi

# 覆盖日志留给 train_wheeldog_lynx.sh；这里只起后台
nohup env TASK="$TASK" NUM_ENVS="$NUM_ENVS" GUI="$GUI" \
  bash "$ROOT/scripts/train_wheeldog_lynx.sh" >/dev/null 2>&1 &
WRAPPER_PID=$!
echo "$WRAPPER_PID" > "$PID_FILE"
# 等 python 起来，PID 写成真正的 train.py（停训用）
for _ in $(seq 1 30); do
  sleep 1
  REAL_PID="$(pgrep -f "$ROOT/upstream_rl_training/scripts/reinforcement_learning/rsl_rl/train.py" | head -1 || true)"
  if [[ -n "${REAL_PID:-}" ]]; then
    echo "$REAL_PID" > "$PID_FILE"
    break
  fi
  kill -0 "$WRAPPER_PID" 2>/dev/null || break
done

cat <<EOF

========== Wheeldog Lynx 训练已启动 ==========
ROOT     : $ROOT
TASK     : $TASK
NUM_ENVS : $NUM_ENVS
GUI      : $GUI
PID_FILE : $PID_FILE  ($(cat "$PID_FILE"))
LOG      : $LOG   （每次开训覆盖）
LATEST   : $LOG_DIR/LATEST.log

# 实时看日志（固定这条，永远是当前这次训练）
tail -f $LOG_DIR/LATEST.log

# 看最新 ckpt
ls -lt $ROOT/upstream_rl_training/logs/rsl_rl/wheeldog_lynx_flat/*/model_*.pt 2>/dev/null | head -5
ls -lt $ROOT/upstream_rl_training/logs/rsl_rl/wheeldog_lynx_rough/*/model_*.pt 2>/dev/null | head -5

# 导出最新权重 + MuJoCo 回放（训练是 Rough 就用 TASK=rough）
cd $ROOT && TASK=rough SCENE=stairs bash scripts/export_and_play_wheeldog.sh

# 只要导出、不弹窗
cd $ROOT && EXPORT_ONLY=1 TASK=rough bash scripts/export_and_play_wheeldog.sh

# 平地场景回放同一套 Rough 权重
cd $ROOT && TASK=rough SCENE=flat bash scripts/export_and_play_wheeldog.sh

# 停训
kill \$(cat $PID_FILE)
==============================================
EOF
