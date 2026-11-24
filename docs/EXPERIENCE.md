# Wheeldog 训练经验笔记

实机 / MuJoCo 多轮踩坑后的可复用结论（Isaac Lab 5.1 + RSL-RL PPO）。

## 1. 观测与网络

- **盲策略**（无 height scan）可部署：`ang_vel | gravity | cmd | q_rel(轮位清零) | qd | [gait_clock 8D] | last_a`。
- 引入 **开环 gait clock（8D）** 后，侧移/转向才容易条件抬腿；obs 从 57 → 65。
- 部署端必须与训练 **同布局**（含时钟相位：对角 trot / 横移 crawl）。

## 2. 转向必须「抬腿滤波」，不能轮胎拖地

- 侧移/转向时：**放开 H2/K**，锁轮速（`wheel_vel_l2_side_yaw`），逼迈步。
- **速度跟踪门控**：`track_*_gait_gated` —— side/yaw 时跟对角时钟才给满跟踪分，否则几乎没分（HIM 思路）。
- `rotation_gait` + `equal_swing_lift` + `dual_hold_air`：对角交替、等抬高、禁止趴地蹭 yaw。
- 抬腿高度宜 **略低**（净空 ~4 cm），过高真机别扭。

## 3. 四条腿要差不多

- 对角关节镜像 `joint_mirror`、直行左右 `joint_lr_mirror`。
- `feet_gait` 同步对角接触；`rotation_gait_symmetry` 占空对称。
- `equal_swing_lift`：摆动对角两脚抬高接近，禁止单腿独抬。

## 4. 越训越差：务必存 best

| 现象 | 原因 |
|------|------|
| mean_reward 在 ~3k 峰值后掉 | 地形课程变难 + 动作抖罚堆叠 |
| 跑满 20k 末尾崩盘 | `action_l2_without_cmd` / rate/acc 爆炸，站着猛抽 |
| 直接部署 latest | 经常不是最好 |

**做法**：`model_best.pt`（按 `Train/mean_reward`）、默认训到 ~8k、峰值后早停；部署用 best 不是 final。

站立动作罚不宜过重（例如 `action_l2_without_cmd` ~−0.25），否则后期螺旋。

## 5. 摩擦与站立

- 真机摩擦高：仿真接触 μ 宜偏高；零指令禁滑（`lin_vel_xy_l2_without_cmd` + stand pose）。
- 不要加机身随机外力扰动（会学歪）。

## 6. 导出与回放

```bash
# 导出 ONNX 到 weights_hub/deploys/
EXPORT_ONLY=1 TASK=rough bash scripts/export_and_play_wheeldog.sh /path/to/model_best.pt

# MuJoCo 楼梯
TASK=rough SCENE=stairs bash scripts/export_and_play_wheeldog.sh /path/to/model_best.pt
```

- 57-D 与 65-D 策略不要混用；play 脚本已按 ONNX 输入维自动识别。
- 实机包：目录内 `policy.onnx` + `META.yaml`（obs 维、iter、备注）。

## 7. 推荐训练顺序

1. Rough 混合地形（楼梯偏重），≥4096 envs，headless。
2. 盯日志：`terrain_levels`、`error_vel_xy/yaw`、`equal_swing_lift`、`dual_hold_air`、`action_l2_without_cmd`。
3. 取 `model_best.pt` → ONNX → 楼梯 MuJoCo → 实机。
