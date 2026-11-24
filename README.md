# Wheeldog Lynx RSL

**Isaac Lab + RSL-RL 轮足（Wheeldog）训练与部署栈**：平地滚轮、楼梯/崎岖抬腿、侧移转向对角迈步，支持 ONNX 导出与 MuJoCo 回放。

> Project name: **wheeldog-lynx-rsl**  
> One-liner: Wheel-legged Wheeldog locomotion in Isaac Lab (RSL-RL PPO) with stair curriculum, lift-leg turn/strafe, and ONNX deploy.

## 简介

面向 DeepRobotics **Wheeldog** 的仿真训练流水线，对齐云深处 Lynx/M20 与开源 DeepRoboticsLab `rl_training` 路线：

- **Rough 课程地形**（楼梯偏重）+ 高摩擦站立稳定  
- **侧移/转向抬腿**：开环 gait clock、跟踪门控、对角等抬、禁止轮胎拖地  
- **四腿对称**：对角镜像 + 占空对称 + equal swing  
- **防后期塌缩**：`model_best.pt`、控制站立动作惩罚、默认 8k iter  
- **一键导出**：ONNX → `weights_hub/deploys/` → MuJoCo flat/stairs

## 环境

- Isaac Lab 5.1 + conda `himlab`（按本机路径改脚本）
- GPU 建议 ≥ RTX 4090，`NUM_ENVS≥4096` headless

## 快速开始

```bash
# 训练（Rough，无头，覆盖写 logs/LATEST.log）
TASK=Rough-Wheeldog-Lynx-v0 NUM_ENVS=4096 GUI=0 bash scripts/oneclick_train.sh

# 看日志
tail -f logs/LATEST.log   # 或 ~/wheeldog_lynx_LATEST.log

# 导出最好权重（优先 model_best.pt）
EXPORT_ONLY=1 TASK=rough bash scripts/export_and_play_wheeldog.sh \
  upstream_rl_training/logs/rsl_rl/wheeldog_lynx_rough/<run>/model_best.pt

# MuJoCo 楼梯回放
TASK=rough SCENE=stairs bash scripts/export_and_play_wheeldog.sh \
  upstream_rl_training/logs/rsl_rl/wheeldog_lynx_rough/<run>/model_best.pt
```

## 目录

```
wheeldog_lynx_rsl/
  upstream_rl_training/   # DeepRoboticsLab/rl_training + Wheeldog 任务配置
  scripts/                # 一键训练 / 导出 / MuJoCo
  mujoco_play/            # ONNX 回放（flat / stairs）
  weights_hub/            # 部署落盘（gitignore 大文件）
  docs/
    STRATEGY.md           # 策略对照
    EXPERIENCE.md         # 训练与部署经验（必读）
  assets/                 # Wheeldog URDF/网格
```

## 观测约定（部署务必对齐）

| 版本 | 维数 | 内容 |
|------|------|------|
| 旧盲策略 | 57 | ang×0.25 \| grav \| cmd \| q_rel \| qd×0.05 \| last_a |
| + gait clock | 65 | 同上，在 last_a 前加 8D 相位时钟 |

动作 16：12 腿位置 + 4 轮速度。

## 经验摘要

详见 [docs/EXPERIENCE.md](docs/EXPERIENCE.md)：

1. 转向靠 **抬腿相位滤波**，不要轮子拖地拧  
2. **四条腿对称**（镜像 / 等抬 / 占空）  
3. **永远部署 best**，不要无脑用最后一个 ckpt  
4. 站立动作惩罚过重 → 后期 mean_reward 崩盘  

## License / 上游

训练框架基于 [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) 与 Isaac Lab；本仓库的 Wheeldog 配置、奖励与部署脚本为迁移与工程化改动。
