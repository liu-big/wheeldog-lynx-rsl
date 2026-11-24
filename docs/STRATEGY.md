# 开源轮足策略对照

## 候选仓库

| 项目 | 栈 | 适合本机 | 备注 |
|---|---|---|---|
| [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) | Isaac Lab 5.1 + RSL-RL | **首选** | 官方 M20/Lynx，与现有 Isaac 一致 |
| [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab) | Isaac Lab | 参考 | Go2W/B2W 奖励与部署成熟 |
| [ShengqianChen/DreamWaQ_Go2W](https://github.com/ShengqianChen/DreamWaQ_Go2W) | Isaac Gym | 备选 | DreamWaQ 估计器，需另装 Isaac Gym |
| [Albusgive/wheel_legged_genesis](https://github.com/Albusgive/wheel_legged_genesis) | Genesis | 备选 | 双轮足为主，非四轮足 |
| Swiss-Mile | 闭源 | — | 思路：能滚不走、省电、RL 学混合模式 |

## 采纳的策略骨架（Lynx/M20）

1. **Actor**：本体感觉 + 指令（可加高度扫描给学生/教师）  
2. **Critic**：特权（高度图、真实线速度）— Teacher 课程  
3. **动作**：12 腿位置 + 4 轮速度（与 Wheeldog 16-DoF 同构）  
4. **奖励主线**（稳、轻）：  
   - `track_lin_vel_xy` / `track_ang_vel_z`  
   - `flat_orientation`（roll 重、pitch 上楼放宽）  
   - `wheel_roll` 直行；上楼 **不** `wheel_lock`  
   - 腿：平地 hold；上楼 clearance / 抬腿  
   - 严禁：横移进度与外八/侧倾/拧腰多项同权互殴  
5. **地形**：平面 → 单级 → **倒三角** 3–20 cm 课程  

## 相对旧 HIM 基线的改动

- 去掉 HIM 估计器依赖；用 RSL-RL 标准 actor-critic  
- 分阶段 curriculum，不在同一套权重里同时榨 flat strafe + stair climb  
- 先在官方 `Rough-Deeprobotics-M20-v0` 上验证栈，再换 Wheeldog URDF  

## 观测/部署注意

Wheeldog 实机若仍要 62-D 历史槽，可在 RSL 策略外再包一层；**训练先用 M20/robot_lab 观测约定**，跑稳后再对齐部署 schema。
