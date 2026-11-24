# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Global curriculum scalar in [0, 1], updated from terrain-level mean.
gait_level: float = 0.0

def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
    """Update global gait_level from mean terrain level.

    Mapping rule:
    - mean <= 0.0 -> 0.0
    - 0.0 < mean < 3.0 -> 使用 exp 函数映射
    - mean == 3.0 -> 1.0
    - mean >= 3.0 -> 1.0
    """
    global gait_level

    mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
    if mean_tensor.numel() == 0:
        mean_val = 0.0
    else:
        mean_val = float(torch.mean(mean_tensor).item())

    if math.isnan(mean_val) or math.isinf(mean_val):
        mean_val = 0.0

    if mean_val <= 0.0:
        gait_level = 0.0
    elif mean_val < 3.0:
        # exp 映射：mean=0 时接近 0，mean=3 时恰好为 1
        gait_level = math.exp(mean_val - 3.0)
    else:  # mean_val >= 3.0
        gait_level = 1.0

    return gait_level

def get_gait_level_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return global gait_level as tensor matching environment batch size."""
    return torch.full((env.num_envs,), gait_level, device=env.device)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_thr: float = 0.08,
    active_when_near_zero: bool = False,
    require_vx_thr: float | None = None,
) -> torch.Tensor:
    """Reward tracking of lateral velocity only.

    Default: gated when |vy_cmd| is significant（横移跟踪）.
    active_when_near_zero=True: gated when |vy_cmd|≈0 — 压直行侧漂.
    require_vx_thr: 仅当 |vx_cmd| 足够大才计（避免站桩刷分）.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.square(cmd[:, 1] - asset.data.root_lin_vel_b[:, 1])
    reward = torch.exp(-err / std**2)
    vy = cmd[:, 1].abs()
    if active_when_near_zero:
        gate = vy < float(cmd_thr)
    else:
        gate = vy > float(cmd_thr)
    if require_vx_thr is not None:
        gate = gate & (cmd[:, 0].abs() > float(require_vx_thr))
    return reward * gate.float()


def lin_vel_y_l2_straight(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vx_thr: float = 0.15,
    vy_thr: float = 0.10,
    yaw_thr: float = 0.10,
) -> torch.Tensor:
    """Penalty |vy|² on pure forward/back — kills crab drift."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    straight = (
        (cmd[:, 0].abs() > float(vx_thr))
        & (cmd[:, 1].abs() < float(vy_thr))
        & (cmd[:, 2].abs() < float(yaw_thr))
    )
    return torch.square(asset.data.root_lin_vel_b[:, 1]) * straight.float()


def ang_vel_z_l2_straight(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vx_thr: float = 0.15,
    vy_thr: float = 0.10,
    yaw_thr: float = 0.10,
) -> torch.Tensor:
    """Penalty |ωz|² on pure forward/back — kills heading wander."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    straight = (
        (cmd[:, 0].abs() > float(vx_thr))
        & (cmd[:, 1].abs() < float(vy_thr))
        & (cmd[:, 2].abs() < float(yaw_thr))
    )
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * straight.float()


def wheel_action_lr_sync_straight(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    vx_thr: float = 0.15,
    side_yaw_thr: float = 0.12,
    wheel_signs: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
) -> torch.Tensor:
    """Penalize L/R wheel-action mismatch on straight — equal signed ω."""
    cmd = env.command_manager.get_command(command_name)
    gate = (cmd[:, 0].abs() > float(vx_thr)) & ((cmd[:, 1].abs() + cmd[:, 2].abs()) < float(side_yaw_thr))
    sl = _get_action_term_slice(env, "joint_vel")
    a = env.action_manager.action[:, sl]
    signs = torch.tensor(wheel_signs, device=a.device, dtype=a.dtype)
    signed = a * signs.unsqueeze(0)
    # FL/FR/RL/RR → all should match mean signed speed
    mean = signed.mean(dim=1, keepdim=True)
    return torch.mean(torch.square(signed - mean), dim=1) * gate.float()


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_thr: float = 0.08,
) -> torch.Tensor:
    """Reward tracking of forward/back velocity only (gated when |vx_cmd| is significant)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.square(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-err / std**2)
    return reward * (cmd[:, 0].abs() > float(cmd_thr)).float()


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques (curriculum-scaled by gait_level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward * get_gait_level_tensor(env)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (curriculum-scaled by gait_level)."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward * get_gait_level_tensor(env)

def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action changes using an env-local action history."""
    cache_name = "_action_smooth_prev_prev_action"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))

    prev_prev_action = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev_action - 2 * env.action_manager.prev_action)
    # Ignore the first two steps of each episode where action history is incomplete.
    diff = diff * (env.action_manager.prev_action != 0)
    diff = diff * (prev_prev_action != 0)
    reward = torch.sum(diff, dim=1)

    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return reward * get_gait_level_tensor(env)


def _wheeldog_action_dim_weights(
    dim: int,
    device: torch.device,
    leg_dim: int = 12,
    leg_weight: float = 3.0,
    hipy_knee_weight: float = 4.0,
    wheel_weight: float = 1.0,
) -> torch.Tensor:
    """Per-dim weights: H2/K heaviest, then H1/legs, wheels lightest."""
    w = torch.ones(dim, device=device)
    w[:leg_dim] = leg_weight
    for i in range(4):
        w[i * 3 + 1] = hipy_knee_weight  # H2
        w[i * 3 + 2] = hipy_knee_weight  # K
    if dim > leg_dim:
        w[leg_dim:] = wheel_weight
    return w


def _wheeldog_side_yaw_scale(
    env: ManagerBasedRLEnv,
    command_name: str,
    side_yaw_boost: float,
    side_yaw_threshold: float,
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    side = cmd[:, 1].abs() + cmd[:, 2].abs()
    return torch.where(side > side_yaw_threshold, side_yaw_boost, 1.0)


def action_rate_l2_wheeldog(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    leg_dim: int = 12,
    leg_weight: float = 3.0,
    hipy_knee_weight: float = 4.0,
    wheel_weight: float = 1.0,
    side_yaw_boost: float = 2.0,
    side_yaw_threshold: float = 0.15,
) -> torch.Tensor:
    """Ungated ‖Δa‖² with leg>wheel weights; boost when |vy|+|yaw| large."""
    da = env.action_manager.action - env.action_manager.prev_action
    w = _wheeldog_action_dim_weights(
        da.shape[-1], da.device, leg_dim, leg_weight, hipy_knee_weight, wheel_weight
    )
    penalty = torch.sum(torch.square(da) * w, dim=1)
    return penalty * _wheeldog_side_yaw_scale(env, command_name, side_yaw_boost, side_yaw_threshold)


def action_acc_l2_wheeldog(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    leg_dim: int = 12,
    leg_weight: float = 3.0,
    hipy_knee_weight: float = 4.0,
    wheel_weight: float = 1.0,
    side_yaw_boost: float = 2.0,
    side_yaw_threshold: float = 0.15,
) -> torch.Tensor:
    """Ungated ‖Δ²a‖² — kills ~25 Hz sign-flip chatter (Nyquist of 50 Hz)."""
    cache_name = "_action_acc_prev_prev_action_wd"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))

    prev_prev = getattr(env, cache_name)
    d2 = env.action_manager.action + prev_prev - 2.0 * env.action_manager.prev_action
    mask = (env.action_manager.prev_action != 0) & (prev_prev != 0)
    w = _wheeldog_action_dim_weights(
        d2.shape[-1], d2.device, leg_dim, leg_weight, hipy_knee_weight, wheel_weight
    )
    penalty = torch.sum(torch.square(d2) * w * mask, dim=1)
    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return penalty * _wheeldog_side_yaw_scale(env, command_name, side_yaw_boost, side_yaw_threshold)


def action_l2_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    leg_dim: int = 12,
    leg_weight: float = 1.0,
    wheel_weight: float = 4.0,
) -> torch.Tensor:
    """Penalize nonzero actions when cmd≈0 — forces stand equilibrium a→0."""
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    a = env.action_manager.action
    w = torch.ones(a.shape[-1], device=a.device)
    w[:leg_dim] = leg_weight
    if a.shape[-1] > leg_dim:
        w[leg_dim:] = wheel_weight
    return torch.sum(torch.square(a) * w, dim=1) * (cmd < command_threshold).float()


def lin_vel_xy_l2_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty |v_xy|² when cmd≈0 — kill standing skate/slide."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    vxy = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return vxy * (cmd < float(command_threshold)).float()


def _wheel_drive_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    vx_threshold: float,
    side_yaw_threshold: float,
) -> torch.Tensor:
    """True when |vx| dominates and |vy|+|yaw| small — roll on wheels, don't step."""
    cmd = env.command_manager.get_command(command_name)
    return (cmd[:, 0].abs() > vx_threshold) & ((cmd[:, 1].abs() + cmd[:, 2].abs()) < side_yaw_threshold)


def wheel_action_track_lin_x(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    wheel_scale: float = 5.0,
    std: float = 0.25,
    vx_threshold: float = 0.15,
    side_yaw_threshold: float = 0.2,
    # FL/RL axis -Y, FR/RR +Y → left/right opposite joint sign for +x
    wheel_signs: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
) -> torch.Tensor:
    """Reward wheel actions matching vx when in wheel-drive mode (flat forward/back)."""
    cmd = env.command_manager.get_command(command_name)
    vx = cmd[:, 0]
    sl = _get_action_term_slice(env, "joint_vel")
    wheel_a = env.action_manager.action[:, sl]
    signs = torch.tensor(wheel_signs, device=wheel_a.device, dtype=wheel_a.dtype)
    target = (vx / wheel_scale).unsqueeze(1) * signs.unsqueeze(0)
    err = torch.mean(torch.square(wheel_a - target), dim=1)
    reward = torch.exp(-err / std**2)
    return reward * _wheel_drive_gate(env, command_name, vx_threshold, side_yaw_threshold).float()


def leg_action_l2_wheel_drive(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    vx_threshold: float = 0.15,
    side_yaw_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize leg actions during pure vx — force wheel-driven forward/back."""
    sl = _get_action_term_slice(env, "joint_pos")
    penalty = torch.sum(torch.square(env.action_manager.action[:, sl]), dim=1)
    return penalty * _wheel_drive_gate(env, command_name, vx_threshold, side_yaw_threshold).float()


def feet_air_time_wheel_drive_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    vx_threshold: float = 0.15,
    side_yaw_threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize feet leaving ground during wheel-drive vx (no stepping for forward)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0
    penalty = torch.sum(in_air.float(), dim=1)
    return penalty * _wheel_drive_gate(env, command_name, vx_threshold, side_yaw_threshold).float()


def _get_action_term_slice(env: ManagerBasedRLEnv, action_term_name: str) -> slice:
    """Resolve and cache a slice of concatenated action vector for a given action term name."""
    cache_name = "_action_term_slices_cache"
    if not hasattr(env, cache_name):
        term_names = list(env.action_manager.active_terms)
        term_dims = list(env.action_manager.action_term_dim)
        start = 0
        cache: dict[str, slice] = {}
        for name, dim in zip(term_names, term_dims, strict=False):
            cache[name] = slice(start, start + dim)
            start += dim
        setattr(env, cache_name, cache)

    cache = getattr(env, cache_name)
    if action_term_name not in cache:
        raise ValueError(
            f"Action term '{action_term_name}' not found. Available terms: {list(cache.keys())}"
        )
    return cache[action_term_name]

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact force violations (curriculum-scaled by gait_level)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    reward = torch.sum(violation.clip(min=0.0), dim=1)
    return reward * get_gait_level_tensor(env)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    near_zero_thr: float | None = None,
    near_zero_std: float | None = None,
    near_zero_require_vx_thr: float | None = None,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel.

    Optional near_zero_*: when |yaw_cmd|<thr use tighter std — 直行压偏航.
    near_zero_require_vx_thr: 仅前进时启用更紧 std（站桩不刷分）.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    if near_zero_thr is not None and near_zero_std is not None:
        use_tight = cmd[:, 2].abs() < float(near_zero_thr)
        if near_zero_require_vx_thr is not None:
            use_tight = use_tight & (cmd[:, 0].abs() > float(near_zero_require_vx_thr))
        std_t = torch.where(
            use_tight,
            torch.full_like(ang_vel_error, float(near_zero_std)),
            torch.full_like(ang_vel_error, float(std)),
        )
        return torch.exp(-ang_vel_error / (std_t**2))
    return torch.exp(-ang_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward * get_gait_level_tensor(env)


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty_except_turn_side_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    turn_side_penalty_scale: float = 0.0,
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_height_threshold: float = 0.05,
    high_terrain_penalty_scale: float = 0.5,
) -> torch.Tensor:
    """Joint position penalty reduced (not always zero) on turn/side commands.

    When turn/side gate is active, multiply by ``turn_side_penalty_scale`` (0 = M20
    fully open; Wheeldog uses ~0.4 so yaw cannot straighten knees and stand up).

    If ``sensor_cfg`` is provided, further reduce the penalty on high terrain.
    """
    reward = joint_pos_penalty(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold,
        command_threshold=command_threshold,
    )

    cmd = env.command_manager.get_command(command_name)
    # 只要有横移/转向就放开（含前进+转向）；不再要求「纯原地」否则边走边转学不会
    del xy_norm_max, xz_norm_max
    gate_turn = torch.abs(cmd[:, 2]) > ang_cmd_threshold
    gate_side = torch.abs(cmd[:, 1]) > y_cmd_threshold
    disable_gate = gate_turn | gate_side
    scale_ts = float(turn_side_penalty_scale)
    reward = torch.where(disable_gate, reward * scale_ts, reward)

    if sensor_cfg is not None:
        height_sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = height_sensor.data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(ray_hits) & (torch.abs(ray_hits) <= 1e6)
        valid_count = torch.sum(valid_hits, dim=1)
        safe_hits = torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits))
        terrain_height = torch.sum(safe_hits, dim=1) / torch.clamp(valid_count, min=1)
        terrain_height = terrain_height - env.scene.env_origins[:, 2]
        high_terrain = (terrain_height > terrain_height_threshold) & (valid_count > 0)
        scale = torch.where(
            high_terrain,
            torch.full_like(reward, high_terrain_penalty_scale),
            torch.ones_like(reward),
        )
        reward = reward * scale

    return reward


def knee_extension_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_K_L_j3"]),
) -> torch.Tensor:
    """Penalize knee straightening past stand pose (|q| < |q0|) — blocks yaw stand-up."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # bent stand has larger |q|; extension toward 0 raises the base
    extension = torch.clamp(torch.abs(q0) - torch.abs(q), min=0.0)
    return torch.sum(torch.square(extension), dim=1)


def base_height_l2_ungated(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Same as base_height_l2 but not scaled by gait_level (keep crouch early)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


class GaitRewardSideYaw(GaitReward):
    """Lite3-style diagonal trot, only when strafing or yawing (not pure forward wheel-roll)."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        side_thr: float = 0.10,
        yaw_thr: float = 0.10,
    ) -> torch.Tensor:
        reward = super().__call__(
            env,
            std,
            command_name,
            max_err,
            velocity_threshold,
            command_threshold,
            synced_feet_pair_names,
            asset_cfg,
            sensor_cfg,
        )
        cmd = env.command_manager.get_command(command_name)
        gate = (cmd[:, 1].abs() > float(side_thr)) | (cmd[:, 2].abs() > float(yaw_thr))
        return reward * gate.float()


def feet_slide_side_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side_thr: float = 0.10,
    yaw_thr: float = 0.10,
) -> torch.Tensor:
    """Penalize foot drag during strafe/yaw — kills one-diagonal-scrape gait."""
    reward = feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    cmd = env.command_manager.get_command(command_name)
    gate = (cmd[:, 1].abs() > float(side_thr)) | (cmd[:, 2].abs() > float(yaw_thr))
    return reward * gate.float()


def phase_foot_trajectory_side_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
    command_threshold: float = 0.1,
    cycle_time: float = 0.4,
    phase_offsets: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0),
    gait_span: float = -0.008,
    gait_psi: float = 0.05,
    gait_delta: float = 0.02,
    x_offset: float = 0.0,
    stance_span: float = 0.20,
    stand_ref_z_offset: float = -0.2,
    velocity_weight: float = 0.5,
    side_thr: float = 0.10,
    yaw_thr: float = 0.10,
) -> torch.Tensor:
    """Point-foot phase swing, gated to strafe/yaw only."""
    reward = phase_foot_trajectory_exp(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        std=std,
        command_threshold=command_threshold,
        cycle_time=cycle_time,
        phase_offsets=phase_offsets,
        gait_span=gait_span,
        gait_psi=gait_psi,
        gait_delta=gait_delta,
        x_offset=x_offset,
        stance_span=stance_span,
        stand_ref_z_offset=stand_ref_z_offset,
        velocity_weight=velocity_weight,
    )
    cmd = env.command_manager.get_command(command_name)
    gate = (cmd[:, 1].abs() > float(side_thr)) | (cmd[:, 2].abs() > float(yaw_thr))
    return reward * gate.float()


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    scale_by_gait_level: bool = True,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    if scale_by_gait_level:
        reward = reward * get_gait_level_tensor(env)
    return reward


def joint_lr_mirror_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    h1_names: list[str] | None = None,
    h2k_pairs: list[list[str]] | None = None,
    straight_thr: float = 0.12,
    yaw_thr: float = 0.10,
) -> torch.Tensor:
    """Bilateral L/R symmetry (morphological): FL↔FR, RL↔RR.

    Online practice (morphosymm / symmetry-guided RL): enforce sagittal mirror on
    straight motion; release when |vy| or |yaw| is large so turn/strafe can break
    symmetry. H1 uses outward scalars (axis signs flip across midline).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if h1_names is None:
        h1_names = ["FL_H1_L_j1", "FR_H1_L_j1", "RL_H1_L_j1", "RR_H1_L_j1"]
    if h2k_pairs is None:
        h2k_pairs = [
            ["FL_H2_L_j2", "FR_H2_L_j2"],
            ["RL_H2_L_j2", "RR_H2_L_j2"],
            ["FL_K_L_j3", "FR_K_L_j3"],
            ["RL_K_L_j3", "RR_K_L_j3"],
        ]
    cache_key = "_joint_lr_mirror_cache"
    if not hasattr(env, cache_key):
        name_to_id = {n: i for i, n in enumerate(asset.data.joint_names)}
        h1_ids = [name_to_id[n] for n in h1_names if n in name_to_id]
        # outward: FL/RR +q, FR/RL −q
        h1_signs = []
        for n in h1_names:
            if n not in name_to_id:
                continue
            h1_signs.append(1.0 if n.startswith(("FL_", "RR_")) else -1.0)
        pair_ids = []
        for a, b in h2k_pairs:
            if a in name_to_id and b in name_to_id:
                pair_ids.append((name_to_id[a], name_to_id[b]))
        setattr(
            env,
            cache_key,
            {
                "h1_ids": h1_ids,
                "h1_signs": torch.tensor(h1_signs, device=env.device, dtype=torch.float32),
                "pair_ids": pair_ids,
            },
        )
    cache = getattr(env, cache_key)
    pen = torch.zeros(env.num_envs, device=env.device)
    if len(cache["h1_ids"]) == 4:
        q = asset.data.joint_pos[:, cache["h1_ids"]]
        outward = q * cache["h1_signs"].unsqueeze(0)
        # indices follow h1_names order FL,FR,RL,RR
        pen = pen + torch.square(outward[:, 0] - outward[:, 1])
        pen = pen + torch.square(outward[:, 2] - outward[:, 3])
    for i, j in cache["pair_ids"]:
        # H2/K: left/right often opposite URDF sign → compare magnitudes of deviation from default
        q0 = asset.data.default_joint_pos
        d_i = asset.data.joint_pos[:, i] - q0[:, i]
        d_j = asset.data.joint_pos[:, j] - q0[:, j]
        pen = pen + torch.square(torch.abs(d_i) - torch.abs(d_j))
    cmd = env.command_manager.get_command(command_name)
    # Full weight on near-straight; soften on side/yaw (allow intentional asymmetry)
    side = torch.abs(cmd[:, 1])
    yaw = torch.abs(cmd[:, 2])
    straight = (side < straight_thr) & (yaw < yaw_thr)
    scale = torch.where(straight, torch.ones_like(pen), torch.full_like(pen, 0.15))
    return pen * scale


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     # print(torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1), "command norm")
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # return reward

# def feet_air_time(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     mode_time: float,
#     velocity_threshold: float,
# ) -> torch.Tensor:
#     """Reward longer feet air and contact time."""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     asset: Articulation = env.scene[asset_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

#     t_max = torch.max(current_air_time, current_contact_time)
#     t_min = torch.clip(t_max, max=mode_time)
#     stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
#     cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
#     body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
#     reward = torch.where(
#         torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
#         torch.where(t_max < mode_time, t_min, 0),
#         stance_cmd_reward,
#     )
#     return torch.sum(reward, dim=1)


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1)
    # print(last_air_time, "last air time")
    # print(last_contact_time, "last contact time")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward




def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.5
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # print(contact, "contact")
    reward = torch.sum(contact, dim=-1).float()
    # print(reward, "reward after sum")
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.5
    # print(env.command_manager.get_command(command_name), "env.command_manager.get_command(command_name)")
    # print(reward, "reward after multiply")
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(
    #     tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # )
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward = torch.sum(foot_z_target_error, dim=1)
    # print(foot_z_target_error, "foot_z_target_error")
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_slide_ang_z_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_lin_threshold: float = 0.1,
    cmd_ang_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize feet sliding, only active when linear xy command is small and angular z command is large.

    This is a variant of ``feet_slide`` that is gated by the velocity command: the penalty is only applied
    when ``norm(cmd_xy) < cmd_lin_threshold`` AND ``abs(cmd_z) > cmd_ang_threshold``, i.e. during pure
    yaw-rotation commands where foot sliding is most problematic.
    """
    reward = feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)

    # --- command gating: only active for pure yaw commands ---
    command = env.command_manager.get_command(command_name)
    lin_xy_norm = torch.norm(command[:, :2], dim=1)
    ang_z_abs = torch.abs(command[:, 2])
    gate = (lin_xy_norm < cmd_lin_threshold) & (ang_z_abs > cmd_ang_threshold)
    reward = reward * gate.float()

    return reward * get_gait_level_tensor(env)

def _bernstein_torch(n: int, k: int, t: torch.Tensor) -> torch.Tensor:
    """Bernstein basis B_k^n(t) for tensor t in [0, 1]."""
    coeff = float(math.comb(n, k))
    return coeff * (1.0 - t) ** (n - k) * t**k


def _bezier_curve_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve points for batched parameter t.

    Args:
        control_points: Tensor of shape [m, 2].
        t: Tensor of shape [N, L] in [0, 1].

    Returns:
        Tensor of shape [N, L, 2].
    """
    n = control_points.shape[0] - 1
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n + 1):
        out = out + _bernstein_torch(n, k, t).unsqueeze(-1) * control_points[k]
    return out


def _bezier_curve_derivative_torch(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate d/dt of Bezier curve points for batched parameter t."""
    n = control_points.shape[0] - 1
    delta_ctrl = n * (control_points[1:] - control_points[:-1])
    out = torch.zeros(*t.shape, 2, device=t.device, dtype=t.dtype)
    for k in range(n):
        out = out + _bernstein_torch(n - 1, k, t).unsqueeze(-1) * delta_ctrl[k]
    return out


def phase_foot_trajectory_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
    command_threshold: float = 0.1,
    cycle_time: float = 0.4,
    phase_offsets: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0),
    gait_span: float = -0.008,
    gait_psi: float = 0.15,
    gait_delta: float = 0.03,
    x_offset: float = 0.0,
    stance_span: float = 0.20,
    stand_ref_z_offset: float = -0.2,
    velocity_weight: float = 0.5,
) -> torch.Tensor:
    """Track MuJoCo-style phase foot trajectory in body frame with exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_feet = len(body_ids)

    if num_feet == 0:
        return torch.zeros(env.num_envs, device=env.device)
    if len(phase_offsets) != num_feet:
        raise ValueError(f"phase_offsets length ({len(phase_offsets)}) must match tracked feet ({num_feet}).")

    # Build and cache base-fixed stand references from the first call.
    if (not hasattr(env, "phase_foot_ref_body")) or (env.phase_foot_ref_body.shape[1] != num_feet):
        rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
        foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
        for i in range(num_feet):
            foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        ref = foot_pos_b[0].detach().clone()
        ref[:, 2] += stand_ref_z_offset
        env.phase_foot_ref_body = ref.unsqueeze(0)

    stand_ref_body = env.phase_foot_ref_body.to(env.device).expand(env.num_envs, -1, -1)

    # Build phase S in [0, 2).
    phase_time = env.episode_length_buf.float() * env.step_dt
    phase_offsets_t = torch.tensor(phase_offsets, device=env.device, dtype=phase_time.dtype).unsqueeze(0)
    S = torch.remainder((2.0 * phase_time / max(cycle_time, 1e-6)).unsqueeze(1) + phase_offsets_t, 2.0)

    # MuJoCo-like piecewise trajectory in local (q, z).
    tau = float(gait_span)
    psi = float(gait_psi)
    delta = float(gait_delta)
    stance_span = float(stance_span)
    stance_span = min(max(stance_span, 1e-6), 2.0 - 1e-6)

    q = torch.zeros_like(S)
    z = torch.zeros_like(S)
    dq_dS = torch.zeros_like(S)
    dz_dS = torch.zeros_like(S)

    stance_mask = S < stance_span
    if stance_mask.any():
        s_stance = S / stance_span
        q_stance = tau * (1.0 - 2.0 * s_stance)
        z_stance = torch.full_like(S, delta)
        dq_dS_stance = torch.full_like(S, -2.0 * tau / stance_span)
        dz_dS_stance = torch.zeros_like(S)

        q = torch.where(stance_mask, q_stance, q)
        z = torch.where(stance_mask, z_stance, z)
        dq_dS = torch.where(stance_mask, dq_dS_stance, dq_dS)
        dz_dS = torch.where(stance_mask, dz_dS_stance, dz_dS)

    swing_mask = ~stance_mask
    if swing_mask.any():
        t_bezier = torch.clamp((S - stance_span) / (2.0 - stance_span), 0.0, 1.0)
        ctrl = torch.tensor(
            [
                [-tau, 0.0],
                [-0.95 * tau, 0.80 * psi],
                [-0.55 * tau, 1.00 * psi],
                [0.55 * tau, 1.00 * psi],
                [0.95 * tau, 0.80 * psi],
                [tau, 0.0],
            ],
            device=env.device,
            dtype=S.dtype,
        )
        qz_swing = _bezier_curve_torch(ctrl, t_bezier)
        dqz_dt = _bezier_curve_derivative_torch(ctrl, t_bezier)
        dt_dS = 1.0 / (2.0 - stance_span)

        q = torch.where(swing_mask, qz_swing[..., 0], q)
        z = torch.where(swing_mask, qz_swing[..., 1] + delta, z)
        dq_dS = torch.where(swing_mask, dqz_dt[..., 0] * dt_dS, dq_dS)
        dz_dS = torch.where(swing_mask, dqz_dt[..., 1] * dt_dS, dz_dS)

    dS_dt = 2.0 / max(cycle_time, 1e-6)
    dq_dt = dq_dS * dS_dt
    dz_dt = dz_dS * dS_dt

    ref_pos_b = stand_ref_body + torch.stack(
        [q + float(x_offset), torch.zeros_like(q), z],
        dim=-1,
    )
    ref_vel_b = torch.stack(
        [dq_dt, torch.zeros_like(dq_dt), dz_dt],
        dim=-1,
    )

    # Actual foot states in body frame.
    rel_foot_pos_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    rel_foot_vel_w = asset.data.body_lin_vel_w[:, body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    foot_vel_b = torch.zeros(env.num_envs, num_feet, 3, device=env.device)
    for i in range(num_feet):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_pos_w[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, rel_foot_vel_w[:, i, :])

    pos_offset = foot_pos_b - ref_pos_b
    vel_offset = foot_vel_b - ref_vel_b

    # Per-dimension (x, y, z) errors over feet for each environment.
    pos_err = torch.sum(torch.square(pos_offset), dim=1)
    vel_err = torch.sum(torch.square(vel_offset), dim=1)

    # Scalar total error for reward computation.
    total_err = torch.sum(pos_err, dim=1) + float(velocity_weight) * torch.sum(vel_err, dim=1)
    reward = torch.exp(-total_err / max(std, 1e-6) ** 2)

    # Command-gating follows full (x, y, yaw) command magnitude.
    command = env.command_manager.get_command(command_name)
    gate = torch.linalg.norm(command[:, :3], dim=1) > command_threshold

    # info for debugging
    pos_offset_xyz_mean = torch.mean(pos_offset, dim=(0, 1))
    vel_offset_xyz_mean = torch.mean(vel_offset, dim=(0, 1))
    # print(
    #     "Offset xyz mean | "
    #     f"pos(x,y,z)=({pos_offset_xyz_mean[0].item():.4f}, {pos_offset_xyz_mean[1].item():.4f}, {pos_offset_xyz_mean[2].item():.4f}) | "
    #     f"vel(x,y,z)=({vel_offset_xyz_mean[0].item():.4f}, {vel_offset_xyz_mean[1].item():.4f}, {vel_offset_xyz_mean[2].item():.4f})"
    # )
    # print("Reward:", reward * gate.float())
    return reward * gate.float() * get_gait_level_tensor(env)

def foot_impact_velocity(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    foot_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    downward_speed = torch.clamp(-foot_lin_vel[:, :, 2], min=0.0)
    downward_speed = torch.clamp(downward_speed - speed_threshold, min=0.0)

    penalty = torch.sum(first_contact * torch.square(downward_speed), dim=1)
    return penalty * get_gait_level_tensor(env)

def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from default joint pose when command ≈ 0 (return-to-stand)."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd < command_threshold).float()


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def joint_acc_l2_new(env: ManagerBasedRLEnv) -> torch.Tensor:

# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     # print(torch.sum(diff, dim=1), "smoothness l2")
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward





def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_z_l2_lateral_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    y_cmd_threshold: float = 0.3,
    ang_cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize z-axis angular velocity when there is a lateral (y) command but no rotation command.

    This discourages the robot from rotating around the z-axis to satisfy lateral velocity
    commands instead of actually translating laterally.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_z_sq = torch.square(asset.data.root_ang_vel_b[:, 2])

    cmd = env.command_manager.get_command(command_name)
    gate = (torch.abs(cmd[:, 1]) > y_cmd_threshold) & (torch.abs(cmd[:, 2]) < ang_cmd_threshold)
    return ang_vel_z_sq * gate.float()


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_lin_xy_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by planar linear velocity command (x, y)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by planar linear velocity command only
    cmd_lin_xy = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= cmd_lin_xy > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_x_neg_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by negative x velocity command only."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate: x command must be negative and exceed magnitude threshold
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward *= cmd_x < 0.0

    return reward

def feet_air_time_ang_z_cmd_lite3(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Air-time reward gated by yaw angular velocity command (z)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Core logic unchanged
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Gate by angular velocity command only
    cmd_ang_z = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    reward *= cmd_ang_z > cmd_threshold
    return reward * get_gait_level_tensor(env)

def feet_air_time_ang_z_cmd_M20(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    cmd_threshold: float = 0.1,
    foot_height_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Air-time reward gated by yaw command and minimum foot clearance."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    state_name = "_feet_air_time_ang_z_height_state"
    state = getattr(env, state_name, None)
    if state is None or state["ground_height"].shape != foot_height.shape:
        state = {
            "ground_height": foot_height.clone(),
            "max_air_height": foot_height.clone(),
        }
        setattr(env, state_name, state)

    ground_height = state["ground_height"].to(device=foot_height.device)
    max_air_height = state["max_air_height"].to(device=foot_height.device)
    max_air_height = torch.where(in_contact, max_air_height, torch.maximum(max_air_height, foot_height))
    cleared_ground = (max_air_height - ground_height) > foot_height_threshold
    reward = torch.sum((last_air_time - threshold) * first_contact * cleared_ground, dim=1)

    state["ground_height"] = torch.where(in_contact, foot_height, ground_height)
    state["max_air_height"] = torch.where(in_contact, foot_height, max_air_height)

    # 转向 + 横移都要抬腿（旧版只 gate yaw → 横移只会拖地）
    cmd = env.command_manager.get_command(command_name)
    gate = (cmd[:, 2].abs() > float(cmd_threshold)) | (cmd[:, 1].abs() > float(cmd_threshold))
    reward *= gate.float()
    # 不乘 gait_level：早期也要学会抬腿，否则永远拖着走
    return reward


def feet_clearance_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    air_threshold: float = 0.12,
    foot_clearance: float = 0.08,
    mode: str = "side_yaw",
    side_thr: float = 0.10,
    yaw_thr: float = 0.10,
    vx_thr: float = 0.15,
    xy_thr: float = 0.12,
) -> torch.Tensor:
    """Air-time with min foot clearance — no gait_level gate.

    mode:
      - side_yaw: |vy| or |yaw| dominate → 横移/转向用腿，抬高防轮胎蹭地
      - forward:  |vx| dominate, small vy/yaw → 前进对角迈步（轮腿混合）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)

    cache = f"_feet_clear_{mode}"
    state = getattr(env, cache, None)
    if state is None or state["g"].shape != foot_z.shape:
        state = {"g": foot_z.clone(), "m": foot_z.clone()}
        setattr(env, cache, state)
    g = state["g"].to(device=foot_z.device)
    m = state["m"].to(device=foot_z.device)
    m = torch.where(in_contact, m, torch.maximum(m, foot_z))
    cleared = (m - g) > float(foot_clearance)
    reward = torch.sum((last_air - float(air_threshold)) * first_contact.float() * cleared.float(), dim=1)
    state["g"] = torch.where(in_contact, foot_z, g)
    state["m"] = torch.where(in_contact, foot_z, m)

    cmd = env.command_manager.get_command(command_name)
    vx, vy, yaw = cmd[:, 0].abs(), cmd[:, 1].abs(), cmd[:, 2].abs()
    if mode == "side_yaw":
        # 有横移/转向就抬腿；不再用 vx 上限把混合指令门死
        gate = (vy > side_thr) | (yaw > yaw_thr)
    elif mode == "forward":
        gate = (vx > vx_thr) & (vy < side_thr) & (yaw < yaw_thr)
    else:
        gate = torch.linalg.norm(cmd, dim=1) > xy_thr
    return reward * gate.float()


def wheel_vel_l2_side_yaw(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_W_L_j4"]),
    side_thr: float = 0.10,
    yaw_thr: float = 0.10,
    vx_exclude: float = 0.20,
) -> torch.Tensor:
    """横移/转向时锁轮：罚 |ω|，逼着用腿迈步而不是轮胎蹭地拧。

    vx_exclude<=0：只要有 side/yaw 就锁轮（混合指令也不开拖轮捷径）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        names = list(asset.data.joint_names)
        ids = [i for i, n in enumerate(names) if n.endswith("_W_L_j4")]
    omega = asset.data.joint_vel[:, ids]
    pen = torch.sum(torch.square(omega), dim=1)
    cmd = env.command_manager.get_command(command_name)
    side_yaw = (cmd[:, 1].abs() > side_thr) | (cmd[:, 2].abs() > yaw_thr)
    if float(vx_exclude) <= 0.0:
        gate = side_yaw
    else:
        gate = side_yaw & (cmd[:, 0].abs() < float(vx_exclude))
    return pen * gate.float()


def feet_air_time_including_ang_z(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward

def lin_vel_xy_l2_with_ang_z_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Penalize xy-axis base linear velocity using L2 squared kernel if command is ang_vel_z."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    command = env.command_manager.get_command(command_name)
    reward *= (torch.sum(torch.square(command[:, 2:]), dim=1) > command_threshold) & \
            (torch.sum(torch.square(command[:, :2]), dim=1) < command_threshold)
    # reward *= torch.sum(torch.square(env.command_manager.get_command(command_name)[:, 2:]), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def rotation_gait_status(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    group_a_body_names: list[str],
    group_b_body_names: list[str],
    target_height: float = 0.05,
    lin_vel_threshold: float = 0.5,
    ang_vel_threshold: float = 0.05,
    scale_by_gait_level: bool = True,
) -> torch.Tensor:
    """Reward alternating contact pattern during rotation.

    When linear velocity command is small and angular velocity command is significant,
    rewards the robot for having one diagonal group grounded while the other is lifted
    by at least `target_height` above the grounded group.

    Groups are diagonal leg pairs (e.g., FL+HR vs FR+HL).
    Flat trains must set scale_by_gait_level=False (gait_level stays 0).
    """
    # Cache body IDs on first call
    cache_key = "_rotation_gait_status_cache"
    if not hasattr(env, cache_key):
        _sensor = env.scene.sensors[sensor_cfg.name]
        _asset = env.scene[asset_cfg.name]
        group_a_ids = [_sensor.find_bodies(n)[0][0] for n in group_a_body_names]
        group_b_ids = [_sensor.find_bodies(n)[0][0] for n in group_b_body_names]
        group_a_body_ids = [_asset.find_bodies(n)[0][0] for n in group_a_body_names]
        group_b_body_ids = [_asset.find_bodies(n)[0][0] for n in group_b_body_names]
        setattr(env, cache_key, {
            "group_a_sensor_ids": group_a_ids,
            "group_b_sensor_ids": group_b_ids,
            "group_a_body_ids": group_a_body_ids,
            "group_b_body_ids": group_b_body_ids,
        })

    cache = getattr(env, cache_key)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    # Gate: only active when rotating in place
    command = env.command_manager.get_command(command_name)
    lin_vel_cmd = torch.linalg.norm(command[:, :2], dim=1)
    ang_vel_cmd = torch.abs(command[:, 2])
    gate = (lin_vel_cmd < lin_vel_threshold) & (ang_vel_cmd > ang_vel_threshold)

    # Contact state: True if in contact
    contact_time = contact_sensor.data.current_contact_time
    group_a_contact = contact_time[:, cache["group_a_sensor_ids"]] > 0  # [N, 2]
    group_b_contact = contact_time[:, cache["group_b_sensor_ids"]] > 0  # [N, 2]

    # Height of each group's feet (world z)
    z_a = asset.data.body_pos_w[:, cache["group_a_body_ids"], 2]  # [N, 2]
    z_b = asset.data.body_pos_w[:, cache["group_b_body_ids"], 2]  # [N, 2]
    mean_z_a = z_a.mean(dim=1)  # [N]
    mean_z_b = z_b.mean(dim=1)  # [N]

    # Pattern 1: Group A grounded, Group B lifted
    a_grounded = group_a_contact.all(dim=1).float()  # [N]
    b_lifted_height = torch.clamp(mean_z_b - mean_z_a - target_height, min=0.0)
    pattern_1 = a_grounded * torch.exp(-torch.square(b_lifted_height - target_height) / (0.02**2))
    # Simpler: reward if height diff >= target_height
    pattern_1 = a_grounded * (mean_z_b - mean_z_a > target_height).float()

    # Pattern 2: Group B grounded, Group A lifted
    b_grounded = group_b_contact.all(dim=1).float()  # [N]
    pattern_2 = b_grounded * (mean_z_a - mean_z_b > target_height).float()

    # Reward is max of either valid pattern
    reward = torch.maximum(pattern_1, pattern_2)
    reward *= gate.float()
    if scale_by_gait_level:
        reward = reward * get_gait_level_tensor(env)
    return reward


class RotationGaitSymmetry(ManagerTermBase):
    """Reward symmetric contact duty cycle during rotation.

    Tracks contact history over a 5-second window and rewards each diagonal leg group
    for spending roughly equal time in contact vs air (50/50 duty cycle).

    Only active when linear velocity command is small and angular velocity command
    is significant (rotating in place).
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name: str = cfg.params["command_name"]
        self.lin_vel_threshold: float = cfg.params["lin_vel_threshold"]
        self.ang_vel_threshold: float = cfg.params["ang_vel_threshold"]
        self.target_duty: float = cfg.params["target_duty"]
        self.std: float = cfg.params["std"]
        self.scale_by_gait_level: bool = bool(cfg.params.get("scale_by_gait_level", True))
        self.side_thr: float = float(cfg.params.get("side_thr", 0.0) or 0.0)

        # Resolve body IDs from contact sensor
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        group_a_body_names = cfg.params["group_a_body_names"]
        group_b_body_names = cfg.params["group_b_body_names"]
        self.group_a_ids = [self.contact_sensor.find_bodies(n)[0][0] for n in group_a_body_names]
        self.group_b_ids = [self.contact_sensor.find_bodies(n)[0][0] for n in group_b_body_names]

        # Circular buffer: [num_envs, num_feet, buffer_size]
        # buffer_size = 5 seconds / step_dt
        self.buffer_size = int(5.0 / env.step_dt)
        num_feet = len(self.group_a_ids) + len(self.group_b_ids)
        self.all_ids = self.group_a_ids + self.group_b_ids
        self.contact_buffer = torch.zeros(
            env.num_envs, num_feet, self.buffer_size, device=env.device
        )
        self.buffer_idx = 0
        self.buffer_filled = False  # Track if buffer has filled at least once

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear buffer for reset envs."""
        if env_ids is not None and len(env_ids) > 0:
            self.contact_buffer[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        group_a_body_names: list[str],
        group_b_body_names: list[str],
        target_duty: float,
        std: float,
        lin_vel_threshold: float,
        ang_vel_threshold: float,
        scale_by_gait_level: bool = True,
        side_thr: float = 0.0,
    ) -> torch.Tensor:
        del scale_by_gait_level, side_thr  # read from self in __init__
        # Update buffer with current contact state
        contact_time = self.contact_sensor.data.current_contact_time
        in_contact = (contact_time[:, self.all_ids] > 0).float()  # [N, num_feet]
        self.contact_buffer[:, :, self.buffer_idx] = in_contact
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if not self.buffer_filled and self.buffer_idx == 0:
            self.buffer_filled = True

        # Gate: yaw in place, and optionally strafe (side_thr>0)
        command = env.command_manager.get_command(self.command_name)
        lin_vel_cmd = torch.linalg.norm(command[:, :2], dim=1)
        ang_vel_cmd = torch.abs(command[:, 2])
        gate = (lin_vel_cmd < self.lin_vel_threshold) & (ang_vel_cmd > self.ang_vel_threshold)
        side_thr = float(getattr(self, "side_thr", 0.0) or 0.0)
        if side_thr > 0.0:
            gate = gate | ((command[:, 1].abs() > side_thr) & (command[:, 0].abs() < self.lin_vel_threshold))

        # Compute duty cycle for each group over the buffer window
        num_a = len(self.group_a_ids)
        # Group A: first num_a entries, Group B: remaining
        duty_a = self.contact_buffer[:, :num_a, :].mean(dim=(1, 2))  # [N]
        duty_b = self.contact_buffer[:, num_a:, :].mean(dim=(1, 2))  # [N]

        # Reward: exp kernel on deviation from target duty (0.5)
        reward_a = torch.exp(-torch.square(duty_a - self.target_duty) / (self.std**2))
        reward_b = torch.exp(-torch.square(duty_b - self.target_duty) / (self.std**2))
        reward = reward_a * reward_b

        # Only give reward once buffer has meaningful data
        if not self.buffer_filled:
            reward = reward * (self.buffer_idx / self.buffer_size)

        reward *= gate.float()
        if self.scale_by_gait_level:
            reward = reward * get_gait_level_tensor(env)
        return reward

def bad_orientation_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty based on bad_orientation_2 condition.

    Returns 1.0 when orientation is considered bad (including back-up flip),
    else returns 0.0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    bad = (asset.data.projected_gravity_b[:, 2] > 0) | (asset.data.projected_gravity_b[:, :2].abs() > 0.7).any(-1)
    return bad.float()

# ---------------------------------------------------------------------------
# HIM-style lift-leg gating (from isaaclab_him/wheeldog/mdp.py)
# ---------------------------------------------------------------------------

def _cmd_mode_gates_him(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    xy_thr: float = 0.15,
    yaw_thr: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(roll_gate, hybrid_gate): roll=pure vx; hybrid=side|yaw."""
    cmd = env.command_manager.get_command(command_name)
    vx = torch.abs(cmd[:, 0])
    vy = torch.abs(cmd[:, 1])
    yaw = torch.abs(cmd[:, 2])
    hybrid = ((vy > xy_thr) | (yaw > yaw_thr)).float()
    roll = ((vx > xy_thr) & (vy <= xy_thr) & (yaw <= yaw_thr)).float()
    return roll, hybrid


def _diag_gait_want_a(env: ManagerBasedRLEnv, period_s: float = 0.90) -> torch.Tensor:
    t = env.episode_length_buf.float() * float(env.step_dt)
    phase = torch.fmod(t / float(period_s), 1.0)
    return phase < 0.5


def _diag_phase_match(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    group_a_body_names: list[str] | None = None,
    group_b_body_names: list[str] | None = None,
    period_s: float = 0.90,
) -> torch.Tensor:
    """1 if contact matches diagonal clock, else 0."""
    if group_a_body_names is None:
        group_a_body_names = ["FL_W_L", "RR_W_L"]
    if group_b_body_names is None:
        group_b_body_names = ["FR_W_L", "RL_W_L"]
    cache_key = "_wd_diag_phase_match_cache"
    if not hasattr(env, cache_key):
        sensor = env.scene.sensors[sensor_cfg.name]
        setattr(
            env,
            cache_key,
            {
                "a_s": [sensor.find_bodies(n)[0][0] for n in group_a_body_names],
                "b_s": [sensor.find_bodies(n)[0][0] for n in group_b_body_names],
            },
        )
    cache = getattr(env, cache_key)
    ct = env.scene.sensors[sensor_cfg.name].data.current_contact_time
    a_air = (ct[:, cache["a_s"]] <= 0).all(dim=1)
    b_air = (ct[:, cache["b_s"]] <= 0).all(dim=1)
    a_gnd = (ct[:, cache["a_s"]] > 0).all(dim=1)
    b_gnd = (ct[:, cache["b_s"]] > 0).all(dim=1)
    want_a = _diag_gait_want_a(env, period_s)
    return torch.where(want_a, a_air & b_gnd, b_air & a_gnd).float()


def track_lin_vel_xy_gait_gated_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    xy_thr: float = 0.12,
    yaw_thr: float = 0.12,
    period_s: float = 0.90,
    hybrid_floor: float = 0.35,
) -> torch.Tensor:
    """Velocity track: full on roll; on side/yaw must match lift clock else near-zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - cmd[:, :2]), dim=1)
    base = torch.exp(-err / (std**2))
    roll, hybrid = _cmd_mode_gates_him(env, command_name, xy_thr, yaw_thr)
    match = _diag_phase_match(env, sensor_cfg=sensor_cfg, period_s=period_s)
    scale = roll + hybrid * (hybrid_floor + (1.0 - hybrid_floor) * match)
    standing = (roll < 0.5) & (hybrid < 0.5)
    scale = torch.where(standing, torch.ones_like(scale), scale)
    return base * scale


def track_ang_vel_z_gait_gated_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    xy_thr: float = 0.12,
    yaw_thr: float = 0.12,
    period_s: float = 0.90,
    hybrid_floor: float = 0.55,
) -> torch.Tensor:
    """Yaw track: no alternating lift → almost no turn reward (HIM lift filter)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = torch.square(asset.data.root_ang_vel_b[:, 2] - cmd[:, 2])
    base = torch.exp(-err / (std**2))
    roll, hybrid = _cmd_mode_gates_him(env, command_name, xy_thr, yaw_thr)
    match = _diag_phase_match(env, sensor_cfg=sensor_cfg, period_s=period_s)
    scale = roll + hybrid * (hybrid_floor + (1.0 - hybrid_floor) * match)
    standing = (roll < 0.5) & (hybrid < 0.5)
    scale = torch.where(standing, torch.ones_like(scale), scale)
    return base * scale


def dual_hold_air_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    xy_thr: float = 0.12,
    yaw_thr: float = 0.12,
    hold_s: float = 0.35,
    yaw_only: bool = True,
    vx_exclude: float = 0.20,
    side_exclude: float = 0.12,
) -> torch.Tensor:
    """Penalize hold-in-air / flat-drag turn: must swing diagonals (HIM dual_hold)."""
    contact: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ids = sensor_cfg.body_ids
    in_air = contact.data.current_contact_time[:, ids] <= 0.0
    n_air = in_air.float().sum(dim=1)
    air_t = contact.data.current_air_time[:, ids] * in_air.float()
    mean_hold = torch.where(n_air > 0, air_t.sum(dim=1) / n_air.clamp(min=1.0), torch.zeros_like(n_air))
    first_contact = contact.compute_first_contact(env.step_dt)[:, ids].any(dim=1)
    cmd = env.command_manager.get_command(command_name)
    if yaw_only:
        gate = (
            (cmd[:, 2].abs() > yaw_thr)
            & (cmd[:, 1].abs() <= side_exclude)
            & (cmd[:, 0].abs() <= vx_exclude)
        ).float()
    else:
        _, gate = _cmd_mode_gates_him(env, command_name, xy_thr, yaw_thr)
    bad = (n_air >= 1.0) & (mean_hold > hold_s) & (~first_contact)
    scale = torch.where(n_air < 1.5, torch.full_like(n_air, 2.5), torch.ones_like(n_air))
    hold_pen = bad.float() * (mean_hold - hold_s).clamp(min=0.0) * scale
    flat_pen = (n_air < 0.5).float() * 2.0
    single_pen = ((n_air > 0.5) & (n_air < 1.5)).float() * 0.8
    return (hold_pen + flat_pen + single_pen) * gate


def equal_swing_lift_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    group_a_body_names: list[str] | None = None,
    group_b_body_names: list[str] | None = None,
    target_height: float = 0.04,
    eq_sigma: float = 0.025,
    height_sigma: float = 0.035,
    xy_thr: float = 0.10,
    yaw_thr: float = 0.08,
    period_s: float = 0.90,
    yaw_only: bool = False,
) -> torch.Tensor:
    """Reward equal lift within the clock-selected diagonal pair (4-leg symmetry).

    Ported from HIM: only the swinging diagonal scores; both feet should clear
    similarly — kills one-leg-high / one-leg-drag.
    """
    if group_a_body_names is None:
        group_a_body_names = ["FL_W_L", "RR_W_L"]
    if group_b_body_names is None:
        group_b_body_names = ["FR_W_L", "RL_W_L"]
    cache_key = "_equal_swing_lift_cache"
    if not hasattr(env, cache_key):
        sensor = env.scene.sensors[sensor_cfg.name]
        asset = env.scene[asset_cfg.name]
        setattr(
            env,
            cache_key,
            {
                "a_s": [sensor.find_bodies(n)[0][0] for n in group_a_body_names],
                "b_s": [sensor.find_bodies(n)[0][0] for n in group_b_body_names],
                "a_b": [asset.find_bodies(n)[0][0] for n in group_a_body_names],
                "b_b": [asset.find_bodies(n)[0][0] for n in group_b_body_names],
            },
        )
    cache = getattr(env, cache_key)
    contact = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    ct = contact.data.current_contact_time
    z = asset.data.body_pos_w[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    want_a = _diag_gait_want_a(env, period_s)
    # heights of each foot in swinging diagonal
    z_a = z[:, cache["a_b"]]
    z_b = z[:, cache["b_b"]]
    # support z as reference (grounded diagonal mean)
    support_z = torch.where(
        want_a.unsqueeze(1),
        z[:, cache["b_b"]].mean(dim=1, keepdim=True),
        z[:, cache["a_b"]].mean(dim=1, keepdim=True),
    )
    swing_z = torch.where(want_a.unsqueeze(1), z_a, z_b)
    lift = (swing_z - support_z).clamp(min=0.0)
    eq = torch.exp(-torch.square(lift[:, 0] - lift[:, 1]) / (eq_sigma**2))
    h_ok = torch.exp(-torch.square(lift.mean(dim=1) - float(target_height)) / (height_sigma**2))
    # both must be air-ish
    swing_air = torch.where(
        want_a,
        (ct[:, cache["a_s"]] <= 0).all(dim=1),
        (ct[:, cache["b_s"]] <= 0).all(dim=1),
    ).float()
    cmd = env.command_manager.get_command(command_name)
    if yaw_only:
        gate = (
            (cmd[:, 2].abs() > yaw_thr)
            & (cmd[:, 1].abs() <= xy_thr)
            & (cmd[:, 0].abs() <= 0.20)
        ).float()
    else:
        _, hybrid = _cmd_mode_gates_him(env, command_name, xy_thr, yaw_thr)
        gate = hybrid
    return eq * h_ok * swing_air * gate
