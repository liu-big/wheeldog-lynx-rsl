# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def gait_phase_clock(
    env: ManagerBasedRLEnv,
    period: float = 0.90,
    theta_fl: float = 0.0,
    theta_fr: float = 0.5,
    theta_rl: float = 0.5,
    theta_rr: float = 0.0,
    command_name: str = "base_velocity",
    stand_thr: float = 0.10,
    side_period: float = 1.20,
    side_theta_pos: tuple[float, float, float, float] = (0.75, 0.25, 0.0, 0.50),
    side_theta_neg: tuple[float, float, float, float] = (0.25, 0.75, 0.50, 0.0),
    side_thr: float = 0.10,
    yaw_exclude: float = 0.08,
    vx_exclude: float = 0.15,
) -> torch.Tensor:
    """Open-loop foot timing clock (8-D): sin/cos × {FL,FR,RL,RR}.

    Ported from HIMLoco isaaclab_him — policy conditions lift on this clock.
    Pure side → crawl offsets; else diagonal trot. Standing freezes to 0.
    """
    cmd = env.command_manager.get_command(command_name)
    vy = cmd[:, 1]
    yaw = cmd[:, 2].abs()
    vx = cmd[:, 0].abs()
    side = (
        (vy.abs() > float(side_thr))
        & (yaw <= float(yaw_exclude))
        & (vx <= float(vx_exclude))
    ).float()
    side_pos = side * (vy >= 0.0).float()
    side_neg = side * (vy < 0.0).float()
    t = env.episode_length_buf.float() * float(env.step_dt)
    period_v = side * float(side_period) + (1.0 - side) * float(period)
    phi = torch.fmod(t / period_v.clamp(min=1e-3), 1.0)
    yaw_th = torch.tensor(
        [theta_fl, theta_fr, theta_rl, theta_rr], device=env.device, dtype=phi.dtype
    )
    pos_th = torch.tensor(list(side_theta_pos), device=env.device, dtype=phi.dtype)
    neg_th = torch.tensor(list(side_theta_neg), device=env.device, dtype=phi.dtype)
    theta = (
        (1.0 - side).unsqueeze(1) * yaw_th.unsqueeze(0)
        + side_pos.unsqueeze(1) * pos_th.unsqueeze(0)
        + side_neg.unsqueeze(1) * neg_th.unsqueeze(0)
    )
    phase = torch.fmod(phi.unsqueeze(1) + theta, 1.0)
    ang = 2.0 * torch.pi * phase
    clock = torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)
    moving = (torch.linalg.norm(cmd[:, :3], dim=1) >= float(stand_thr)).float().unsqueeze(1)
    return clock * moving
