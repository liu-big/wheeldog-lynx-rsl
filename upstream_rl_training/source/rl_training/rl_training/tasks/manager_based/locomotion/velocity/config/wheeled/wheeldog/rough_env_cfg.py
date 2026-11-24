# Copyright (c) 2025
# Wheeldog = DeepRobotics M20/Lynx reward recipe with Wheeldog URDF only.
# No custom strafe/yaw/jitter reward stack — keep parity with deeprobotics_m20.

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from rl_training.assets.wheeldog import WHEELDOG_CFG  # isort: skip


@configclass
class WheeldogActionsCfg(ActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class WheeldogRewardsCfg(RewardsCfg):
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    # Flat/早期 gait_level≡0 → 门控 action_rate/smooth 恒 0；用不门控轻量腿加权压抖（勿过重）
    action_rate_l2_wheeldog = RewTerm(
        func=mdp.action_rate_l2_wheeldog,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "leg_dim": 12,
            "leg_weight": 3.0,
            "hipy_knee_weight": 4.5,
            "wheel_weight": 0.5,
            "side_yaw_boost": 1.5,
            "side_yaw_threshold": 0.15,
        },
    )
    action_acc_l2_wheeldog = RewTerm(
        func=mdp.action_acc_l2_wheeldog,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "leg_dim": 12,
            "leg_weight": 3.0,
            "hipy_knee_weight": 4.5,
            "wheel_weight": 0.5,
            "side_yaw_boost": 1.5,
            "side_yaw_threshold": 0.15,
        },
    )
    action_l2_without_cmd = RewTerm(
        func=mdp.action_l2_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.15,
            "leg_dim": 12,
            "leg_weight": 1.0,
            "wheel_weight": 2.0,
        },
    )
    # 防转向伸膝起立
    knee_extension_l2 = RewTerm(
        func=mdp.knee_extension_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )
    # 横移/转向：点足式对角交替 + 拖地罚 + 相位（仅 side/yaw 门控）
    feet_slide_side_yaw = RewTerm(
        func=mdp.feet_slide_side_yaw,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "side_thr": 0.10,
            "yaw_thr": 0.10,
        },
    )
    phase_foot_side_yaw = RewTerm(
        func=mdp.phase_foot_trajectory_side_yaw,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "std": 0.12,
            "command_threshold": 0.1,
            # 慢周期；抬腿略降（用户反馈太高）
            "cycle_time": 0.70,
            "phase_offsets": (0.0, 1.0, 1.0, 0.0),
            "gait_span": 0.0,
            "gait_psi": 0.07,
            "gait_delta": 0.025,
            "x_offset": 0.0,
            "stance_span": 1.0,
            "stand_ref_z_offset": 0.0,
            "velocity_weight": 0.0,
            "side_thr": 0.10,
            "yaw_thr": 0.10,
        },
    )
    # 直行压侧漂：仅 |vy|≈0 且 |vx| 够大（不站桩刷分）
    track_lin_vel_y_exp = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.10),
            "cmd_thr": 0.10,
            "active_when_near_zero": True,
            "require_vx_thr": 0.15,
        },
    )
    # 横移跟踪：|vy_cmd| 显著时单独盯 y
    track_lin_vel_y_strafe_exp = RewTerm(
        func=mdp.track_lin_vel_y_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.18),
            "cmd_thr": 0.10,
            "active_when_near_zero": False,
        },
    )
    track_lin_vel_x_exp = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "cmd_thr": 0.08},
    )
    lin_vel_y_l2_straight = RewTerm(
        func=mdp.lin_vel_y_l2_straight,
        weight=0.0,
        params={"command_name": "base_velocity", "vx_thr": 0.15, "vy_thr": 0.10, "yaw_thr": 0.10},
    )
    ang_vel_z_l2_straight = RewTerm(
        func=mdp.ang_vel_z_l2_straight,
        weight=0.0,
        params={"command_name": "base_velocity", "vx_thr": 0.15, "vy_thr": 0.10, "yaw_thr": 0.10},
    )
    # 纯前进：轮速动作对齐 vx + L/R 同步
    wheel_action_track_lin_x = RewTerm(
        func=mdp.wheel_action_track_lin_x,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "wheel_scale": 5.0,
            "std": 0.25,
            "vx_threshold": 0.15,
            "side_yaw_threshold": 0.20,
        },
    )
    wheel_action_lr_sync_straight = RewTerm(
        func=mdp.wheel_action_lr_sync_straight,
        weight=0.0,
        params={"command_name": "base_velocity", "vx_thr": 0.15, "side_yaw_thr": 0.12},
    )
    # 零指令禁滑：真机摩擦大，仿真也不能站着滑冰
    lin_vel_xy_l2_without_cmd = RewTerm(
        func=mdp.lin_vel_xy_l2_without_cmd,
        weight=0.0,
        params={"command_name": "base_velocity", "command_threshold": 0.12},
    )
    # HIM：转向不迈步/空抬过久 → 罚（堵住拖地蹭 yaw）
    dual_hold_air = RewTerm(
        func=mdp.dual_hold_air_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "xy_thr": 0.12,
            "yaw_thr": 0.12,
            "hold_s": 0.35,
            "yaw_only": True,
        },
    )
    # 对角摆腿等抬高：四条腿幅度差不多，禁止单腿独抬
    equal_swing_lift = RewTerm(
        func=mdp.equal_swing_lift_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "group_a_body_names": ["FL_W_L", "RR_W_L"],
            "group_b_body_names": ["FR_W_L", "RL_W_L"],
            "target_height": 0.04,
            "eq_sigma": 0.025,
            "height_sigma": 0.035,
            "period_s": 0.90,
            "yaw_only": False,
        },
    )


@configclass
class WheeldogRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """M20 rough recipe; only robot asset + joint/body names differ."""

    actions: WheeldogActionsCfg = WheeldogActionsCfg()
    rewards: WheeldogRewardsCfg = WheeldogRewardsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_W_L"

    # fmt: off
    leg_joint_names = [
        "FL_H1_L_j1", "FL_H2_L_j2", "FL_K_L_j3",
        "FR_H1_L_j1", "FR_H2_L_j2", "FR_K_L_j3",
        "RL_H1_L_j1", "RL_H2_L_j2", "RL_K_L_j3",
        "RR_H1_L_j1", "RR_H2_L_j2", "RR_K_L_j3",
    ]
    wheel_joint_names = [
        "FL_W_L_j4", "FR_W_L_j4", "RL_W_L_j4", "RR_W_L_j4",
    ]
    hipx_joint_names = ["FL_H1_L_j1", "FR_H1_L_j1", "RL_H1_L_j1", "RR_H1_L_j1"]
    hipy_joint_names = ["FL_H2_L_j2", "FR_H2_L_j2", "RL_H2_L_j2", "RR_H2_L_j2"]
    knee_joint_names = ["FL_K_L_j3", "FR_K_L_j3", "RL_K_L_j3", "RR_K_L_j3"]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene (URDF swap)------------------------------
        self.scene.robot = WHEELDOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations (M20)------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # HIM：开环对角/横移时钟进观测（57→65），策略才能按相位抬腿
        _gait_clock = {
            "period": 0.90,
            "command_name": "base_velocity",
            "stand_thr": 0.10,
            "side_period": 1.20,
            "side_thr": 0.10,
            "yaw_exclude": 0.08,
            "vx_exclude": 0.15,
        }
        self.observations.policy.gait_phase = ObsTerm(func=mdp.gait_phase_clock, params=dict(_gait_clock))
        self.observations.critic.gait_phase = ObsTerm(func=mdp.gait_phase_clock, params=dict(_gait_clock))

        # ------------------------------Actions (H1≡hipx; clip 对齐 runner clip_actions=2)------------------------------
        self.actions.joint_pos.scale = {".*_H1_L_j1": 0.125, "^(?!.*_H1_L_j1).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-2.0, 2.0)}
        self.actions.joint_vel.clip = {".*": (-2.0, 2.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events: PD 偏差优先（延迟在 DelayedPDActuator 0–4）------------------------------
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.7, 1.3)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.7, 1.3)

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            # 别在 reset 时甩出初速 → 高摩擦场景也不该一落地就滑
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 稳定优先：不加底座外力/力矩扰动
        self.events.randomize_apply_external_force_torque = None

        # 混合地形：楼梯为主；阶高放缓。init 用 1 避免全挤在 0 又被降级刷均值
        self.scene.terrain.max_init_terrain_level = 1
        gen = self.scene.terrain.terrain_generator
        gen.curriculum = True
        st = gen.sub_terrains
        st["pyramid_stairs"].proportion = 0.25
        st["pyramid_stairs"].step_height_range = (0.02, 0.14)
        st["pyramid_stairs"].step_width = 0.30
        st["pyramid_stairs_inv"].proportion = 0.45
        st["pyramid_stairs_inv"].step_height_range = (0.02, 0.14)
        st["pyramid_stairs_inv"].step_width = 0.30
        st["boxes"].proportion = 0.10
        st["boxes"].grid_height_range = (0.025, 0.16)
        st["random_rough"].proportion = 0.12
        st["random_rough"].noise_range = (0.01, 0.12)
        st["random_rough"].noise_step = 0.01
        st["hf_pyramid_slope"].proportion = 0.04
        st["hf_pyramid_slope_inv"].proportion = 0.04

        # 真机摩擦很大：仿真下限抬高，禁止 0.35 那种“冰面”域随机
        self.events.randomize_rigid_body_material.params["static_friction_range"] = [1.2, 2.5]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [1.0, 2.2]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.0, 0.05]
        self.scene.terrain.physics_material.static_friction = 1.5
        self.scene.terrain.physics_material.dynamic_friction = 1.5
        self.scene.terrain.physics_material.restitution = 0.0

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -60.0
        # 锁机身高度≈stand_q0；权重放轻，避免压死横移/转向抬腿
        self.rewards.base_height_l2.func = mdp.base_height_l2_ungated
        self.rewards.base_height_l2.weight = -4.0
        self.rewards.base_height_l2.params["target_height"] = 0.505
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -0.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -0.0
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        # 零指令：回 stand_q0 + 禁止站着滑
        self.rewards.stand_still.weight = -4.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.params["command_threshold"] = 0.12
        self.rewards.stand_still_without_cmd.weight = -3.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.params["command_threshold"] = 0.12
        self.rewards.lin_vel_xy_l2_without_cmd.weight = -3.0
        self.rewards.lin_vel_xy_l2_without_cmd.params["command_threshold"] = 0.12
        self.rewards.action_l2_without_cmd.weight = -0.25
        # hipx 横移必须外展；原先恒 -3 把横移锁死。side/yaw（含边走边转）放开
        self.rewards.hipx_joint_pos_penalty.weight = -2.0
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipx_joint_pos_penalty.params["stand_still_scale"] = 8.0
        self.rewards.hipx_joint_pos_penalty.func = mdp.joint_pos_penalty_except_turn_side_cmd
        self.rewards.hipx_joint_pos_penalty.params["ang_cmd_threshold"] = 0.1
        self.rewards.hipx_joint_pos_penalty.params["y_cmd_threshold"] = 0.08
        self.rewards.hipx_joint_pos_penalty.params["turn_side_penalty_scale"] = 0.1
        # H2/K：直行锁姿；横移/转向完全放开（否则幅度偏小、符号乱、真机拖地）
        self.rewards.hipy_joint_pos_penalty.weight = -0.8
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.hipy_joint_pos_penalty.func = mdp.joint_pos_penalty_except_turn_side_cmd
        self.rewards.hipy_joint_pos_penalty.params["ang_cmd_threshold"] = 0.1
        self.rewards.hipy_joint_pos_penalty.params["y_cmd_threshold"] = 0.08
        self.rewards.hipy_joint_pos_penalty.params["stand_still_scale"] = 8.0
        self.rewards.hipy_joint_pos_penalty.params["turn_side_penalty_scale"] = 0.0
        self.rewards.hipy_joint_pos_penalty.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.hipy_joint_pos_penalty.params["terrain_height_threshold"] = 0.06
        self.rewards.hipy_joint_pos_penalty.params["high_terrain_penalty_scale"] = 0.05
        self.rewards.knee_joint_pos_penalty.weight = -0.5
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.knee_joint_pos_penalty.func = mdp.joint_pos_penalty_except_turn_side_cmd
        self.rewards.knee_joint_pos_penalty.params["ang_cmd_threshold"] = 0.1
        self.rewards.knee_joint_pos_penalty.params["y_cmd_threshold"] = 0.08
        self.rewards.knee_joint_pos_penalty.params["stand_still_scale"] = 8.0
        self.rewards.knee_joint_pos_penalty.params["turn_side_penalty_scale"] = 0.0
        self.rewards.knee_joint_pos_penalty.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.knee_joint_pos_penalty.params["terrain_height_threshold"] = 0.06
        self.rewards.knee_joint_pos_penalty.params["high_terrain_penalty_scale"] = 0.05
        self.rewards.knee_extension_l2.weight = -0.6
        self.rewards.knee_extension_l2.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 四腿差不多：对角关节对称加严
        self.rewards.joint_mirror.weight = -0.12
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(H1|H2|K).*", "RR_(H1|H2|K).*"],
            ["FR_(H1|H2|K).*", "RL_(H1|H2|K).*"],
        ]
        # 直行 L/R 对称；横移/转向放开
        self.rewards.joint_lr_mirror.weight = -0.40
        self.rewards.joint_lr_mirror.params["straight_thr"] = 0.08
        self.rewards.joint_lr_mirror.params["yaw_thr"] = 0.08

        # 后期塌缩主因：站立抽搐惩罚堆叠爆炸 → 放轻，别让负项主导
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.action_smooth_l2.weight = -0.02
        self.rewards.action_rate_l2_wheeldog.weight = -0.03
        self.rewards.action_acc_l2_wheeldog.weight = -0.02
        self.rewards.action_l2_without_cmd.weight = -0.25

        # HIM 抬腿滤波：保留门控但抬高 floor，避免步态一偏就饿死正分
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_gait_gated_exp
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_W_L", "FR_W_L", "RL_W_L", "RR_W_L"], preserve_order=True),
            "xy_thr": 0.12,
            "yaw_thr": 0.12,
            "period_s": 0.90,
            "hybrid_floor": 0.55,
        }
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_gait_gated_exp
        self.rewards.track_ang_vel_z_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.params = {
            "command_name": "base_velocity",
            "std": 0.45,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_W_L", "FR_W_L", "RL_W_L", "RR_W_L"], preserve_order=True),
            "xy_thr": 0.12,
            "yaw_thr": 0.12,
            "period_s": 0.90,
            "hybrid_floor": 0.70,
        }
        self.rewards.track_lin_vel_x_exp.weight = 8.5
        self.rewards.track_lin_vel_x_exp.params["std"] = math.sqrt(0.18)
        self.rewards.wheel_action_track_lin_x.weight = 3.5
        self.rewards.wheel_action_lr_sync_straight.weight = -2.5
        self.rewards.track_lin_vel_y_exp.weight = 6.0
        self.rewards.track_lin_vel_y_exp.params["std"] = math.sqrt(0.08)
        self.rewards.track_lin_vel_y_exp.params["cmd_thr"] = 0.08
        self.rewards.track_lin_vel_y_exp.params["active_when_near_zero"] = True
        self.rewards.track_lin_vel_y_exp.params["require_vx_thr"] = 0.12
        self.rewards.lin_vel_y_l2_straight.weight = -4.5
        self.rewards.ang_vel_z_l2_straight.weight = -3.5
        self.rewards.track_lin_vel_y_strafe_exp.weight = 7.0
        self.rewards.track_lin_vel_y_strafe_exp.params["std"] = math.sqrt(0.15)
        self.rewards.track_lin_vel_y_strafe_exp.params["cmd_thr"] = 0.08
        self.rewards.track_lin_vel_y_strafe_exp.params["active_when_near_zero"] = False

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.4
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]

        # HIM 量级：空气/净空别刷分；对角时钟 + dual_hold 才是转向主线
        self.rewards.feet_air_time_ang_z_M20.weight = 6.0
        self.rewards.feet_air_time_ang_z_M20.params["threshold"] = 0.22
        self.rewards.feet_air_time_ang_z_M20.params["foot_height_threshold"] = 0.040
        self.rewards.feet_air_time_ang_z_M20.params["cmd_threshold"] = 0.08
        self.rewards.feet_air_time_ang_z_M20.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z_M20.params["asset_cfg"].body_names = [self.foot_link_name]

        self.rewards.wheel_vel_l2_side_yaw.weight = -0.08
        self.rewards.wheel_vel_l2_side_yaw.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_vel_l2_side_yaw.params["side_thr"] = 0.08
        self.rewards.wheel_vel_l2_side_yaw.params["yaw_thr"] = 0.08
        self.rewards.wheel_vel_l2_side_yaw.params["vx_exclude"] = 0.0
        self.rewards.feet_clearance_air_side_yaw.weight = 2.0
        self.rewards.feet_clearance_air_side_yaw.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_clearance_air_side_yaw.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_clearance_air_side_yaw.params["foot_clearance"] = 0.040
        self.rewards.feet_clearance_air_side_yaw.params["air_threshold"] = 0.12
        self.rewards.feet_clearance_air_forward.weight = 0.0

        self.rewards.feet_gait.func = mdp.GaitRewardSideYaw
        self.rewards.feet_gait.weight = 5.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_W_L", "RR_W_L"), ("FR_W_L", "RL_W_L"))
        self.rewards.feet_gait.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.params["velocity_threshold"] = 0.10
        self.rewards.feet_gait.params["command_threshold"] = 0.08
        self.rewards.feet_gait.params["side_thr"] = 0.08
        self.rewards.feet_gait.params["yaw_thr"] = 0.08
        self.rewards.feet_slide_side_yaw.weight = -1.6
        self.rewards.feet_slide_side_yaw.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_side_yaw.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.phase_foot_side_yaw.weight = 4.0
        self.rewards.phase_foot_side_yaw.params["asset_cfg"].body_names = [
            "FL_W_L",
            "FR_W_L",
            "RL_W_L",
            "RR_W_L",
        ]
        self.rewards.phase_foot_side_yaw.params["cycle_time"] = 0.90
        self.rewards.phase_foot_side_yaw.params["gait_psi"] = 0.040
        self.rewards.phase_foot_side_yaw.params["gait_delta"] = 0.025
        self.rewards.phase_foot_side_yaw.params["side_thr"] = 0.08
        self.rewards.phase_foot_side_yaw.params["yaw_thr"] = 0.08
        self.rewards.feet_slide_ang_z_cmd.weight = -2.0
        self.rewards.equal_swing_lift.weight = 5.0
        self.rewards.equal_swing_lift.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.equal_swing_lift.params["asset_cfg"].body_names = [
            "FL_W_L",
            "FR_W_L",
            "RL_W_L",
            "RR_W_L",
        ]
        self.rewards.equal_swing_lift.params["target_height"] = 0.04
        self.rewards.equal_swing_lift.params["period_s"] = 0.90
        self.rewards.equal_swing_lift.params["yaw_only"] = False

        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_ang_z_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_ang_z_cmd.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.0

        # HIM 对角交替主奖 + dual_hold 堵拖地
        self.rewards.rotation_gait_status.weight = 8.0
        self.rewards.rotation_gait_status.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.rotation_gait_status.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.rotation_gait_status.params["group_a_body_names"] = ["FL_W_L", "RR_W_L"]
        self.rewards.rotation_gait_status.params["group_b_body_names"] = ["FR_W_L", "RL_W_L"]
        self.rewards.rotation_gait_status.params["target_height"] = 0.040
        self.rewards.rotation_gait_status.params["lin_vel_threshold"] = 0.55
        self.rewards.rotation_gait_status.params["ang_vel_threshold"] = 0.08
        self.rewards.rotation_gait_status.params["scale_by_gait_level"] = False
        self.rewards.dual_hold_air.weight = -6.0
        self.rewards.dual_hold_air.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.dual_hold_air.params["hold_s"] = 0.40
        self.rewards.dual_hold_air.params["yaw_only"] = True
        self.rewards.rotation_gait_symmetry.weight = 6.5
        self.rewards.rotation_gait_symmetry.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.rotation_gait_symmetry.params["group_a_body_names"] = ["FL_W_L", "RR_W_L"]
        self.rewards.rotation_gait_symmetry.params["group_b_body_names"] = ["FR_W_L", "RL_W_L"]
        self.rewards.rotation_gait_symmetry.params["scale_by_gait_level"] = False
        self.rewards.rotation_gait_symmetry.params["side_thr"] = 0.10
        self.rewards.rotation_gait_symmetry.params["lin_vel_threshold"] = 0.55
        self.rewards.joint_mirror.params["scale_by_gait_level"] = False

        self.rewards.bad_orientation_penalty.weight = -1e3

        if self.__class__.__name__ == "WheeldogRoughEnvCfg":
            self.disable_zero_weight_rewards()

        self.terminations.illegal_contact = None
        self.curriculum.command_levels = None

        # ------------------------------Commands：直行 + 横移 + 转向都要练够------------------------------
        self.commands.base_velocity.knee_joint_names_expr = ".*_K_L_j3"
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.04
        self.commands.base_velocity.rel_zero_vel_envs = 0.06
        self.commands.base_velocity.rel_only_lin_y_envs = 0.22
        self.commands.base_velocity.rel_only_lin_x_envs = 0.18
        self.commands.base_velocity.rel_only_ang_z_envs = 0.20
        self.commands.base_velocity.rel_lin_x_ang_z_envs = 0.10
