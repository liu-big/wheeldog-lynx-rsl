# Copyright (c) 2025
# Wheeldog articulation for Lynx-style RSL training (RS03 legs + RS00 wheels).

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# portable: <repo>/assets/wheeldog  (this file → …/rl_training/assets/wheeldog.py)
_ASSET_ROOT = Path(__file__).resolve().parents[5] / "assets" / "wheeldog"
_URDF = _ASSET_ROOT / "urdf" / "wheeldog.urdf"
_USD_DIR = _ASSET_ROOT / "usd"
_USD = _USD_DIR / "wheeldog.usd"

# Pinocchio-optimized dog-stance (wheeldog_ext/stand_q0_pinocchio.yaml)
_H1_OUT = 0.02
_WHEELDOG_STAND = {
    "FL_H1_L_j1": _H1_OUT,
    "FL_H2_L_j2": 0.721326,
    "FL_K_L_j3": 1.442678,
    "FR_H1_L_j1": -_H1_OUT,
    "FR_H2_L_j2": -0.721296,
    "FR_K_L_j3": -1.442566,
    "RL_H1_L_j1": -_H1_OUT,
    "RL_H2_L_j2": -0.721466,
    "RL_K_L_j3": -1.442939,
    "RR_H1_L_j1": _H1_OUT,
    "RR_H2_L_j2": 0.721459,
    "RR_K_L_j3": 1.442911,
    ".*_W_L_j4": 0.0,
}

_TARGET_HEIGHT = 0.505

_RIGID = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=0.4,
)
_ART = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
)

_SPAWN = (
    sim_utils.UsdFileCfg(
        usd_path=str(_USD),
        activate_contact_sensors=True,
        rigid_props=_RIGID,
        articulation_props=_ART,
    )
    if _USD.is_file()
    else sim_utils.UrdfFileCfg(
        asset_path=str(_URDF),
        usd_dir=str(_USD_DIR),
        usd_file_name="wheeldog.usd",
        force_usd_conversion=True,
        activate_contact_sensors=True,
        fix_base=False,
        merge_fixed_joints=False,
        collider_type="convex_hull",
        collision_from_visuals=False,
        rigid_props=_RIGID,
        articulation_props=_ART,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=125.0, damping=6.6),
        ),
    )
)

WHEELDOG_CFG = ArticulationCfg(
    spawn=_SPAWN,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, _TARGET_HEIGHT),
        joint_pos=_WHEELDOG_STAND,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # delay 0–4 控制步（~0–80 ms @50 Hz），逼近板端跟踪滞后，避免仿真瞬时跟上喂闭环抖
        "hips": DelayedPDActuatorCfg(
            joint_names_expr=[".*_H1_L_j1"],
            effort_limit=60.0,
            velocity_limit=20.0,
            stiffness=125.0,
            damping=6.6,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "thigh_knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_H2_L_j2", ".*_K_L_j3"],
            effort_limit=60.0,
            velocity_limit=20.0,
            stiffness=95.0,
            damping=6.6,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_W_L_j4"],
            effort_limit=14.0,
            velocity_limit=33.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)
