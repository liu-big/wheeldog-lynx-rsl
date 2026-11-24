# Copyright (c) 2025
# Flat stage: M20 flat recipe + local procedural plane (avoid Nucleus USD).

from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import WheeldogRoughEnvCfg


@configclass
class WheeldogFlatEnvCfg(WheeldogRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # Local procedural flat — same as prior Wheeldog flat (Nucleus-safe)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0,
            num_rows=10,
            num_cols=20,
            curriculum=False,
            size=(8.0, 8.0),
            border_width=20.0,
            color_scheme="height",
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=1.0),
            },
        )
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.rewards.hipy_joint_pos_penalty.params["sensor_cfg"] = None
        self.rewards.knee_joint_pos_penalty.params["sensor_cfg"] = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "WheeldogFlatEnvCfg":
            self.disable_zero_weight_rewards()
