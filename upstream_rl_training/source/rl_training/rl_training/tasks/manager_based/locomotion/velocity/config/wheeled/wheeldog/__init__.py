# Copyright (c) 2025

import gymnasium as gym

from . import agents

gym.register(
    id="Flat-Wheeldog-Lynx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheeldogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeldogFlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Wheeldog-Lynx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:WheeldogRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeldogRoughPPORunnerCfg",
    },
)
