# Copyright (c) 2025
# Align with DeepRobotics M20: MLP + log-std exploration (not GRU/scalar lock).
# 去抖：clip_actions=2 + 低 init_noise（Δa≈1.5–1.9 → Δq≈0.4rad 的旧坑）

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WheeldogRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 历史峰值 ~3k；跑满 20k 后期塌。留 model_best.pt，默认训到 8k
    max_iterations = 8000
    save_interval = 50
    experiment_name = "wheeldog_lynx_rough"
    empirical_normalization = False
    # 与 env action clip±2 对齐；±100 会让探索期学猛抽
    clip_actions = 2.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.40,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheeldogFlatPPORunnerCfg(WheeldogRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "wheeldog_lynx_flat"
        self.save_interval = 50
