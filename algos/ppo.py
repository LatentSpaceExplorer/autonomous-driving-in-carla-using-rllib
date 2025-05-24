from ray.rllib.algorithms.ppo import PPOConfig


def get_ppo_config():
    config = PPOConfig()
    config.training(
        train_batch_size=5000,
        model = {"fcnet_hiddens": [64, 128, 64]},
    )
    return config
