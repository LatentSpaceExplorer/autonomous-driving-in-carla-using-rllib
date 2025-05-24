from ray.rllib.algorithms.dqn.dqn import DQNConfig

def get_dqn_config():
    config = DQNConfig()
    config.training(
        num_steps_sampled_before_learning_starts=4000,
        replay_buffer_config={"replay_buffer_size": 100_000},
        model = {
            "fcnet_hiddens": [128, 64, 32, 16]
        },
        train_batch_size=128,    
    )


    # Timesteps collected per rollout fragment
    config.rollouts(rollout_fragment_length=16)  


    config.exploration(
        exploration_config={
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 500_000,
        }
    )

    return config