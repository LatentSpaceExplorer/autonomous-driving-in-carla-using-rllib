import os
import argparse

import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.tune.tuner import Tuner
from ray.tune import Trainable

import algos.dqn
import algos.ppo

from envs.road_center_env import RoadCenterEnv


class RLlibTrainable(Trainable):

    def setup(self, config):
        self.config = config
        print(f'TRAINABLE CONFIG: {self.config}')

        self.algo = self.config["algorithm_config"].build()
        self.env = self.algo.workers.local_worker().env


    def step(self):
        result = self.algo.train()

        return {
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_len_mean": result["episode_len_mean"],
            "timesteps_total": result["timesteps_total"],
        }


def main(env="RoadCenterEnv", algo="DQN", _num_cpus=4, _num_gpus=0, debug=False, iterations=100):

    print(f"Running {env} with {algo} | cpus: {_num_cpus} | gpus: {_num_gpus} | debug: {debug} | iterations: {iterations}")

    # launch tensorboard
    import tensorboard
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, "--logdir", f"logs/{env}_{algo}", "--host", "localhost", "--port", "6006"])
    url = tb.launch()
    print(f"TensorBoard running at {url}")

    # init ray
    ray.init(num_cpus=_num_cpus, num_gpus=_num_gpus)

    config = get_algo_config(algo)
    config.environment(env)
    config.framework("torch")

    trainable = tune.with_resources(RLlibTrainable, {"gpu": _num_gpus, "cpu": _num_cpus})

    tuner = Tuner(
        trainable,
        run_config=air.RunConfig(
            name=f"{env}_{algo}_experiment",
            storage_path=os.path.abspath(f'logs/{env}_{algo}'),
            log_to_file=True,
            local_dir=os.path.abspath(f'logs/{env}_{algo}')
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
        ),
        param_space={
            "algorithm_config": config,
        },
    )

    results = tuner.fit()
    print("Best trial:")
    print(results.get_best_result(metric="episode_reward_mean", mode="max"))

    ray.shutdown()


def get_algo_config(algo: str = "DQN"):
    """Return the algorithm config object based on the algo name."""
    algo = algo.upper()
    if algo == "DQN":
        return algos.dqn.get_dqn_config()
    elif algo == "PPO":
        return algos.ppo.get_ppo_config()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def register_all_envs(debug=False, max_episode_steps=1000):
    """Register all custom environments with Ray."""
    from gymnasium.wrappers import TimeLimit
    register_env("RoadCenterEnv", lambda config: TimeLimit(RoadCenterEnv(config, debug), max_episode_steps))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Carla RL Trainer")
    parser.add_argument("-c", "--cpus", type=int, default=4, help="Number of CPUs (default: 4)")
    parser.add_argument("-g", "--gpus", type=int, default=0, help="Number of GPUs (default: 0)")
    parser.add_argument("-e", "--env", type=str, default="EnvX3", help="Environment (default: EnvX3)")
    parser.add_argument("-a", "--algo", type=str, default="DQN", help="Algorithm (default: DQN)")
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("-i", "--iterations", type=int, default=100, help="Number of iterations (default: 100)")
    parser.add_argument("-m", "--max_episode_steps", type=int, default=1000, help="Maximum number of steps per episode (default: 1000)")

    args = parser.parse_args()

    register_all_envs(args.debug, max_episode_steps=args.max_episode_steps)
    main(
        env=args.env,
        algo=args.algo,
        _num_cpus=args.cpus,
        _num_gpus=args.gpus,
        debug=args.debug,
        iterations=args.iterations,
    )

