import argparse

from ray import air, tune, train
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import custom_environment
import numpy as np
import os

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    def env_creator(args):
        v_max = {"att1":1.5, "def1":0.75, "def2":0.75}
        target_pos = np.array([0,0])
        dt = 0.1
        a_max = {"att1":4, "def1":1.5, "def2":1.5}
        drone_rad = {"att1":0.5, "def1":0.25, "def2":0.25}
        return PettingZooEnv(custom_environment.env(a_max, v_max, target_pos, drone_rad, dt))

    env = env_creator({})
    register_env("drones", env_creator)
    config = (
        PPOConfig()
        .environment("drones")
        .framework("torch")
        .training(num_sgd_iter=10)
        .multi_agent(
            policies={agent_id: (None, env.observation_space, env.action_space, {}) for agent_id in env.get_agent_ids()},
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: agent_id),
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .resources(num_gpus=1)
    )


    if args.as_test:
        # Only a compilation test of running the environment with DQN.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 20000}



    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),),
    )
    # tuner.restore("/home/ubuntu/ray_results/PPO_2024-04-26_19-21-47/PPO_drones_da6c1_00000_0_2024-04-26_19-21-51/", trainable=True)
    results = tuner.fit()

