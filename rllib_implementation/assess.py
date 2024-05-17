import argparse
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
import custom_environment
import numpy as np
import os

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for training.")
parser.add_argument("--as-test", action="store_true", help="Whether this script should be run as a test.")
parser.add_argument("--checkpoint-path", type=str, default="", help="Path to the checkpoint to restore from.")
args = parser.parse_args()

# Environment creator function
def env_creator(env_config):
    v_max = {"att1":1.5, "def1":0.75, "def2":0.75}
    target_pos = np.array([0,0])
    dt = 0.1
    a_max = {"att1":4, "def1":1.5, "def2":1.5}
    drone_rad = {"att1":0.5, "def1":0.25, "def2":0.25}
    return PettingZooEnv(custom_environment.env(a_max, v_max, target_pos, drone_rad, dt, render_mode="human"))

# Register the custom environment
env = env_creator({})
register_env("my_custom_env", env_creator)

# Initialize the algorithm with the appropriate config
trainer = PPO(config={
    "env": "my_custom_env",
    # Add other configuration parameters here
    "framework": "torch",
})

# Correct path to the checkpoint file
checkpoint_path = "/home/ubuntu/ray_results/PPO_2024-05-15_18-12-30/PPO_drones_51c12_00000_0_2024-05-15_18-12-33/checkpoint_000033"

# Load the checkpoint
trainer.restore(checkpoint_path)

# Now you can continue training or evaluate the model
results = trainer.train()