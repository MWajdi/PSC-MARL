# Multi-Agent Reinforcement Learning for Drone Defense

## Project Overview

This project demonstrates the application of multi-agent reinforcement learning (MARL) to a drone defense scenario. The environment consists of one target, two defending drones, and one attacking drone. The objective of the attacking drone is to reach the target, while the defending drones aim to intercept it. We used the PettingZoo library to model our custom environment and trained agents using two distinct implementations: one using CleanRL, and the other using RLlib.

## Team Members

- Maatouk Wajdi
- Eilles Chan-Way Stéphane
- Delaunay Paul-Adrien
- Chapellier Antonin
- Boulan Baptiste

## Requirements

The following libraries are required to run this project:

ray[rllib]
numpy
torch
matplotlib
json
gymnasium
pettingzoo
moviepy.editor
pygame


## Project Structure

### CleanRL Implementation
- `cleanrl_implementation/custom_environment.py`: Contains the definition of the custom environment modeled using PettingZoo.
- `cleanrl_implementation/models.py`: Contains the definition of the agent models used in the training.
- `cleanrl_implementation/plotting_utils.py`: Contains utility functions for plotting training results and logging episode data.
- `cleanrl_implementation/ppo_implement.py`: Script for training agents using the CleanRL implementation.
- `cleanrl_implementation/recording.py`: Script for recording the agent interactions.

### RLlib Implementation
- `rllib_implementation/assess.py`: Script for visualizing models trained using RLlib.
- `rllib_implementation/custom_environment.py`: Contains the definition of the custom environment modeled using PettingZoo.
- `rllib_implementation/training.py`: Script for training agents using the RLlib implementation.

## Usage

### Training

You can train the agents using either CleanRL or RLlib.

#### CleanRL

To train using CleanRL, run:

`python cleanrl_implementation/ppo_implement.py`


#### RLlib

To train using RLlib, run:

`python rllib_implementation/training.py`


### Visualization

To visualize the models trained using RLlib, run:

`python rllib_implementation/assess.py`


### Configuration

The training and environment parameters can be configured within the respective training scripts (`ppo_implement.py` for CleanRL and `training.py` for RLlib). Adjust the parameters as needed to experiment with different settings.

### Custom Environment

The custom environment is defined in `custom_environment.py` files within both the CleanRL and RLlib implementation directories. It includes the logic for the attacking and defending drones, their maximum acceleration, velocity, and attack range, as well the reward functions.

## Results

The results of the training, including the performance of the agents and their interactions within the environment, can be visualized using the plotting utilities provided in the `plotting_utils.py` script. This script provides functions to calculate rolling outcomes, log episode data, and plot graphs.

