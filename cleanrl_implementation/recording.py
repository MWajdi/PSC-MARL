import torch
from custom_environment import CustomEnvironment
import argparse
import numpy as np
from models import Agent
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('v_max_att1',type=float)
    parser.add_argument('v_max_def1',type=float)
    parser.add_argument('v_max_def2',type=float)
    parser.add_argument('a_max_att1',type=float)
    parser.add_argument('a_max_def1',type=float)
    parser.add_argument('a_max_def2',type=float)
    parser.add_argument('drone_rad_att1',type=float)
    parser.add_argument('drone_rad_def1',type=float)
    parser.add_argument('drone_rad_def2',type=float)
    parser.add_argument('episode_length', type=int)
    parser.add_argument('num_episodes', type=int)
    parser.add_argument('-o', '--dt', type=float, default=0.1)

    args = parser.parse_args()

    v_max = {"att1":args.v_max_att1,"def1":args.v_max_def1,"def2":args.v_max_def2}   # Define the maximum velocity
    target_pos = np.array([0,0])  # Define the target position
    dt = args.dt  # Define the time step
    a_max = {"att1":args.a_max_att1,"def1":args.a_max_def1,"def2":args.a_max_def2}
    drone_rad = {"att1":args.drone_rad_att1,"def1":args.drone_rad_def1,"def2":args.drone_rad_def2}  # Define the drone radius
    episode_length = args.episode_length
    num_episodes = args.num_episodes


    num_actions = 15
    observation_size = 12
    
    
    env = CustomEnvironment(a_max, v_max, target_pos, drone_rad, dt)

    total_episodes = 1
    p1,p2 = 1, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading agents
    agents = {
        drone: Agent(observation_size, num_actions).to(device)
        for drone in env.agents
    }
    for agent_name, agent in agents.items():
        try:
            agent.load_state_dict(torch.load(f"{agent_name}_ppo_agent_latest.pth"))
            agent.to(device)
            print(f"Loaded saved model for {agent_name}")
        except FileNotFoundError:
            print(f"No saved model found for {agent_name}, starting from scratch")

    for agent_name in env.agents:
        agents[agent_name].eval()

    with torch.no_grad():
        env.start_recording()

        for episode in range(num_episodes):
            obs, infos = env.reset(episode, total_episodes, p1, p2)
            terms = [False] * 3
            truncs = [False] * 3
            total_episodic_returns = { 
                drone: 0 
                for drone in env.agents
            }


            end_step = 0
            while not any(terms) and not any(truncs):
                actions = {}    
                logprobs = {}
                values = {}

                for i, agent_name in enumerate(env.agents):
                    agent = agents[agent_name]
                    agent_obs = torch.tensor(obs[agent_name], dtype=torch.float32).unsqueeze(0).to(device)
                    action, logprob, _, value = agent.get_action_and_value(agent_obs)
                    values[agent_name] = value
                    logprobs[agent_name] = logprob
                    actions[agent_name] = action.item()

                obs, rewards, terms, truncs, infos = env.step(actions)
                
                for i, agent_name in enumerate(env.agents):
                    total_episodic_returns[agent_name] += rewards[agent_name]

                env.render()

                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
                end_step += 1

            for agent_name in env.agents:
                print(f"Agent name : {agent_name} - Ending: {env.ending}")
                print(f"Training episode {episode}")
                print(f"Episodic Return: {np.mean(total_episodic_returns[agent_name])}")
                print(f"Episode Length: {end_step}")
                print("")
                print("\n-------------------------------------------\n")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f'episode_{episode}_{timestamp}.mp4'
        env.stop_recording(video_filename)