import numpy as np
import torch
import torch.optim as optim
from custom_environment import CustomEnvironment
import matplotlib.pyplot as plt
from plotting_utils import calculate_rolling_outcomes, log_episode_data, plot_graphs
from models import Agent



def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x

def convert_to_tensor(x, device):
    return torch.tensor(x).to(device)

if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.0005
    ent_coef = 0.05
    vf_coef = 0.75
    clip_coef = 0.2
    gamma = 0.9
    batch_size = 300
    p1 = 0.5
    p2 = 0.95
    total_episodes = 0

    """ ENV SETUP """
    v_max = {"att1":1,"def1":0.5,"def2":0.5}   # Define the maximum velocity
    target_pos = np.array([0,0])  # Define the target position
    dt = 0.1  # Define the time step
    a_max = {"att1":1,"def1":2,"def2":2} 
    drone_rad = {"att1":0.5,"def1":0.25,"def2":0.25}  # Define the drone radius
    num_actions = 15
    episode_length = 10

    env = CustomEnvironment(a_max, v_max, target_pos, drone_rad, dt)

    max_cycles = 300
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape[0]

    """ LEARNER SETUP """
    agents = {
        drone: Agent(observation_size, num_actions).to(device)
        for drone in env.agents
    }
    
    optimizers = {
        agent_name: optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
        for agent_name, agent in agents.items()
    }

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_returns = { 
        drone: 0 
        for drone in env.agents
    }
    outcome_history = []
    
    total_loss = { 
        drone: 0 
        for drone in env.agents
    }


    rb_obs = torch.zeros((max_cycles, num_agents, observation_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    # Load previously saved models
    for agent_name, agent in agents.items():
        try:
            agent.load_state_dict(torch.load(f"{agent_name}_ppo_agent_latest.pth"))
            agent.to(device)
            print(f"Loaded saved model for {agent_name}")
        except FileNotFoundError:
            print(f"No saved model found for {agent_name}, starting from scratch")

    

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(0, total_episodes):
        with torch.no_grad():
            next_obs, info = env.reset(episode, total_episodes, p1, p2) # Reset l'environnement
            total_episodic_returns = { # Initialise à 0 le total des rewards de chaque agent sur un épisode
                drone: 0 
                for drone in env.agents
            }
            total_loss = { 
                drone: 0 
                for drone in env.agents
            }
            total_updates = 0

            # start of episode loop
            for step in range(1, max_cycles):
                actions = {}
                logprobs = {}
                values = {}

                for i, agent_name in enumerate(env.agents):
                    agent = agents[agent_name]
                    agent_obs = next_obs[agent_name]
                    obs = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0).to(device)
                    rb_obs[step,i,:] = obs
                    action, logprobs[agent_name], _, values[agent_name] = agent.get_action_and_value(obs)
                    actions[agent_name] = action.item()

                next_obs, rewards, terms, truncs, infos = env.step(actions)
                ending = env.ending

                # In training rendering
                # if episode % 100 == 0:
                #     env.render()

                # add to episode storage
                for i, agent_name in enumerate(env.agents):
                    rb_rewards[step, i] = convert_to_tensor(rewards[agent_name], device)
                    rb_terms[step,i] = convert_to_tensor(terms[agent_name],device)
                    rb_actions[step,i] = convert_to_tensor(actions[agent_name],device)
                    rb_logprobs[step,i] = logprobs[agent_name]
                    rb_values[step,i] = values[agent_name].flatten()

                    # compute episodic return
                    total_episodic_returns[agent_name] += rb_rewards[step, i].cpu().numpy()

                # Check for termination or truncation
                if any(terms.values()) or any(truncs.values()):
                    end_step = step
                    ending = env.ending
                    break


            if end_step == 0:
                end_step = max_cycles - 1

        
        # Bootstrap value if not done and prepare batches for each agent
        for i, agent_name in enumerate(env.agents):
            agent = agents[agent_name]
            optimizer = optimizers[agent_name]    

            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards[:,i]).to(device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t,i]
                        + gamma * rb_values[t + 1,i] * rb_terms[t + 1,i]
                        - rb_values[t,i]
                    )
                    rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
                rb_returns = rb_advantages + rb_values[:,i]

            # convert our episodes to batch of individual transitions
            b_obs = rb_obs[:end_step, i, :]
            b_logprobs = rb_logprobs[:end_step, i].flatten()
            b_actions = rb_actions[:end_step, i].flatten()
            b_values = rb_values[:end_step,i].flatten()
            b_returns = rb_returns[:end_step].flatten()
            b_advantages = rb_advantages[:end_step].flatten()


            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(3):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[batch_index]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[batch_index] * ratio
                    pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    total_loss[agent_name] += loss.item()
                    total_updates += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            print(f"Agent name : {agent_name} - Ending: {ending}")
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(total_episodic_returns[agent_name])}")
            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")



        for agent_name, agent in agents.items():
            torch.save(agent.state_dict(), f"{agent_name}_ppo_agent_latest.pth")

        if ending=="att - cible":
            outcome_history.append('attacker_win')
        elif ending=="def - att":
            outcome_history.append('defender_win')
        elif ending=="att wrong direction":
            outcome_history.append('wrong_direction_att')


        average_loss_per_episode = {
            agent: total_loss[agent] / total_updates
            for agent in env.agents
        }

        attacker_wins, defender_wins , wrong_direction_att= calculate_rolling_outcomes(outcome_history)

        log_episode_data("/home/ubuntu/Projects/PSC-MARL/logs/training_logs.json", episode, total_episodic_returns, end_step, outcome_history, average_loss_per_episode, 20)


    for agent_name in env.agents:
        agents[agent_name].eval()

    plot_graphs("/home/ubuntu/Projects/PSC-MARL/logs/training_logs.json", 100)

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(100):
            obs, infos = env.reset(episode, total_episodes, p1, p2)
            terms = [False] * num_agents
            truncs = [False] * num_agents
            total_episodic_returns = { 
                drone: 0 
                for drone in env.agents
            }

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

            for agent_name in env.agents:
                print(f"Agent name : {agent_name} - Ending: {env.ending}")
                print(f"Training episode {episode}")
                print(f"Episodic Return: {np.mean(total_episodic_returns[agent_name])}")
                print(f"Episode Length: {end_step}")
                print("")
                print("\n-------------------------------------------\n")




