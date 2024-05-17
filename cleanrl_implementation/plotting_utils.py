import matplotlib.pyplot as plt
import json

def calculate_rolling_outcomes(outcome_history, rolling_window=100):
    recent_history = outcome_history[-rolling_window:]
    
    attacker_wins = recent_history.count('attacker_win')
    defender_wins = recent_history.count('defender_win')
    wrong_direction_att = recent_history.count('wrong_direction_att')
    

    return attacker_wins, defender_wins, wrong_direction_att

def moving_average(data, window_size):
    return [sum(data[i:i+window_size]) / window_size for i in range(len(data) - window_size + 1)]


def plot_graphs(log_file_path, window_size=10):  # window_size can be adjusted
    episodes, rewards, lengths, outcomes, losses = [], {'att1': [], 'def1': [], 'def2': []}, [], [], {'att1': [], 'def1': [], 'def2': []}

    with open(log_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            episodes.append(data['episode'])
            for agent in ['att1', 'def1', 'def2']:
                rewards[agent].append(data['episode_rewards'].get(agent, 0))
                losses[agent].append(data['losses'].get(agent, 0))
            lengths.append(data['episode_length'])
            outcomes.append(data['recent_outcomes'])

    # Calculate moving averages
    avg_lengths = moving_average(lengths, window_size)
    avg_rewards = {agent: moving_average(rewards[agent], window_size) for agent in rewards}
    avg_losses = {agent: moving_average(losses[agent], window_size) for agent in losses}

    # Process outcomes for plotting
    attacker_wins, defender_wins, wrong_direction_att = [], [], []
    for outcome_list in outcomes:
        attacker_wins.append(outcome_list.count('attacker_win'))
        defender_wins.append(outcome_list.count('defender_win'))
        wrong_direction_att.append(outcome_list.count('wrong_direction_att'))

    plt.figure(figsize=(15, 12))

    # Plotting average rewards for each agent
    plt.subplot(2, 2, 1)
    for agent in ['att1', 'def1', 'def2']:
        plt.plot(episodes[window_size-1:], avg_rewards[agent], label=f'{agent} Avg Rewards')
    plt.title('Average Episode Rewards')
    plt.legend()

    # Plotting average episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(episodes[window_size-1:], avg_lengths)
    plt.title('Average Episode Lengths')

    # Plotting outcomes
    plt.subplot(2, 2, 3)
    plt.plot(episodes, attacker_wins, label='Attacker Wins')
    plt.plot(episodes, defender_wins, label='Defender Wins')
    plt.plot(episodes, wrong_direction_att, label='Wrong Direction Attackers')
    plt.title('Outcomes over Episodes')
    plt.legend()

    # Plotting average losses for each agent
    plt.subplot(2, 2, 4)
    for agent in ['att1', 'def1', 'def2']:
        plt.plot(episodes[window_size-1:], avg_losses[agent], label=f'{agent} Avg Loss')
    plt.title('Average Loss per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()






def log_episode_data(file_path, episode, episode_rewards, episode_length, outcome_history, losses, rolling_window=100):
    # Ensure we don't exceed the number of episodes when slicing the history
    recent_outcomes = outcome_history[-rolling_window:]

    data = {
        'episode': episode,
        'episode_rewards': episode_rewards,  # Dictionary of rewards for each agent
        'episode_length': episode_length,
        'recent_outcomes': recent_outcomes,  # List of recent outcomes
        'losses': losses  # Dictionary of losses for each agent
    }
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')