import gymnasium as gym
from pettingzoo import ParallelEnv
import functools
import random
from copy import copy
import numpy as np
import pygame
from numpy import sqrt
from numpy import exp
from moviepy.editor import ImageSequenceClip



def relative_position(pos1, pos2):
    return [pos2[0] - pos1[0], pos2[1] - pos1[1]]

def normalize_to(v,v_max):
    return (v_max/np.linalg.norm(v))*v

def norm(x):
    return np.linalg.norm(x)
        #coucou les gars

class CustomEnvironment(ParallelEnv):

    def start_recording(self):
        self.frames = []

    def stop_recording(self, filename='replay.mp4', fps=30):
        clip = ImageSequenceClip(self.frames, fps=fps)
        print(f"FPS before video creation: {fps}")
        clip.write_videofile(filename, codec='libx264', fps=fps)






    def _create_action_mapping(self,agent):
        norme = np.linspace(-self.a_max[agent], self.a_max[agent], 15, endpoint=False)
        vx,vy=self.drones_v[agent][0],self.drones_v[agent][1]
        direction=normalize_to(np.array([-vy,vx]),1)
        return {i: direction * norme[i] for i in range(len(norme))}


    def __init__(self, a_max, v_max,target_pos,drone_rad,dt,recording=False):
        self.possible_agents = ["def1", "def2", "att1"] #liste des agents
        self.agents = ["def1", "def2", "att1"] # au cas où
        self.target_pos = target_pos #position de la cible
        self.v_max = v_max #vitesses max (frottements de l'air tout ça tout ça)
        self.drone_rad=drone_rad #rayon des drones
        self.dt=dt #dt
        self.a_max = a_max
        self.trajectories = {drone:[] for drone in self.agents} #ADDTHIS
        


    def reward_attacker(self):
        param = max(min(1, (-self.p1 * self.total_episodes + self.p2 * self.total_episodes) / ((self.total_episodes+1) * (self.p2 - self.p1))),0)
        alpha = 1  # Tunable parameter
        beta = 10   # Large reward for successful attack
        gamma = 10 # Large penalty for being caught
        zeta = 10 # Penalty for wrong direction 

        current_distance = norm(self.drones_pos["att1"])
        attack_success = int(current_distance <= self.drone_rad["att1"])
        wrong_direction = int(np.dot(self.drones_v["att1"], -self.drones_pos["att1"])<0 )
        caught = int( (norm(self.drones_pos["att1"]-self.drones_pos["def1"])<self.drone_rad["def1"]) or (norm(self.drones_pos["att1"]-self.drones_pos["def2"])<self.drone_rad["def2"]) )

        reward = beta * attack_success + param * alpha * np.dot(self.drones_v["att1"], -self.drones_pos["att1"]) * self.dt - zeta * wrong_direction - gamma * caught
        return reward
    
    def reward_defender(self, agent):
        param = max(min(1, (-self.p1 * self.total_episodes + self.p2 * self.total_episodes) / ((self.total_episodes+1) * (self.p2 - self.p1))),0)
        alpha = 1  # Tunable parameter
        beta = 10  # Large reward for interception
        gamma = 10 # Large pernalty for succesful attack
        zeta = 4 # Penalty for wrong direction 

        current_distance_to_attacker = norm(self.drones_pos[agent]-self.drones_pos["att1"])
        interception = int(current_distance_to_attacker <= self.drone_rad[agent])
        # wrong_direction = int(np.dot(self.drones_v[agent], self.drones_pos["att1"]-self.drones_pos[agent])<0 )
        attacker_distance = norm(self.drones_pos["att1"])
        attack_success = int(attacker_distance <= self.drone_rad["att1"])

        reward = param * alpha * np.dot(self.drones_v[agent], self.drones_pos["att1"]-self.drones_pos[agent]) * self.dt + beta * interception- gamma * attack_success
        return reward

    def reset(self, episode, total_episodes, p1, p2):
        self.ending = "Episode ongoing"
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.rewards = {agent: 0 for agent in self.agents}
        self.episode = episode
        self.total_episodes = total_episodes  
        self.p1 = p1
        self.p2 = p2
        
        self.drones_pos = { 
            "def1": np.array([0,0]),
            "def2": np.array([0,0]),
            "att1": np.array([10,0])
        }

        # Clear trajectories for each drone ADDTHIS
        for drone in self.agents:
            self.trajectories[drone].clear()

        self.drones_v = { #vecteurs vitesses des drones
            "def1": normalize_to(np.array([0.9, 0.1]),self.v_max["def1"]),
            "def2": normalize_to(np.array([0.9, -0.1]),self.v_max["def2"]),
            "att1": np.array([-self.v_max["att1"], 0])
        }

        observations = {} #infos auquels les drones ont accès, j'ai choisi les positions relatives et les vitesses de tous les drones

        for drone, pos in self.drones_pos.items():
            teta = np.arctan2(self.drones_pos[drone][1], self.drones_pos[drone][0])
            R = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])
            drones_pos_rotated = {drone: self.drones_pos[drone] @ R for drone in self.drones_pos}
            positions_flat = np.concatenate(list(drones_pos_rotated.values())) 
            drones_v_rotated = {drone: self.drones_v[drone] @ R for drone in self.drones_v}
            velocities_flat = np.concatenate(list(drones_v_rotated.values())) 
            observations[drone]=tuple(np.concatenate((positions_flat, velocities_flat)))


        #     # le code en dessus concatène les vecteurs positions relatives d'un drone ac les autres drones et la cible et les vecteurs vitesses de tous les drones

        infos = {a: {} for a in self.agents}

        self.last_observation = {}
        self.last_reward = {}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {}

        return observations, infos
    


    def step(self, actions_param):
        self.action_to_vector = {
            agent: self._create_action_mapping(agent)
            for agent in self.agents
        }
        actions = {
            drone: self.action_to_vector[drone][actions_param[drone]]
            for drone in self.agents
        }

        
        for drone in self.agents: #update les positions des drones
            self.drones_v[drone]=self.drones_v[drone]+self.dt*actions[drone]
            self.drones_v[drone]=normalize_to(self.drones_v[drone],self.v_max[drone])
            self.drones_pos[drone] = self.drones_pos[drone] + self.dt * self.drones_v[drone]  #+ 0.5 * (self.dt)**2 * actions[drone]

        terminations = {a: False for a in self.agents} #prend True à une valeur si la simulation se termine

        rewards = {
            "att1": self.reward_attacker(),
            "def1": self.reward_defender("def1"),
            "def2": self.reward_defender("def2")
        }

        
        if norm(self.drones_pos["att1"] - self.target_pos) < self.drone_rad["att1"]:  # Attacker reaches target
            self.ending = "att - cible"
            terminations = {a: True for a in self.agents}

        elif any([norm(self.drones_pos[drone]-self.drones_pos["att1"]) < self.drone_rad[drone] for drone in self.agents if drone != "att1"]):  # Defender touches attacker
            self.ending = "def - att"
            terminations = {a: True for a in self.agents}

        elif np.dot(self.drones_v["att1"], -self.drones_pos["att1"])<0 :
            self.ending = "att wrong direction"
            terminations = {a: True for a in self.agents}

        truncations = {a: False for a in self.agents}    
            
        self.timestep += self.dt

        observations = {} #recalc les observations

        for drone, pos in self.drones_pos.items():
            teta = np.arctan2(self.drones_pos[drone][1], self.drones_pos[drone][0])
            R = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])
            drones_pos_rotated = {drone: self.drones_pos[drone] @ R for drone in self.drones_pos}
            positions_flat = np.concatenate(list(drones_pos_rotated.values())) 
            drones_v_rotated = {drone: self.drones_v[drone] @ R for drone in self.drones_v}
            velocities_flat = np.concatenate(list(drones_v_rotated.values())) 
            observations[drone]=tuple(np.concatenate((positions_flat, velocities_flat)))

        infos = {a: {} for a in self.agents}

        self.rewards = rewards

        for agent in self.agents:
            self.last_reward[agent] = rewards[agent]
            self.last_observation[agent] = observations[agent]
            self.terminations[agent]=terminations[agent]
            self.truncations[agent]=truncations

        # Update trajectories ADDTHIS
        for drone, pos in self.drones_pos.items():
            self.trajectories[drone].append(pos.copy())

        return observations, rewards, terminations, truncations, infos


    def render(self, mode='human'):
        # Initialize Pygame screen if not already initialized
        if not hasattr(self, 'screen'):
            pygame.init()
            screen_size = 800
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.font.init()
            # Initialize Pygame and its font module


        font = pygame.font.Font(None, 24) 
        # Define colors
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        TRANSLUCENT_BLUE = (0, 0, 255, 128)  # Translucent blue
        TRANSLUCENT_RED = (255, 0, 0, 128)   # Translucent red

        # Convert environment coordinates to screen coordinates
        def convert_coords(pos):
            # return int((pos[0] + 10) * self.screen.get_width() / 20), int((pos[1] + 10) * self.screen.get_height() / 20)
             # Convert environment coordinates to screen coordinates
            screen_x = int((pos[0] + 10) * self.screen.get_width() / 20)
            screen_y = int((pos[1] + 10) * self.screen.get_height() / 20)

            # Clamp the coordinates to the screen size
            screen_x = max(0, min(screen_x, self.screen.get_width()))
            screen_y = max(0, min(screen_y, self.screen.get_height()))

            return screen_x, screen_y

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((255, 255, 255))  # Fill screen with white

        # Draw trajectories ADDTHIS
        for drone, trajectory in self.trajectories.items():
            if len(trajectory) > 1:  # Ensure there are at least two points to connect
                points = [convert_coords(pos) for pos in trajectory]
                pygame.draw.lines(self.screen, RED if 'att' in drone else BLUE, False, points, 2)

        # Draw drones and their attack range
        for drone, pos in self.drones_pos.items():
            drone_color = BLUE if 'def' in drone else RED
            attack_range_color = TRANSLUCENT_BLUE if 'def' in drone else TRANSLUCENT_RED
            attack_range = int(self.drone_rad[drone] * self.screen.get_width() / 20)

            # Draw attack range (translucent circle)
            surface = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
            pygame.draw.circle(surface, attack_range_color, convert_coords(pos), attack_range)
            self.screen.blit(surface, (0, 0))

            # Draw drone (solid color dot)
            pygame.draw.circle(self.screen, drone_color, convert_coords(pos), 5)  # 5 is the drone dot size

        # Draw target
        pygame.draw.circle(self.screen, GREEN, convert_coords(self.target_pos), 5)  # 5 is the target dot size

        for drone, reward in self.rewards.items():
            reward_text = f"{drone}: {reward:.2f}"  # Format the reward to 2 decimal places
            text_surface = font.render(reward_text, True, (0, 0, 0))  # Black color for text
            self.screen.blit(text_surface, convert_coords(self.drones_pos[drone]) + np.array([10, -10]))
            # Adjust the position offset ([10, -10]) as needed


        pygame.display.flip()  # Update the screen with what we've drawn

        # Save the current frame if recording
        if hasattr(self, 'frames'):
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))  # Transpose to correct shape (height, width, channels)
            self.frames.append(frame)


        pygame.time.delay(10)


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,))

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(15)
    
