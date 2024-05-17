
import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from moviepy.editor import ImageSequenceClip
import pygame


def norm(x):
    return np.linalg.norm(x)

def normalize_to(v,v_max):
    return (v_max/np.linalg.norm(v))*v

def env(a_max, v_max,target_pos,drone_rad,dt,render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(a_max, v_max,target_pos,drone_rad,dt,render_mode=internal_render_mode)
    env = wrappers.OrderEnforcingWrapper(env)
    return env



class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "mixedenv",
        "is_parallelizable": False,
        "render_fps": 30,
    }

    def __init__(self, a_max, v_max,target_pos,drone_rad,dt,render_mode=None):
        super().__init__()
        self.agents = ["att1", "def1", "def2"]
        self.possible_agents = ["att1", "def1", "def2"]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        

        self.target_pos = target_pos #position de la cible
        self.v_max = v_max #vitesses max (frottements de l'air tout ça tout ça)
        self.drone_rad=drone_rad #rayon des drones
        self.dt=dt #dt
        self.a_max = a_max
        self.trajectories = {drone:[] for drone in self.agents} #ADDTHIS
        self.last_acc = {drone:np.array([0,0]) for drone in self.possible_agents}

        self.action_spaces = {i: spaces.Box(low=0, high=1, shape=(1,)) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
            for i in self.agents
        }
        self.has_reset = False
        self.render_mode = render_mode

    def start_recording(self):
        self.frames = []

    def stop_recording(self, filename='replay.mp4', fps=30):
        clip = ImageSequenceClip(self.frames, fps=fps)
        print(f"FPS before video creation: {fps}")
        clip.write_videofile(filename, codec='libx264', fps=fps)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reward_attacker(self):
        alpha = 1  # Tunable parameter
        beta = 100   # Large reward for successful attack
        gamma = 100 # Large penalty for being caught

        current_distance = norm(self.drones_pos["att1"])
        attack_success = int(current_distance <= self.drone_rad["att1"])
        caught = int( (norm(self.drones_pos["att1"]-self.drones_pos["def1"])<self.drone_rad["def1"]) or (norm(self.drones_pos["att1"]-self.drones_pos["def2"])<self.drone_rad["def2"]) )

        reward = - gamma * caught + beta * attack_success - alpha * (1-caught) * (1-attack_success) * current_distance
        return reward
    
    def reward_defender(self, agent):
        alpha = 1  # Tunable parameter
        beta = 100  # Large reward for interception
        gamma = 100 # Large pernalty for succesful attack

        current_distance_to_attacker = norm(self.drones_pos[agent]-self.drones_pos["att1"])
        interception = int(current_distance_to_attacker <= self.drone_rad[agent])
        attacker_distance = norm(self.drones_pos["att1"])
        attack_success = int(attacker_distance <= self.drone_rad["att1"])

        reward = beta * interception - gamma * attack_success - alpha * (1-interception) * (1-attack_success) * current_distance_to_attacker
        return reward


    def observe(self, drone):
        # Calculate the angle for rotation based on the drone's position
        teta = np.arctan2(self.drones_pos[drone][1], self.drones_pos[drone][0])
        R = np.array([[np.cos(teta), -np.sin(teta)], [np.sin(teta), np.cos(teta)]])
        
        # Rotate positions and velocities for relative positioning and movement
        drones_pos_rotated = {k: v @ R for k, v in self.drones_pos.items()}
        positions_flat = np.concatenate(list(drones_pos_rotated.values()))
        
        drones_v_rotated = {k: v @ R for k, v in self.drones_v.items()}
        velocities_flat = np.concatenate(list(drones_v_rotated.values()))
        
        # Combine positions and velocities into a single observation vector
        observation = np.concatenate((positions_flat, velocities_flat), dtype=np.float64)
        return observation

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))



    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        drone = self.agent_selection
        is_last = self._agent_selector.is_last()
        action = action[0]

        acc = np.array([np.cos(2*np.pi*action), np.sin(2*np.pi*action)])
        self.last_acc[drone] = acc

        self.drones_v[drone]=self.drones_v[drone]+self.dt*self.a_max[drone]*acc
        self.drones_v[drone]=normalize_to(self.drones_v[drone],self.v_max[drone])
        self.drones_pos[drone] = self.drones_pos[drone] + self.dt * self.drones_v[drone]         

        for agent in self.agents:
            if agent!=drone:
                self.drones_v[agent]=self.drones_v[agent]+self.dt*self.a_max[agent]*self.last_acc[agent]
                self.drones_v[agent]=normalize_to(self.drones_v[agent],self.v_max[agent])
                self.drones_pos[agent] = self.drones_pos[agent] + self.dt * self.drones_v[agent]  #+ 0.5 * (self.dt)**2 * actions[drone]

        self.terminations = {a: False for a in self.agents} #prend True à une valeur si la simulation se termine

        self.rewards = {
            "att1": self.reward_attacker(),
            "def1": self.reward_defender("def1"),
            "def2": self.reward_defender("def2")
        }

        
        if norm(self.drones_pos["att1"] - self.target_pos) < self.drone_rad["att1"]:  # Attacker reaches target
            self.ending = "att - cible"
            self.terminations = {a: True for a in self.agents}

        elif any([norm(self.drones_pos[drone]-self.drones_pos["att1"]) < self.drone_rad[drone] for drone in self.agents if drone != "att1"]):  # Defender touches attacker
            self.ending = "def - att"
            self.terminations = {a: True for a in self.agents}

        elif self.timestep > 10:
            self.terminations = {a: True for a in self.agents}


        # elif np.dot(self.drones_v["att1"], -self.drones_pos["att1"])<0 :
        #     self.ending = "att wrong direction"
        #     self.terminations = {a: True for a in self.agents}

        self.truncations = {a: False for a in self.agents}    
            
        self.timestep += self.dt
        for agent, pos in self.drones_pos.items():
            self.trajectories[agent].append(pos.copy())

        next_agent = self._agent_selector.next()
        self._cumulative_rewards[drone] = 0
        self.agent_selection = next_agent
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        obs = {agent: self.observe(agent) for agent in self.agents}
        
        return obs, self.rewards, self.terminations, self.truncations, {}

        


    def reset(self,seed=None, options=None):
        # reset environment

        self.has_reset = True

        self.timestep = 0   

        self.last_acc = {drone:np.array([0,0]) for drone in self.possible_agents}

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

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

        obs = {agent: self.observe(agent) for agent in self.agents}

        return obs, {}

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
            pos = np.array(pos)  # Ensure pos is an array for uniform processing
            try:
                screen_x = int((pos[0] + 10) * self.screen.get_width() / 20)
            except:
                raise ValueError("wrong format, pos = "+str(pos))
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

    def close(self):
        pass
