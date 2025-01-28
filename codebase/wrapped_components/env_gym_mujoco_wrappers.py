from es_utilities.wrappers import EsEnvironmentWrapper
from novelty_utilities.ns_wrappers import NsEnvironmentWrapper
from novelty_utilities import behavior

import numpy as np


class GymMujocoWrapper(EsEnvironmentWrapper):
    def __init__(self, env, seed, reward_scale):
        super().__init__(env, seed)
        self.reward_scale = reward_scale
        
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return next_state, reward / self.reward_scale, terminated, truncated
    
    def set_seed(self, seed):
        self.env.reset(seed=seed)
    
    @property
    def timestep_limit(self):
        return 1000


class GymMujocoWrapperNsEndXY(GymMujocoWrapper, NsEnvironmentWrapper):
    def __init__(self, env, seed, reward_scale):
        super().__init__(env, seed, reward_scale)
        
    def get_behavior_characteristic(self):
        # Access the underlying MuJoCo environment object and retrieve the x, y coordinates.
        last_position = self.env.unwrapped.data.qpos
        x, y = last_position[0], last_position[1]
        
        return behavior.EndXYPositionCharacteristic(x, y)


class GymMujocoWrapperNsEndXYWithNondirectionalDistanceReward(GymMujocoWrapperNsEndXY):
    def __init__(self, env, seed):
        super().__init__(env, seed, None)
        self.starting_position = None
        
    def reset(self):
        state = super().reset()
        
        starting_position = self.env.unwrapped.data.qpos
        self.starting_position = (starting_position[0], starting_position[1])
        
        return state
    
    def step(self, action):
        next_state, _, terminated, truncated, _ = self.env.step(action)
        
        done = terminated or truncated
        if not done:
            fitness = 0
        else:
            last_position = self.env.unwrapped.data.qpos
            x, y = last_position[0], last_position[1]
            x_diff, y_diff = x - self.starting_position[0], y - self.starting_position[1]
            fitness = np.sqrt(x_diff**2 + y_diff**2)
            
        return next_state, fitness, terminated, truncated
