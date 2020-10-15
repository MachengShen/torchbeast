#!/usr/bin/env python3
"""
Classes for wrapping a multi-agent environment into a single-agent
environment so that single-agent RL framework can be applied
"""
from typing import List, Sequence, Tuple, Dict
import gym
from gym import spaces
import numpy as np

class Multi2SingleWrapper(gym.Env):
    """
    This class wraps around a multi-agent environment so that the
    interface looks like a single-agent environment
    """
    def __init__(self, multi_env: gym.Env) -> None:
        super(Multi2SingleWrapper, self).__init__()
        self._multi_env = multi_env
        self._set_action_obs_spaces()

    def _set_action_obs_spaces(self) -> None:
        """
        Set the action and observation spaces according to the multi-agent env,
        by stacking the agents along the first dim
        """
        list_of_obs_space = self._multi_env.observation_space.spaces
        list_of_action_space = self._multi_env.action_space.spaces
        obs_low = np.stack([obs_space.low for obs_space in list_of_obs_space])
        obs_high = np.stack([obs_space.high for obs_space in list_of_obs_space])
        action_low = np.stack([action_space.low for action_space in list_of_action_space])
        action_high = np.stack([action_space.high for action_space in list_of_action_space])
        self.observation_space = spaces.Box(low=obs_low,
                                            high=obs_high)
        self.action_space = spaces.Box(low=action_low,
                                       high=action_high)

    def step(self,
             actions: List[np.ndarray]
             ) -> Tuple[np.ndarray,
                           np.ndarray,
                           np.ndarray,
                           Tuple[Dict, Dict]]:
        next_obs, rewards, dones, infos = self._multi_env.step(actions)

        next_obs = np.stack(next_obs).astype(np.float32)
        rewards = np.stack(rewards).astype(np.float32)
        done = np.array(dones[0]).astype(np.bool)

        return next_obs, rewards, done, infos

    def reset(self) -> np.ndarray:
        return np.stack(self._multi_env.reset()).astype(np.float32)

    def render(self, mode='human', close=False) -> None:
        self._multi_env.render(mode=mode, close=close)