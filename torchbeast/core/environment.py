# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch
from typing import Union, List
import numpy as np
from torchbeast.core.utils import check_nan_and_inf


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).

def _get_gym_action(action: torch.Tensor) -> Union[np.ndarray, List[np.ndarray]]:
    action_list = [action[:, i].numpy() for i in range(action.shape[-1])]
    if len(action_list) == 1:
        return action_list[0]
    return action_list

class Environment:
    def __init__(self, gym_env, num_agents: int):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self._num_agents = num_agents

    def initial(self):
        initial_reward = torch.zeros(1, self._num_agents)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, self._num_agents, dtype=torch.int64)
        self.episode_return = torch.zeros(1, self._num_agents)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame = _format_frame(self.gym_env.reset())

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        gym_env_action = _get_gym_action(action)
        frame, reward, done, unused_info = self.gym_env.step(gym_env_action)
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, self._num_agents)
            self.episode_step = torch.zeros(1, self._num_agents, dtype=torch.int32)

        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, self._num_agents)
        done = torch.tensor(done).view(1, 1)

        check_nan_and_inf([frame, reward, done, episode_return, episode_step, action])

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

    def close(self):
        self.gym_env.close()
