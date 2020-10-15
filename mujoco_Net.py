#!/usr/bin/env python3
"""
Mujoco multi-agent network, include two actors and one centralized critic
"""
from torch import nn
import torch
from typing import List, Tuple, Union, Dict
import torch.nn.functional as F


class MujocoNet(nn.Module):
    def __init__(self, observation_shape,
                 action_space,
                 use_lstm=False,
                 policy_internal_dims: List[int]=[128, 128],
                 value_internal_dims: List[int] =[256, 256]
                 ):
        super(MujocoNet, self).__init__()
        if use_lstm:
            raise Exception("currently does not support lstm")
        self._use_lstm = use_lstm
        self._observation_shape = observation_shape
        self._action_shape = action_space.shape
        self._check_space_dimension()
        self._obs_dim = self._observation_shape[-1]
        self._act_dim = self._action_shape[-1]
        (self._policy0,
         self._policy1,
         self._critic_net) = self._build_policy_value_nets(
                                    policy_internal_dims,
                                    value_internal_dims)

    def initial_state(self, batch_size):
        if not self._use_lstm:
            return tuple()
        raise Exception("currently does not support lstm")
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()
                ) -> Tuple[Dict[str,torch.Tensor], tuple]:
        # [T, B, num_agents, obs_dim] for mojoco
        # core_state is the lstm state, which is not used in mojoco
        # return dim: [T, B, num_agents, *]
        x = inputs["frame"]
        T, B, num_agents, obs_dim = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        critic_obs_input = torch.flatten(x, start_dim=-2, end_dim=-1)
        policy_inputs = [x[:, i, :] for i in range(num_agents)]
        # flatten the T and B dimension
        if T * B > 1:
            joint_action_last = torch.flatten(inputs["last_action"], 0, 1)
        else:
            joint_action_last = inputs["last_action"]
        #if len(joint_action_last.shape) == 1:
        #    joint_action_last = joint_action_last.view(1, num_agents, 1)
        joint_action_one_hot = torch.cat([F.one_hot(
                                joint_action_last[:, i], self._act_dim
                                ).float() for i in range(num_agents)], dim=-1)

        # joint_reward = torch.flatten(inputs['reward'], 0, 1)
        critic_input = torch.cat([critic_obs_input, joint_action_one_hot], dim=-1)

        policy_logits = [getattr(self, f'_policy{i}')(policy_inputs[i])
                         for i in range(num_agents)]

        baseline = self._critic_net(critic_input)

        if self.training:
            actions = [torch.multinomial(
                        F.softmax(policy_logits[i], dim=1), num_samples=1
                        )
                       for i in range(num_agents)]
        else:
            # Don't sample when testing.
            actions = [torch.argmax(policy_logits[i], dim=1)
                       for i in range(num_agents)]

        policy_logits = torch.cat([policy_logits[i].view(T, B, self._act_dim).unsqueeze(-2)
                         for i in range(num_agents)], dim=-2)
        baseline = baseline.view(T, B, 2)
        actions = torch.cat([actions[i].view(T, B) for i in range(num_agents)],
                            dim=-1)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=actions),
            core_state,
        )

    def _check_space_dimension(self) -> None:
        """
        check if the dimension of the spaces are valid
        """
        if (not self._observation_shape[0] == 2 or
                not self._action_shape[0] == 2):
            raise ValueError("Have to be two agents")

    def _build_policy_value_nets(self,
                                 policy_internal_dims: List[int],
                                 value_internal_dims: List[int]
                                 ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Build policy and value network based on the observation and action dim
        :return:
        """
        return (self._construct_policy_net(policy_internal_dims),
                self._construct_policy_net(policy_internal_dims),
                self._construct_value_net(value_internal_dims)
                )

    def _construct_value_net(self,
                             value_internal_dims: List[int]
                             ) -> nn.Module:
        # value net take joint action and joint obs, (MADDPG)
        value_stack = [nn.Linear(2 * (self._obs_dim + self._act_dim), value_internal_dims[0])]
        value_stack.append(nn.ReLU())
        for i, dim in enumerate(value_internal_dims[:-1]):
            value_stack.append(
                                nn.Linear(value_internal_dims[i], value_internal_dims[i + 1])
                                )
            value_stack.append(nn.ReLU())
        value_stack.append(nn.Linear(value_internal_dims[-1], 2))
        value_net = nn.Sequential(*value_stack)
        return value_net

    def _construct_policy_net(self,
                              policy_internal_dims: List[int]
                              ) -> nn.Module:
        policy_stack = [nn.Linear(self._obs_dim, policy_internal_dims[0])]
        policy_stack.append(nn.ReLU())
        for i, dim in enumerate(policy_internal_dims[:-1]):
            policy_stack.append(
                                nn.Linear(policy_internal_dims[i], policy_internal_dims[i + 1])
                                )
            policy_stack.append(nn.ReLU())
        policy_stack.append(nn.Linear(policy_internal_dims[-1], self._act_dim))
        policy_stack.append(nn.Softmax())
        policy_net = nn.Sequential(*policy_stack)
        return policy_net




