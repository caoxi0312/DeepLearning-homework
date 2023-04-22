import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Agent:
    """
    A custom agent that uses a neural network to approximate the Q-function.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((observation_space.shape[0], action_space.n))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        returns the action with the highest Q-value for the current state, or a random action with probability
        """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        state = self._discretize(observation)
        q_values = self.q_table[state]
        return np.argmax(q_values)

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        updates the Q-value for the (state, action) pair using the Q-learning update rule.
        """
        state = self._discretize(observation)
        if terminated or truncated:
            target = reward
        else:
            next_state = self._discretize(observation)
            next_q_values = self.q_table[next_state]
            target = reward + self.gamma * np.max(next_q_values)
        current_q_value = self.q_table[state]
        current_q_value[action] += self.learning_rate * (
            target - current_q_value[action]
        )
        self.q_table[state] = current_q_value
        if truncated or terminated:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _discretize(self, observation):
        upper_bounds = [
            self.observation_space.high[0],
            0.5,
            self.observation_space.high[2],
            0.5,
            self.observation_space.high[4],
            self.observation_space.high[5],
        ]
        lower_bounds = [
            self.observation_space.low[0],
            -0.5,
            self.observation_space.low[2],
            -0.5,
            self.observation_space.low[4],
            self.observation_space.low[5],
        ]
        ratios = [
            (observation[i] + abs(lower_bounds[i]))
            / (upper_bounds[i] - lower_bounds[i])
            for i in range(len(observation))
        ]
        discretized = [
            int(round((self.observation_space.nvec[i] - 1) * ratios[i]))
            for i in range(len(observation))
        ]
        discretized = [
            min(self.observation_space.nvec[i] - 1, max(0, discretized[i]))
            for i in range(len(observation))
        ]
        return tuple(discretized)
