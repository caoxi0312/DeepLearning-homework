import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class Policy(nn.Module):
    """
    a neural network that takes observations as input and outputs a probability distribution over actions
    """

    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    """
    a policy-based agent that learns to maximize the expected return
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        obs_dim = observation_space.shape[0]
        action_dim = action_space.n
        hidden_dim = 64
        lr = 1e-3

        self.policy = Policy(obs_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        uses the policy to choose an action by sampling from this distribution
        """
        with torch.no_grad():
            x = torch.FloatTensor(observation).unsqueeze(0)
            probs = self.policy(x)
            action = probs.multinomial(1)
        return action.item()
        # return self.action_space.sample()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        updates the policy parameters based on the observed reward and the chosen action
        """
        x = torch.FloatTensor(observation).unsqueeze(0)
        probs = self.policy(x)
        log_probs = torch.log(probs)
        action = probs.multinomial(1)
        loss = -log_probs[0, action] * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # pass
