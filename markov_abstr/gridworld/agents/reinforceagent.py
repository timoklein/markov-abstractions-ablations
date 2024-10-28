from typing import Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
from torch.types import Number

class EpisodicReplayBuffer:
    """Basic replay buffer for storing a single rollout for training."""

    def __init__(self):

        self.keys = ("state", "action", "reward", "done")

        for key in self.keys:
            setattr(self, key, [])

        self.current_episode = {key: [] for key in self.keys}

        self.episodes = 0
        self.total_experiences = 0

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        """Add the experience of the current time step to the buffer."""

        # Add the current step
        step_data = (state, action, reward, done)
        for i, key in enumerate(self.keys):
            self.current_episode[key].append(step_data[i])

        self.total_experiences += 1

        if done:
            self.episodes += 1

    def get_episode_and_clear(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample an episode from the buffer for training. Currently returns the single stored episode."""
        # Dicts are ordered with Python > 3.6, so this works
        states = np.array(self.current_episode["state"])
        actions = np.array(self.current_episode["action"])
        rewards = np.array(self.current_episode["reward"])
        dones = np.array(self.current_episode["done"])

        # Reset and clear memory
        for key in self.keys:
            self.current_episode[key] = []
        self.episodes -= 1

        return states, actions, rewards, dones

    def __len__(self) -> int:
        return self.episodes

class PolicyNet(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=32):
        super().__init__()
                
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        x = self.out_layer(x)
        return x
        


class DiscreteReinforceAgent:
    """Discrete REINFORCE Agent for VisGrid."""

    def __init__(
        self,
        n_features,
        n_actions,
        phi,
        lr=0.001,
        train_phi=False,
        n_hidden_layers=1,
        n_units_per_layer=32,
        gamma=0.9,
        center_returns: bool = False,
        alpha: float = 0.0,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.n_units_per_layer = n_units_per_layer
        self.lr = lr
        self.phi = phi
        self.gamma = gamma
        self.alpha = alpha
        self.train_phi = train_phi
        self.center_returns = center_returns
        
        self.reset()
        

    def reset(self):
        """ Reset the agent and the optimizer. """
        self.replay = EpisodicReplayBuffer()
        self.policy_net = PolicyNet(self.n_features, self.n_actions, self.n_units_per_layer)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

    def act(self, x: np.ndarray) -> Number:
        """Sample an action for acting in the environment."""
        with torch.no_grad():
            x = torch.from_numpy(x).to(torch.float)
            z = self.phi(x)
            logits = self.policy_net(z)
            distribution = D.Categorical(logits=logits)
            action = distribution.sample()

        return action.item()

    def get_log_probs(self, states: np.ndarray, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action log probabilities for estimating the policy gradient."""
        x = torch.from_numpy(states).to(torch.float)
        z = self.phi(x)
        if not self.train_phi:
            z = z.detach()
        logits = self.policy_net(z)
        distribution = D.Categorical(logits=logits)
        tensor_actions = torch.from_numpy(actions)
        log_probs = distribution.log_prob(tensor_actions)
        return log_probs, distribution.entropy()

    def calculate_episode_returns(self, episode_returns: np.ndarray) -> torch.Tensor:
        """Calculate the discounted future returns."""
        T = len(episode_returns)
        rets = np.empty(T, dtype=np.float)
        future_ret = 0
        for t in reversed(range(T)):
            future_ret = episode_returns[t] + self.gamma * future_ret
            rets[t] = future_ret

        # Use episode mean as baseline when enabled
        if self.center_returns:
            rets -= rets.mean()

        return torch.from_numpy(rets)

    def train(self) -> Number:  # type: ignore[override]
        

        states, actions, rewards, _ = self.replay.get_episode_and_clear()

        logp, entropy = self.get_log_probs(states, actions)
        returns = self.calculate_episode_returns(rewards)

        loss = (-returns * logp).mean()

        if not math.isclose(self.alpha, 0.0):
            loss += self.alpha * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()