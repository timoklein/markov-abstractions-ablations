from typing import Tuple


import numpy as np
import torch
from torch.types import Number
from torch import nn
import torch.nn.functional as F
from torch import optim
import torch.distributions as D



class RolloutReplayBuffer:
    """Basic replay buffer for storing a single rollout for training."""

    def __init__(self):

        self.keys = ("state", "action", "reward", "next_state", "done")

        for key in self.keys:
            setattr(self, key, [])

        self.current_episode = {key: [] for key in self.keys}

        self.episodes = 0
        self.total_experiences = 0

    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add the experience of the current time step to the buffer."""

        # Add the current step
        step_data = (state, action, reward, next_state, done)
        for i, key in enumerate(self.keys):
            self.current_episode[key].append(step_data[i])

        self.total_experiences += 1

        if done:
            self.episodes += 1

    def get_samples_and_clear(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all transitions that have been collected so far, return them and clear the buffer."""
        # Dicts are ordered with Python > 3.6, so this works
        states = np.array(self.current_episode["state"])
        actions = np.array(self.current_episode["action"])
        rewards = np.array(self.current_episode["reward"])
        next_states = np.array(self.current_episode["next_state"])
        dones = np.array(self.current_episode["done"])

        # Reset and clear memory
        for key in self.keys:
            self.current_episode[key] = []
        self.episodes -= 1

        return (
            torch.from_numpy(states).to(torch.float),
            torch.from_numpy(actions),
            torch.from_numpy(rewards).to(torch.float),
            torch.from_numpy(next_states).to(torch.float),
            torch.from_numpy(dones),
        )

    def __len__(self) -> int:
        return self.episodes


class Actor(nn.Module):
    """Discrete A2C actor."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_policy(self, x: torch.Tensor) -> D.Categorical:
        logits = self(x)
        policy_dist = D.Categorical(logits=logits)
        return policy_dist


class Critic(nn.Module):
    """Discrete A2C critic."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DiscreteA2CAgent:
    """Discrete A2C Agent for Visgrid."""

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
        alpha: float = 0.1,
        use_gae: bool = False,
        n_steps: int = 5,
        gae_lambda: float = 1.0,
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
        self.use_gae = use_gae
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        
        self.reset()

    def reset(self) -> None:
        """Reset the agent."""
        self.buffer = RolloutReplayBuffer()
        self.actor = Actor(obs_dim=self.n_features, n_actions=self.n_actions)
        self.critic = Critic(obs_dim=self.n_features)

        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=self.lr)


    def act(self, x: np.ndarray) -> int:
        """Sample an action for acting in the environment."""
        with torch.no_grad():
            x = torch.from_numpy(x).to(torch.float)
            z = self.phi(x)
            policy_dist = self.actor.get_policy(z[None, ...])
            action = policy_dist.sample()

        return int(action.item())

    def calculate_nstep_returns(self, rewards: torch.Tensor, dones: torch.Tensor, next_v_pred: torch.Tensor) -> torch.Tensor:
        """Calculate the discounted future returns."""
        rets = torch.zeros_like(rewards)
        # Value estimate for last next_state for bootstrapping
        future_ret = next_v_pred
        not_dones = ~dones

        # Recursive target calculation for n steps
        for t in reversed(range(self.n_steps)):
            future_ret = rewards[t] + self.gamma * future_ret * not_dones[t]
            rets[t] = future_ret

        return rets[..., None]

    def calculate_gaes(
        self, rewards: torch.Tensor, dones: torch.Tensor, v_preds: torch.Tensor, last_v_pred: torch.Tensor
    ) -> torch.Tensor:
        """Calculate GAEs."""
        # GAE directly calculates the advantages
        # Needs the value predictions for all states
        # + last next_state for bootstrapping
        v_preds_gae = torch.cat((v_preds.squeeze(), last_v_pred), dim=0)
        assert self.n_steps + 1 == len(v_preds_gae)
        gaes = torch.zeros_like(rewards)
        # Create scalar for the future GAEs
        future_gae = torch.tensor(0.0, dtype=rewards.dtype)
        not_dones = (1 - dones).to(torch.float)
        for t in reversed(range(self.n_steps)):
            # 1-step Bellman error for t
            delta = rewards[t] + self.gamma * v_preds_gae[t + 1] * not_dones[t] - v_preds_gae[t]
            # Accumulate deltas for steps t, t+1, ...
            future_gae = delta + self.gamma * self.gae_lambda * not_dones[t] * future_gae
            gaes[t] = future_gae

        return gaes[..., None]

    def train(self) -> Number:
        """Train on a batch of data."""

        states, actions, rewards, next_states, dones= self.buffer.get_samples_and_clear()

        # Bootstrapping target after n steps
        next_state = next_states[-1]
        with torch.no_grad():
            z = self.phi(next_state[None, ...])
            next_v_pred = self.critic(z)

        z = self.phi(states)
        if not self.train_phi:
            z = z.detach()
        v_preds = self.critic(z)
        adv_pred = v_preds.detach()

        if self.use_gae:
            # GAE calculates advantages directly
            advantage = self.calculate_gaes(rewards=rewards, dones=dones, v_preds=adv_pred, last_v_pred=next_v_pred)
            v_targets = advantage + adv_pred
        else:
            # N-step bootstrapping calculates only value targets
            v_targets = self.calculate_nstep_returns(rewards=rewards, dones=dones, next_v_pred=next_v_pred)
            advantage = v_targets - adv_pred

        policy = self.actor.get_policy(z)
        logp = policy.log_prob(actions)

        # Actor loss with entropy bonus
        actor_loss = (-advantage * logp + self.alpha * logp.mean()).mean()
        critic_loss = F.smooth_l1_loss(v_preds, v_targets, reduction="mean")
        loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return loss.item()

