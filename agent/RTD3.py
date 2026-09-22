"""Robust twin delayed deep deterministic policy gradient (RTD3) on the portfolio simplex."""
import copy
import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

from env.features import project_to_simplex
from .buffer import ReplayBuffer
from .nu_sampler import NuSampler
from .robust import hq_opt_continuous

ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh}


class SimplexActor(nn.Module):
    """Policy network with a softmax output over the assets.

    Args:
        obs_dim: Dimension of the flattened state.
        action_dim: Number of assets.
        hidden_sizes: Widths of the hidden layers.
        activation: Activation module class of the hidden layers.
        spec: StateSpec used to flatten state dicts.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int],
                 activation, spec):
        super().__init__()
        self.spec = spec
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state) -> torch.Tensor:
        """Compute portfolio weights.

        Args:
            state: State dict.

        Returns:
            Weights on the simplex of shape (..., action_dim).
        """
        return torch.softmax(self.net(self.spec.flatten(state)), dim=-1)


class SimplexCritic(nn.Module):
    """Q-network over a state and centred portfolio weights.

    Args:
        obs_dim: Dimension of the flattened state.
        action_dim: Number of assets.
        hidden_sizes: Widths of the hidden layers.
        activation: Activation module class of the hidden layers.
        action_centre: Value subtracted from the action weights.
        spec: StateSpec used to flatten state dicts.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int],
                 activation, action_centre: float, spec):
        super().__init__()
        self.spec = spec
        layers = []
        in_dim = obs_dim + action_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.action_centre = float(action_centre)

    def forward(self, state, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            state: State dict.
            action: Weights of shape (..., action_dim).

        Returns:
            Tensor of shape (..., 1).
        """
        return self.net(torch.cat([self.spec.flatten(state), action - self.action_centre], dim=-1))


class RTD3:
    """RTD3 agent with a Sinkhorn robust critic target.

    Args:
        obs_dim: Dimension of the flattened state.
        action_dim: Number of assets.
        model: PortfolioModel shared with the environment.
        discount: Discount factor.
        nu_loc: Location of nu.
        nu_scale: Scalar or per-asset scale of nu.
        nu_df: Degrees of freedom of nu.
        nu_corr: Correlation matrix of nu.
        nu_bounds: Support (lower, upper) of nu in log returns.
        nu_ratio: Budget shares [fixed, local] of the nu candidates.
        epsilon: Sinkhorn radius.
        delta: Sinkhorn entropic regularisation.
        n_nu: Number of nu candidates per transition.
        lamda_init: Initial log dual variable of new transitions.
        lamda_max_iter: Maximum Newton iterations of the dual solve.
        lamda_tol: Newton tolerance on log(lam).
        hidden_sizes: Hidden layer widths of the actor and critics.
        activation: 'relu' or 'tanh'.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.
        buffer_size: Replay buffer capacity.
        batch_size: Transitions per update.
        train_steps: Environment steps between updates.
        policy_delay: Critic updates per actor update.
        polyak_tau: Polyak averaging coefficient of the target networks.
        entropy_beta: Coefficient of the policy entropy bonus.
        expl_sigma: Initial standard deviation of the exploration noise.
        expl_clip: Clip of the exploration noise.
        expl_rho: Autocorrelation of the Ornstein-Uhlenbeck exploration noise.
        expl_end_sigma: Final standard deviation of the exploration noise.
        expl_sigma_steps: Environment steps over which the noise decays linearly.
        robust: Use the Sinkhorn robust target; otherwise the TD3 target.
        device: Torch device.
        seed: Seed of the global RNGs and the nu candidates.
    """

    def __init__(self, obs_dim: int, action_dim: int, model, discount: float,
                 nu_loc, nu_scale, nu_df: float, nu_corr, nu_bounds, nu_ratio,
                 epsilon: float, delta: float, n_nu: int, lamda_init: float,
                 lamda_max_iter: int, lamda_tol: float, hidden_sizes: List[int],
                 activation: str, actor_lr: float, critic_lr: float, buffer_size: int,
                 batch_size: int, train_steps: int, policy_delay: int, polyak_tau: float,
                 entropy_beta: float, expl_sigma: float, expl_clip: float, expl_rho: float,
                 expl_end_sigma: float, expl_sigma_steps: int, robust: bool,
                 device: torch.device, seed: int):
        self.model = model
        self.spec = model.spec
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.epsilon = epsilon
        self.delta = delta
        self.n_nu = n_nu
        self.lamda_init = lamda_init
        self.lamda_max_iter = lamda_max_iter
        self.lamda_tol = lamda_tol
        self.reward_scale = model.reward_scale
        self.robust = robust
        self.entropy_beta = entropy_beta
        self.policy_delay = policy_delay
        self.polyak_tau = polyak_tau
        self.expl_sigma = expl_sigma
        self.expl_clip = expl_clip
        self.expl_rho = expl_rho
        self.expl_sigma_steps = expl_sigma_steps
        self.expl_sigma_decay = (expl_end_sigma - expl_sigma) / expl_sigma_steps
        self.ou_state = None

        self.buffer = ReplayBuffer(buffer_size, self.spec, action_dim)
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.steps = 0
        self.q_updates = 0
        self.actor_updates = 0
        self.training_mode = True

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        act_fn = ACTIVATIONS[activation]
        self.actor = SimplexActor(obs_dim, action_dim, hidden_sizes, act_fn, self.spec).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        centre = self.spec.weight_centre
        self.critic1 = SimplexCritic(obs_dim, action_dim, hidden_sizes, act_fn, centre, self.spec).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2 = SimplexCritic(obs_dim, action_dim, hidden_sizes, act_fn, centre, self.spec).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.nu_sampler = NuSampler(n_nu=n_nu, seq_dim=action_dim, nu_df=nu_df,
                                    nu_scale=nu_scale, nu_loc=nu_loc, nu_bounds=nu_bounds,
                                    nu_ratio=nu_ratio, nu_corr=nu_corr, seed=seed) if robust else None
        self.prev_state = None
        self.prev_action = None

    def agent_start(self, observation: dict) -> torch.Tensor:
        """Reset the exploration noise and act on the first observation of an episode.

        Args:
            observation: Batched state dict.

        Returns:
            Weights of shape (B, action_dim).
        """
        self.ou_state = torch.randn(observation['weight'].shape[0], self.action_dim) * self.expl_sigma
        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        return action

    def agent_step(self, reward: torch.Tensor, observation: dict) -> torch.Tensor:
        """Store the last transition, train if due, and act on the new observation.

        Args:
            reward: Rewards of the last step, shape (B, 1).
            observation: Batched state dict.

        Returns:
            Weights of shape (B, action_dim).
        """
        if self.training_mode:
            self.train_mode_actions(reward.reshape(-1), observation, False)
        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        return action

    def agent_end(self, reward: torch.Tensor, observation: dict):
        """Store the last transition of an episode and train if due.

        The transition is stored with terminal=False. The episode ends because the
        path bank runs out of generated days, not because the process reaches an
        absorbing state, so the continuation value is bootstrapped from observation
        as usual and not_terminal is 1 for every stored transition.

        Args:
            reward: Rewards of the last step, shape (B, 1).
            observation: Batched state dict after the last step.
        """
        if self.training_mode:
            self.train_mode_actions(reward.reshape(-1), observation, False)

    def _ou_noise(self) -> torch.Tensor:
        """Advance the Ornstein-Uhlenbeck exploration noise and return its clipped value."""
        eps = torch.randn_like(self.ou_state) * self.expl_sigma
        self.ou_state = self.expl_rho * self.ou_state + math.sqrt(1. - self.expl_rho ** 2) * eps
        return self.ou_state.clamp(-self.expl_clip, self.expl_clip)

    def get_action(self, observation: dict, deterministic: bool = False) -> torch.Tensor:
        """Compute portfolio weights, with exploration noise in training mode.

        Args:
            observation: Batched state dict.
            deterministic: Disable exploration noise.

        Returns:
            Weights of shape (B, action_dim).
        """
        with torch.no_grad():
            w = self.actor({k: v.to(self.device) for k, v in observation.items()}).cpu()
        if (not deterministic) and self.training_mode:
            return project_to_simplex(w + self._ou_noise())[..., :self.action_dim]
        return w[..., :self.action_dim]

    def train_mode_actions(self, reward: torch.Tensor, observation: dict, terminal: bool):
        """Store a transition, update the networks when due and decay the exploration noise.

        Args:
            reward: Rewards of shape (B,).
            observation: Next-state dict.
            terminal: Whether observation is terminal.
        """
        self.steps += 1
        n = self.prev_state['weight'].shape[0]
        self.buffer.add(self.prev_state, self.prev_action, reward, observation,
                        torch.tensor([terminal], dtype=torch.bool).repeat(n, 1),
                        torch.tensor([self.lamda_init], dtype=torch.float32).repeat(n, 1))
        if self.steps % self.train_steps == 0 and len(self.buffer) >= self.batch_size:
            idx_list, f = self.buffer.sample(self.batch_size)
            self.train_batch(f['states'], f['actions'], f['rewards'], f['next_states'],
                             f['terminals'], f['lamdas'], idx_list)
        if self.steps < self.expl_sigma_steps:
            self.expl_sigma += self.expl_sigma_decay

    def target_action(self, states: dict) -> torch.Tensor:
        """Compute target-policy weights projected onto the simplex.

        Args:
            states: State dict.

        Returns:
            Weights of shape (..., action_dim).
        """
        w = self.target_actor(states)
        torch.randn_like(w)  # advances the global RNG exactly as the published runs did
        return project_to_simplex(w)[..., :self.action_dim]

    def target_value_fn(self, states: dict) -> torch.Tensor:
        """Compute the clipped double-Q continuation value.

        Args:
            states: State dict with leading dimension (N,).

        Returns:
            Tensor of shape (N, 1).
        """
        actions = self.target_action(states)
        q1 = self.target_critic1(states, actions).cpu()
        q2 = self.target_critic2(states, actions).cpu()
        return torch.min(q1, q2)

    def td3_target(self, rewards: torch.Tensor, next_states: dict, not_terminal: torch.Tensor):
        """Compute the non-robust TD3 target on the realised next state.

        Returns:
            Tuple (targets (B,), zero dual variables (B,), all-true mask (B,)).
        """
        with torch.no_grad():
            next_a = self.target_action(next_states)
            q = torch.min(self.target_critic1(next_states, next_a),
                          self.target_critic2(next_states, next_a))
            targets = (rewards.reshape(-1, 1) + self.discount * not_terminal.reshape(-1, 1)
                       * q.reshape(-1, 1)).squeeze(-1)
        n = targets.shape[0]
        return targets, torch.zeros(n), torch.ones(n, dtype=torch.bool)

    def polyak_update_target_networks(self):
        """Move every target network towards its online network by polyak_tau."""
        pairs = [(self.critic1, self.target_critic1), (self.actor, self.target_actor),
                 (self.critic2, self.target_critic2)]
        with torch.no_grad():
            for online, target in pairs:
                for p, p_targ in zip(online.parameters(), target.parameters()):
                    p_targ.data.mul_(1. - self.polyak_tau).add_(self.polyak_tau * p.data)

    def train_batch(self, states: dict, actions: torch.Tensor, rewards: torch.Tensor,
                    next_states: dict, terminals: torch.Tensor, lamda_inits: torch.Tensor,
                    buffer_indices: List[int]):
        """Update the critics, and every policy_delay updates the actor and targets.

        Args:
            states: State dict of the batch.
            actions: Executed weights of shape (B, action_dim).
            rewards: Scaled rewards of shape (B,).
            next_states: Next-state dict of the batch.
            terminals: Terminal flags of shape (B, 1).
            lamda_inits: Cached log dual variables of shape (B, 1).
            buffer_indices: Buffer positions of the batch.
        """
        actions, rewards = actions.to(self.device), rewards.to(self.device)
        terminals, lamda_inits = terminals.to(self.device), lamda_inits.to(self.device)
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        not_terminal = torch.logical_not(terminals)

        if self.robust:
            targets, lambdas, mask = hq_opt_continuous(
                value_fn=self.target_value_fn, states=states, nu=self.nu_sampler,
                model=self.model, epsilon=self.epsilon, delta=self.delta, n_inner=self.n_nu,
                discount=self.discount, lamda_init=lamda_inits,
                lamda_max_iter=self.lamda_max_iter, lamda_tol=self.lamda_tol,
                not_terminal=not_terminal, act_values=actions, next_states=next_states,
                target_scale=self.reward_scale, device=self.device)
        else:
            targets, lambdas, mask = self.td3_target(rewards, next_states, not_terminal)

        if mask.any():
            valid = mask if (~mask).any() else slice(None)
            target_values = targets[valid].to(self.device)
            current_q1 = self.critic1(states, actions).squeeze(-1)
            critic_loss = self.loss_fn(current_q1[valid], target_values)
            self.critic1_optimizer.zero_grad()
            current_q2 = self.critic2(states, actions).squeeze(-1)
            critic_loss = critic_loss + self.loss_fn(current_q2[valid], target_values)
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

        if self.q_updates % self.policy_delay == 0:
            w = self.actor(states)
            actor_loss = -self.critic1(states, w[..., :self.action_dim]).mean()
            if self.entropy_beta > 0.:
                entropy = -(w.clamp(min=1e-12).log() * w).sum(-1).mean()
                actor_loss = actor_loss + -self.entropy_beta * entropy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_updates += 1
            self.polyak_update_target_networks()

        if self.robust:
            self.buffer.set_lamda(buffer_indices, lambdas.cpu().detach(), mask)
        self.q_updates += 1

    def save_agent(self, path: str):
        """Save network weights, optimiser states and counters.

        Args:
            path: Output file.
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'ou_state': self.ou_state,
            'expl_sigma': self.expl_sigma,
            'steps': self.steps,
            'q_updates': self.q_updates,
            'actor_updates': self.actor_updates,
        }, path)
