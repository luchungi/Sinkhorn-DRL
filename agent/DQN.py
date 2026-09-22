"""Double DQN and its Sinkhorn robust counterpart for a discrete action grid."""
import copy
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .buffer import ReplayBuffer
from .nu_sampler import NuSampler
from .robust import hq_opt


class DQN:
    """Double DQN with epsilon-greedy exploration and hard target updates.

    Args:
        qfunc: Q-network.
        num_actions: Number of discrete actions.
        spec: StateSpec describing the state layout.
        discount: Discount factor.
        epsilon: Probability of a uniformly random action in training mode.
        buffer_size: Replay buffer capacity.
        clone_steps: Environment steps between target network updates.
        train_steps: Environment steps between updates.
        batch_size: Transitions per minibatch.
        n_batches: Minibatches per epoch.
        n_epochs: Epochs per update.
        lr: Learning rate.
        device: Torch device.
        seed: Seed of Python's global RNG.
    """

    def __init__(self, qfunc: nn.Module, num_actions: int, spec, discount: float,
                 epsilon: float, buffer_size: int, clone_steps: int, train_steps: int,
                 batch_size: int, n_batches: int, n_epochs: int, lr: float,
                 device: torch.device, seed: int):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.discount = discount
        self.q = qfunc
        self.batch_size = batch_size
        self.clone_steps = clone_steps
        self.train_steps = train_steps
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.device = device
        self.q.to(self.device)
        self.target_q = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.q_updates = 0
        self.training_mode = True
        self.steps = 0
        self.lamda_init = 0.
        self.buffer = ReplayBuffer(buffer_size, spec, 1, action_dtype=torch.long)
        self.loss_fn = nn.MSELoss()
        random.seed(seed)

    def agent_start(self, observation: dict) -> torch.Tensor:
        """Act on the first observation of an episode.

        Args:
            observation: Batched state dict.

        Returns:
            Action indices of shape (B, 1).
        """
        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        return action

    def agent_step(self, reward: torch.Tensor, observation: dict) -> torch.Tensor:
        """Act on a new observation, then store the last transition and train if due.

        Args:
            reward: Rewards of the last step, shape (B, 1).
            observation: Batched state dict.

        Returns:
            Action indices of shape (B, 1).
        """
        action = self.get_action(observation)
        if self.training_mode:
            self.train_mode_actions(reward.reshape(-1), observation, False)
        self.prev_state = observation
        self.prev_action = action
        return action

    def agent_end(self, reward: torch.Tensor, observation: dict):
        """Store the last transition of an episode and train if due.

        The transition is stored with terminal=False. The episode ends because the
        simulated calendar runs out, not because the process reaches an absorbing
        state, so the continuation value is bootstrapped from observation as usual.

        Args:
            reward: Rewards of the last step, shape (B, 1).
            observation: Batched state dict after the last step.
        """
        if self.training_mode:
            self.train_mode_actions(reward.reshape(-1), observation, False)

    def get_action(self, observation: dict, greedy: bool = False) -> torch.Tensor:
        """Select actions greedily, with epsilon-greedy exploration in training mode.

        Args:
            observation: Batched state dict.
            greedy: Disable exploration.

        Returns:
            Action indices of shape (B, 1).
        """
        with torch.no_grad():
            q_values = self.q({k: v.to(self.device) for k, v in observation.items()})
        actions = torch.argmax(q_values, dim=-1, keepdim=True)
        if self.epsilon > 0 and not greedy and self.training_mode:
            explore = torch.rand(actions.shape[0]) < self.epsilon
            if explore.sum() > 0:
                actions[explore] = torch.randint(0, self.num_actions, (explore.sum(), 1),
                                                 device=self.device)
        return actions.cpu()

    def train_mode_actions(self, reward: torch.Tensor, observation: dict, terminal: bool):
        """Store a transition, update the target network and train when due.

        Args:
            reward: Rewards of shape (B,).
            observation: Next-state dict.
            terminal: Whether observation is terminal.
        """
        self.steps += 1
        n = self.prev_state['weight'].shape[0]
        self.buffer.add(self.prev_state, self.prev_action, reward, observation,
                        torch.tensor([terminal], dtype=torch.bool).repeat(n, 1),
                        torch.tensor([float(self.lamda_init)], dtype=torch.float32).repeat(n, 1))
        ready = len(self.buffer) >= self.batch_size * self.n_batches
        if self.steps % self.clone_steps == 0 and ready:
            self.target_q.load_state_dict(self.q.state_dict())
        if self.steps % self.train_steps == 0 and ready:
            self.update_q()

    def update_q(self):
        """Train the Q-network on minibatches drawn from the replay buffer."""
        for _ in range(self.n_epochs):
            idx_list, f = self.buffer.sample(self.batch_size * self.n_batches)
            for j in range(self.n_batches):
                sl = slice(j * self.batch_size, (j + 1) * self.batch_size)
                self.train_batch({k: v[sl] for k, v in f['states'].items()},
                                 f['actions'][sl], f['rewards'][sl],
                                 {k: v[sl] for k, v in f['next_states'].items()},
                                 f['terminals'][sl], f['lamdas'][sl], idx_list[sl])

    def train_batch(self, states: dict, actions: torch.Tensor, rewards: torch.Tensor,
                    next_states: dict, terminals: torch.Tensor, lamda_inits: torch.Tensor,
                    buffer_indices: List[int]):
        """Take one gradient step towards the double DQN target.

        Args:
            states: State dict of the batch.
            actions: Action indices of shape (B, 1).
            rewards: Rewards of shape (B,).
            next_states: Next-state dict of the batch.
            terminals: Terminal flags of shape (B, 1).
            lamda_inits: Unused cached dual variables.
            buffer_indices: Unused buffer positions.
        """
        rewards = rewards.unsqueeze(-1).to(self.device)
        actions, terminals = actions.to(self.device), terminals.to(self.device)
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}

        next_actions = self.get_action(next_states, greedy=True)
        row_indices = np.arange(next_actions.shape[0])
        with torch.no_grad():
            next_state_q = self.target_q(next_states)
            next_state_q = next_state_q[row_indices, next_actions.squeeze().to(torch.int32)].unsqueeze(-1)
        targets = (rewards + self.discount * next_state_q * torch.logical_not(terminals)).squeeze()

        current_state_q = self.q(states)[np.arange(actions.shape[0]), actions.squeeze(-1)]
        loss = self.loss_fn(current_state_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_updates += 1


class RobustDQN(DQN):
    """DQN whose target is the Sinkhorn distributionally robust Bellman operator.

    Args:
        qfunc: Q-network.
        num_actions: Number of discrete actions.
        spec: StateSpec describing the state layout.
        model: PortfolioModel shared with the environment.
        action_values: Portfolio weight of each action, shape (num_actions,).
        discount: Discount factor.
        epsilon: Probability of a uniformly random action in training mode.
        buffer_size: Replay buffer capacity.
        clone_steps: Environment steps between target network updates.
        train_steps: Environment steps between updates.
        batch_size: Transitions per minibatch.
        n_batches: Minibatches per epoch.
        n_epochs: Epochs per update.
        lr: Learning rate.
        device: Torch device.
        seed: Seed of Python's global RNG and the nu candidates.
        nu_loc: Location of nu.
        nu_scale: Scale of nu.
        nu_df: Degrees of freedom of nu.
        nu_bounds: Support (lower, upper) of nu in log returns.
        nu_ratio: Budget shares [fixed, local, far] of the nu candidates.
        sinkhorn_dist: Sinkhorn radius.
        delta: Sinkhorn entropic regularisation.
        n_nu: Number of nu candidates per transition.
        lamda_init: Initial log dual variable of new transitions.
        lamda_max_iter: Maximum Newton iterations of the dual solve.
        lamda_tol: Newton tolerance on log(lam).
    """

    def __init__(self, qfunc: nn.Module, num_actions: int, spec, model,
                 action_values: torch.Tensor, discount: float, epsilon: float,
                 buffer_size: int, clone_steps: int, train_steps: int, batch_size: int,
                 n_batches: int, n_epochs: int, lr: float, device: torch.device, seed: int,
                 nu_loc: float, nu_scale: float, nu_df: float, nu_bounds, nu_ratio,
                 sinkhorn_dist: float, delta: float, n_nu: int, lamda_init: float,
                 lamda_max_iter: int, lamda_tol: float):
        super().__init__(qfunc, num_actions, spec, discount, epsilon, buffer_size,
                         clone_steps, train_steps, batch_size, n_batches, n_epochs, lr,
                         device, seed)
        self.model = model
        self.action_values = action_values
        self.sinkhorn_dist = sinkhorn_dist
        self.delta = delta
        self.n_nu = n_nu
        self.lamda_init = lamda_init
        self.lamda_max_iter = lamda_max_iter
        self.lamda_tol = lamda_tol
        self.nu_sampler = NuSampler(n_nu=n_nu, seq_dim=1, nu_df=nu_df, nu_scale=nu_scale,
                                    nu_loc=nu_loc, nu_bounds=nu_bounds, nu_ratio=nu_ratio,
                                    nu_corr=None, seed=seed)

    def train_batch(self, states: dict, actions: torch.Tensor, rewards: torch.Tensor,
                    next_states: dict, terminals: torch.Tensor, lamda_inits: torch.Tensor,
                    buffer_indices: List[int]):
        """Take one gradient step towards the robust target and cache the dual variables.

        Args:
            states: State dict of the batch.
            actions: Action indices of shape (B, 1).
            rewards: Rewards of shape (B,).
            next_states: Next-state dict of the batch.
            terminals: Terminal flags of shape (B, 1).
            lamda_inits: Cached log dual variables of shape (B, 1).
            buffer_indices: Buffer positions of the batch.
        """
        act_values = self.action_values[actions].to(self.device)
        actions, terminals = actions.to(self.device), terminals.to(self.device)
        lamda_inits = lamda_inits.to(self.device)
        states = {k: v.to(self.device) for k, v in states.items()}
        next_states = {k: v.to(self.device) for k, v in next_states.items()}
        not_terminal = torch.logical_not(terminals)

        current_state_q = self.q(states)[np.arange(actions.shape[0]), actions.squeeze(-1)]
        targets, lambdas, mask = hq_opt(
            self.target_q, states, self.nu_sampler, self.model, self.sinkhorn_dist,
            self.delta, self.n_nu, self.discount, lamda_inits, self.lamda_max_iter,
            self.lamda_tol, not_terminal, act_values, next_states, device=self.device)

        if (~mask).any():
            loss = self.loss_fn(current_state_q[mask], targets[mask].to(self.device))
        else:
            loss = self.loss_fn(current_state_q, targets.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.set_lamda(buffer_indices, lambdas.cpu().detach(), mask)
        self.q_updates += 1
