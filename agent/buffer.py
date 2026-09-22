"""Preallocated replay buffer for batched state-dict transitions."""
import random
from typing import Dict, List, Tuple

import torch


class ReplayBuffer:
    """Fixed-capacity ring buffer storing transitions and cached dual variables.

    Args:
        capacity: Maximum number of transitions.
        spec: StateSpec describing the state layout.
        action_dim: Width of the stored action.
        action_dtype: dtype of the stored action (float for weights, long for indices).
    """

    def __init__(self, capacity: int, spec, action_dim: int,
                 action_dtype: torch.dtype = torch.float32):
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.action_dtype = action_dtype
        self.shapes = {'returns': (spec.state_len, spec.seq_dim),
                       'weight': (spec.seq_dim,),
                       'dt': (1,)}
        self.states = {k: torch.zeros(self.capacity, *sh, dtype=torch.float32)
                       for k, sh in self.shapes.items()}
        self.next_states = {k: torch.zeros(self.capacity, *sh, dtype=torch.float32)
                            for k, sh in self.shapes.items()}
        self.actions = torch.zeros(self.capacity, self.action_dim, dtype=self.action_dtype)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32)
        self.terminals = torch.zeros(self.capacity, 1, dtype=torch.bool)
        self.lamdas = torch.zeros(self.capacity, 1, dtype=torch.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, states: Dict[str, torch.Tensor], actions: torch.Tensor,
            rewards: torch.Tensor, next_states: Dict[str, torch.Tensor],
            terminals: torch.Tensor, lamdas: torch.Tensor):
        """Append a batch of transitions, overwriting the oldest when full.

        Args:
            states: State dict with leading dimension (B,).
            actions: Actions of shape (B, action_dim).
            rewards: Rewards with B elements.
            next_states: Next-state dict with leading dimension (B,).
            terminals: Terminal flags with B elements.
            lamdas: Initial log dual variables with B elements.
        """
        B = actions.shape[0]
        idx = (torch.arange(B) + self.pos) % self.capacity
        for k in self.shapes:
            self.states[k][idx] = states[k].detach().to(torch.float32)
            self.next_states[k][idx] = next_states[k].detach().to(torch.float32)
        self.actions[idx] = actions.detach().to(self.action_dtype)
        self.rewards[idx] = rewards.detach().reshape(B).to(torch.float32)
        self.terminals[idx] = terminals.detach().reshape(B, 1).to(torch.bool)
        self.lamdas[idx] = lamdas.detach().reshape(B, 1).to(torch.float32)
        if self.pos + B >= self.capacity:
            self.full = True
        self.pos = (self.pos + B) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[int], dict]:
        """Draw transitions uniformly without replacement using Python's global RNG.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            Tuple (indices, fields) where fields holds 'states', 'actions', 'rewards',
            'next_states', 'terminals' and 'lamdas'.
        """
        idx_list = random.sample(range(len(self)), batch_size)
        idx = torch.as_tensor(idx_list, dtype=torch.long)
        fields = {
            'states': {k: v[idx] for k, v in self.states.items()},
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'next_states': {k: v[idx] for k, v in self.next_states.items()},
            'terminals': self.terminals[idx],
            'lamdas': self.lamdas[idx],
        }
        return idx_list, fields

    def set_lamda(self, indices, values: torch.Tensor, mask: torch.Tensor):
        """Overwrite cached log dual variables for the selected transitions.

        Args:
            indices: Buffer positions returned by sample.
            values: New log dual variables, one per index.
            mask: Boolean tensor selecting which entries to overwrite.
        """
        idx = torch.as_tensor(indices, dtype=torch.long)
        vals = values.detach().reshape(-1, 1).to(torch.float32)
        m = mask.detach().cpu().reshape(-1)
        idx, vals = idx[m], vals[m]
        if idx.numel():
            self.lamdas[idx] = vals
