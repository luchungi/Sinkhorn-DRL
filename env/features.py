"""Observation layout and simplex projection shared by the environments and agents."""
import math
from typing import Optional

import torch

DATATYPE = torch.float32
TRADING_DAYS = 252.0
ANN = math.sqrt(TRADING_DAYS)


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project each row onto the probability simplex {w : w >= 0, sum(w) = 1}.

    Rows whose projection is numerically degenerate (a coordinate so large that the
    closed form underflows) are mapped to the vertex of their largest coordinate.

    Args:
        v: Tensor of shape (..., n).

    Returns:
        Tensor of shape (..., n) whose rows lie on the simplex.
    """
    n = v.shape[-1]
    u, _ = torch.sort(v, dim=-1, descending=True)
    css = u.cumsum(dim=-1) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = (u - css / ind) > 0
    rho = (cond.to(torch.long).sum(dim=-1, keepdim=True) - 1).clamp(min=0)
    theta = css.gather(-1, rho) / (rho + 1).to(v.dtype)
    w = (v - theta).clamp(min=0.)
    vertex = torch.zeros_like(v).scatter_(-1, v.argmax(dim=-1, keepdim=True),
                                          torch.ones_like(v[..., :1]))
    tol = ((8 * n * torch.finfo(v.dtype).eps)
           * u[..., :1].abs().clamp(min=1.0)).clamp(max=0.5)
    ok = ((w.sum(dim=-1, keepdim=True) - 1.0).abs() <= tol)
    return torch.where(ok, w, vertex)


class StateSpec:
    """Layout of the state dict and its flattening into the network input.

    A state is a dict of raw quantities:
        'returns': (..., state_len, seq_dim) daily log returns, latest last.
        'weight':  (..., seq_dim) portfolio weights held before trading.
        'dt':      (..., 1) length in years of the upcoming holding period.

    The network input is [returns * ret_scale | weight - weight_centre | dt (optional)].

    Args:
        seq_dim: Number of assets.
        state_len: Number of daily log-return lags in the state.
        ret_gain: Gain applied to annualised returns (returns are multiplied by
            sqrt(252) * ret_gain).
        weight_centre: Value subtracted from the weights; defaults to 1 / seq_dim.
        include_dt: Whether dt is part of the network input.
    """

    def __init__(self, seq_dim: int, state_len: int, ret_gain: float = 1.0,
                 weight_centre: Optional[float] = None, include_dt: bool = False):
        self.seq_dim = seq_dim
        self.state_len = state_len
        self.ret_gain = ret_gain
        self.weight_centre = (1.0 / seq_dim) if weight_centre is None else float(weight_centre)
        self.include_dt = include_dt
        self.ret_scale = ANN * ret_gain
        self.dim = state_len * seq_dim + seq_dim + int(include_dt)

    def to_dict(self) -> dict:
        """Return the constructor arguments and the input dimension.

        Returns:
            Dict of the layout parameters.
        """
        return {'seq_dim': self.seq_dim, 'state_len': self.state_len,
                'ret_gain': self.ret_gain, 'weight_centre': self.weight_centre,
                'include_dt': self.include_dt, 'dim': self.dim}

    def encode_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Centre portfolio weights for use as a network input.

        Args:
            w: Weights of shape (..., seq_dim).

        Returns:
            Centred weights of the same shape.
        """
        return w - self.weight_centre

    def build(self, window: torch.Tensor, position: torch.Tensor,
              dt: Optional[torch.Tensor] = None) -> dict:
        """Assemble a state dict.

        Args:
            window: Daily log returns of shape (B, state_len, seq_dim), latest last.
            position: Portfolio weights of shape (B, seq_dim).
            dt: Holding-period lengths of shape (B, 1); defaults to 1 / 252.

        Returns:
            State dict with keys 'returns', 'weight' and 'dt'.
        """
        B = window.shape[0]
        return {
            'returns': window,
            'weight': position,
            'dt': window.new_full((B, 1), 1. / TRADING_DAYS) if dt is None else dt.reshape(B, 1),
        }

    def flatten(self, state: dict) -> torch.Tensor:
        """Flatten a state dict into the network input.

        Args:
            state: State dict with any number of leading batch dimensions.

        Returns:
            Tensor of shape (..., dim).
        """
        lead = state['returns'].shape[:-2]
        blocks = [(state['returns'] * self.ret_scale).reshape(*lead, -1),
                  self.encode_weights(state['weight'])]
        if self.include_dt:
            blocks.append(state['dt'])
        return torch.cat(blocks, dim=-1)

    def latest(self, state: dict) -> torch.Tensor:
        """Return the most recent daily log-return vector of a state.

        Args:
            state: State dict.

        Returns:
            Tensor of shape (..., seq_dim).
        """
        return state['returns'][..., -1, :]
