"""Portfolio accounting and performance metrics shared by the environments and agents."""
import numpy as np
import pandas as pd
import torch

DATATYPE = torch.float32


def transaction_cost_return(old_w: torch.Tensor, new_w: torch.Tensor,
                            trans_cost: float) -> torch.Tensor:
    """Compute the self-financing rebalancing cost as a fraction of portfolio value.

    Solves k = c * sum_i |(1 - k) new_w_i - old_w_i| by iterating over the sign vector,
    which converges in at most seq_dim + 1 sweeps.

    Args:
        old_w: Weights before trading, shape (..., seq_dim).
        new_w: Weights after trading, broadcastable against old_w.
        trans_cost: Proportional transaction cost c.

    Returns:
        Tensor of shape (..., 1) with the cost fraction k.
    """
    s = torch.where(new_w > old_w, 1.0, -1.0)
    for _ in range(new_w.shape[-1] + 1):
        num = torch.sum(trans_cost * s * (new_w - old_w), dim=-1, keepdim=True)
        den = 1.0 + torch.sum(trans_cost * s * new_w, dim=-1, keepdim=True)
        k = num / den
        s = torch.where((1.0 - k) * new_w > old_w, 1.0, -1.0)
    num = torch.sum(trans_cost * s * (new_w - old_w), dim=-1, keepdim=True)
    den = 1.0 + torch.sum(trans_cost * s * new_w, dim=-1, keepdim=True)
    return num / den


def interest_return(int_rate: float, dt: torch.Tensor) -> torch.Tensor:
    """Compute the simple return on cash over a holding period.

    Args:
        int_rate: Continuously compounded annual interest rate.
        dt: Holding-period lengths in years.

    Returns:
        Tensor of the same shape and dtype as dt.
    """
    out = torch.exp(dt.to(torch.float64) * float(int_rate)) - 1.0
    return out.to(dt.dtype)


def rf_log_return(int_rate: float, dt):
    """Compute the risk-free log return over one or more holding periods.

    Args:
        int_rate: Continuously compounded annual interest rate.
        dt: Scalar or array of holding-period lengths in years.

    Returns:
        int_rate * dt as a numpy value.
    """
    return float(int_rate) * np.asarray(dt, dtype=float)


def perf_metrics(excess, ann_factor: int = 252, axis: int = 0):
    """Compute annualised volatility, Sharpe ratio, downside deviation and Sortino ratio.

    Args:
        excess: Excess log returns, 1-D or 2-D.
        ann_factor: Number of periods per year.
        axis: Axis along which time runs.

    Returns:
        Tuple (volatility, sharpe, downside_deviation, sortino), reduced along axis.
    """
    excess = np.asarray(excess, dtype=float)
    root = np.sqrt(ann_factor)
    vol = np.std(excess, axis=axis, ddof=1) * root
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2, axis=axis))
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpe = np.mean(excess, axis=axis) / np.std(excess, axis=axis, ddof=1) * root
        sortino = np.mean(excess, axis=axis) / downside * root
    return vol, sharpe, downside * root, sortino


def drawdown_backward(levels: pd.Series) -> pd.Series:
    """Compute the drawdown of a wealth series from its running peak.

    Args:
        levels: Strictly positive wealth levels indexed by time.

    Returns:
        Series of drawdowns (<= 0) with the same index.
    """
    arr = levels.to_numpy(dtype=float)
    return pd.Series(arr / np.maximum.accumulate(arr, axis=-1) - 1.0,
                     index=levels.index, name=levels.name)


def gross_return(action: torch.Tensor, non_cash_return: torch.Tensor,
                 cash_return: torch.Tensor) -> torch.Tensor:
    """Compute the portfolio gross return before transaction costs.

    Args:
        action: Executed weights, shape (..., seq_dim); the remainder is held in cash.
        non_cash_return: Simple asset returns, shape (..., seq_dim).
        cash_return: Simple cash return, shape (..., 1).

    Returns:
        Tensor of shape (..., 1).
    """
    cash_weight = 1.0 - action.sum(dim=-1, keepdim=True)
    return 1.0 + cash_weight * cash_return + (action * non_cash_return).sum(dim=-1, keepdim=True)


def drift_weights(action: torch.Tensor, non_cash_return: torch.Tensor,
                  cash_return: torch.Tensor) -> torch.Tensor:
    """Compute the weights held after asset prices move over one period.

    Args:
        action: Executed weights, shape (..., seq_dim).
        non_cash_return: Simple asset returns, shape (..., seq_dim).
        cash_return: Simple cash return, shape (..., 1).

    Returns:
        Drifted weights of shape (..., seq_dim).
    """
    return action * (1.0 + non_cash_return) / gross_return(action, non_cash_return, cash_return)


def _align(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Insert singleton axes after the batch axis until x has ndim dimensions."""
    while x.ndim < ndim:
        x = x.unsqueeze(1)
    return x


class PortfolioModel:
    """Reward and state transition of a portfolio, shared by environments and agents.

    Args:
        spec: StateSpec describing the state layout.
        int_rate: Continuously compounded annual interest rate on cash.
        trans_cost: Proportional transaction cost.
        reward_scale: Factor applied to rewards by the environment.
    """

    def __init__(self, spec, int_rate: float = 0.024, trans_cost: float = 5e-4,
                 reward_scale: float = 1.0):
        self.spec = spec
        self.int_rate = float(int_rate)
        self.trans_cost = float(trans_cost)
        self.reward_scale = float(reward_scale)

    def reward(self, state: dict, action: torch.Tensor, next_state: dict,
               strict: bool = False) -> torch.Tensor:
        """Compute the log return of trading into action and holding for one period.

        state and action may have one fewer leading axis than next_state, in which
        case they are broadcast across it.

        Args:
            state: State dict with 'weight' and 'dt'.
            action: New weights, shape (..., seq_dim).
            next_state: State dict whose latest returns are realised over the period.
            strict: Raise if the gross return is not positive.

        Returns:
            Tensor of shape (..., 1) with the unscaled log return.
        """
        r_next = self.spec.latest(next_state)
        nd = r_next.ndim
        old_w = _align(state['weight'], nd)
        new_w = _align(action, nd)
        dt = _align(state['dt'], nd)

        cash_return = interest_return(self.int_rate, dt)
        non_cash_return = r_next.exp() - 1.
        k = transaction_cost_return(old_w, new_w, self.trans_cost)
        simple_return = (1.0 - k) * gross_return(new_w, non_cash_return, cash_return)
        if strict and bool(simple_return.amin() <= 0.0):
            raise ValueError(f'gross return {float(simple_return.amin()):.6g} <= 0 on a '
                             f'realised transition')
        return torch.log(simple_return)

    def advance(self, state: dict, action: torch.Tensor, next_returns: torch.Tensor) -> dict:
        """Compute the next state given executed weights and next-period returns.

        Args:
            state: State dict with leading dimension (B,).
            action: Executed weights, shape (B, seq_dim).
            next_returns: Daily log returns, shape (B, seq_dim) or (B, N, seq_dim).

        Returns:
            State dict with leading dimensions matching next_returns[..., 0].
        """
        nd = next_returns.ndim
        window = _align(state['returns'], nd + 1)
        window = window.expand(*next_returns.shape[:-1], *window.shape[-2:])
        nxt = {'returns': torch.cat([window[..., 1:, :], next_returns.unsqueeze(-2)], dim=-2)}
        dt = _align(state['dt'], nd).expand(*next_returns.shape[:-1], 1)
        nxt['weight'] = drift_weights(_align(action, nd), next_returns.exp() - 1.,
                                      interest_return(self.int_rate, dt))
        nxt['dt'] = dt
        return nxt
