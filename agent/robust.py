"""Sinkhorn distributionally robust Bellman targets and their dual solver."""
from typing import Callable

import numpy as np
import torch

LAM_MIN, LAM_MAX = 1e-8, 1e8


class _SinkhornDual:
    """Dual objective h(lam) = -lam * (eps + delta * outer(lam)) and its derivatives.

    Args:
        r: Candidate rewards of shape (B, 1, N).
        q_max: Continuation values at the candidates, shape (B, 1, N).
        cost: Transport costs to the candidates, shape (B, 1, N).
        not_terminal: Non-terminal flags of shape (B, 1).
        discount: Discount factor.
        delta: Sinkhorn entropic regularisation.
        epsilon: Sinkhorn radius.
        log_w_nu: Importance-sampling log-weights of shape (B, 1, N).
    """

    def __init__(self, r, q_max, cost, not_terminal, discount, delta, epsilon, log_w_nu):
        dt = torch.float64
        r = r.detach().to('cpu', dt)
        q_max = q_max.detach().to('cpu', dt)
        nt = not_terminal.detach().to('cpu', dt)
        self.numer = -r - discount * q_max * nt.unsqueeze(-1)
        self.delta = float(delta)
        self.eps = float(epsilon)
        self.exp_c = -cost.detach().to('cpu', dt) / self.delta
        self.exp_c = self.exp_c + log_w_nu.detach().to('cpu', dt)

    def _weights(self, lam: torch.Tensor):
        """Return the normalised candidate weights and the log-mean-exp at lam."""
        e = self.numer / (self.delta * lam.view(-1, 1, 1)) + self.exp_c
        c = e.amax(dim=2, keepdim=True)
        w = torch.exp(e - c)
        s = w.sum(dim=2, keepdim=True)
        lse = c.squeeze(-1) + torch.log(s.squeeze(-1) / e.shape[2])
        return w / s, lse

    def h(self, lam: torch.Tensor) -> torch.Tensor:
        """Evaluate the dual objective.

        Args:
            lam: Dual variables of shape (B,).

        Returns:
            Tensor of shape (B,).
        """
        _, lse = self._weights(lam)
        return -lam * (self.eps + self.delta * lse.mean(dim=1))

    def derivs(self, lam: torch.Tensor):
        """Evaluate the dual objective and its first two derivatives.

        Args:
            lam: Dual variables of shape (B,).

        Returns:
            Tuple (h, h', h'') of tensors of shape (B,).
        """
        wn, lse = self._weights(lam)
        n_inner = self.numer.shape[2]
        neg_ent = torch.special.xlogy(wn, wn).sum(dim=2)
        e_expc = (wn * self.exp_c).sum(dim=2)
        ew = (wn * self.numer).sum(dim=2)
        var = ((wn * self.numer ** 2).sum(dim=2) - ew ** 2).clamp_min(0)
        h = -lam * (self.eps + self.delta * lse.mean(dim=1))
        h1 = -self.eps + self.delta * (np.log(n_inner) + neg_ent - e_expc).mean(dim=1)
        h2 = -var.mean(dim=1) / (self.delta * lam ** 3)
        return h, h1, h2


def solve_lamda_newton(dual: _SinkhornDual, lam0: torch.Tensor, mask: torch.Tensor,
                       tol: float = 1e-6, max_iter: int = 50) -> torch.Tensor:
    """Maximise the dual per sample by safeguarded Newton iterations on log(lam).

    Args:
        dual: The dual objective.
        lam0: Initial dual variables of shape (B,).
        mask: Boolean (B,) selecting the samples to optimise; others keep lam0.
        tol: Tolerance on the width of the log(lam) bracket.
        max_iter: Maximum number of iterations.

    Returns:
        float64 tensor of shape (B,) with the optimised dual variables.
    """
    lo, hi = np.log(LAM_MIN), np.log(LAM_MAX)
    u = lam0.clamp(LAM_MIN, LAM_MAX).log()
    a = torch.full_like(u, lo)
    b = torch.full_like(u, hi)
    _, g0, _ = dual.derivs(torch.full_like(u, LAM_MIN))
    at_lo = (g0 <= 0) & mask
    done = at_lo | ~mask
    conv_phi = torch.zeros_like(at_lo)
    for _ in range(max_iter):
        lam = u.exp()
        h, h1, h2 = dual.derivs(lam)
        a = torch.where(h1 > 0, u, a)
        b = torch.where(h1 < 0, u, b)
        phi1 = lam * h1
        phi2 = lam * h1 + lam ** 2 * h2
        newt = u - phi1 / phi2
        ok = (phi2 < 0) & (newt > a) & (newt < b) & torch.isfinite(newt)
        u_next = torch.where(ok, newt, (a + b) / 2)
        conv_phi = conv_phi | (~done & (phi1.abs() <= 1e-9 * (h.abs() + dual.eps)))
        done = done | ((b - a) < tol) | conv_phi
        u = torch.where(done, u, u_next)
        if bool(done.all()):
            break
    u = torch.where(conv_phi | at_lo | ~mask, u, (a + b) / 2)
    lam = u.exp()
    lam = torch.where(at_lo, torch.full_like(lam, LAM_MIN), lam)
    return torch.where(mask, lam, lam0)


def _solve_lamda(nu_rewards, q_max, cost, not_terminal, discount, delta, epsilon,
                 log_w_nu, lamda_init, ebar_pos, device, lamda_tol, lamda_max_iter):
    """Solve the dual from warm-started log dual variables and evaluate the robust target.

    Returns:
        Tuple (target (B,) on device in the dtype of nu_rewards, log dual variables
        (B,) float32 on CPU).
    """
    lam0 = lamda_init.detach().reshape(-1).to('cpu', torch.float64).exp()
    lam0 = lam0.clamp(LAM_MIN, LAM_MAX)
    dual = _SinkhornDual(nu_rewards, q_max, cost, not_terminal, discount, delta,
                         epsilon, log_w_nu)
    mask = ebar_pos.detach().to('cpu', torch.bool).reshape(-1)
    lam = solve_lamda_newton(dual, lam0, mask, tol=lamda_tol, max_iter=lamda_max_iter)
    hq = dual.h(lam).to(device=device, dtype=nu_rewards.dtype)
    return hq, lam.log().to(torch.float32)


def _nu_candidates(states: dict, act_values: torch.Tensor, nu, model, delta: float,
                   next_states: dict, n_inner: int):
    """Build the candidate next states, their rewards and transport costs.

    Returns:
        Tuple (eval_states with leading dims (B, n_inner), rewards (B, 1, n_inner),
        costs (B, 1, n_inner), log-weights (B, 1, n_inner)).
    """
    spec = model.spec
    batch_size = states['weight'].shape[0]
    next_returns = spec.latest(next_states)
    if next_returns.ndim == 2:
        next_returns = next_returns.unsqueeze(1)
    nu_returns, log_w_nu = nu(next_returns, delta)
    nu_returns = nu_returns.reshape(batch_size, n_inner, spec.seq_dim)

    eval_states = model.advance(states, act_values, nu_returns)
    eval_states['dt'] = next_states['dt'].unsqueeze(1).expand(batch_size, n_inner, 1).contiguous()
    nu_rewards = model.reward(states, act_values, eval_states).reshape(batch_size, 1, n_inner)
    nu_returns = nu_returns.reshape(batch_size, 1, n_inner, spec.seq_dim)
    cost = torch.linalg.norm(next_returns.unsqueeze(2) - nu_returns, dim=-1)
    return eval_states, nu_rewards, cost, log_w_nu


def _ebar_positive(cost: torch.Tensor, delta: float, epsilon: float,
                   log_w_nu: torch.Tensor) -> torch.Tensor:
    """Return a (B,) mask of samples with epsilon + delta * log E_nu[exp(-c/delta)] > 0."""
    exponent = -(cost / delta) + log_w_nu
    c = exponent.amax(dim=2, keepdim=True)
    ebar = epsilon + delta * (c.squeeze(-1) + torch.log(torch.exp(exponent - c).mean(dim=2))).mean(dim=1)
    return (ebar > 0).reshape(-1)


def hq_opt(qfunc: torch.nn.Module, states: dict, nu, model, epsilon: float, delta: float,
           n_inner: int, discount: float, lamda_init: torch.Tensor, lamda_max_iter: int,
           lamda_tol: float, not_terminal: torch.Tensor, act_values: torch.Tensor,
           next_states: dict, device='cpu'):
    """Compute the robust Bellman target for a discrete-action Q-function.

    The continuation value at each candidate next state is max_a Q(x, a).

    Args:
        qfunc: Target Q-network.
        states: State dict with leading dimension (B,).
        nu: NuSampler supplying candidates and log-weights.
        model: PortfolioModel pricing the candidates.
        epsilon: Sinkhorn radius.
        delta: Sinkhorn entropic regularisation.
        n_inner: Number of candidates per sample.
        discount: Discount factor.
        lamda_init: Warm-start log dual variables of shape (B, 1).
        lamda_max_iter: Maximum Newton iterations.
        lamda_tol: Newton tolerance on log(lam).
        not_terminal: Non-terminal flags of shape (B, 1).
        act_values: Executed actions of shape (B, action_dim).
        next_states: Realised next-state dict with leading dimension (B,).
        device: Device of the returned target.

    Returns:
        Tuple (targets (B,), log dual variables (B,), boolean mask (B,) of samples
        with a feasible dual).
    """
    batch_size = states['weight'].shape[0]
    with torch.no_grad():
        eval_states, nu_rewards, cost, log_w_nu = _nu_candidates(
            states, act_values, nu, model, delta, next_states, n_inner)
        nu_q_value = qfunc({k: v.reshape(batch_size, 1, n_inner, *v.shape[2:])
                            for k, v in eval_states.items()}).cpu()
        nu_act_idx = nu_q_value.argmax(dim=-1).cpu().unsqueeze(-1)
        q_max = torch.gather(nu_q_value, -1, nu_act_idx).squeeze(-1)
        ebar_pos = _ebar_positive(cost, delta, epsilon, log_w_nu)
    hq, log_lam = _solve_lamda(nu_rewards, q_max, cost, not_terminal, discount, delta,
                               epsilon, log_w_nu, lamda_init, ebar_pos, device,
                               lamda_tol, lamda_max_iter)
    return hq, log_lam, ebar_pos


def hq_opt_continuous(value_fn: Callable, states: dict, nu, model, epsilon: float,
                      delta: float, n_inner: int, discount: float,
                      lamda_init: torch.Tensor, lamda_max_iter: int, lamda_tol: float,
                      not_terminal: torch.Tensor, act_values: torch.Tensor,
                      next_states: dict, target_scale: float = 1.0, device='cpu'):
    """Compute the robust Bellman target for a continuous-action agent.

    Args:
        value_fn: Maps a flat state dict with leading dimension (N,) to values (N, 1).
        states: State dict with leading dimension (B,).
        nu: NuSampler supplying candidates and log-weights.
        model: PortfolioModel pricing the candidates.
        epsilon: Sinkhorn radius.
        delta: Sinkhorn entropic regularisation.
        n_inner: Number of candidates per sample.
        discount: Discount factor.
        lamda_init: Warm-start log dual variables of shape (B, 1).
        lamda_max_iter: Maximum Newton iterations.
        lamda_tol: Newton tolerance on log(lam).
        not_terminal: Non-terminal flags of shape (B, 1).
        act_values: Executed actions of shape (B, action_dim).
        next_states: Realised next-state dict with leading dimension (B,).
        target_scale: Reward scale the values are expressed in.
        device: Device of the returned target.

    Returns:
        Tuple (targets (B,), log dual variables (B,), boolean mask (B,) of samples
        with a feasible dual).
    """
    batch_size = states['weight'].shape[0]
    with torch.no_grad():
        eval_states, nu_rewards, cost, log_w_nu = _nu_candidates(
            states, act_values, nu, model, delta, next_states, n_inner)
        nu_values = value_fn({k: v.flatten(0, 1) for k, v in eval_states.items()})
        q_max = nu_values.reshape(batch_size, 1, n_inner)
        if target_scale != 1.0:
            q_max = q_max / target_scale
        ebar_pos = _ebar_positive(cost, delta, epsilon, log_w_nu)
    hq, log_lam = _solve_lamda(nu_rewards, q_max, cost, not_terminal, discount, delta,
                               epsilon, log_w_nu, lamda_init, ebar_pos, device,
                               lamda_tol, lamda_max_iter)
    if target_scale != 1.0:
        hq = hq * target_scale
    return hq, log_lam, ebar_pos
