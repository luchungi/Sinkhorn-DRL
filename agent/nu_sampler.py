"""Truncated Student-t reference measure nu and its importance-sampling candidate set."""
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.special import gammaln, stdtr, stdtrit

DATATYPE = torch.float32
FAR_TAIL_FRAC = 1e-5


def _log_sphere_area(d: int) -> float:
    """Return the log surface area of the unit sphere in R^d."""
    return float(np.log(2.0) + 0.5 * d * np.log(np.pi) - gammaln(d / 2.0))


def as_nu_param(v, d: int, name: str):
    """Convert a nu location or scale to a float or a (d,) float64 array.

    Args:
        v: Scalar or array-like value.
        d: Dimension of nu.
        name: Parameter name used in error messages.

    Returns:
        A Python float if v has one element, otherwise a (d,) numpy array.
    """
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return float(a)
    a = a.reshape(-1)
    if a.size == 1:
        return float(a[0])
    if a.size != d:
        raise ValueError(f'{name} must be a scalar or a ({d},) vector, got shape {a.shape}')
    return a


def sample_multivariate_t_qmc(n_samples: int, corr, df: float, scale, loc,
                              seed: Optional[int] = None) -> torch.Tensor:
    """Draw an antithetic scrambled-Sobol' sample from a correlated multivariate Student-t.

    Args:
        n_samples: Number of points.
        corr: Correlation matrix of shape (d, d).
        df: Degrees of freedom.
        scale: Scalar or (d,) per-coordinate scale.
        loc: Scalar location added to every coordinate.
        seed: Seed of the Sobol' scrambling.

    Returns:
        Tensor of shape (n_samples, d) whose sample mean equals loc.
    """
    from scipy.stats import chi2, norm, qmc

    corr_np = (corr.to(torch.float64).numpy() if isinstance(corr, torch.Tensor)
               else np.asarray(corr, dtype=np.float64))
    dim = corr_np.shape[0]
    L = np.linalg.cholesky(corr_np)
    n_base = (n_samples + 1) // 2
    sampler = qmc.Sobol(d=dim + 1, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = sampler.random(n_base)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    z = norm.ppf(u[:, :dim])
    g = chi2.ppf(u[:, dim], df).reshape(-1, 1)
    x = (z @ L.T) / np.sqrt(g / df)
    x = np.concatenate([x, -x], axis=0)[:n_samples]
    return torch.tensor(scale * x + loc, dtype=DATATYPE)


class NuSampler:
    """Candidate points from nu and their multiple-importance-sampling log-weights.

    nu is a Student-t truncated to the box [nu_bounds[0], nu_bounds[1]] in every
    coordinate. The candidate set of size n_nu is the union of three blocks:
        fixed: a quadrature set of nu (a quantile grid in 1-D, an antithetic Sobol'
            sample otherwise);
        local: a ball around the observed return x with Gamma(d, delta) radius;
        far: barycentre nodes of the tail mass the 1-D grid leaves uncovered (1-D only).
    Each candidate carries the weight log p_nu - log q_mix, where q_mix is the
    balance-heuristic mixture of the blocks.

    Args:
        n_nu: Total number of candidates.
        seq_dim: Dimension d of nu.
        nu_df: Degrees of freedom.
        nu_scale: Scalar or (d,) per-coordinate scale.
        nu_loc: Scalar or (d,) location.
        nu_bounds: (lower, upper) support of every coordinate, in log returns.
        nu_ratio: Budget shares [fixed, local] or [fixed, local, far] of n_nu.
        nu_corr: (d, d) correlation matrix; None for d = 1.
        seed: Seed of the Sobol' sequences.
    """

    def __init__(self, n_nu: int, seq_dim: int, nu_df: float, nu_scale, nu_loc,
                 nu_bounds: Sequence[float], nu_ratio: Sequence[float],
                 nu_corr=None, seed: Optional[int] = None):
        self.n_nu = int(n_nu)
        self.seq_dim = int(seq_dim)
        self.nu_df = float(nu_df)
        self.nu_scale = as_nu_param(nu_scale, self.seq_dim, 'nu_scale')
        self.nu_loc = as_nu_param(nu_loc, self.seq_dim, 'nu_loc')
        self._vec = not (np.isscalar(self.nu_scale) and np.isscalar(self.nu_loc))
        if nu_corr is None and self.seq_dim != 1:
            raise ValueError(f'seq_dim={self.seq_dim} needs a correlation matrix')
        self.far = nu_corr is None
        d = self.seq_dim

        self.nu_lo, self.nu_hi = float(nu_bounds[0]), float(nu_bounds[1])
        if not np.all((self.nu_lo < self.nu_loc) & (self.nu_loc < self.nu_hi)):
            raise ValueError(f'nu_bounds {(self.nu_lo, self.nu_hi)} must bracket nu_loc {self.nu_loc}')

        self._z_lo = (self.nu_lo - self.nu_loc) / self.nu_scale
        self._z_hi = (self.nu_hi - self.nu_loc) / self.nu_scale
        _f = (lambda v: np.asarray(v, dtype=np.float64)) if self._vec else float
        self._F_lo = _f(stdtr(self.nu_df, self._z_lo))
        self._F_hi = _f(stdtr(self.nu_df, self._z_hi))
        self._trunc_mass = self._F_hi - self._F_lo
        tail_1d = 1.0 - self._trunc_mass
        if d == 1:
            self._log_Z_trunc = float(np.log(self._trunc_mass))
        else:
            bound = float(np.sum(tail_1d)) if self._vec else d * tail_1d
            if bound > 1e-3:
                raise ValueError(f'nu_bounds cut up to {bound:.3e} of nu; widen nu_bounds')
            self._log_Z_trunc = 0.0

        r = np.zeros(3)
        r[:len(nu_ratio)] = nu_ratio
        if not self.far:
            r[2] = 0.0
        self.nu_ratio = tuple(r / r.sum())
        self.n_loc, self.n_fix, self.n_far = self._split_budget()

        self.nu_points = self._build_fixed(self.n_fix, nu_corr, seed)
        self._fix_restricted = bool(self.n_far)

        corr = np.eye(d, dtype=np.float64) if nu_corr is None else (
            nu_corr.to(torch.float64).numpy() if isinstance(nu_corr, torch.Tensor)
            else np.asarray(nu_corr, np.float64))
        _, logdet_corr = np.linalg.slogdet(corr)
        corr_inv = np.linalg.inv(corr)
        if self._vec:
            inv_s = 1.0 / np.asarray(self.nu_scale, dtype=np.float64)
            self._Sinv = torch.tensor(corr_inv * np.outer(inv_s, inv_s), dtype=torch.float64)
            self._logdet_S = float(logdet_corr + 2.0 * np.sum(np.log(np.asarray(self.nu_scale))))
        else:
            self._Sinv = torch.tensor(corr_inv / self.nu_scale ** 2, dtype=torch.float64)
            self._logdet_S = float(logdet_corr + 2.0 * d * np.log(self.nu_scale))
        self._lp_const = float(
            gammaln((self.nu_df + d) / 2.0) - gammaln(self.nu_df / 2.0)
            - 0.5 * d * np.log(self.nu_df * np.pi) - 0.5 * self._logdet_S
        ) - self._log_Z_trunc
        self._log_cd = _log_sphere_area(d) + float(gammaln(d))
        self._loc_t = (torch.tensor(np.asarray(self.nu_loc, dtype=np.float64), dtype=torch.float64)
                       if self._vec else None)

        self._r_unit, self._dirs = self._build_proposal(self.n_loc, d, seed)
        self.max_r_unit = float(self._r_unit.max())

        self._assert_support(self.nu_points, 'the fixed nu block')
        if self.n_far:
            self._fix_lo, self._fix_hi, self._log_m_central = self._fixed_support()
            self.far_points, far_edges, far_off = self._build_far()
            self._far_edges = torch.tensor(far_edges, dtype=torch.float64)
            self._far_off = torch.tensor(far_off, dtype=torch.float64)
            self._assert_support(self.far_points, 'the far nu block')

        self._dev_cache = {}

    def _split_budget(self) -> Tuple[int, int, int]:
        """Convert the budget shares into integer block sizes (n_loc, n_fix, n_far)."""
        _, f_loc, f_far = self.nu_ratio
        n_far = int(round(f_far * self.n_nu))
        if n_far:
            n_far = max(2, n_far - (n_far % 2))
        n_loc = int(round(f_loc * self.n_nu))
        n_fix = self.n_nu - n_far - n_loc
        if n_fix <= 1 or n_loc < 1:
            raise ValueError(f'nu_ratio {self.nu_ratio} at n_nu={self.n_nu} leaves '
                             f'n_fix={n_fix}, n_loc={n_loc}')
        return n_loc, n_fix, n_far

    def _q_trunc(self, u):
        """Map probabilities in (0, 1) to CDF levels of the truncated law."""
        return np.asarray(u) * self._trunc_mass + self._F_lo

    def _build_fixed(self, n: int, nu_corr, seed) -> torch.Tensor:
        """Build the fixed quadrature block of shape (n, d)."""
        if nu_corr is None:
            u = np.linspace(0, 1, n + 2)[1:-1]
            z = stdtrit(self.nu_df, self._q_trunc(u)) * self.nu_scale + self.nu_loc
            return torch.tensor(z, dtype=DATATYPE).unsqueeze(-1)
        return sample_multivariate_t_qmc(n, nu_corr, self.nu_df, self.nu_scale,
                                         loc=self.nu_loc, seed=seed)

    def _build_proposal(self, n: int, d: int, seed) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build unit-scale radii and directions of the local block.

        Returns:
            Tuple of radii (n,) and unit directions (n, d).
        """
        from scipy.stats import gamma as gamma_dist
        from scipy.stats import norm, qmc
        n_base = (n + 1) // 2
        sampler = qmc.Sobol(d=d + 1, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            u = sampler.random(n_base)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        r = gamma_dist.ppf(u[:, 0], a=d, scale=1.0)
        z = norm.ppf(u[:, 1:])
        dirs = z / np.linalg.norm(z, axis=-1, keepdims=True) # unit directions on S^{d-1}
        r = np.concatenate([r, r])[:n]
        dirs = np.concatenate([dirs, -dirs])[:n] # antithetic pairs
        return (torch.tensor(r, dtype=DATATYPE), torch.tensor(dirs, dtype=DATATYPE))

    @staticmethod
    def _geom_edges(F_near: float, F_end: float, n: int, tail_frac: float) -> np.ndarray:
        """Return n + 1 CDF edges from F_near to F_end, geometric in exceedance of F_end."""
        mass = F_end - F_near
        if n < 2:
            return np.array([F_near, F_end])
        rho = tail_frac ** (1.0 / (n - 1))
        edges = F_end - mass * rho ** np.arange(n)
        return np.concatenate([edges, [F_end]])

    def _bary_z(self, z_lo: np.ndarray, z_hi: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Return the conditional mean of a standard Student-t on each bucket [z_lo, z_hi]."""
        df = self.nu_df
        C = np.exp(gammaln((df + 1) / 2.0) - gammaln(df / 2.0) - 0.5 * np.log(df * np.pi))
        G = lambda z: (1.0 + z * z / df) ** (-(df - 1.0) / 2.0)
        return -C * (df / (df - 1.0)) * (G(z_hi) - G(z_lo)) / w

    def _fixed_support(self):
        """Return the interval covered by the strata of the 1-D grid and its log mass."""
        n = self.n_fix
        u = np.array([0.5, n + 0.5]) / (n + 1.0)
        lo, hi = stdtrit(self.nu_df, self._q_trunc(u)) * self.nu_scale + self.nu_loc
        return float(lo), float(hi), float(np.log(n / (n + 1.0)))

    def _build_far(self):
        """Build the far block in 1-D.

        Returns:
            Tuple (points (n_far, 1), standardised bucket edges, log density offsets).
        """
        n_side = self.n_far // 2
        F_near_lo = float(stdtr(self.nu_df, (self._fix_lo - self.nu_loc) / self.nu_scale))
        F_near_hi = float(stdtr(self.nu_df, (self._fix_hi - self.nu_loc) / self.nu_scale))
        nodes, edges, w_all = [], [], []
        e_lo = self._geom_edges(F_near_lo, self._F_lo, n_side, FAR_TAIL_FRAC)[::-1]
        e_hi = self._geom_edges(F_near_hi, self._F_hi, n_side, FAR_TAIL_FRAC)
        for e in (e_lo, e_hi):
            w = np.diff(e) # bucket mass
            zb = stdtrit(self.nu_df, np.clip(e, 1e-300, 1.0 - 1e-16))
            nodes.append(self._bary_z(zb[:-1], zb[1:], w))
            edges.append(zb)
            w_all.append(w)
        z_edges = np.concatenate([edges[0], edges[1]])
        w = np.concatenate(w_all)
        y = np.concatenate(nodes) * self.nu_scale + self.nu_loc
        log_off = np.full(z_edges.size + 1, -np.inf)
        log_off[1:n_side + 1] = -np.log(self.n_far * w[:n_side])
        log_off[n_side + 2:n_side + 2 + n_side] = -np.log(self.n_far * w[n_side:])
        pts = torch.tensor(y, dtype=DATATYPE).reshape(-1, 1)
        return pts, z_edges, log_off

    def _assert_support(self, pts: torch.Tensor, what: str) -> None:
        """Raise if any point lies outside the declared support."""
        lo, hi = float(pts.min()), float(pts.max())
        if lo < self.nu_lo or hi > self.nu_hi:
            raise ValueError(f'{what} reaches [{lo:.5f}, {hi:.5f}], outside the nu support '
                             f'[{self.nu_lo:.5f}, {self.nu_hi:.5f}]')

    def reach(self, delta: float) -> float:
        """Return the largest distance of a local candidate from the observed return.

        Args:
            delta: Sinkhorn entropic regularisation.

        Returns:
            max radius * delta.
        """
        return self.max_r_unit * float(delta)

    def assert_in_support(self, x: torch.Tensor, delta: float) -> None:
        """Raise if a local candidate around any observed return could leave the support.

        Args:
            x: Observed daily log returns of any shape.
            delta: Sinkhorn entropic regularisation.
        """
        r = self.reach(delta)
        lo, hi = self.nu_lo + r, self.nu_hi - r
        if lo >= hi:
            raise ValueError(f'delta={delta:g} gives a local reach {r:.4f} wider than the nu support')
        xmin, xmax = float(x.min()), float(x.max())
        if xmin < lo or xmax > hi:
            raise ValueError(f'observed returns reach [{xmin:.5f}, {xmax:.5f}] but delta={delta:g} '
                             f'requires them inside [{lo:.5f}, {hi:.5f}]')

    def log_q_fixed(self, y: torch.Tensor, log_p_nu: torch.Tensor) -> torch.Tensor:
        """Return the log density of the fixed block at y."""
        if not self._fix_restricted:
            return log_p_nu
        k = y.reshape(*y.shape[:-1])
        inside = (k >= self._fix_lo) & (k <= self._fix_hi)
        return torch.where(inside, log_p_nu - self._log_m_central,
                           torch.full_like(log_p_nu, -float('inf')))

    def _far_stratum_offset(self, y: torch.Tensor) -> torch.Tensor:
        """Return -log(n_far * bucket mass) at y, and -inf outside the far buckets."""
        edges = self._on(self._far_edges, y.device, torch.float64)
        off = self._on(self._far_off, y.device, torch.float64)
        key = (y.reshape(*y.shape[:-1]).to(torch.float64) - self.nu_loc) / self.nu_scale
        return off[torch.bucketize(key.contiguous(), edges)]

    def log_p_nu(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the truncated multivariate Student-t log density.

        Args:
            y: Points of shape (..., d).

        Returns:
            float64 tensor of shape (...), -inf outside the support.
        """
        Sinv = self._on(self._Sinv, y.device, torch.float64)
        c = (y.to(torch.float64) - self._loc_on(y))
        q = torch.einsum('...i,ij,...j->...', c, Sinv, c)
        lp = self._lp_const - 0.5 * (self.nu_df + self.seq_dim) * torch.log1p(q / self.nu_df)
        inside = ((y >= self.nu_lo) & (y <= self.nu_hi)).all(dim=-1)
        return torch.where(inside, lp, torch.full_like(lp, -float('inf')))

    def _loc_on(self, y: torch.Tensor):
        """Return the location as a float, or as a float64 tensor on y's device."""
        return self.nu_loc if self._loc_t is None else self._on(self._loc_t, y.device, torch.float64)

    def _on(self, t, device, dtype=None):
        """Return t on device (and dtype), cached per tensor."""
        key = (id(t), str(device), str(dtype))
        got = self._dev_cache.get(key)
        if got is None:
            got = t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)
            self._dev_cache[key] = got
        return got

    def __call__(self, x: torch.Tensor, delta: float):
        """Build the candidate set around the observed returns.

        Args:
            x: Observed next-period daily log returns of shape (B, n_outer, d).
            delta: Sinkhorn entropic regularisation.

        Returns:
            Tuple (y, log_w_nu) with candidates y of shape (B, n_outer, n_nu, d) and
            log-weights log p_nu - log q_mix of shape (B, n_outer, n_nu).
        """
        batch_size, n_outer, d = x.shape[0], x.shape[1], self.seq_dim
        self.assert_in_support(x, delta)
        dev = x.device
        r = self._on(self._r_unit, dev) * float(delta)
        dirs = self._on(self._dirs, dev)
        y_loc = x.unsqueeze(2) + (r[:, None] * dirs).unsqueeze(0).unsqueeze(0)
        y_fix = self._on(self.nu_points, dev).unsqueeze(0).unsqueeze(0) \
                    .expand(batch_size, n_outer, -1, -1)
        blocks = [y_loc, y_fix]
        if self.n_far:
            blocks.append(self._on(self.far_points, dev).unsqueeze(0).unsqueeze(0)
                              .expand(batch_size, n_outer, -1, -1))
        y = torch.cat(blocks, dim=2)

        cost = torch.linalg.norm(y - x.unsqueeze(2), dim=-1)
        log_q1 = -cost / delta - self._log_cd - d * float(np.log(delta))
        log_p = self.log_p_nu(y)
        log_w_nu = (log_p - self._log_q_mix(y, log_p, log_q1)).to(y.dtype)
        return y, log_w_nu

    def _log_q_mix(self, y: torch.Tensor, log_p_nu: torch.Tensor,
                   log_q1: torch.Tensor) -> torch.Tensor:
        """Return the log density of the balance-heuristic mixture of the blocks."""
        terms = [log_q1.to(torch.float64) + float(np.log(self.n_loc / self.n_nu)),
                 self.log_q_fixed(y, log_p_nu) + float(np.log(self.n_fix / self.n_nu))]
        if self.n_far:
            terms.append(log_p_nu + self._far_stratum_offset(y)
                         + float(np.log(self.n_far / self.n_nu)))
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)
