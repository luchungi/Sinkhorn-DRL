"""Maximum-likelihood fit of a multivariate Student-t by ECME."""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, stdtr

DF_FLOOR = 2.1
DF_CEIL = 200.0
FIT_SEED = 0


@dataclass
class NuFit:
    """Fitted multivariate Student-t parameters.

    Attributes:
        loc: Common scalar location.
        scale: Per-coordinate scale of shape (d,).
        corr: Correlation matrix of shape (d, d).
        df: Degrees of freedom.
        loglik: Log-likelihood at the fit.
        n_obs: Number of observations.
    """
    loc: float
    scale: np.ndarray
    corr: np.ndarray
    df: float
    loglik: float
    n_obs: int


def loglik(X: np.ndarray, mu: np.ndarray, S: np.ndarray, df: float) -> float:
    """Compute the log-likelihood of a multivariate Student-t.

    Args:
        X: Observations of shape (n, d).
        mu: Location of shape (d,).
        S: Scatter matrix of shape (d, d).
        df: Degrees of freedom.

    Returns:
        The log-likelihood.
    """
    n, d = X.shape
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T).T
    delta = np.einsum('ni,ni->n', z, z)
    const = (gammaln((df + d) / 2.0) - gammaln(df / 2.0)
             - 0.5 * d * np.log(df * np.pi) - np.log(np.diag(L)).sum())
    return float(n * const - 0.5 * (df + d) * np.log1p(delta / df).sum())


def _df_step(u: np.ndarray, df_old: float, d: int, floor: float) -> float:
    """Update the degrees of freedom by solving the score equation with Brent's method."""
    c = 1.0 + float(np.mean(np.log(u) - u)) + digamma((df_old + d) / 2.0) - np.log((df_old + d) / 2.0)
    f = lambda v: -digamma(v / 2.0) + np.log(v / 2.0) + c
    if f(floor) < 0.0:
        return floor
    if f(DF_CEIL) > 0.0:
        return DF_CEIL
    return float(brentq(f, floor, DF_CEIL, xtol=1e-10, rtol=1e-12))


def _ecme(X: np.ndarray, mu: np.ndarray, S: np.ndarray, df: float, fit_loc: bool,
          floor: float, max_iter: int, tol: float):
    """Run ECME from one initialisation.

    Returns:
        Tuple (mu, S, df, loglik).
    """
    n, d = X.shape
    one = np.ones(d)
    prev = -np.inf
    for _ in range(max_iter):
        L = np.linalg.cholesky(S)
        z = np.linalg.solve(L, (X - mu).T).T
        delta = np.einsum('ni,ni->n', z, z)
        u = (df + d) / (df + delta)

        if fit_loc:
            Sinv = np.linalg.inv(S)
            xbar = (u[:, None] * X).sum(0) / u.sum()
            mu = np.full(d, float(one @ Sinv @ xbar) / float(one @ Sinv @ one))

        Xc = X - mu
        S = (Xc * u[:, None]).T @ Xc / n
        df = _df_step(u, df, d, floor)

        cur = loglik(X, mu, S, df)
        if abs(cur - prev) < tol * abs(cur):
            prev = cur
            break
        prev = cur
    return mu, S, df, prev


def fit_nu_t(X: np.ndarray, loc: Optional[float], n_init: int = 4, df_init: float = 4.0,
             scale_init: Optional[float] = None, df_floor: float = DF_FLOOR,
             seed: int = FIT_SEED, max_iter: int = 2000, tol: float = 1e-13) -> NuFit:
    """Fit the scale, correlation and degrees of freedom of a multivariate Student-t.

    The location is a common scalar: fitted by generalised least squares if loc is
    None, otherwise held at loc. The best of n_init initialisations is returned.

    Args:
        X: Observations of shape (n, d).
        loc: Fixed common location, or None to fit it.
        n_init: Number of initialisations.
        df_init: Degrees of freedom of the first initialisation.
        scale_init: Scale of the first initialisation; None uses the sample scale.
        df_floor: Lower bound on the degrees of freedom.
        seed: Seed of the randomised initialisations.
        max_iter: Maximum ECME iterations per initialisation.
        tol: Relative log-likelihood tolerance.

    Returns:
        NuFit with the fitted parameters.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    n_init = max(1, int(n_init))
    fit_loc = loc is None
    rng = np.random.default_rng(seed)
    S_mom = np.cov(X.T)
    if scale_init is not None:
        S_mom = S_mom / np.sqrt(np.outer(np.diag(S_mom), np.diag(S_mom))) * float(scale_init) ** 2

    best = None
    for k in range(n_init):
        if k == 0:
            mu0 = X.mean(0) if fit_loc else np.full(d, float(loc))
            S0, df0 = S_mom.copy(), float(df_init)
        else:
            idx = rng.choice(n, size=max(d + 2, n // 2), replace=False)
            sub = np.cov(X[idx].T) * rng.uniform(0.5, 2.0)
            mu0 = X[idx].mean(0) if fit_loc else np.full(d, float(loc))
            S0, df0 = sub, float(rng.uniform(2.5, 20.0))
        try:
            mu, S, df, ll = _ecme(X, mu0, S0, df0, fit_loc, df_floor, max_iter, tol)
        except np.linalg.LinAlgError:
            continue
        if best is None or ll > best[3]:
            best = (mu, S, df, ll)
    if best is None:
        raise RuntimeError(f'every nu-fit initialisation failed on data of shape {X.shape}')

    mu, S, df, ll = best
    if df <= df_floor * (1.0 + 1e-9):
        raise ValueError(f'the fit converged to df={df:.4f}, at the floor {df_floor}')
    scale = np.sqrt(np.diag(S))
    corr = S / np.outer(scale, scale)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return NuFit(loc=float(mu[0]) if fit_loc else float(loc), scale=scale, corr=corr,
                 df=float(df), loglik=float(ll), n_obs=n)


def assert_box_negligible(fit: NuFit, nu_lo: float, nu_hi: float, tol: float = 1e-3) -> float:
    """Check that the fitted law puts negligible mass outside the support box.

    Args:
        fit: Fitted parameters.
        nu_lo: Lower bound of every coordinate.
        nu_hi: Upper bound of every coordinate.
        tol: Maximum allowed union bound on the mass outside the box.

    Returns:
        The union bound on the mass outside the box.
    """
    z_lo = (nu_lo - fit.loc) / fit.scale
    z_hi = (nu_hi - fit.loc) / fit.scale
    tail = float(np.sum(1.0 - (stdtr(fit.df, z_hi) - stdtr(fit.df, z_lo))))
    if tail > tol:
        raise ValueError(f'the fitted nu puts {tail:.3e} of its mass outside [{nu_lo}, {nu_hi}]')
    return tail


def fit_nu_stocks(csv_path: str, tickers: Sequence[str], insample_end: str,
                  loc: Optional[float], n_init: int = 4, df_init: float = 4.0,
                  scale_init: Optional[float] = None) -> NuFit:
    """Fit nu to in-sample daily log returns of the given tickers.

    Args:
        csv_path: Price CSV with one close-price column per ticker.
        tickers: Tickers in bank order.
        insample_end: Last date of the fitting window.
        loc: Fixed common location, or None to fit it.
        n_init: Number of initialisations.
        df_init: Degrees of freedom of the first initialisation.
        scale_init: Scale of the first initialisation.

    Returns:
        NuFit with the fitted parameters.
    """
    from env.stocks import load_stock_returns
    _, log_returns = load_stock_returns(csv_path, list(tickers))
    X = log_returns.loc[:insample_end].values
    return fit_nu_t(X, loc=loc, n_init=n_init, df_init=df_init, scale_init=scale_init)
