"""Path-bank portfolio environment, evaluation on real prices and the RTD3 training loop."""
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from env.common import PortfolioModel, drawdown_backward, perf_metrics, rf_log_return
from env.plot import overview

DATATYPE = torch.float32
DT = 1. / 252.
INVESTED_TOL = 1e-5


def load_path_bank(npz_path: str) -> dict:
    """Load a bank of generated multi-asset return paths.

    Args:
        npz_path: File with 'log_returns', 'seed_returns', 'seed_dates' and 'tickers'.

    Returns:
        Dict with 'log_returns' (n_paths, horizon, seq_dim), 'seed_returns'
        (n_paths, seed_len, seq_dim), 'tickers' and 'paths_per_seed', the number of
        consecutive paths generated from the same real seed window.
    """
    data = np.load(npz_path)
    seed_dates = data['seed_dates']
    first = seed_dates[0]
    paths_per_seed = (int(np.argmax(seed_dates != first)) if (seed_dates != first).any()
                      else len(seed_dates))
    return {'log_returns': torch.tensor(data['log_returns'], dtype=DATATYPE),
            'seed_returns': torch.tensor(data['seed_returns'], dtype=DATATYPE),
            'tickers': [str(t) for t in data['tickers']],
            'paths_per_seed': paths_per_seed}


def split_bank(n_paths: int, paths_per_seed: int, n_holdout_seeds: int, seed: int):
    """Split path indices into training and validation sets by seed window.

    Args:
        n_paths: Number of paths in the bank.
        paths_per_seed: Number of consecutive paths per seed window.
        n_holdout_seeds: Number of seed windows held out for validation.
        seed: Seed of the split.

    Returns:
        Tuple (train_indices, validation_indices) of numpy arrays.
    """
    n_seeds = n_paths // paths_per_seed
    rng = np.random.default_rng(seed)
    holdout_seeds = rng.choice(n_seeds, size=n_holdout_seeds, replace=False)
    holdout_mask = np.zeros(n_seeds, dtype=bool)
    holdout_mask[holdout_seeds] = True
    path_mask = np.repeat(holdout_mask, paths_per_seed)
    all_idx = np.arange(n_paths)
    return all_idx[~path_mask], all_idx[path_mask]


def project_action(action: torch.Tensor) -> torch.Tensor:
    """Validate fully invested long-only weights.

    Args:
        action: Weights of shape (B, seq_dim).

    Returns:
        Weights clamped to [0, 1] and rescaled so that no row sums above 1.
    """
    action = action.clamp(0., 1.)
    total = action.sum(dim=-1, keepdim=True)
    if float((total.detach() - 1.).abs().max()) > INVESTED_TOL:
        raise ValueError('portfolio weights must sum to 1')
    return action / total.clamp(min=1.)


class PathBankEnv:
    """Batched environment replaying generated return paths.

    Each reset draws paths uniformly with replacement; the state starts from the path's
    real seed window and fully in cash. An episode lasts the length of a path.

    Args:
        bank: Path bank returned by load_path_bank.
        model: PortfolioModel pricing the trades.
        path_indices: Paths that resets draw from.
        batch_size: Number of paths stepped in parallel.
        seed: Seed of the path draws.
    """

    def __init__(self, bank: dict, model: PortfolioModel, path_indices: np.ndarray,
                 batch_size: int, seed: Optional[int] = None):
        self.log_returns = bank['log_returns']
        self.seed_returns = bank['seed_returns']
        self.n_paths, self.horizon, self.seq_dim = self.log_returns.shape
        self.model = model
        self.spec = model.spec
        self.path_indices = np.asarray(path_indices)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def reset(self, paths: Optional[np.ndarray] = None) -> dict:
        """Start new episodes.

        Args:
            paths: Path indices to replay; drawn at random if None.

        Returns:
            Initial batched state dict.
        """
        self.paths = (np.asarray(paths) if paths is not None else
                      self.rng.choice(self.path_indices, size=self.batch_size, replace=True))
        idx = torch.as_tensor(self.paths, dtype=torch.long)
        window = self.seed_returns[idx].clone()[:, -self.spec.state_len:, :].clone()
        self.episode_returns = self.log_returns[idx]
        position = torch.zeros((self.batch_size, self.seq_dim), dtype=DATATYPE)
        self.curr_step = 0
        self._state = self.spec.build(window, position)
        return self._state

    def step(self, action: torch.Tensor):
        """Trade into action and advance one day.

        Args:
            action: Portfolio weights of shape (B, seq_dim).

        Returns:
            Tuple (next_state, reward (B, 1), done), where done signals that
            the path bank runs out of generated days. It ends the episode loop and is
            not a terminal-state flag: no state of this environment is absorbing.
        """
        action = project_action(action.to(DATATYPE))
        next_return = self.episode_returns[:, self.curr_step, :]
        state = self._state
        next_state = self.model.advance(state, action, next_return)
        reward = self.model.reward(state, action, next_state, strict=True)
        self._state = next_state
        self.curr_step += 1
        return next_state, reward * self.model.reward_scale, self.curr_step == self.horizon


def load_stock_returns(csv_path: str, tickers: list):
    """Load close prices and daily log returns.

    Args:
        csv_path: CSV indexed by date with one close-price column per ticker.
        tickers: Columns to load, in order.

    Returns:
        Tuple (prices, log_returns) of DataFrames.
    """
    close = pd.read_csv(csv_path, index_col=0, parse_dates=True)[list(tickers)]
    log_returns = np.log(close).diff().dropna()
    return close, log_returns


def insample_correlation(csv_path: str, tickers: list, insample_end: str) -> torch.Tensor:
    """Compute the correlation of daily log returns up to insample_end.

    Args:
        csv_path: Price CSV.
        tickers: Tickers in bank order.
        insample_end: Last date used.

    Returns:
        Correlation matrix of shape (d, d).
    """
    _, log_returns = load_stock_returns(csv_path, tickers)
    return torch.tensor(log_returns.loc[:insample_end].corr().values, dtype=DATATYPE)


def ebar_fraction(returns: torch.Tensor, sampler, epsilon: float, delta: float,
                  n_draws: int = 5000, seed: int = 0) -> float:
    """Estimate the fraction of transitions for which the Sinkhorn dual is feasible.

    Feasibility means epsilon + delta * log E_nu[exp(-||x - y|| / delta)] > 0 for a
    next-period return x drawn uniformly from returns.

    Args:
        returns: Daily log returns of shape (n_paths, T, d).
        sampler: NuSampler.
        epsilon: Sinkhorn radius.
        delta: Sinkhorn entropic regularisation.
        n_draws: Number of sampled returns.
        seed: Seed of the draws.

    Returns:
        The feasible fraction.
    """
    rng = np.random.default_rng(seed)
    p_idx = rng.integers(0, returns.shape[0], size=n_draws)
    t_idx = rng.integers(0, returns.shape[1], size=n_draws)
    r_next = returns[p_idx, t_idx]
    ebar = []
    for i in range(0, n_draws, 512):
        xb = r_next[i:i + 512].unsqueeze(1)
        y, lw = sampler(xb, delta)
        exponent = -torch.linalg.norm(y - xb.unsqueeze(2), dim=-1).squeeze(1) / delta + lw.squeeze(1)
        c = exponent.amax(dim=1, keepdim=True)
        ebar.append(epsilon + delta * (c.squeeze(1) + torch.log(torch.exp(exponent - c).mean(dim=-1))))
    return (torch.cat(ebar) > 0).float().mean().item()


def _append_csv(path: str, row: dict):
    """Append one row to a CSV file, writing the header if the file is new."""
    pd.DataFrame(row, index=[0]).to_csv(path, index=False, mode='a',
                                        header=not os.path.exists(path))


def simulate_agent_stocks(agent, tickers: list, csv_path: str, model: PortfolioModel,
                          start_date: str = '2021-07-01', benchmarks: bool = False,
                          fig_path: Optional[str] = None) -> dict:
    """Backtest the deterministic policy on real prices from start_date to the end of the data.

    Args:
        agent: Agent exposing get_action(state, deterministic=True).
        tickers: Tickers in bank order.
        csv_path: Price CSV.
        model: PortfolioModel pricing the trades.
        start_date: First trading day.
        benchmarks: Also report equal-weight rebalanced ('ew_') and equal-weight
            buy-and-hold ('bh_') metrics.
        fig_path: File for the overview figure; no figure if None.

    Returns:
        Dict with final_wealth, max_drawdown, sharpe, volatility and sortino.
    """
    prices, log_returns = load_stock_returns(csv_path, tickers)
    spec = model.spec
    seq_dim = len(tickers)
    oos_pos = log_returns.index.searchsorted(pd.Timestamp(start_date))
    end_pos = len(log_returns)
    returns_np = log_returns.values.astype(np.float32)

    def run_policy(policy: Callable):
        log_wealth = 0.
        position = torch.zeros((1, seq_dim), dtype=DATATYPE)
        log_wealth_seq = [0.]
        weight_seq = [np.zeros(seq_dim, dtype=np.float32)]
        for i in range(oos_pos, end_pos):
            window = torch.tensor(returns_np[i - spec.state_len:i], dtype=DATATYPE).unsqueeze(0)
            state = spec.build(window, position)
            weights = project_action(policy(state))
            next_return = torch.tensor(returns_np[i], dtype=DATATYPE).reshape(1, -1)
            next_state = model.advance(state, weights, next_return)
            log_wealth += float(model.reward(state, weights, next_state).squeeze())
            position = next_state['weight']
            log_wealth_seq.append(log_wealth)
            weight_seq.append(weights.numpy().squeeze(0))
        return np.array(log_wealth_seq), np.array(weight_seq)

    df = pd.DataFrame(index=prices.index[oos_pos:end_pos + 1])
    log_wealth, weights = run_policy(lambda s: agent.get_action(s, deterministic=True))
    series = {'agent': log_wealth}
    df['agent'] = np.exp(log_wealth)
    if benchmarks:
        ew = torch.full((1, seq_dim), 1. / seq_dim, dtype=DATATYPE)
        series['ew_rebalanced'], _ = run_policy(lambda s: ew)
        df['ew_rebalanced'] = np.exp(series['ew_rebalanced'])
        bh_wealth = (prices.iloc[oos_pos:end_pos + 1] / prices.iloc[oos_pos]).mean(axis=1).values
        series['ew_buy_hold'] = np.log(bh_wealth)
        df['ew_buy_hold'] = bh_wealth
    for name in series:
        df[f'{name}_mdd'] = drawdown_backward(df[name])

    if fig_path is not None:
        for i, ticker in enumerate(tickers):
            df[ticker] = weights[:, i]
        fig, _ = overview(df, list(series), list(tickers), [f'{n}_mdd' for n in series])
        fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
        plt.close(fig)

    def metrics(name: str) -> dict:
        excess = np.diff(series[name]) - rf_log_return(model.int_rate, DT)
        vol, sharpe, _, sortino = perf_metrics(excess)
        return {'final_wealth': float(df[name].iloc[-1]),
                'max_drawdown': float(df[f'{name}_mdd'].min()),
                'sharpe': float(sharpe), 'volatility': float(vol), 'sortino': float(sortino)}

    result = metrics('agent')
    if benchmarks:
        for prefix, name in (('ew', 'ew_rebalanced'), ('bh', 'ew_buy_hold')):
            result.update({f'{prefix}_{k}': v for k, v in metrics(name).items()})
    return result


def bank_ew_rebalanced_policy(seq_dim: int) -> Callable:
    """Return a policy holding equal weights, rebalanced every day.

    Args:
        seq_dim: Number of assets.

    Returns:
        Callable mapping a batched state dict to weights (B, seq_dim).
    """
    def policy(states: dict) -> torch.Tensor:
        return torch.full((states['weight'].shape[0], seq_dim), 1. / seq_dim, dtype=DATATYPE)
    return policy


def bank_buy_hold_policy(seq_dim: int) -> Callable:
    """Return a policy buying equal weights on the first day and holding them.

    Args:
        seq_dim: Number of assets.

    Returns:
        Callable mapping a batched state dict to weights (B, seq_dim).
    """
    def policy(states: dict) -> torch.Tensor:
        w = states['weight']
        held = w.sum(dim=-1, keepdim=True) > 0
        return torch.where(held, w, torch.full_like(w, 1. / seq_dim))
    return policy


def evaluate_on_bank(policy_fn: Callable, bank: dict, path_indices: np.ndarray,
                     model: PortfolioModel) -> dict:
    """Run a deterministic policy once over every given bank path.

    Args:
        policy_fn: Maps a batched state dict to weights (B, seq_dim).
        bank: Path bank.
        path_indices: Paths to evaluate.
        model: PortfolioModel pricing the trades.

    Returns:
        Dict of path-averaged final wealth, final log wealth, Sharpe ratio, Sortino
        ratio and volatility.
    """
    env = PathBankEnv(bank, model, path_indices, batch_size=len(path_indices), seed=0)
    states = env.reset(paths=path_indices)
    rewards_seq = []
    done = False
    while not done:
        with torch.no_grad():
            actions = policy_fn(states)
        states, rewards, done = env.step(actions)
        rewards_seq.append(rewards)
    rewards = torch.cat(rewards_seq, dim=1) / model.reward_scale
    final_log_wealth = rewards.sum(dim=1)
    excess = rewards.detach().cpu().numpy() - rf_log_return(model.int_rate, DT)
    vol, sharpe, _, sortino = perf_metrics(excess, axis=1)
    return {'val_mean_final_wealth': final_log_wealth.exp().mean().item(),
            'val_mean_final_log_wealth': final_log_wealth.mean().item(),
            'val_mean_sharpe': float(sharpe.mean()),
            'val_mean_sortino': float(sortino.mean()),
            'val_mean_vol': float(vol.mean())}


def train_rtd3_bank(agent, env: PathBankEnv, n_episodes: int, log_dir: str, tickers: list,
                    csv_path: str, val_bank: dict, val_indices: np.ndarray,
                    val_every: int, oos_every: int):
    """Train an agent on the path bank with periodic validation and real-data backtests.

    Every oos_every episodes a backtest row is appended to oos_monitor.csv. Every
    val_every episodes a validation row is appended to val.csv and the agent is saved
    to rtd3_<episode>.pt.

    Args:
        agent: RTD3 agent.
        env: Training environment.
        n_episodes: Number of episodes.
        log_dir: Output directory.
        tickers: Tickers in bank order.
        csv_path: Price CSV for the backtests.
        val_bank: Path bank used for validation.
        val_indices: Validation path indices.
        val_every: Episodes between validations.
        oos_every: Episodes between backtests.

    Returns:
        The trained agent.
    """
    pbar = tqdm(range(n_episodes), desc='RTD3', unit='ep')
    last_val_wealth = float('nan')
    for episode in pbar:
        cum_rewards = torch.zeros(env.batch_size, 1)
        obs = env.reset()
        action = agent.agent_start(obs)
        while True:
            obs, rewards, done = env.step(action)
            cum_rewards += rewards
            if done:
                agent.agent_end(rewards, obs)
                break
            action = agent.agent_step(rewards, obs)

        ep = episode + 1
        oos = None
        if ep % oos_every == 0:
            oos = simulate_agent_stocks(agent, tickers, csv_path, env.model)
            oos['episodes_trained'] = ep
            _append_csv(f'{log_dir}/oos_monitor.csv', oos)
        if ep % val_every == 0:
            val = evaluate_on_bank(lambda s: agent.get_action(s, deterministic=True),
                                   val_bank, val_indices, env.model)
            val['episodes_trained'] = ep
            _append_csv(f'{log_dir}/val.csv', val)
            agent.save_agent(f'{log_dir}/rtd3_{ep}.pt')
            last_val_wealth = val['val_mean_final_wealth']
            msg = f'[ep {ep}] val_wealth={last_val_wealth:.4f} val_sortino={val["val_mean_sortino"]:.3f}'
            if oos is not None:
                msg += f' oos_wealth={oos["final_wealth"]:.3f}'
            tqdm.write(msg)
        pbar.set_postfix(cum_r=f'{cum_rewards.mean():.2f}', val_w=f'{last_val_wealth:.4f}')
    return agent
