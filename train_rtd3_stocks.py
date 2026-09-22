"""Train RTD3 on the five-stock path bank and evaluate it on held-out paths and real prices.

Settings are read from config_rtd3.py.

Usage:
    python train_rtd3_stocks.py

Outputs per seed in runs/<RTD3|TD3>_stocks_<timestamp>/seed<N>/:
    run_meta.json    settings, fitted nu parameters and the feasibility check
    val.csv          validation metrics every VAL_EVERY episodes
    oos_monitor.csv  real-price backtest metrics every OOS_EVERY episodes
    rtd3_<ep>.pt     agent every VAL_EVERY episodes, and rtd3_final.pt
    oos_final.csv    final real-price backtest with benchmarks
    val_final.csv    final validation with benchmarks
    rtd3_final.png   wealth, weights and drawdown of the final backtest
"""
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch

import config_rtd3 as cfg
from agent.nu_fit import assert_box_negligible, fit_nu_stocks
from agent.nu_sampler import NuSampler
from agent.RTD3 import RTD3
from env.common import PortfolioModel
from env.features import StateSpec
from env.stocks import (PathBankEnv, bank_buy_hold_policy, bank_ew_rebalanced_policy,
                        ebar_fraction, evaluate_on_bank, insample_correlation, load_path_bank,
                        simulate_agent_stocks, split_bank, train_rtd3_bank)

INSAMPLE_END = '2021-06-30'
VAL_NAMES = {'val_mean_final_wealth': 'final_wealth',
             'val_mean_final_log_wealth': 'final_log_wealth',
             'val_mean_sharpe': 'sharpe',
             'val_mean_vol': 'volatility',
             'val_mean_sortino': 'sortino'}


def set_global_seeds(seed: int):
    """Seed Python, numpy and torch global RNGs.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rename_val(val: dict) -> dict:
    """Rename evaluate_on_bank outputs to the backtest metric names.

    Args:
        val: Output of evaluate_on_bank.

    Returns:
        Dict with keys final_wealth, final_log_wealth, sharpe, volatility, sortino.
    """
    return {name: val[key] for key, name in VAL_NAMES.items()}


class StocksSetup:
    """Path bank, state layout, portfolio model, data split and nu parameters shared by all seeds."""

    def __init__(self):
        self.bank = load_path_bank(cfg.BANK_PATH)
        self.n_paths, self.horizon, self.seq_dim = self.bank['log_returns'].shape
        self.tickers = self.bank['tickers']
        self.spec = StateSpec(self.seq_dim, cfg.STATE_LEN, ret_gain=cfg.RET_GAIN)
        self.model = PortfolioModel(self.spec, int_rate=cfg.INT_RATE, trans_cost=cfg.TRANS_COST,
                                    reward_scale=cfg.REWARD_SCALE)
        self.train_idx, self.val_idx = split_bank(self.n_paths, self.bank['paths_per_seed'],
                                                  cfg.N_HOLDOUT_SEEDS, cfg.HOLDOUT_SEED)
        if cfg.NU_FIT:
            fit = fit_nu_stocks(cfg.CSV_PATH, self.tickers, INSAMPLE_END, loc=cfg.NU_LOC,
                                n_init=cfg.NU_FIT_N_INIT, df_init=cfg.NU_DF,
                                scale_init=cfg.NU_SCALE)
            assert_box_negligible(fit, cfg.NU_Y_MIN, cfg.NU_Y_MAX)
            self.nu_loc, self.nu_scale, self.nu_df, self.corr = fit.loc, fit.scale, fit.df, fit.corr
        else:
            self.nu_loc, self.nu_scale, self.nu_df = cfg.NU_LOC, cfg.NU_SCALE, cfg.NU_DF
            self.corr = insample_correlation(cfg.CSV_PATH, self.tickers, INSAMPLE_END)
        print(f'Bank: {self.n_paths} paths ({len(self.train_idx)} train, {len(self.val_idx)} val), '
              f'tickers {", ".join(self.tickers)}')
        print(f'nu: loc {self.nu_loc:.6g}, scale {np.round(np.asarray(self.nu_scale), 6)}, '
              f'df {self.nu_df:.4f}')

    def make_env(self, seed: int) -> PathBankEnv:
        """Build the training environment.

        Args:
            seed: Seed of the path draws.

        Returns:
            PathBankEnv over the training paths.
        """
        return PathBankEnv(self.bank, self.model, self.train_idx, cfg.ENV_BATCH_SIZE, seed=seed)

    def nu_sampler(self, seed: int) -> NuSampler:
        """Build the nu candidate sampler used by the agent.

        Args:
            seed: Seed of the candidate construction.

        Returns:
            NuSampler.
        """
        return NuSampler(n_nu=cfg.N_NU, seq_dim=self.seq_dim, nu_df=self.nu_df,
                         nu_scale=self.nu_scale, nu_loc=self.nu_loc,
                         nu_bounds=(cfg.NU_Y_MIN, cfg.NU_Y_MAX), nu_ratio=cfg.NU_RATIO,
                         nu_corr=self.corr, seed=seed)

    def val_benchmarks(self) -> dict:
        """Evaluate the equal-weight rebalanced and buy-and-hold policies on the validation paths.

        Returns:
            Dict of 'ew_' and 'bh_' prefixed metrics.
        """
        out = {}
        for prefix, policy in (('ew', bank_ew_rebalanced_policy(self.seq_dim)),
                               ('bh', bank_buy_hold_policy(self.seq_dim))):
            m = rename_val(evaluate_on_bank(policy, self.bank, self.val_idx, self.model))
            out.update({f'{prefix}_{"log_wealth" if k == "final_log_wealth" else k}': v
                        for k, v in m.items()})
        return out


def build_agent(setup: StocksSetup, seed: int) -> RTD3:
    """Build the RTD3 agent.

    Args:
        setup: Shared setup.
        seed: Seed of the networks and nu candidates.

    Returns:
        RTD3 agent.
    """
    return RTD3(obs_dim=setup.spec.dim, action_dim=setup.seq_dim, model=setup.model,
                discount=cfg.DISCOUNT, nu_loc=setup.nu_loc, nu_scale=setup.nu_scale,
                nu_df=setup.nu_df, nu_corr=setup.corr, nu_bounds=(cfg.NU_Y_MIN, cfg.NU_Y_MAX),
                nu_ratio=cfg.NU_RATIO, epsilon=cfg.EPSILON, delta=cfg.DELTA, n_nu=cfg.N_NU,
                lamda_init=cfg.LAMDA_INIT, lamda_max_iter=cfg.LAMDA_MAX_ITER,
                lamda_tol=cfg.LAMDA_TOL, hidden_sizes=list(cfg.HIDDEN),
                activation=cfg.ACTIVATION, actor_lr=cfg.ACTOR_LR, critic_lr=cfg.CRITIC_LR,
                buffer_size=cfg.BUFFER_SIZE, batch_size=cfg.AGENT_BATCH_SIZE,
                train_steps=cfg.TRAIN_STEPS, policy_delay=cfg.POLICY_DELAY,
                polyak_tau=cfg.POLYAK_TAU, entropy_beta=cfg.ENTROPY_BETA,
                expl_sigma=cfg.EXPL_SIGMA, expl_clip=cfg.EXPL_CLIP, expl_rho=cfg.EXPL_RHO,
                expl_end_sigma=cfg.EXPL_END_SIGMA, expl_sigma_steps=cfg.EXPL_SIGMA_STEPS,
                robust=cfg.ROBUST, device=torch.device(cfg.DEVICE), seed=seed)


def check_feasibility(setup: StocksSetup, seeds: list) -> dict:
    """Abort unless the Sinkhorn dual is feasible for enough bank transitions for every seed.

    Args:
        setup: Shared setup.
        seeds: Seeds to be trained.

    Returns:
        Dict mapping each seed to its feasible fraction.
    """
    fracs = {}
    for seed in seeds:
        fracs[seed] = ebar_fraction(setup.bank['log_returns'], setup.nu_sampler(seed),
                                    cfg.EPSILON, cfg.DELTA, seed=seed)
        print(f'seed {seed}: feasible dual fraction {fracs[seed]:.4f}')
    failed = [s for s, f in fracs.items() if f < cfg.EBAR_MIN_FRAC]
    if failed:
        raise SystemExit(f'feasible dual fraction below {cfg.EBAR_MIN_FRAC} for seeds {failed}; '
                         f'increase EPSILON or decrease DELTA')
    return fracs


def write_run_meta(log_dir: str, setup: StocksSetup, seed: int, fracs: dict):
    """Write run_meta.json.

    Args:
        log_dir: Output directory.
        setup: Shared setup.
        seed: Seed of the run.
        fracs: Feasible dual fractions per seed.
    """
    meta = {'seed': seed,
            'config': {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
            'tickers': setup.tickers,
            'state_spec': setup.spec.to_dict(),
            'nu_loc': setup.nu_loc,
            'nu_scale': np.asarray(setup.nu_scale).tolist(),
            'nu_df': setup.nu_df,
            'nu_corr': np.asarray(setup.corr, dtype=np.float64).tolist(),
            'feasible_dual_fraction': fracs.get(seed)}
    with open(f'{log_dir}/run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def run_once(setup: StocksSetup, seed: int, log_dir: str):
    """Train and evaluate one seed.

    Args:
        setup: Shared setup.
        seed: Seed of the run.
        log_dir: Output directory.
    """
    agent = build_agent(setup, seed)
    env = setup.make_env(seed)
    agent = train_rtd3_bank(agent, env, cfg.N_EPISODES, log_dir, setup.tickers, cfg.CSV_PATH,
                            setup.bank, setup.val_idx, cfg.VAL_EVERY, cfg.OOS_EVERY)
    agent.save_agent(f'{log_dir}/rtd3_final.pt')
    agent.training_mode = False

    oos = simulate_agent_stocks(agent, setup.tickers, cfg.CSV_PATH, setup.model, benchmarks=True,
                                fig_path=f'{log_dir}/rtd3_final.png')
    pd.DataFrame(oos, index=[0]).to_csv(f'{log_dir}/oos_final.csv', index=False)
    val = rename_val(evaluate_on_bank(lambda s: agent.get_action(s, deterministic=True),
                                      setup.bank, setup.val_idx, setup.model))
    val.update(setup.val_benchmarks())
    pd.DataFrame(val, index=[0]).to_csv(f'{log_dir}/val_final.csv', index=False)
    print(f'final backtest wealth {oos["final_wealth"]:.4f}, '
          f'final validation wealth {val["final_wealth"]:.4f}')


def main(root: Optional[str] = None):
    """Train every configured seed.

    Args:
        root: Output directory; defaults to runs/<RTD3|TD3>_stocks_<timestamp>.
    """
    setup = StocksSetup()
    seeds = [cfg.SEED + i for i in range(cfg.N_SEEDS)]
    fracs = check_feasibility(setup, seeds) if cfg.ROBUST else {}
    name = 'RTD3' if cfg.ROBUST else 'TD3'
    root = root or f'runs/{name}_stocks_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    for seed in seeds:
        log_dir = f'{root}/seed{seed}'
        os.makedirs(log_dir, exist_ok=True)
        print(f'===== seed {seed} -> {log_dir} =====')
        set_global_seeds(seed)
        write_run_meta(log_dir, setup, seed, fracs)
        run_once(setup, seed, log_dir)


if __name__ == '__main__':
    main()
