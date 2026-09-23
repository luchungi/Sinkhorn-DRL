"""Train RobustDQN on the generated SPX environment and evaluate it on the simulator and real SPX data.

Settings are read from config_rdqn.py.

Usage:
    python train_rdqn_spx.py

Outputs in runs/<RDQN|DQN>_spx_<timestamp>/:
    results.csv      final-weight metrics of every seed, one row per (seed, split, policy):
                     split 'val' is EVAL_BATCH_SIZE simulated paths and 'oos' the real-SPX
                     backtest; policy 'agent' is the trained agent, and the benchmarks
                     'simulator' (val) and 'spx' (oos) hold SPX and have a blank seed
    seed<N>/run_meta.json         settings and the feasibility check
    seed<N>/<rdqn|dqn>_final.pt   final Q-network
    seed<N>/spx_oos.png           wealth, weights and drawdown of the final backtest
"""
import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch

import config_rdqn as cfg
from agent.DQN import DQN, RobustDQN
from agent.nu_sampler import NuSampler
from agent.q import QFunc
from env.common import append_results
from env.spx import MMDSimulator, load_generator, simulate_agent_spx, spx_spec, train_robustdqn
from env.stocks import ebar_fraction

EBAR_N_DRAWS = 5000
EBAR_GATE_PATHS = 64


def set_global_seeds(seed: int):
    """Seed Python, numpy and torch global RNGs.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SpxSetup:
    """Generator, noise parameters, action grid and state layout shared by all seeds."""

    def __init__(self):
        with open(cfg.MA_PARAMS_PATH) as f:
            self.ma_params = json.load(f)
        self.generator = load_generator(cfg.GENERATOR_DIR)
        self.action_values = torch.linspace(cfg.ACTION_MIN, cfg.ACTION_MAX, cfg.N_ACTIONS)
        self.spec = spx_spec()

    def make_env(self, batch_size: int, logging: bool = False) -> MMDSimulator:
        """Build a simulator.

        Args:
            batch_size: Number of paths stepped in parallel.
            logging: Record the episode for evaluation_metrics.

        Returns:
            MMDSimulator.
        """
        return MMDSimulator(self.generator, self.ma_params, self.action_values, cfg.INT_RATE,
                            cfg.TRANS_COST, batch_size, device=cfg.DEVICE, logging=logging)

    def nu_sampler(self, seed: int) -> NuSampler:
        """Build the nu candidate sampler used by the agent.

        Args:
            seed: Seed of the candidate construction.

        Returns:
            NuSampler.
        """
        return NuSampler(n_nu=cfg.N_NU, seq_dim=1, nu_df=cfg.NU_DF, nu_scale=cfg.NU_SCALE,
                         nu_loc=cfg.NU_LOC, nu_bounds=(cfg.NU_Y_MIN, cfg.NU_Y_MAX),
                         nu_ratio=cfg.NU_RATIO, nu_corr=None, seed=seed)


def build_agent(setup: SpxSetup, seed: int, env: MMDSimulator):
    """Build RobustDQN, or DQN when ROBUST is False.

    Args:
        setup: Shared setup.
        seed: Seed of Python's RNG and the nu candidates.
        env: Training environment.

    Returns:
        The agent.
    """
    qfunc = QFunc(setup.spec.dim, list(cfg.ARCHITECTURE), cfg.N_ACTIONS, spec=env.spec).to(cfg.DEVICE)
    common = dict(qfunc=qfunc, num_actions=cfg.N_ACTIONS, spec=env.spec, discount=cfg.DISCOUNT,
                  epsilon=cfg.EPS_GREEDY, buffer_size=cfg.BUFFER_MAX_LENGTH,
                  clone_steps=cfg.CLONE_STEPS, train_steps=cfg.TRAIN_STEPS,
                  batch_size=cfg.AGENT_BATCH_SIZE, n_batches=cfg.N_BATCHES,
                  n_epochs=cfg.N_EPOCHS, lr=cfg.LR, device=torch.device(cfg.DEVICE), seed=seed)
    if not cfg.ROBUST:
        return DQN(**common)
    return RobustDQN(model=env.model, action_values=setup.action_values, nu_loc=cfg.NU_LOC,
                     nu_scale=cfg.NU_SCALE, nu_df=cfg.NU_DF, nu_bounds=(cfg.NU_Y_MIN, cfg.NU_Y_MAX),
                     nu_ratio=cfg.NU_RATIO, sinkhorn_dist=cfg.EPSILON, delta=cfg.DELTA,
                     n_nu=cfg.N_NU, lamda_init=cfg.LAMDA_INIT, lamda_max_iter=cfg.LAMDA_MAX_ITER,
                     lamda_tol=cfg.LAMDA_TOL, **common)


def rollout_returns(setup: SpxSetup, seed: int) -> torch.Tensor:
    """Collect generated daily log returns of one episode holding a zero position.

    Args:
        setup: Shared setup.
        seed: Seed of the simulation.

    Returns:
        Tensor of shape (EBAR_GATE_PATHS, T, 1).
    """
    torch.manual_seed(seed)
    env = setup.make_env(EBAR_GATE_PATHS)
    env.reset()
    flat = torch.full((EBAR_GATE_PATHS,), int(setup.action_values.abs().argmin()), dtype=torch.long)
    rets = []
    while True:
        obs, _, done = env.step(flat)
        if done:
            break
        rets.append(env.spec.latest(obs).detach().clone())
    return torch.stack(rets, dim=1)


def check_feasibility(setup: SpxSetup, seeds: list) -> dict:
    """Abort unless generated returns keep nu candidates in the support and the dual is feasible.

    Args:
        setup: Shared setup.
        seeds: Seeds to be trained.

    Returns:
        Dict mapping each seed to its feasible dual fraction.
    """
    returns = rollout_returns(setup, seeds[0])
    setup.nu_sampler(seeds[0]).assert_in_support(returns, cfg.DELTA)
    fracs = {}
    for seed in seeds:
        fracs[seed] = ebar_fraction(returns, setup.nu_sampler(seed), cfg.EPSILON, cfg.DELTA,
                                    n_draws=EBAR_N_DRAWS, seed=seed)
        print(f'seed {seed}: feasible dual fraction {fracs[seed]:.4f}')
    failed = [s for s, f in fracs.items() if f < cfg.EBAR_MIN_FRAC]
    if failed:
        raise SystemExit(f'feasible dual fraction below {cfg.EBAR_MIN_FRAC} for seeds {failed}; '
                         f'increase EPSILON or decrease DELTA')
    return fracs


def write_run_meta(log_dir: str, setup: SpxSetup, seed: int, fracs: dict):
    """Write run_meta.json.

    Args:
        log_dir: Output directory.
        setup: Shared setup.
        seed: Seed of the run.
        fracs: Feasible dual fractions per seed.
    """
    meta = {'seed': seed,
            'config': {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
            'state_spec': setup.spec.to_dict(),
            'action_values': setup.action_values.tolist(),
            'feasible_dual_fraction': fracs.get(seed)}
    with open(f'{log_dir}/run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def run_once(setup: SpxSetup, seed: int, log_dir: str) -> dict:
    """Train and evaluate one seed.

    Args:
        setup: Shared setup.
        seed: Seed of the run.
        log_dir: Output directory.

    Returns:
        Nested dict {split: {policy: metrics}} of the final evaluations.
    """
    prefix = 'rdqn' if cfg.ROBUST else 'dqn'
    env = setup.make_env(cfg.ENV_BATCH_SIZE)
    agent = build_agent(setup, seed, env)
    agent = train_robustdqn(agent, env, cfg.N_EPISODES)
    torch.save(agent.q.state_dict(), f'{log_dir}/{prefix}_final.pt')

    agent.training_mode = False
    agent.epsilon = 0.
    torch.manual_seed(cfg.EVAL_SEED)
    eval_env = setup.make_env(cfg.EVAL_BATCH_SIZE, logging=True)
    obs = eval_env.reset()
    action = agent.get_action(obs)
    while True:
        obs, _, done = eval_env.step(action)
        if done:
            break
        action = agent.get_action(obs)
    val = eval_env.evaluation_metrics()

    oos = simulate_agent_spx(agent.q, setup.action_values, cfg.SPX_CSV_PATH,
                             cfg.SPX_EVAL_START_DATE, cfg.SPX_EVAL_END_DATE, cfg.INT_RATE,
                             cfg.TRANS_COST, fig_path=f'{log_dir}/spx_oos.png')
    print(f'final log return p.a.: validation {val["agent"]["log_return_pa"]:.4f}, '
          f'backtest {oos["agent"]["log_return_pa"]:.4f}')
    return {'val': val, 'oos': oos}


def main(root: Optional[str] = None):
    """Train every configured seed.

    Args:
        root: Output directory; defaults to runs/<RDQN|DQN>_spx_<timestamp>.
    """
    setup = SpxSetup()
    seeds = [cfg.SEED + i for i in range(cfg.N_SEEDS)]
    fracs = check_feasibility(setup, seeds) if cfg.ROBUST else {}
    name = 'RDQN' if cfg.ROBUST else 'DQN'
    root = root or f'runs/{name}_spx_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(root, exist_ok=True)
    results_path = f'{root}/results.csv'
    if os.path.exists(results_path):
        os.remove(results_path)
    ids = {'model': name, 'epsilon': cfg.EPSILON if cfg.ROBUST else None,
           'delta': cfg.DELTA if cfg.ROBUST else None}
    for seed in seeds:
        log_dir = f'{root}/seed{seed}'
        os.makedirs(log_dir, exist_ok=True)
        print(f'===== seed {seed} -> {log_dir} =====')
        set_global_seeds(seed)
        write_run_meta(log_dir, setup, seed, fracs)
        results = run_once(setup, seed, log_dir)
        append_results(results_path, ids, seed, results, benchmarks=seed == seeds[0])


if __name__ == '__main__':
    main()
