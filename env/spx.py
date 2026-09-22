"""SPX environment driven by a pretrained LSTM generator, real-SPX backtest and the DQN training loop."""
import json
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
from tqdm import tqdm

from env.common import PortfolioModel, drawdown_backward, perf_metrics, rf_log_return
from env.features import ANN, StateSpec
from env.plot import overview

DATATYPE = torch.float32
TRADING_CALENDAR = 'NYSE'
CAL_START_DATE = '1995-01-01'
CAL_END_DATE = '2024-12-31'
BURN_IN = 500
STATE_LEN = 60


def spx_spec() -> StateSpec:
    """Return the state layout of the SPX environment.

    Returns:
        StateSpec with 60 daily log returns, the position and dt.
    """
    return StateSpec(seq_dim=1, state_len=STATE_LEN, ret_gain=1.0 / ANN, weight_centre=0.0,
                     include_dt=True)


class GenLSTM(nn.Module):
    """LSTM generator of daily log returns driven by noise and time increments.

    Args:
        noise_dim: Dimension of the noise input.
        seq_dim: Number of assets.
        hidden_size: LSTM hidden size.
        n_lstm_layers: Number of LSTM layers.
    """

    def __init__(self, noise_dim: int, seq_dim: int, hidden_size: int, n_lstm_layers: int):
        super().__init__()
        self.seq_dim = seq_dim
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.rnn = nn.LSTM(input_size=seq_dim + noise_dim + 1, hidden_size=hidden_size,
                           num_layers=n_lstm_layers, batch_first=True, bidirectional=False)
        self.output_net = nn.Linear(hidden_size, seq_dim)

    def forward(self, noise: torch.Tensor, dts: torch.Tensor, h: Optional[torch.Tensor] = None,
                c: Optional[torch.Tensor] = None, last_return: Optional[torch.Tensor] = None):
        """Generate log returns one step at a time.

        Starts from a zero state when h, c and last_return are None.

        Args:
            noise: Noise of shape (B, n, noise_dim).
            dts: Time increments of shape (B, n, 1).
            h: LSTM hidden state of shape (n_lstm_layers, B, hidden_size).
            c: LSTM cell state of shape (n_lstm_layers, B, hidden_size).
            last_return: Previous log return of shape (B, 1, seq_dim).

        Returns:
            Tuple (log returns (B, n, seq_dim), h, c).
        """
        if h is None:
            batch_size = noise.shape[0]
            last_return = torch.zeros(batch_size, 1, self.seq_dim, device=noise.device)
            h = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, device=noise.device)
            c = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_size, device=noise.device)
        gen_seq = []
        for i in range(noise.shape[1]):
            x = torch.cat([last_return, noise[:, i:i + 1, :], dts[:, i:i + 1, :]], dim=-1)
            output, (h, c) = self.rnn(x, (h, c))
            last_return = self.output_net(output)
            gen_seq.append(last_return)
        generated = torch.cat(gen_seq, dim=1) if len(gen_seq) > 1 else gen_seq[0]
        return generated, h, c


def load_generator(generator_dir: str) -> GenLSTM:
    """Build the generator from generator_params.json and load generator.pt.

    Args:
        generator_dir: Directory holding both files.

    Returns:
        The loaded GenLSTM.
    """
    with open(os.path.join(generator_dir, 'generator_params.json')) as f:
        params = json.load(f)
    generator = GenLSTM(**params)
    generator.load_state_dict(torch.load(os.path.join(generator_dir, 'generator.pt'),
                                         map_location='cpu', weights_only=True))
    return generator


class MMDSimulator:
    """Batched SPX trading environment on generated returns over the NYSE calendar.

    Each episode generates BURN_IN + STATE_LEN warm-up days and then trades every
    remaining session between CAL_START_DATE and CAL_END_DATE.

    Args:
        generator: Pretrained GenLSTM.
        ma_params: Dict with 'omega' and 'alpha' of the moving-average noise scale.
        action_values: Portfolio weight of each discrete action.
        int_rate: Continuously compounded annual interest rate on cash.
        trans_cost: Proportional transaction cost.
        batch_size: Number of paths stepped in parallel.
        device: Torch device.
        logging: Record actions, rewards and market returns for evaluation_metrics.
    """

    def __init__(self, generator: GenLSTM, ma_params: dict, action_values: torch.Tensor,
                 int_rate: float, trans_cost: float, batch_size: int, device: str = 'cpu',
                 logging: bool = False):
        self.generator = generator.to(device)
        self.noise_dim = generator.noise_dim
        self.seq_dim = generator.seq_dim
        self.bias = torch.tensor(ma_params['omega'], dtype=DATATYPE)
        self.lags = torch.tensor(ma_params['alpha'], dtype=DATATYPE).flip(0).unsqueeze(-1)
        self.ma_p = len(self.lags)
        self.action_values = action_values
        self.batch_size = batch_size
        self.device = device
        self.logging = logging
        self.spec = spx_spec()
        self.model = PortfolioModel(self.spec, int_rate=int_rate, trans_cost=trans_cost)

        schedule = mcal.get_calendar(TRADING_CALENDAR).schedule(start_date=CAL_START_DATE,
                                                                end_date=CAL_END_DATE)
        t = np.zeros(len(schedule))
        t[1:] = (schedule.index.to_series().diff()[1:].dt.days / 365).values.cumsum()
        self.t = torch.tensor(t, dtype=DATATYPE, device=self.device)
        self.dts = self.t.diff(dim=0)
        self.total_steps = len(self.dts)

    def reset(self) -> dict:
        """Generate the warm-up days and return the initial state.

        Returns:
            Batched state dict.
        """
        position = torch.zeros((self.batch_size, self.seq_dim), dtype=DATATYPE)
        self.curr_step = BURN_IN + STATE_LEN
        self.ma_noise = self.generate_ma_noise(self.batch_size, self.total_steps)
        noise = self.ma_noise[:, :self.curr_step, :]
        dts = self.dts[:self.curr_step].expand(self.batch_size, -1).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            seq, self.h, self.c = self.generator(noise, dts)
        window = seq[:, BURN_IN:].clone()
        self._state = self.spec.build(window, position, dt=self._dt(self.curr_step))
        if self.logging:
            self.episode_rewards = []
            self.episode_log_returns = []
        return self._state

    def step(self, action: torch.Tensor):
        """Trade into the weights of the chosen actions and generate the next day.

        Args:
            action: Action indices with B elements.

        Returns:
            Tuple (next_state, reward (B, 1), done), where done signals that
            the simulated calendar runs out. It ends the episode loop and is
            not a terminal-state flag: no state of this environment is absorbing.
        """
        action = self.action_values[action.reshape(-1).to(torch.long)].reshape(
            self.batch_size, self.seq_dim).to(DATATYPE)
        noise = self.ma_noise[:, self.curr_step:self.curr_step + 1, :]
        dts = self.dts[self.curr_step:self.curr_step + 1].expand(self.batch_size, -1).unsqueeze(-1).to(self.device)
        last_return = self.spec.latest(self._state).unsqueeze(1)
        with torch.no_grad():
            generated, self.h, self.c = self.generator(noise, dts, h=self.h, c=self.c,
                                                       last_return=last_return)
        next_return = generated[:, 0, :]

        state = self._state
        next_state = self.model.advance(state, action, next_return)
        reward = self.model.reward(state, action, next_state, strict=True)
        self.curr_step += 1
        next_state['dt'] = self._dt(self.curr_step)
        self._state = next_state
        if any(v.isnan().any() for v in next_state.values()):
            raise ValueError('next state contains NaN values')

        if self.logging:
            self.episode_rewards.append(reward)
            self.episode_log_returns.append(self.spec.latest(next_state))
        return next_state, reward, self.curr_step == self.total_steps

    def _dt(self, step: int) -> torch.Tensor:
        """Return the length of the holding period starting at step, shape (B, 1)."""
        return self.dts[min(step, len(self.dts) - 1)].repeat(self.batch_size, 1)

    def generate_ma_noise(self, batch_size: int, length: int) -> torch.Tensor:
        """Generate generator noise with a moving-average conditional scale.

        Args:
            batch_size: Number of paths.
            length: Number of days.

        Returns:
            Noise of shape (batch_size, length, noise_dim).
        """
        seq = torch.randn(batch_size, self.noise_dim, self.ma_p, dtype=DATATYPE)
        noise = []
        for _ in range(length):
            sigma = (seq ** 2 @ self.lags.expand(batch_size, -1, 1) + self.bias).sqrt()
            noise.append(sigma * torch.randn_like(sigma))
            seq = seq.roll(-1, dims=2)
            seq[:, :, -1:] = noise[-1]
        return torch.cat(noise, dim=2).permute(0, 2, 1)

    def evaluation_metrics(self) -> dict:
        """Summarise a logged episode for the agent ('eval_') and for holding SPX ('env_').

        Returns:
            Dict of path-averaged final log wealth, final wealth, volatility, Sharpe
            ratio, downside deviation and Sortino ratio.
        """
        rewards = torch.cat(self.episode_rewards, dim=1)
        market = torch.cat(self.episode_log_returns, dim=1)
        step0 = BURN_IN + STATE_LEN
        dt = self.dts[step0:step0 + rewards.shape[1]].detach().cpu().numpy()
        rf = rf_log_return(self.model.int_rate, dt)
        out = {}
        for prefix, log_returns in (('eval', rewards), ('env', market)):
            final_log_wealth = log_returns.sum(dim=1)
            vol, sharpe, downside, sortino = perf_metrics(log_returns.detach().cpu().numpy() - rf, axis=1)
            out.update({f'{prefix}_mean_final_log_wealth': final_log_wealth.mean(dim=0).item(),
                        f'{prefix}_mean_final_wealth': final_log_wealth.exp().mean(dim=0).item(),
                        f'{prefix}_mean_return_vol': float(vol.mean()),
                        f'{prefix}_mean_sharpe': float(sharpe.mean()),
                        f'{prefix}_mean_downside_dev': float(downside.mean()),
                        f'{prefix}_mean_sortino': float(sortino.mean())})
        return out


def simulate_agent_spx(q: nn.Module, action_values: torch.Tensor, csv_path: str,
                       start_date: str, end_date: str, int_rate: float, trans_cost: float,
                       fig_path: Optional[str] = None) -> dict:
    """Backtest the greedy policy of a Q-network on real SPX data.

    The first STATE_LEN days of the window form the initial state.

    Args:
        q: Q-network.
        action_values: Portfolio weight of each discrete action.
        csv_path: CSV with 'Date', 'spx_normalised' and 'log_return'.
        start_date: First date of the window.
        end_date: Last date of the window.
        int_rate: Continuously compounded annual interest rate on cash.
        trans_cost: Proportional transaction cost.
        fig_path: File for the overview figure; no figure if None.

    Returns:
        Dict of agent metrics and 'spx_'-prefixed buy-and-hold metrics.
    """
    spec = spx_spec()
    hist_len = spec.state_len
    spx_df = pd.read_csv(csv_path, index_col=0, parse_dates=True).loc[start_date:end_date].copy()
    spx_df['spx_normalised'] = spx_df['spx_normalised'] / spx_df['spx_normalised'].iloc[hist_len]
    dt = (spx_df.index.diff().days / 365).values
    ret_col = spx_df.columns.get_loc('log_return')
    model = PortfolioModel(spec, int_rate=int_rate, trans_cost=trans_cost)
    q.cpu()

    log_wealth = torch.zeros((1, 1), dtype=DATATYPE)
    position = torch.zeros((1, 1), dtype=DATATYPE)
    log_wealth_seq = [0.]
    position_seq = [0.]
    for i in range(hist_len + 1, len(spx_df)):
        window = torch.tensor(spx_df.iloc[i - hist_len:i, ret_col].values,
                              dtype=DATATYPE).reshape(1, hist_len, 1)
        state = spec.build(window, position, dt=torch.tensor([[dt[i]]], dtype=DATATYPE))
        with torch.no_grad():
            act_idx = q(state).argmax(dim=-1).squeeze()
        action = torch.tensor(np.asarray(action_values[act_idx].numpy()), dtype=DATATYPE).reshape(1, 1)
        next_return = torch.tensor([[spx_df.iloc[i, ret_col]]], dtype=DATATYPE)
        next_state = model.advance(state, action, next_return)
        log_wealth = log_wealth + model.reward(state, action, next_state, strict=True)
        position = next_state['weight']
        log_wealth_seq.append(float(log_wealth.squeeze()))
        position_seq.append(float(action.squeeze()))

    spx_df['agent'] = np.nan
    spx_df.iloc[hist_len:, spx_df.columns.get_loc('agent')] = np.exp(log_wealth_seq)
    spx_df['position'] = np.nan
    spx_df.iloc[hist_len:, spx_df.columns.get_loc('position')] = position_seq
    spx_df['agent_max_drawdown'] = np.nan
    spx_df['spx_max_drawdown'] = np.nan
    first_idx = spx_df['agent'].first_valid_index()
    spx_df.loc[first_idx:, 'agent_max_drawdown'] = drawdown_backward(spx_df.loc[first_idx:, 'agent'])
    spx_df.loc[first_idx:, 'spx_max_drawdown'] = drawdown_backward(spx_df.loc[first_idx:, 'spx_normalised'])

    if fig_path is not None:
        fig, _ = overview(spx_df.loc[first_idx:], ['agent', 'spx_normalised'], ['position'],
                          ['agent_max_drawdown', 'spx_max_drawdown'], path=fig_path,
                          wealth_label=['Agent', 'S&P 500'])
        plt.close(fig)

    rf_dt = dt[hist_len + 1:]
    agent_excess = np.asarray(np.diff(log_wealth_seq), dtype=float) - rf_log_return(int_rate, rf_dt)
    vol, sharpe, downside, sortino = perf_metrics(agent_excess)
    spx_returns = spx_df.iloc[hist_len + 1:, ret_col].values
    spx_vol, spx_sharpe, spx_downside, spx_sortino = perf_metrics(
        np.asarray(spx_returns, dtype=float) - rf_log_return(int_rate, rf_dt))
    return {'final_wealth': float(spx_df['agent'].iloc[-1]),
            'final_log_wealth': float(log_wealth_seq[-1]),
            'max_drawdown': float(spx_df['agent_max_drawdown'].min()),
            'sharpe': float(sharpe), 'volatility': float(vol),
            'down_deviation': float(downside), 'sortino': float(sortino),
            'spx_final_wealth': float(spx_df['spx_normalised'].iloc[-1]),
            'spx_final_log_wealth': float(np.log(spx_df['spx_normalised'].iloc[-1])),
            'spx_max_drawdown': float(spx_df['spx_max_drawdown'].min()),
            'spx_sharpe': float(spx_sharpe), 'spx_volatility': float(spx_vol),
            'spx_down_deviation': float(spx_downside), 'spx_sortino': float(spx_sortino)}


def train_robustdqn(agent, env: MMDSimulator, n_episodes: int, log_dir: str,
                    backtest: Callable[[], dict], oos_every: int, ckpt_prefix: str):
    """Train a DQN agent on the simulator.

    After every episode the Q-network is saved to <ckpt_prefix>_<episode>.pt, and every
    oos_every episodes a backtest row is appended to spx.csv.

    Args:
        agent: DQN or RobustDQN agent.
        env: Training environment.
        n_episodes: Number of episodes.
        log_dir: Output directory.
        backtest: Callable returning the real-SPX backtest metrics.
        oos_every: Episodes between backtests.
        ckpt_prefix: Prefix of the checkpoint files.

    Returns:
        The trained agent.
    """
    for episode in tqdm(range(n_episodes)):
        cum_rewards = torch.zeros(env.batch_size, 1)
        obs = env.reset()
        act_idx = agent.agent_start(obs)
        while True:
            obs, rewards, done = env.step(act_idx)
            cum_rewards += rewards
            if done:
                agent.agent_end(rewards, obs)
                if (episode + 1) % oos_every == 0:
                    metrics = backtest()
                    metrics['episodes_trained'] = episode + 1
                    path = f'{log_dir}/spx.csv'
                    pd.DataFrame(metrics, index=[0]).to_csv(path, index=False, mode='a',
                                                            header=not os.path.exists(path))
                torch.save(agent.q.state_dict(), f'{log_dir}/{ckpt_prefix}_{episode + 1}.pt')
                break
            act_idx = agent.agent_step(rewards, obs)
        print(f'Episode {episode + 1} mean of summed rewards: {cum_rewards.mean():.3f}')
    return agent
