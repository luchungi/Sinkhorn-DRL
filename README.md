# Code for "Distributionally Robust Deep Q-learning"

## Chung I Lu, Julian Sester, Aijia Zhang

Sinkhorn distributionally robust reinforcement learning for portfolio choice, with two experiments:

- **SPX with RobustDQN**: a discrete leverage grid on the S&P 500, trained on returns from a pretrained LSTM generator and backtested on real SPX data.
- **Stocks with RTD3**: a long-only, fully invested portfolio of F, JNJ, JPM, KO and WMT, trained on a bank of generated return paths and backtested on real prices.

## Installation

Requires Python 3.12.

```bash
uv sync
```

or

```bash
pip install -e .
```

## Running

Run from the repository root. Settings are read from the config files.

```bash
python train_rdqn_spx.py      # settings in config_rdqn.py
python train_rtd3_stocks.py   # settings in config_rtd3.py
```

Results are written to `runs/`. The output files are listed in the docstring of each training script.

## Repository layout

```
agent/     DQN.py, RTD3.py, robust.py (robust targets), nu_sampler.py, nu_fit.py, buffer.py, q.py
env/       spx.py, stocks.py, common.py (portfolio accounting and metrics), features.py, plot.py
data/      spx_generator/ (generator weights and noise parameters), spx.csv,
           stocks_path_bank.npz, stock_prices.csv
```
