"""Settings of train_rdqn_spx.py: RobustDQN on the generated SPX environment."""

# ---------------------------------------------------------------------------
# Generator and market data
# ---------------------------------------------------------------------------
GENERATOR_DIR = 'data/spx_generator/'                   # generator weights and shape parameters
MA_PARAMS_PATH = 'data/spx_generator/ma_params.json'    # moving-average parameters of the generator noise
SPX_CSV_PATH = 'data/spx.csv'                           # real SPX index levels and daily log returns

# ---------------------------------------------------------------------------
# Real-SPX backtest window
# ---------------------------------------------------------------------------
SPX_EVAL_START_DATE = '2020-10-05'  # first date; the first 60 days form the initial state
SPX_EVAL_END_DATE = '2025-12-31'    # last date

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
INT_RATE = 0.024          # continuously compounded annual interest rate on cash
TRANS_COST = 5e-4         # proportional transaction cost

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
N_ACTIONS = 9             # number of evenly spaced SPX weights
ACTION_MIN = -1.0         # smallest weight
ACTION_MAX = 1.0          # largest weight

# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
ROBUST = True             # True: Sinkhorn robust target; False: double DQN target

EPSILON = 2e-3            # Sinkhorn radius of the ambiguity set
DELTA = 5e-5              # Sinkhorn entropic regularisation
NU_Y_MIN = -2.99573227355399    # lower bound of the support of nu in log returns, log(0.05)
NU_Y_MAX = 0.667829372575655    # upper bound of the support of nu in log returns, log(1.95)
NU_DF = 2.7818            # degrees of freedom of nu
NU_LOC = 3.10884618e-04   # location of nu
NU_SCALE = 7.12279856e-03 # scale of nu
N_NU = 500                # number of nu candidates per transition
NU_RATIO = [0.8, 0.1, 0.1]  # shares of N_NU for the [fixed, local, far] candidate blocks

LAMDA_INIT = -1.1607      # initial log dual variable of new transitions
LAMDA_MAX_ITER = 50       # maximum Newton iterations of the dual solve
LAMDA_TOL = 1e-6          # Newton tolerance on the log dual variable
EBAR_MIN_FRAC = 0.995     # minimum fraction of feasible duals required before training starts

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
ARCHITECTURE = [64, 64]   # hidden layer widths of the Q-network
DISCOUNT = 0.90           # discount factor
EPS_GREEDY = 0.1          # probability of a uniformly random action during training
BUFFER_MAX_LENGTH = int(1e6)  # replay buffer capacity
CLONE_STEPS = 50          # environment steps between target network updates
TRAIN_STEPS = 2           # environment steps between updates
AGENT_BATCH_SIZE = 256    # transitions per minibatch
N_BATCHES = 1             # minibatches per epoch
N_EPOCHS = 1              # epochs per update
LR = 1e-4                 # learning rate

# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
ENV_BATCH_SIZE = 16       # paths stepped in parallel during training
N_EPISODES = 5            # training episodes per seed
SEED = 1                  # first seed
N_SEEDS = 5               # number of seeds trained sequentially: SEED, SEED + 1, ...
DEVICE = 'cpu'            # torch device

OOS_EVERY = 1             # episodes between real-SPX backtests
EVAL_BATCH_SIZE = 1000    # paths in the final evaluation on the simulator
EVAL_SEED = 12345         # seed of the final evaluation on the simulator
