"""Settings of train_rtd3_stocks.py: RTD3 on the five-stock generated path bank."""

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BANK_PATH = 'data/stocks_path_bank.npz'  # generated daily log-return paths of F, JNJ, JPM, KO, WMT
CSV_PATH = 'data/stock_prices.csv'       # real daily close prices of the same tickers

# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
EPSILON = 4e-2            # Sinkhorn radius of the ambiguity set
DELTA = 5e-4              # Sinkhorn entropic regularisation
NU_LOC = None             # common location of nu; None fits it (requires NU_FIT = True)
NU_SCALE = 0.011          # scale of nu; starting value of the fit when NU_FIT is True
NU_DF = 4.62              # degrees of freedom of nu; starting value of the fit when NU_FIT is True
NU_FIT = True             # fit the scale, correlation and degrees of freedom of nu to in-sample returns
NU_FIT_N_INIT = 4         # number of initialisations of the nu fit
N_NU = 1000               # number of nu candidates per transition
NU_Y_MIN = -6.90775528    # lower bound of the support of nu in log returns, log(0.001)
NU_Y_MAX = 6.90775528     # upper bound of the support of nu in log returns, log(1000)
NU_RATIO = [0.8, 0.20]    # shares of N_NU for the [fixed, local] candidate blocks
LAMDA_INIT = -1.1607      # initial log dual variable of new transitions
LAMDA_MAX_ITER = 50       # maximum Newton iterations of the dual solve
LAMDA_TOL = 1e-6          # Newton tolerance on the log dual variable
EBAR_MIN_FRAC = 0.995     # minimum fraction of feasible duals required before training starts

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
INT_RATE = 0.024          # continuously compounded annual interest rate on cash
TRANS_COST = 5e-4         # proportional transaction cost
REWARD_SCALE = 1.         # factor applied to the rewards used for training
ENV_BATCH_SIZE = 32       # paths stepped in parallel during training
N_HOLDOUT_SEEDS = 100     # real seed windows whose generated paths form the validation set
HOLDOUT_SEED = 0          # seed of the train/validation split

STATE_LEN = 21            # daily log-return lags in the state (at most 29)
RET_GAIN = 0.3            # gain applied to annualised returns in the network input

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPISODES = 200          # training episodes per seed
SEED = 1                  # first seed
N_SEEDS = 5               # number of seeds trained sequentially: SEED, SEED + 1, ...
DISCOUNT = 0.9            # discount factor
HIDDEN = [128, 128]       # hidden layer widths of the actor and critics
ACTIVATION = 'relu'       # hidden layer activation, 'relu' or 'tanh'
ACTOR_LR = 1e-4           # actor learning rate
CRITIC_LR = 1e-4          # critic learning rate
DEVICE = 'cpu'            # torch device

# ---------------------------------------------------------------------------
# Off-policy updates
# ---------------------------------------------------------------------------
BUFFER_SIZE = 1_000_000   # replay buffer capacity
AGENT_BATCH_SIZE = 256    # transitions per update
TRAIN_STEPS = 2           # environment steps between updates

# ---------------------------------------------------------------------------
# RTD3
# ---------------------------------------------------------------------------
POLICY_DELAY = 2          # critic updates per actor and target update
POLYAK_TAU = 0.005        # Polyak averaging coefficient of the target networks
ENTROPY_BETA = 0.0005     # coefficient of the policy entropy bonus in the actor loss
ROBUST = True             # True: Sinkhorn robust critic target; False: TD3 target
EXPL_SIGMA = 0.1          # initial standard deviation of the exploration noise
EXPL_CLIP = 0.2           # clip of the exploration noise
EXPL_RHO = 0.9            # autocorrelation of the Ornstein-Uhlenbeck exploration noise
EXPL_END_SIGMA = 0.01     # final standard deviation of the exploration noise
EXPL_SIGMA_STEPS = 50000  # environment steps over which the noise decays linearly
