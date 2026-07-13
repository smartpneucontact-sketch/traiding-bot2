"""ML track configuration (pre-registered — do not tune outside the grid)."""

DEV_END = "2022-12-31"
HORIZON = 21            # forward-return horizon in trading days
EMBARGO = 21            # trading days dropped before each test fold start
TRAIN_GRID = 5          # row-sampling grid for TRAINING rows (trading days)
REBAL = 21              # portfolio decision grid (trading days)
COST_BPS = 5
LEV = 2.0
MAX_LOOKBACK_BARS = 273  # hard cap on any feature lookback
CAND_POOL = 80          # top-N per momentum leg for the candidate union
TOP_N = 30              # portfolio size
BUF_IN = 25             # entries need model rank <= BUF_IN
BUF_OUT = 45            # incumbents stay unless model rank > BUF_OUT

HP_S = dict(
    num_leaves=15, max_depth=4, min_child_samples=50, learning_rate=0.05,
    n_estimators=300, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=1, reg_lambda=5.0,
)
HP_M = dict(
    num_leaves=31, max_depth=6, min_child_samples=30, learning_rate=0.03,
    n_estimators=600,
)

# Test folds: anchored walk-forward, test on calendar years
TEST_YEARS = [2018, 2019, 2020, 2021, 2022]

# Stage 1 experiment registry — the CORE 12 (pre-registered; runs 1-6 are the
# {arch} x {label} matrix, 7-10 the pool/weighting probes on the two
# non-ranker archs, 11 the monotone-constraint probe, 12 the training-grid
# density probe).
STAGE1_EXPERIMENTS = {
    "S1_01_Arank_L1":        dict(arch="A_rank", label="L1", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_02_Arank_L2":        dict(arch="A_rank", label="L2", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_03_Areg_L1":         dict(arch="A_reg",  label="L1", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_04_Areg_L2":         dict(arch="A_reg",  label="L2", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_05_Bcls_L1":         dict(arch="B_cls",  label="L1", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_06_Bcls_L2":         dict(arch="B_cls",  label="L2", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_07_Areg_L1_p120":    dict(arch="A_reg",  label="L1", pool=120, weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_08_Bcls_L1_p120":    dict(arch="B_cls",  label="L1", pool=120, weighting="EW", hp="HP_S", grid=5,  monotone=False),
    "S1_09_Areg_L1_IV":      dict(arch="A_reg",  label="L1", pool=80,  weighting="IV", hp="HP_S", grid=5,  monotone=False),
    "S1_10_Bcls_L1_IV":      dict(arch="B_cls",  label="L1", pool=80,  weighting="IV", hp="HP_S", grid=5,  monotone=False),
    "S1_11_Areg_L1_mono":    dict(arch="A_reg",  label="L1", pool=80,  weighting="EW", hp="HP_S", grid=5,  monotone=True),
    "S1_12_Arank_L1_g21":    dict(arch="A_rank", label="L1", pool=80,  weighting="EW", hp="HP_S", grid=21, monotone=False),
}

# Signal Gate 1 thresholds
G1_RANK_IC = 0.03
G1_IC_EDGE_OVER_PRIOR = 0.01
G1_PCT_POS_YEARS = 0.60
G1_HIT30 = 0.52
