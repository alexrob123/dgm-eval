from .authpct import compute_authpct
from .ct import compute_CTscore, compute_CTscore_mem, compute_CTscore_mode
from .fd import (
    compute_efficient_FD_with_reps,
    compute_FD_infinity,
    compute_FD_with_reps,
    compute_FD_with_stats,
)
from .fls import compute_fls, compute_fls_overfit
from .inception_score import compute_inception_score
from .mmd import compute_mmd
from .pr_curve import compute_pr_curve
from .prdc import compute_prdc
from .sw import sw_approx
from .vendi import compute_per_class_vendi_scores, compute_vendi_score

METRICS = [
    "authpct",
    "ct",
    "ct_test",
    "ct_modified",
    "fd",
    "fd_eff",
    "fd_infinity",
    "fls",
    "fls_overfit",
    "is",
    "kd",
    "prdc",
    "pr_curve",
    "sw_approx",
    "vendi",
]
