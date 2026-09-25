"""Power versus sample allocation with reference weight rho=0.5.

Uses P_X=N(-0.5, 1), Q_X=N(0.5, 1), and the existing different-marginal
conditional test. Each requested CMMD level is tested on the full samples.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD_rho, Test_Diff_Marginal
from src.kernels import gaussian_kernel, median_heuristic
from src.models import (
    conditional_y, conditional_z, sample_joint,
    sample_covariate_p, sample_covariate_q, propensity,
)


def run_mixture_power_experiment(
    allocations: list[float] | np.ndarray,
    total_sample_size: int = 200,
    n_trials: int = 100,
    alpha: float = 0.05,
    B: int = 200,
    lam_p: float = 0.1,
    lam_q: float = 0.1,
    noise_std: float = 0.5,
    seed: int = 42,
    cmmd_levels: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> dict[str, np.ndarray]:
    """Power versus sample allocation with reference weight rho=0.5.

    allocations specifies n/(n+m), rounded to integer sample sizes.
    Each group's reference contribution is 0.5 regardless of its size.
    """
    allocations = np.asarray(allocations, dtype=float)
    if allocations.ndim != 1 or allocations.size == 0 or np.any(~np.isfinite(allocations)):
        raise ValueError("allocations must be a nonempty one-dimensional array of finite values")
    if np.any((allocations <= 0) | (allocations >= 1)):
        raise ValueError("Every allocations value must lie strictly between 0 and 1")
    if not isinstance(total_sample_size, (int, np.integer)) or total_sample_size < 4:
        raise ValueError("total_sample_size must be an integer of at least 4")
    if not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError("B must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if any(not np.isfinite(lam) or lam <= 0 for lam in (lam_p, lam_q)):
        raise ValueError("lam_p and lam_q must be finite and positive")
    if len(cmmd_levels) == 0 or any(not np.isfinite(s) or s < 0 for s in cmmd_levels):
        raise ValueError("cmmd_levels must contain finite, nonnegative levels")
    levels = {f"cmmd{s:g}": float(s) for s in cmmd_levels}
    results = {name: np.zeros(allocations.size) for name in levels}
    algorithm = Test_Diff_Marginal()
    sample_sizes = np.rint(allocations * total_sample_size).astype(int)
    if np.any(sample_sizes < 2) or np.any(total_sample_size - sample_sizes < 2):
        raise ValueError("Each allocation must leave at least two observations per group")
    for i, n_p in enumerate(sample_sizes):
        n_q = total_sample_size - n_p
        rho = 0.5
        print(f"n_P={n_p}, n_Q={n_q}, rho={rho:g}")
        statistics = {name: CMMD_rho(rho=rho, s=s) for name, s in levels.items()}
        rng = np.random.default_rng(seed)
        for trial in range(n_trials):
            seed_p, seed_q, seed_test = (int(rng.integers(1, 2**31)) for _ in range(3))
            X_P, Y = sample_joint(n_p, sample_covariate_p, conditional_y,
                                  noise_std=noise_std, seed=seed_p)
            X_Q, Z = sample_joint(n_q, sample_covariate_q, conditional_z,
                                  noise_std=noise_std, seed=seed_q)
            X_P, Y = X_P.reshape(-1, 1), Y.reshape(-1, 1)
            X_Q, Z = X_Q.reshape(-1, 1), Z.reshape(-1, 1)
            bandwidth = median_heuristic(np.vstack([X_P, X_Q]))
            for name, statistic in statistics.items():
                _, p_value = algorithm.test(
                    X_P, Y, X_Q, Z, statistic, gaussian_kernel,
                    algo_kwargs={
                        "alpha": alpha, "B": B, "lam_p": lam_p, "lam_q": lam_q,
                        "random_state": seed_test, "propensity_fn": propensity,
                    },
                    stat_kwargs={"bandwidth": bandwidth},
                )
                results[name][i] += int(p_value < alpha)
        print("Power:", end=" ")
        for name in levels:
            print(f"{name.upper()} = {results[name][i] / n_trials:.3f}", end="  ")
        print(flush=True)
    return {name: counts / n_trials for name, counts in results.items()}


def plot_power_vs_rho(allocations, results, total_sample_size=None):
    """Plot estimated power for each requested CMMD level."""
    if total_sample_size is not None:
        allocations = np.rint(np.asarray(allocations) * total_sample_size) / total_sample_size
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, power in results.items():
        ax.plot(allocations, power, marker="o", label=f"CMMD$_{{{name[4:]}}}$")
    ax.set_title("Power vs. Mixture ($\\rho=0.5$)", fontsize=24)
    ax.set_xlabel("Proportion from $P$ ($n/(n+m)$)", fontsize=20)
    ax.set_ylabel("Power", fontsize=20)
    ax.set_xticks(allocations)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    allocations = np.arange(1, 10) / 10
    results = run_mixture_power_experiment(
        allocations=allocations,
        total_sample_size=200,
        n_trials=200,
        B=200,
        cmmd_levels=(0.0, 1.0, 2.0),
    )
    fig, _ = plot_power_vs_rho(allocations, results, total_sample_size=200)
    figs_dir = os.path.join(project_root, "figs/synthetic")
    os.makedirs(figs_dir, exist_ok=True)
    fig_path = os.path.join(figs_dir, "power_vs_sample_prop.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")
    plt.show()
