"""
Estimate rejection rates for standard and doubly robust CMMD tests.
Reports Type I error or power as a function of sample size under covariate shift.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD0, CMMD1, CMMD2, Test_Diff_Marginal
from src.dr_cmmd import (
	CMMD0_dr,
	CMMD1_dr,
	CMMD2_dr,
	CrossFittedDREstimator,
	polynomial_features,
)
from src.kernels import polynomial_kernel, linear_kernel
from src.dr_models import sample_joint, sample_covariate_p, sample_covariate_q, conditional_y, conditional_z, propensity
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def run_power_experiment(
	error: str,
	sample_sizes: list[int],
	n_trials: int = 250,
	alpha: float = 0.05,
	B: int = 250,
	nfolds: int = 5,
	seed: int = 42
) -> dict[str, np.ndarray]:
	"""
	Estimate power (rejection rate) across sample sizes for CMMD tests.
	"""
	cmmd0_stat = CMMD0()
	cmmd1_stat = CMMD1()
	cmmd2_stat = CMMD2()
	
	algo = Test_Diff_Marginal()

	rng = np.random.default_rng(seed)

	results_cmmd0 = np.zeros(len(sample_sizes))
	results_cmmd1 = np.zeros(len(sample_sizes))
	results_cmmd2 = np.zeros(len(sample_sizes))
	results_cmmd0_dr = np.zeros(len(sample_sizes))
	results_cmmd1_dr = np.zeros(len(sample_sizes))
	results_cmmd2_dr = np.zeros(len(sample_sizes))

	for i, n in enumerate(sample_sizes):
		# Run all Monte Carlo trials at the current sample size.
		print(f"Running power experiment for sample size: {n}")
		
		reject0 = 0
		reject1 = 0
		reject2 = 0
		reject0_dr = 0
		reject1_dr = 0
		reject2_dr = 0

		for trial in range(n_trials):
			seed_perm = int(rng.integers(1, 2**31))

			# P: Y|X
			X_P, Y = sample_joint(n, sample_covariate_p, conditional_y, seed=trial*2)
			
			# Q: Z|X
			if error == "type1":
				X_Q, Z = sample_joint(n, sample_covariate_q, conditional_y, seed=trial*2 + 1)
			else:
				X_Q, Z = sample_joint(n, sample_covariate_q, conditional_z, seed=trial*2 + 1)

			# Ensure data is 2D for kernel computation
			X_P = X_P.reshape(-1, 1)
			Y = Y.reshape(-1, 1)
			X_Q = X_Q.reshape(-1, 1)
			Z = Z.reshape(-1, 1)

			alpha_grid = {"alpha": np.logspace(-4, 1, 5)}

			# Select regularization for the standard statistics using the full data.
			krr_y_standard = GridSearchCV(
				KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1),
				alpha_grid, cv=min(5, n))
			krr_y_standard.fit(X_P, Y.flatten())

			krr_z_standard = GridSearchCV(
				KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1),
				alpha_grid, cv=min(5, n))
			krr_z_standard.fit(X_Q, Z.flatten())

			alpha_p = float(krr_y_standard.best_params_["alpha"])
			alpha_q = float(krr_z_standard.best_params_["alpha"])
			lam_p = alpha_p / X_P.shape[0]
			lam_q = alpha_q / X_Q.shape[0]

			# Share cross-fitted nuisance and DR operator fits across CMMD levels.
			# Test_Diff_Marginal calls compute again for every resampled label vector,
			# so the complete cross-fitting pipeline is rerun under resampling.
			shared_dr_stats = CrossFittedDREstimator(
				n_folds=nfolds,
				random_state=trial * 2 + 2,
				feature_map=polynomial_features,
			)
			cmmd0_stat_dr = CMMD0_dr(shared_estimator=shared_dr_stats)
			cmmd1_stat_dr = CMMD1_dr(shared_estimator=shared_dr_stats)
			cmmd2_stat_dr = CMMD2_dr(shared_estimator=shared_dr_stats)

			# Prepare kwargs for test method
			algo_kwargs = {
				"alpha": alpha,
				"B": B,
				"lam_p": lam_p,
				"lam_q": lam_q,
				"random_state": seed_perm,
				"propensity_fn": propensity,
			}
			algo_kwargs_dr = {
				**algo_kwargs,
			}
			
			stat_kwargs = {}
			_, p0 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd0_stat,
				polynomial_kernel,
				linear_kernel,
				algo_kwargs=algo_kwargs,
				stat_kwargs=stat_kwargs,
			)
			
			# Compute p-values for standard and DR tests
			_, p1 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd1_stat,
				polynomial_kernel,
				linear_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 1},
				stat_kwargs=stat_kwargs,
			)
			_, p2 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd2_stat,
				polynomial_kernel,
				linear_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 2},
				stat_kwargs={**stat_kwargs, "estimator": "cmmd"},
			)
			_, p0_dr = algo.test(
				X_P, Y, X_Q, Z,
				cmmd0_stat_dr,
				polynomial_kernel,
				linear_kernel,
				algo_kwargs=algo_kwargs_dr,
				stat_kwargs={**stat_kwargs, "propensity": propensity},
			)
			_, p1_dr = algo.test(
				X_P, Y, X_Q, Z,
				cmmd1_stat_dr,
				polynomial_kernel,
				linear_kernel,
				algo_kwargs=algo_kwargs_dr,
				stat_kwargs={**stat_kwargs, "propensity": propensity},
			)
			_, p2_dr = algo.test(
				X_P, Y, X_Q, Z,
				cmmd2_stat_dr,
				polynomial_kernel,
				linear_kernel,
				algo_kwargs=algo_kwargs_dr,
				stat_kwargs={**stat_kwargs, "propensity": propensity},
			)

			# Update rejection counts for standard and DR tests
			reject0 += int(p0 < alpha)
			reject1 += int(p1 < alpha)
			reject2 += int(p2 < alpha)
			reject0_dr += int(p0_dr < alpha)
			reject1_dr += int(p1_dr < alpha)
			reject2_dr += int(p2_dr < alpha)

			# Log progress every 50 trials.
			if (trial + 1) % 50 == 0:
				print(
					f"Trial {trial+1:d}/{n_trials:d} for n={n:d}: "
					f"p0 = {p0:.4f}, p1 = {p1:.4f}, p2 = {p2:.4f}, "
					f"p0_dr = {p0_dr:.4f}, p1_dr = {p1_dr:.4f}, p2_dr = {p2_dr:.4f}"
				)

		results_cmmd0[i] = reject0 / n_trials
		results_cmmd1[i] = reject1 / n_trials
		results_cmmd2[i] = reject2 / n_trials
		results_cmmd0_dr[i] = reject0_dr / n_trials
		results_cmmd1_dr[i] = reject1_dr / n_trials
		results_cmmd2_dr[i] = reject2_dr / n_trials

		# Print rejection rates for the current sample size.
		print(
			f"rejection rate for n={n:.2f}: "
			f"CMMD0 = {results_cmmd0[i]:.3f}, "
			f"CMMD1 = {results_cmmd1[i]:.3f}, "
			f"CMMD2 = {results_cmmd2[i]:.3f}, "
			f"CMMD0_DR = {results_cmmd0_dr[i]:.3f}, "
			f"CMMD1_DR = {results_cmmd1_dr[i]:.3f}, "
			f"CMMD2_DR = {results_cmmd2_dr[i]:.3f}"
		)

	return {
		"cmmd0": results_cmmd0,
		"cmmd1": results_cmmd1,
		"cmmd2": results_cmmd2,
		"cmmd0_dr": results_cmmd0_dr,
		"cmmd1_dr": results_cmmd1_dr,
		"cmmd2_dr": results_cmmd2_dr,
	}


def plot_power_vs_sample_size(
	sample_sizes: list[int],
	results: dict[str, np.ndarray],
	error: str = "type1"
):
	"""
	Plot power/type I error vs sample size for CMMD0, CMMD1, and CMMD2.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))
	
	# Add significance-level reference for Type I error curves.
	if error == "type1":
		ax.axhline(0.05, color="black", linestyle="--")

	if error == "type1":
		linewidth = 1.5
	elif error == "type2":
		linewidth = 2

	ax.plot(sample_sizes, results["cmmd0"], marker="o", label="CMMD$_0$", linewidth=linewidth)
	ax.plot(sample_sizes, results["cmmd1"], marker="s", label="CMMD$_1$", linewidth=linewidth)
	ax.plot(sample_sizes, results["cmmd2"], marker="^", label="CMMD$_2$", linewidth=linewidth)
	ax.plot(sample_sizes, results["cmmd0_dr"], marker="o", linestyle="--", label="CMMD$_0$ DR", color="C0", linewidth=linewidth)
	ax.plot(sample_sizes, results["cmmd1_dr"], marker="s", linestyle="--", label="CMMD$_1$ DR", color="C1", linewidth=linewidth)
	ax.plot(sample_sizes, results["cmmd2_dr"], marker="^", linestyle="--", label="CMMD$_2$ DR", color="C2", linewidth=linewidth)


	if error == "type1":
		ax.set_title("Hypothesis Testing", fontsize=24)
		ax.set_xlabel("Sample size ($n$)", fontsize=20)
		ax.set_ylabel("Type I Error", fontsize=20)
		ax.set_ylim(0.0, 0.3)
		ax.legend(fontsize=16, ncols=2)
		ax.tick_params(axis="both", which="major", labelsize=14)
	if error == "type2":
		ax.set_title("Hypothesis Testing", fontsize=36)
		ax.set_xlabel("Sample size ($n$)", fontsize=30)
		ax.set_ylabel("Power", fontsize=30)
		ax.set_ylim(0.0, 1.1)
		ax.legend(fontsize=18, loc=[0.3, 0.4], ncols=2)
		ax.tick_params(axis="both", which="major", labelsize=21)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	sample_sizes = [100, 200, 300, 400, 500]
	# error = "type1"
	error = "type2"

	results = run_power_experiment(
		error=error,
		sample_sizes=sample_sizes,
		n_trials=200,
		alpha=0.05,
		B=200,
		seed=42,
		nfolds=5,
	)

	fig, _ = plot_power_vs_sample_size(sample_sizes, results, error=error)

	figs_dir = os.path.join(project_root, "figs/dr")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, f"{error}_vs_sample_size.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved to: {fig_path}")

	plt.show()
