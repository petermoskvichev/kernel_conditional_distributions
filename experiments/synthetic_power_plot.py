"""
Estimate power curves for CMMD tests over alternative strength parameters.
Uses same-marginal synthetic data with varying covariate mean shift.
Includes CMMDs with its level selected on a separate sample.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD0, CMMD1, CMMD2, CMMDs, Test_Same_Marginal
from src.kernels import gaussian_kernel, median_heuristic
from src.models import sample_joint_theta, conditional_y, conditional_z


def run_power_experiment(
	thetas: list[float],
	sample_size: int,
	n_trials: int = 100,
	alpha: float = 0.05,
	B: int = 200,
	lam_p: float = 0.1,
	lam_q: float = 0.1,
	noise_std: float = 0.3,
	cmmd2_estimator: str = "jmmd",
	seed: int = 42,
	cmmd_levels: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
	selection_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
	"""
	Estimate power (rejection rate) across theta values for CMMD tests.

	CMMD0/1/2 use the full samples. CMMDs splits each sample, selects the
	level maximizing (statistic - bootstrap quantile) / bootstrap std, 
	then tests only the held-out observations.
	"""
	if not 0 < alpha < 1:
		raise ValueError("alpha must lie strictly between 0 and 1")
	if B < 2:
		raise ValueError("B must be at least 2 to estimate bootstrap variability")
	if not 0 < selection_fraction < 1:
		raise ValueError("selection_fraction must lie strictly between 0 and 1")
	n_selection = int(sample_size * selection_fraction)
	if min(n_selection, sample_size - n_selection) < 2:
		raise ValueError("Each split must contain at least 2 observations per dataset")
	if not cmmd_levels or any(not np.isfinite(s) or s < 0 for s in cmmd_levels):
		raise ValueError("cmmd_levels must contain finite, nonnegative levels")
	if n_trials < 1:
		raise ValueError("n_trials must be positive")
	candidates = [CMMDs(level=s) for s in cmmd_levels]

	cmmd0_stat = CMMD0()
	cmmd1_stat = CMMD1()
	cmmd2_stat = CMMD2()
	
	algo = Test_Same_Marginal()

	rng = np.random.default_rng(seed)
	# A separate stream keeps the original full-sample comparisons reproducible.
	rng_select = np.random.default_rng(np.random.SeedSequence([seed+1, 1]))

	n = sample_size

	results_cmmd0 = np.zeros(len(thetas))
	results_cmmd1 = np.zeros(len(thetas))
	results_cmmd2 = np.zeros(len(thetas))
	results_cmmds = np.zeros(len(thetas))

	for i, theta in enumerate(thetas):
		# print current theta
		print(f"Running power experiment for theta: {theta:.2f}")
		
		reject0 = 0
		reject1 = 0
		reject2 = 0
		rejects = 0

		for trial in range(n_trials):
			seed_y = int(rng.integers(1, 2**31))
			seed_z = int(rng.integers(1, 2**31))
			seed_perm = int(rng.integers(1, 2**31))

			X_P, Y = sample_joint_theta(n, theta, conditional_y, noise_std=noise_std, seed=seed_y)
			X_Q, Z = sample_joint_theta(n, theta, conditional_z, noise_std=noise_std, seed=seed_z)

			X_P = X_P.reshape(-1, 1)
			Y = Y.reshape(-1, 1)
			X_Q = X_Q.reshape(-1, 1)
			Z = Z.reshape(-1, 1)

			# Choose bandwidth for current theta using median heuristic
			bandwidth = median_heuristic(X_P)

			# Prepare kwargs for test method
			algo_kwargs = {
				"alpha": alpha,
				"B": B,
				"lam_p": lam_p,
				"lam_q": lam_q,
				"random_state": seed_perm,
			}
			stat_kwargs = {
				"bandwidth": bandwidth,
			}

			_, p0 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd0_stat,
				gaussian_kernel,
				algo_kwargs=algo_kwargs,
				stat_kwargs=stat_kwargs,
			)
			_, p1 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd1_stat,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 1},
				stat_kwargs=stat_kwargs,
			)
			_, p2 = algo.test(
				X_P, Y, X_Q, Z,
				cmmd2_stat,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_perm + 2},
				stat_kwargs={**stat_kwargs, "estimator": cmmd2_estimator},
			)

			# Split paired observations independently within each dataset.
			idx_p = rng_select.permutation(n)
			idx_q = rng_select.permutation(n)
			sel_p, inf_p = idx_p[:n_selection], idx_p[n_selection:]
			sel_q, inf_q = idx_q[:n_selection], idx_q[n_selection:]
			selection_kwargs = {"bandwidth": median_heuristic(X_P[sel_p])}
			seed_selection = int(rng_select.integers(1, 2**31))
			seed_inference = int(rng_select.integers(1, 2**31))

			scores = []
			for candidate in candidates:
				stat, _, stats_boot = algo.test(
					X_P[sel_p], Y[sel_p], X_Q[sel_q], Z[sel_q],
					candidate,
					gaussian_kernel,
					# Use the same resampling assignments across candidates.
					algo_kwargs={**algo_kwargs, "random_state": seed_selection,
						"return_bootstrap": True},
					stat_kwargs=selection_kwargs,
				)
				q = np.quantile(stats_boot, 1 - alpha)
				v = np.std(stats_boot, ddof=1)
				# Zero-variance candidates have undefined scores; skip them.
				scores.append((stat - q) / v if v > 0 else -np.inf)

			# First candidate wins ties, including when all variances are zero.
			selected = candidates[int(np.argmax(scores))]
			_, ps = algo.test(
				X_P[inf_p], Y[inf_p], X_Q[inf_q], Z[inf_q],
				selected,
				gaussian_kernel,
				algo_kwargs={**algo_kwargs, "random_state": seed_inference},
				stat_kwargs=selection_kwargs,
			)

			reject0 += int(p0 < alpha)
			reject1 += int(p1 < alpha)
			reject2 += int(p2 < alpha)
			rejects += int(ps < alpha)

			# print every 50th trial for progress
			if (trial + 1) % 50 == 0:
				print(
					f"Trial {trial+1:d}/{n_trials:d} for n={n:d}: "
					f"CMMD0 p-value = {p0:.4f}, CMMD1 p-value = {p1:.4f}, CMMD2 p-value = {p2:.4f}, "
					f"CMMDs p-value = {ps:.4f} (s={selected.level:g})"
				)

		results_cmmd0[i] = reject0 / n_trials
		results_cmmd1[i] = reject1 / n_trials
		results_cmmd2[i] = reject2 / n_trials
		results_cmmds[i] = rejects / n_trials

		# print power for current theta
		print(
			f"Power for theta={theta:.2f}: "
			f"CMMD0 = {results_cmmd0[i]:.3f}, "
			f"CMMD1 = {results_cmmd1[i]:.3f}, "
			f"CMMD2 = {results_cmmd2[i]:.3f}, "
			f"CMMDs = {results_cmmds[i]:.3f}"
		)

	return {
		"cmmd0": results_cmmd0,
		"cmmd1": results_cmmd1,
		"cmmd2": results_cmmd2,
		"cmmds": results_cmmds,
	}


def plot_power_vs_theta(
	thetas: list[float],
	results: dict[str, np.ndarray]
):
	"""
	Plot power/type I error vs theta for CMMD tests.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.plot(thetas, results["cmmd0"], marker="o", label="CMMD$_0$")
	ax.plot(thetas, results["cmmd1"], marker="s", label="CMMD$_1$")
	ax.plot(thetas, results["cmmd2"], marker="^", label="CMMD$_2$")
	ax.plot(thetas, results["cmmds"], marker="*", linestyle="--", label="CMMD$_{s^*}$")

	ax.set_title("Power Curve", fontsize=24)
	ax.set_xlabel("Covariate mean ($\\theta$)", fontsize=20)
	ax.set_ylabel("Power", fontsize=20)
	ax.set_ylim(0.0, 1.1)
	ax.legend(fontsize=16)
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	thetas = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

	results = run_power_experiment(
		thetas = thetas,
		sample_size=100,
		n_trials=200,
		alpha=0.05,
		B=200,
		lam_p=0.1,
		lam_q=0.1,
		noise_std=0.5,
		seed=42,
		cmmd_levels=(0.0, 1.0, 2.0),
		selection_fraction=0.2,
	)

	fig, _ = plot_power_vs_theta(thetas, results)

	figs_dir = os.path.join(project_root, "figs/synthetic")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, "power_vs_theta.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"Figure saved to: {fig_path}")

	plt.show()
