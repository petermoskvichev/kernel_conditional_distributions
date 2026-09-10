"""
Evaluate squared-error scaling of standard vs doubly robust CMMD1 estimators.
Runs repeated trials over sample sizes and reports mean error with uncertainty bars.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import sys
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# Add project root to path (robust to different working directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.cmmd import CMMD1
from src.dr_cmmd import CMMD1_dr, polynomial_features
from src.kernels import polynomial_kernel, linear_kernel
from src.dr_models import (
	sample_joint,
	sample_covariate_p,
	sample_covariate_q,
	conditional_y,
	conditional_z,
	propensity,
)


def fit_krr_predict(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
	label: str,
	alpha_grid: np.ndarray | None = None,
	cv: int = 5,
) -> np.ndarray:
	"""
	Fit KernelRidge with CV over alpha and predict on x_test.
	"""
	if alpha_grid is None:
		alpha_grid = np.logspace(-4, 1, 5)

	x_train_2d = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
	x_test_2d = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
	y_train_1d = y_train.ravel()

	krr = KernelRidge(kernel="polynomial", degree=2, coef0=1, gamma=1)

	search = GridSearchCV(
		estimator=krr,
		param_grid={"alpha": alpha_grid},
		cv=min(cv, x_train_2d.shape[0]),
		scoring="neg_mean_squared_error",
	)
	search.fit(x_train_2d, y_train_1d)

	best_model = search.best_estimator_
	return best_model.predict(x_test_2d)


def true_cmmd(n_samples: int = 2000) -> float:
	"""
	Approximate CMMD1^2 between P and Q by Monte Carlo averaging.
	"""
	X_P, _ = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=1)
	X_Q, _ = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=2)

	X_test = np.concatenate([X_P, X_Q])
	cmmd1 = np.mean((0.5 * X_test**2) ** 2)
	return cmmd1


def compute_cmmd1_estimates(
	n_samples: int,
	n_folds: int,
	seed: int
) -> Tuple[float, float]:
	"""
	Compute standard and doubly robust CMMD1^2 estimates for one trial.
	"""

	cmmd1_standard_stat = CMMD1()
	cmmd1_dr_stat = CMMD1_dr(
		n_folds=n_folds,
		random_state=seed + 2,
		feature_map=polynomial_features,
	)

	X_P, Y = sample_joint(n_samples, sample_covariate_p, conditional_y, seed=seed)
	X_Q, Z = sample_joint(n_samples, sample_covariate_q, conditional_z, seed=seed + 1)

	# Ensure data is 2D for kernel computation
	X_P = X_P.reshape(-1, 1)
	Y = Y.reshape(-1, 1)
	X_Q = X_Q.reshape(-1, 1)
	Z = Z.reshape(-1, 1)
	
	# Tune regularization for the standard estimator.
	alpha_grid = {"alpha": np.logspace(-4, 1, 5)}
	krr_y = GridSearchCV(
		KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1),
		alpha_grid, cv=5)
	krr_y.fit(X_P, Y.flatten())

	krr_z = GridSearchCV(
		KernelRidge(kernel='polynomial', degree=2, coef0=1, gamma=1),
		alpha_grid, cv=5)
	krr_z.fit(X_Q, Z.flatten())

	alpha_p = float(krr_y.best_params_["alpha"])
	alpha_q = float(krr_z.best_params_["alpha"])
	lam_p = alpha_p / X_P.shape[0]
	lam_q = alpha_q / X_Q.shape[0]

	stat_standard = cmmd1_standard_stat.compute(
		X_P, Y, X_Q, Z,
		lam_p, lam_q,
		polynomial_kernel,
		linear_kernel
	)

	stat_dr = cmmd1_dr_stat.compute(
		X_P, Y, X_Q, Z,
		lam_p, lam_q,
		polynomial_kernel,
		linear_kernel,
		propensity=propensity,
	)

	return stat_standard, stat_dr


def run_error_experiment(
	sample_sizes: np.ndarray,
	n_trials: int,
	n_folds: int,
	true_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Run squared error experiments across sample sizes.
	"""
	cmmd1_true = true_cmmd(n_samples=true_samples)

	standard_errors = np.zeros_like(sample_sizes, dtype=float)
	dr_errors = np.zeros_like(sample_sizes, dtype=float)
	standard_se = np.zeros_like(sample_sizes, dtype=float)
	dr_se = np.zeros_like(sample_sizes, dtype=float)

	print("Running CMMD1 squared-error experiment")
	print(f"  - trials per size: {n_trials}")
	print()

	for i, n_samples in enumerate(sample_sizes):
		standard_trials = np.zeros(n_trials)
		dr_trials = np.zeros(n_trials)

		for trial in range(n_trials):
			stat_standard, stat_dr = compute_cmmd1_estimates(
				n_samples=n_samples,
				n_folds=n_folds,
				seed=trial * 5 + n_samples,
			)

			standard_trials[trial] = (stat_standard - cmmd1_true) ** 2
			dr_trials[trial] = (stat_dr - cmmd1_true) ** 2

		standard_errors[i] = np.mean(standard_trials)
		dr_errors[i] = np.mean(dr_trials)
		standard_se[i] = np.std(standard_trials, ddof=1) / np.sqrt(n_trials)
		dr_se[i] = np.std(dr_trials, ddof=1) / np.sqrt(n_trials)

		print(
			f"n={n_samples:4d}: SE std={standard_errors[i]:.6f}, "
			f"SE dr={dr_errors[i]:.6f}"
		)

	return standard_errors, dr_errors, standard_se, dr_se


def plot_error_curves(
	sample_sizes: np.ndarray,
	standard_errors: np.ndarray,
	dr_errors: np.ndarray,
	standard_se: np.ndarray,
	dr_se: np.ndarray
):
	"""
	Plot squared error vs number of samples for both estimators.
	"""
	fig, ax = plt.subplots(figsize=(8, 6))

	ax.errorbar(
		sample_sizes,
		standard_errors,
		yerr=standard_se,
		marker="o",
		linestyle="-",
		color="blue",
		label="Standard Estimator",
		linewidth=2,
	)
	ax.errorbar(
		sample_sizes,
		dr_errors,
		yerr=dr_se,
		marker="o",
		linestyle="-",
		color="red",
		label="DR Estimator",
		linewidth=2,
	)

	ax.set_title("CMMD$_1$ Error", fontsize=36)
	ax.set_xlabel("Sample Size ($n$)", fontsize=30)
	ax.set_ylabel("Squared Error", fontsize=30)
	ax.tick_params(axis="both", which="major", labelsize=21)
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=18)

	plt.tight_layout()

	return fig, ax


if __name__ == "__main__":
	sample_sizes = np.array([100, 200, 300, 400, 500])
	n_trials = 100
	n_folds = 5

	standard_errors, dr_errors, standard_se, dr_se = run_error_experiment(
		sample_sizes=sample_sizes,
		n_trials=n_trials,
		n_folds=n_folds
	)

	fig, ax = plot_error_curves(
		sample_sizes, standard_errors, dr_errors, standard_se, dr_se)

	figs_dir = os.path.join(script_dir, "..", "figs", "dr")
	os.makedirs(figs_dir, exist_ok=True)
	fig_path = os.path.join(figs_dir, f"cmmd1_error.pdf")
	fig.savefig(fig_path, dpi=300, bbox_inches="tight")
	print(f"\nFigure saved to: {fig_path}")

	plt.show()
