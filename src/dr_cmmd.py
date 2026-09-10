"""
Cross-fitted doubly robust CMMD test statistics.

The classes in this module implement the same ``TestStatistic`` interface as
the statistics in ``src.cmmd``. Nuisance conditional mean models are fitted
internally by kernel ridge regression, so callers only need to supply a
propensity function.
"""

from typing import Callable

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

from src.cmmd import TestStatistic


def polynomial_features(X: np.ndarray) -> np.ndarray:
    """Explicit feature map for the kernel ``(x x' + 1)^2``."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] != 1:
        raise ValueError("polynomial_features expects one-dimensional covariates")

    x = X[:, 0]
    return np.column_stack([np.ones_like(x), np.sqrt(2.0) * x, x**2])


def _ridge_coefficients(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit ridge regression in an explicit feature space."""
    gram = features.T @ features
    rhs = features.T @ np.asarray(target).reshape(-1)
    return np.linalg.solve(gram + alpha * np.eye(features.shape[1]), rhs)


def _fit_feature_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha_grid: np.ndarray,
    cv: int,
) -> np.ndarray:
    """Fit CV-selected ridge regression in an explicit feature space."""
    target = np.asarray(target).reshape(-1)
    n_splits = min(cv, features.shape[0])
    if n_splits < 2:
        raise ValueError("At least two training observations are required")

    splitter = KFold(n_splits=n_splits, shuffle=False)
    mean_scores = np.empty(len(alpha_grid), dtype=float)

    for alpha_index, alpha in enumerate(alpha_grid):
        scores = []
        for train, validation in splitter.split(features):
            coefficients = _ridge_coefficients(
                features[train], target[train], float(alpha)
            )
            prediction = features[validation] @ coefficients
            residual_sum = np.sum((target[validation] - prediction) ** 2)
            centered = target[validation] - np.mean(target[validation])
            total_sum = np.sum(centered**2)
            if total_sum == 0.0:
                scores.append(1.0 if residual_sum == 0.0 else 0.0)
            else:
                scores.append(1.0 - residual_sum / total_sum)
        mean_scores[alpha_index] = np.mean(scores)

    best_alpha = float(alpha_grid[int(np.argmax(mean_scores))])
    return _ridge_coefficients(features, target, best_alpha)


def _fit_kernel_ridge(
    X: np.ndarray,
    target: np.ndarray,
    kernel_x: Callable,
    alpha_grid: np.ndarray,
    cv: int,
    kernel_kwargs: dict,
) -> KernelRidge:
    """Fit precomputed-kernel ridge regression with CV-selected alpha."""
    n = X.shape[0]
    n_splits = min(cv, n)
    if n_splits < 2:
        raise ValueError("At least two training observations are required")

    K = kernel_x(X, X, **kernel_kwargs)
    search = GridSearchCV(
        estimator=KernelRidge(kernel="precomputed"),
        param_grid={"alpha": alpha_grid},
        cv=n_splits,
    )
    search.fit(K, np.asarray(target).reshape(-1))
    return search.best_estimator_


def _predict_kernel_ridge(
    model: KernelRidge,
    X_test: np.ndarray,
    X_train: np.ndarray,
    kernel_x: Callable,
    kernel_kwargs: dict,
) -> np.ndarray:
    """Predict with a kernel-ridge model fitted on a precomputed kernel."""
    K_test_train = kernel_x(X_test, X_train, **kernel_kwargs)
    return np.asarray(model.predict(K_test_train)).reshape(-1)


class CrossFittedDREstimator:
    """
    Shared implementation of cross-fitted DR CMMD levels zero, one and two.

    The same instance may be shared by ``CMMD0_dr``, ``CMMD1_dr`` and
    ``CMMD2_dr``. Computed values are cached so that the nuisance fits are
    reused when all three levels are evaluated on the same data.
    """

    def __init__(
        self,
        n_folds: int = 5,
        alpha_grid: np.ndarray | None = None,
        cme_cv: int = 5,
        dr_cv: int = 5,
        random_state: int | None = None,
        feature_map: Callable | None = None,
    ):
        if n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if cme_cv < 2 or dr_cv < 2:
            raise ValueError("cme_cv and dr_cv must be at least two")

        if alpha_grid is None:
            alpha_grid = np.logspace(-4, 1, 5)

        self.n_folds = int(n_folds)
        self.alpha_grid = np.asarray(alpha_grid, dtype=float)
        self.cme_cv = int(cme_cv)
        self.dr_cv = int(dr_cv)
        self.random_state = random_state
        self.feature_map = feature_map
        self._cache: dict[tuple[object, ...], tuple[float, float, float]] = {}

    @staticmethod
    def _cache_key(
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        kernel_x: Callable,
        propensity: Callable,
        kernel_kwargs: dict,
    ) -> tuple[object, ...]:
        return (
            id(kernel_x),
            id(propensity),
            repr(sorted(kernel_kwargs.items())),
            np.ascontiguousarray(X_P).tobytes(),
            np.ascontiguousarray(Y).tobytes(),
            np.ascontiguousarray(X_Q).tobytes(),
            np.ascontiguousarray(Z).tobytes(),
        )

    def compute_all(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        kernel_x: Callable,
        propensity: Callable,
        **kernel_kwargs,
    ) -> tuple[float, float, float]:
        """Compute cross-fitted DR CMMD levels zero, one and two."""
        key = self._cache_key(
            X_P, Y, X_Q, Z, kernel_x, propensity, kernel_kwargs
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        n = X_P.shape[0]
        m = X_Q.shape[0]
        if self.n_folds > min(n, m):
            raise ValueError("n_folds cannot exceed either group size")

        y = np.asarray(Y).reshape(-1)
        z = np.asarray(Z).reshape(-1)
        if y.shape[0] != n or z.shape[0] != m:
            raise ValueError(
                "Cross-fitted DR CMMD currently requires scalar outcomes"
            )

        mu_y_p = np.full(n, np.nan)
        mu_y_q = np.full(m, np.nan)
        mu_z_p = np.full(n, np.nan)
        mu_z_q = np.full(m, np.nan)

        features_p = None
        features_q = None
        if self.feature_map is not None:
            features_p = np.asarray(self.feature_map(X_P), dtype=float)
            features_q = np.asarray(self.feature_map(X_Q), dtype=float)

        p_folds = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        q_random_state = (
            None if self.random_state is None else self.random_state + 1
        )
        q_folds = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=q_random_state,
        )

        for (p_train, p_holdout), (q_train, q_holdout) in zip(
            p_folds.split(X_P), q_folds.split(X_Q)
        ):
            if self.feature_map is not None:
                coef_y = _fit_feature_ridge(
                    features_p[p_train],
                    y[p_train],
                    self.alpha_grid,
                    self.cme_cv,
                )
                coef_z = _fit_feature_ridge(
                    features_q[q_train],
                    z[q_train],
                    self.alpha_grid,
                    self.cme_cv,
                )

                mu_y_p[p_holdout] = features_p[p_holdout] @ coef_y
                mu_y_q[q_holdout] = features_q[q_holdout] @ coef_y
                mu_z_p[p_holdout] = features_p[p_holdout] @ coef_z
                mu_z_q[q_holdout] = features_q[q_holdout] @ coef_z
            else:
                model_y = _fit_kernel_ridge(
                    X_P[p_train],
                    y[p_train],
                    kernel_x,
                    self.alpha_grid,
                    self.cme_cv,
                    kernel_kwargs,
                )
                model_z = _fit_kernel_ridge(
                    X_Q[q_train],
                    z[q_train],
                    kernel_x,
                    self.alpha_grid,
                    self.cme_cv,
                    kernel_kwargs,
                )

                mu_y_p[p_holdout] = _predict_kernel_ridge(
                    model_y,
                    X_P[p_holdout],
                    X_P[p_train],
                    kernel_x,
                    kernel_kwargs,
                )
                mu_y_q[q_holdout] = _predict_kernel_ridge(
                    model_y,
                    X_Q[q_holdout],
                    X_P[p_train],
                    kernel_x,
                    kernel_kwargs,
                )
                mu_z_p[p_holdout] = _predict_kernel_ridge(
                    model_z,
                    X_P[p_holdout],
                    X_Q[q_train],
                    kernel_x,
                    kernel_kwargs,
                )
                mu_z_q[q_holdout] = _predict_kernel_ridge(
                    model_z,
                    X_Q[q_holdout],
                    X_Q[q_train],
                    kernel_x,
                    kernel_kwargs,
                )

        nuisance_predictions = (mu_y_p, mu_y_q, mu_z_p, mu_z_q)
        if any(np.isnan(values).any() for values in nuisance_predictions):
            raise RuntimeError("Cross-fitting failed to predict every observation")

        X = np.concatenate([X_P, X_Q], axis=0)
        W = np.concatenate([y, z])
        T = np.concatenate([np.ones(n), np.zeros(m)])
        E = np.asarray(propensity(X), dtype=float).reshape(-1)
        if E.shape[0] != n + m:
            raise ValueError("propensity must return one value per observation")
        E = np.clip(E, 1e-6, 1.0 - 1e-6)

        mu_y = np.concatenate([mu_y_p, mu_y_q])
        mu_z = np.concatenate([mu_z_p, mu_z_q])
        pseudo_outcome = (T - E) / (E * (1.0 - E)) * (
            W - (1.0 - E) * mu_y - E * mu_z
        )

        total = n + m

        if self.feature_map is not None:
            features = np.concatenate([features_p, features_q], axis=0)
            coefficients = _fit_feature_ridge(
                features, pseudo_outcome, self.alpha_grid, self.dr_cv
            )
            feature_gram = features.T @ features
            cmmd0 = float(coefficients @ coefficients)
            cmmd1 = float(coefficients @ feature_gram @ coefficients) / total
            cmmd2 = float(
                coefficients @ feature_gram @ feature_gram @ coefficients
            ) / (total**2)
        else:
            final_model = _fit_kernel_ridge(
                X,
                pseudo_outcome,
                kernel_x,
                self.alpha_grid,
                self.dr_cv,
                kernel_kwargs,
            )
            K = kernel_x(X, X, **kernel_kwargs)
            dual = np.asarray(final_model.dual_coef_).reshape(-1)
            fitted_difference = K @ dual
            cmmd0 = float(dual @ fitted_difference)
            cmmd1 = float(fitted_difference @ fitted_difference) / total
            cmmd2 = float(
                fitted_difference @ K @ fitted_difference
            ) / (total**2)

        statistics = (cmmd0, cmmd1, cmmd2)
        self._cache[key] = statistics
        return statistics


class _CrossFittedDRStatistic(TestStatistic):
    """Base class for one level of cross-fitted DR CMMD."""

    level: int

    def __init__(
        self,
        n_folds: int = 5,
        alpha_grid: np.ndarray | None = None,
        cme_cv: int = 5,
        dr_cv: int = 5,
        random_state: int | None = None,
        feature_map: Callable | None = None,
        shared_estimator: CrossFittedDREstimator | None = None,
    ):
        if shared_estimator is None:
            shared_estimator = CrossFittedDREstimator(
                n_folds=n_folds,
                alpha_grid=alpha_grid,
                cme_cv=cme_cv,
                dr_cv=dr_cv,
                random_state=random_state,
                feature_map=feature_map,
            )
        self.shared_estimator = shared_estimator

    def compute(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        lam_p: float,
        lam_q: float,
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        **kwargs,
    ) -> float:
        """
        Compute one cross-fitted DR CMMD statistic.

        ``lam_p`` and ``lam_q`` are accepted for compatibility with the
        ``TestStatistic`` interface. All nuisance and final ridge parameters
        are selected internally using ``alpha_grid``.
        """
        propensity = kwargs.pop("propensity", None)
        if propensity is None:
            raise ValueError(
                f"{self.__class__.__name__} requires 'propensity' in stat_kwargs"
            )

        # These options belong to other CMMD implementations and are not
        # covariate-kernel parameters.
        kwargs.pop("cme_y", None)
        kwargs.pop("cme_z", None)
        kwargs.pop("estimator", None)

        statistics = self.shared_estimator.compute_all(
            X_P,
            Y,
            X_Q,
            Z,
            kernel_x,
            propensity,
            **kwargs,
        )
        return statistics[self.level]


class CMMD0_dr(_CrossFittedDRStatistic):
    """Level-zero CMMD with cross-fitted DR pseudo-outcomes."""

    level = 0


class CMMD1_dr(_CrossFittedDRStatistic):
    """Level-one CMMD with cross-fitted DR pseudo-outcomes."""

    level = 1


class CMMD2_dr(_CrossFittedDRStatistic):
    """Level-two CMMD with cross-fitted DR pseudo-outcomes."""

    level = 2
