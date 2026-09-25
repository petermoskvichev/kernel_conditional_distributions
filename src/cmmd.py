"""
CMMD test statistics and hypothesis-testing algorithms.
Implements CMMD levels, doubly robust variants, and bootstrap/permutation tests.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np


def _compute_x_kernels(
    X_P: np.ndarray,
    X_Q: np.ndarray,
    kernel_x: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_XpXp = kernel_x(X_P, X_P, **kwargs) # (n, n)
    K_XqXq = kernel_x(X_Q, X_Q, **kwargs) # (m, m)
    K_XqXp = kernel_x(X_Q, X_P, **kwargs) # (m, n)
    return K_XpXp, K_XqXq, K_XqXp


def _compute_yz_kernels(
    Y: np.ndarray,
    Z: np.ndarray,
    kernel_y: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L_YY = kernel_y(Y, Y, **kwargs) # (n, n)
    L_ZZ = kernel_y(Z, Z, **kwargs) # (m, m)
    L_YZ = kernel_y(Y, Z, **kwargs) # (n, m)
    return L_YY, L_ZZ, L_YZ


def _compute_w_matrices(
    K_XpXp: np.ndarray,
    K_XqXq: np.ndarray,
    lam_p: float,
    lam_q: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = K_XpXp.shape[0]
    m = K_XqXq.shape[0]
    I_n = np.eye(n)
    I_m = np.eye(m)
    W_Xp = np.linalg.inv(K_XpXp + lam_p * n * I_n)
    W_Xq = np.linalg.inv(K_XqXq + lam_q * m * I_m)
    return W_Xp, W_Xq


def _compute_tilde_kernels(
    X_P: np.ndarray,
    X_Q: np.ndarray,
    kernel_x: Callable,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_tilde = np.vstack([X_P, X_Q])
    K_Xp_Xtilde = kernel_x(X_P, X_tilde, **kwargs) # (n, n+m)
    K_Xq_Xtilde = kernel_x(X_Q, X_tilde, **kwargs) # (m, n+m)
    K_Xtilde_Xtilde = kernel_x(X_tilde, X_tilde, **kwargs) # (n+m, n+m)
    return K_Xp_Xtilde, K_Xq_Xtilde, K_Xtilde_Xtilde


def _psd_matrix_power(K: np.ndarray, power: float) -> np.ndarray:
    """
    Compute K**power for a PSD matrix K via eigen-decomposition.
    """
    K_sym = 0.5 * (K + K.T)
    eigenvalues, eigenvectors = np.linalg.eigh(K_sym)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    return (eigenvectors * (eigenvalues ** power)) @ eigenvectors.T


class TestStatistic(ABC):
    """
    Abstract base class for CMMD-based test statistics.
    
    A test statistic takes samples from two conditional distributions and
    returns a scalar value. Different implementations may use different
    estimators or kernels.
    """
    
    @abstractmethod
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
        **kwargs
    ) -> float:
        """
        Compute the test statistic.
        
        Parameters
        ----------
        X_P : np.ndarray, shape (n, d)
            Covariates from distribution P.
        Y : np.ndarray, shape (n, k)
            Outcomes from distribution P (conditional on X_P).
        X_Q : np.ndarray, shape (m, d)
            Covariates from distribution Q.
        Z : np.ndarray, shape (m, k)
            Outcomes from distribution Q (conditional on X_Q).
        lam_p : float
            Regularization parameter for W_X = (K_XX + lam_p * n * I_n)^{-1}.
        lam_q : float
            Regularization parameter for W_X' = (K_X'X' + lam_q * m * I_m)^{-1}.
        kernel_x : callable
            Kernel function for covariates X.
        kernel_y : callable, optional
            Kernel function for outcomes Y/Z. If None, uses kernel_x.
        **kwargs
            Additional arguments passed to kernels (e.g., bandwidth).
        
        Returns
        -------
        stat : float
            The computed test statistic value.
        """
        pass


class CMMD0(TestStatistic):
    """
    CMMD0 test statistic.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD0 test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)
        
        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X L_YY W_X K_XX)
        term1 = np.trace(W_Xp @ L_YY @ W_Xp @ K_XpXp)
        
        # Term 2: -2 * Tr(W_X L_YZ W_X' K_X'X)
        term2 = -2.0 * np.trace(W_Xp @ L_YZ @ W_Xq @ K_XqXp)
        
        # Term 3: Tr(W_X' L_ZZ W_X' K_X'X')
        term3 = np.trace(W_Xq @ L_ZZ @ W_Xq @ K_XqXq)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)
    

class CMMD0_dr(TestStatistic):
    """
    CMMD0 DR test statistic.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD0 DR test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        propensity = kwargs.pop("propensity", None)
        if propensity is None:
            raise ValueError("CMMD0_dr requires 'propensity' in stat_kwargs")
        
        cme_y = kwargs.pop("cme_y", None)
        cme_z = kwargs.pop("cme_z", None)

        lam = lam_p
        
        # Merge test set 
        X = np.concatenate([X_P, X_Q])
        YZ_combined = np.concatenate([Y, Z]).reshape(-1)

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)]).reshape(-1)

        # Construct propensity vector
        E = propensity(X)
        E = np.clip(np.asarray(E).reshape(-1), 1e-6, 1 - 1e-6)
        E_tilde = (T - E) / (E * (1 - E))

        # CME models for Y and Z
        mu_Y = cme_y(X)
        mu_Z = cme_z(X)

        # Combine observed outcomes into a single feature vector
        Phi_W = np.asarray(YZ_combined).reshape(-1)

        # Pseudo-outcome operator
        Phi = np.asarray((Phi_W - (1 - E) * mu_Y - E * mu_Z) * E_tilde).reshape(-1)

        # Kernel matrix and inverse for DR test statistic
        K_X = kernel_x(X, X, **kwargs)
        n_x = X.shape[0]
        W_X = np.linalg.inv(K_X + lam * n_x * np.eye(n_x))
        
        # CMMD statistic
        cmmd = float(Phi @ W_X @ K_X @ W_X @ Phi.T)
        
        return cmmd
    

class CMMD1_dr(TestStatistic):
    """
    CMMD1 DR test statistic.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD1 DR test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        propensity = kwargs.pop("propensity", None)

        cme_y = kwargs.pop("cme_y", None)
        cme_z = kwargs.pop("cme_z", None)

        lam = lam_p

        # Merge test set 
        X = np.concatenate([X_P, X_Q])
        YZ_combined = np.concatenate([Y, Z]).reshape(-1)

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)]).reshape(-1)

        # Construct propensity vector
        E = propensity(X)
        E = np.clip(np.asarray(E).reshape(-1), 1e-6, 1 - 1e-6)
        E_tilde = (T - E) / (E * (1 - E))

        # CME models for Y and Z
        mu_Y = cme_y(X)
        mu_Z = cme_z(X)

        # Combine observed outcomes into a single feature vector
        Phi_W = np.asarray(YZ_combined).reshape(-1)

        # Pseudo-outcome operator
        Phi = np.asarray((Phi_W - (1 - E) * mu_Y - E * mu_Z) * E_tilde).reshape(-1)

        # Kernel matrix and inverse for DR test statistic
        K_X = kernel_x(X, X, **kwargs)
        n_x = X.shape[0]
        W_X = np.linalg.inv(K_X + lam * n_x * np.eye(n_x))

        # CMMD1
        cmmd = float(Phi @ W_X @ K_X @ K_X @ W_X @ Phi.T) / n_x
        
        return cmmd
    

class CMMD2_dr(TestStatistic):
    """
    CMMD2 DR test statistic.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD2 DR test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        propensity = kwargs.pop("propensity", None)

        cme_y = kwargs.pop("cme_y", None)
        cme_z = kwargs.pop("cme_z", None)

        lam = lam_p

        # Merge test set 
        X = np.concatenate([X_P, X_Q])
        YZ_combined = np.concatenate([Y, Z]).reshape(-1)

        # T indicates which samples are from P (T=1) vs Q (T=0)
        T = np.concatenate([np.ones_like(Y), np.zeros_like(Z)]).reshape(-1)

        # Construct propensity vector
        E = propensity(X)
        E = np.clip(np.asarray(E).reshape(-1), 1e-6, 1 - 1e-6)
        E_tilde = (T - E) / (E * (1 - E))

        # CME models for Y and Z
        mu_Y = cme_y(X)
        mu_Z = cme_z(X)

        # Combine observed outcomes into a single feature vector
        Phi_W = np.asarray(YZ_combined).reshape(-1)

        # Pseudo-outcome operator
        Phi = np.asarray((Phi_W - (1 - E) * mu_Y - E * mu_Z) * E_tilde).reshape(-1)

        # Kernel matrix and inverse for DR test statistic
        K_X = kernel_x(X, X, **kwargs)
        n_x = X.shape[0]
        W_X = np.linalg.inv(K_X + lam * n_x * np.eye(n_x))

        # CMMD2
        cmmd = float(Phi @ W_X @ K_X @ K_X @ K_X @ W_X @ Phi.T) / (n_x ** 2)
        
        return cmmd


class CMMD1(TestStatistic):
    """
    CMMD1 test statistic.

    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean embeddings.
    
    Uses pooled covariates \tilde{X} = (X, X')
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD1 test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        n = X_P.shape[0]
        m = X_Q.shape[0]

        K_XpXp, K_XqXq, _ = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        K_Xp_Xtilde, K_Xq_Xtilde, _ = _compute_tilde_kernels(
            X_P, X_Q, kernel_x, **kwargs
        )
        K_Xtilde_Xp = K_Xp_Xtilde.T
        K_Xtilde_Xq = K_Xq_Xtilde.T
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X L_YY W_X K_Xtilde K_tildeX)
        term1 = np.trace(W_Xp @ L_YY @ W_Xp @ K_Xp_Xtilde @ K_Xtilde_Xp)

        # Term 2: -2 * Tr(W_X L_YZ W_X' K_X'tilde K_tildeX)
        term2 = -2.0 * np.trace(W_Xp @ L_YZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xp)

        # Term 3: Tr(W_X' L_ZZ W_X' K_X'tilde K_tildeX')
        term3 = np.trace(W_Xq @ L_ZZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xq)

        cmmd = (term1 + term2 + term3) / (n + m)

        return float(cmmd)


class CMMD2(TestStatistic):
    """
    CMMD2 test statistic.

    Compares two conditional distributions P(Y|X) and Q(Z|X) using joint mean embeddings.
    """
    
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
        estimator: str = "cmmd",
        **kwargs
    ) -> float:
        """
        Compute CMMD2 test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        estimator = estimator.lower()
        n = X_P.shape[0]
        m = X_Q.shape[0]

        K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)

        if estimator == "jmmd":
            # Term 1: Tr(L_YY K_XX)
            term1 = np.trace(L_YY @ K_XpXp)

            # Term 2: -2 * Tr(L_YZ K_X'X)
            term2 = -2.0 * np.trace(L_YZ @ K_XqXp)

            # Term 3: Tr(L_ZZ K_X'X')
            term3 = np.trace(L_ZZ @ K_XqXq)

            cmmd = (term1 / n ** 2) + (term2 / (n * m)) + (term3 / m ** 2)
            return float(cmmd)

        if estimator == "cmmd":
            W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)
            K_Xp_Xtilde, K_Xq_Xtilde, K_Xtilde_Xtilde = _compute_tilde_kernels(X_P, X_Q, kernel_x, **kwargs)
            K_Xtilde_Xp = K_Xp_Xtilde.T
            K_Xtilde_Xq = K_Xq_Xtilde.T

            # Term 1: Tr(W_X L_YY W_X K_Xp_Xtilde K_Xtilde_Xtilde K_Xtilde_Xp)
            term1 = np.trace(
                W_Xp @ L_YY @ W_Xp @ K_Xp_Xtilde @ K_Xtilde_Xtilde @ K_Xtilde_Xp
            )

            # Term 2: -2 * Tr(W_X L_YZ W_X' K_Xq_Xtilde K_Xtilde_Xtilde K_Xtilde_Xp)
            term2 = -2.0 * np.trace(
                W_Xp @ L_YZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xtilde @ K_Xtilde_Xp
            )

            # Term 3: Tr(W_X' L_ZZ W_X' K_Xq_Xtilde K_Xtilde_Xtilde K_Xtilde_Xq)
            term3 = np.trace(
                W_Xq @ L_ZZ @ W_Xq @ K_Xq_Xtilde @ K_Xtilde_Xtilde @ K_Xtilde_Xq
            )

            cmmd = (term1 + term2 + term3) / (n + m) ** 2
            return float(cmmd)

        raise ValueError(
            f"Unknown estimator '{estimator}'. Use 'jmmd' or 'cmmd'."
        )
    
class CMMD0_primal(TestStatistic):
    """
    CMMD0 test statistic using primal estimator of CMO.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD0 test statistic.

        X_P, X_Q are categorical so we use kronecker delta kernel and compute CMO in primal form.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        # Flatten X_P and X_Q since they're categorical
        X_P_flat = np.asarray(X_P).flatten()
        X_Q_flat = np.asarray(X_Q).flatten()

        # Build one-hot feature maps for the pooled categorical support
        unique_X = np.unique(np.concatenate([np.unique(X_P_flat), np.unique(X_Q_flat)]))
        Phi_Xp = np.zeros((unique_X.size, X_P_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xp[i, X_P_flat == x] = 1.0

        Phi_Xq = np.zeros((unique_X.size, X_Q_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xq[i, X_Q_flat == x] = 1.0

        # K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(Phi_Xp @ Phi_Xp.T, Phi_Xq @ Phi_Xq.T, lam_p, lam_q)
        
        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X Phi_X L_YY Phi_X^* W_X)
        term1 = np.trace(W_Xp @ Phi_Xp @ L_YY @ Phi_Xp.T @ W_Xp)
        
        # Term 2: -2 * Tr(W_X Phi_X L_YZ Phi_X'^* W_X')
        term2 = -2.0 * np.trace(W_Xp @ Phi_Xp @ L_YZ @ Phi_Xq.T @ W_Xq)
        
        # Term 3: Tr(W_X' Phi_X' L_ZZ Phi_X'^* W_X')
        term3 = np.trace(W_Xq @ Phi_Xq @ L_ZZ @ Phi_Xq.T @ W_Xq)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)
    
class CMMD1_primal(TestStatistic):
    """
    CMMD1 test statistic using primal estimator of CMO.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD1 test statistic.

        X_P, X_Q are categorical so we use kronecker delta kernel and compute CMO in primal form.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Flatten X_P and X_Q since they're categorical
        X_P_flat = np.asarray(X_P).flatten()
        X_Q_flat = np.asarray(X_Q).flatten()

        # Build one-hot feature maps for the pooled categorical support
        unique_X = np.unique(np.concatenate([np.unique(X_P_flat), np.unique(X_Q_flat)]))
        Phi_Xp = np.zeros((unique_X.size, X_P_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xp[i, X_P_flat == x] = 1.0

        Phi_Xq = np.zeros((unique_X.size, X_Q_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xq[i, X_Q_flat == x] = 1.0

        # K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(Phi_Xp @ Phi_Xp.T, Phi_Xq @ Phi_Xq.T, lam_p, lam_q)
        
        # Estimate covariance matrix
        C_XX = (1 / (n + m)) * (Phi_Xp @ Phi_Xp.T + Phi_Xq @ Phi_Xq.T)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X Phi_X L_YY Phi_X^* W_X C_XX)
        term1 = np.trace(W_Xp @ Phi_Xp @ L_YY @ Phi_Xp.T @ W_Xp @ C_XX)
        
        # Term 2: -2 * Tr(W_X Phi_X L_YZ Phi_X'^* W_X' C_XX)
        term2 = -2.0 * np.trace(W_Xp @ Phi_Xp @ L_YZ @ Phi_Xq.T @ W_Xq @ C_XX)
        
        # Term 3: Tr(W_X' Phi_X' L_ZZ Phi_X'^* W_X' C_XX)
        term3 = np.trace(W_Xq @ Phi_Xq @ L_ZZ @ Phi_Xq.T @ W_Xq @ C_XX)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)

class CMMD2_primal(TestStatistic):
    """
    CMMD2 test statistic using primal estimator of CMO.
    
    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean operators.
    """
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMD2 test statistic.

        X_P, X_Q are categorical so we use kronecker delta kernel and compute CMO in primal form.
        """
        if kernel_y is None:
            kernel_y = kernel_x

        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Flatten X_P and X_Q since they're categorical
        X_P_flat = np.asarray(X_P).flatten()
        X_Q_flat = np.asarray(X_Q).flatten()

        # Build one-hot feature maps for the pooled categorical support
        unique_X = np.unique(np.concatenate([np.unique(X_P_flat), np.unique(X_Q_flat)]))
        Phi_Xp = np.zeros((unique_X.size, X_P_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xp[i, X_P_flat == x] = 1.0

        Phi_Xq = np.zeros((unique_X.size, X_Q_flat.shape[0]))
        for i, x in enumerate(unique_X):
            Phi_Xq[i, X_Q_flat == x] = 1.0

        # K_XpXp, K_XqXq, K_XqXp = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(Phi_Xp @ Phi_Xp.T, Phi_Xq @ Phi_Xq.T, lam_p, lam_q)
        
        # Estimate covariance matrix
        C_XX = (1 / (n + m)) * (Phi_Xp @ Phi_Xp.T + Phi_Xq @ Phi_Xq.T)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X Phi_X L_YY Phi_X^* W_X C_XX^2)
        term1 = np.trace(W_Xp @ Phi_Xp @ L_YY @ Phi_Xp.T @ W_Xp @ C_XX @ C_XX)
        
        # Term 2: -2 * Tr(W_X Phi_X L_YZ Phi_X'^* W_X' C_XX^2)
        term2 = -2.0 * np.trace(W_Xp @ Phi_Xp @ L_YZ @ Phi_Xq.T @ W_Xq @ C_XX @ C_XX)
        
        # Term 3: Tr(W_X' Phi_X' L_ZZ Phi_X'^* W_X' C_XX^2)
        term3 = np.trace(W_Xq @ Phi_Xq @ L_ZZ @ Phi_Xq.T @ W_Xq @ C_XX @ C_XX)
        
        # CMMD statistic
        cmmd = term1 + term2 + term3
        
        return float(cmmd)


class CMMDs(TestStatistic):
    """
    CMMDs test statistic.

    Compares two conditional distributions P(Y|X) and Q(Z|X) using conditional mean embeddings.
    
    Uses pooled covariates \tilde{X} = (X, X')
    """

    # Level controls the power used in the pooled kernel smoothing matrix.
    def __init__(self, level: float = 0.5):
        self.level = float(level)
    
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
        **kwargs
    ) -> float:
        """
        Compute CMMDs test statistic.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        n = X_P.shape[0]
        m = X_Q.shape[0]

        # Compute kernel matrices and inverses
        K_XpXp, K_XqXq, _ = _compute_x_kernels(X_P, X_Q, kernel_x, **kwargs)
        X_tilde = np.vstack([X_P, X_Q])
        K_Xtilde_Xtilde = kernel_x(X_tilde, X_tilde, **kwargs) # (n+m, n+m)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_Xp, W_Xq = _compute_w_matrices(K_XpXp, K_XqXq, lam_p, lam_q)

        # Compute smoothing matrix
        s = self.level
        K = _psd_matrix_power(K_Xtilde_Xtilde, s + 1)

        # Get projection matrices
        P_Xp = np.concatenate([np.eye(n), np.zeros((n, m))], axis=1) # (n, n+m)
        P_Xq = np.concatenate([np.zeros((m, n)), np.eye(m)], axis=1) # (m, n+m)

        # Compute the three terms of the CMMD statistic
        # Term 1: Tr(W_X L_YY W_X K_Xtilde K_tildeX)
        term1 = np.trace(W_Xp @ L_YY @ W_Xp @ P_Xp @ K @ P_Xp.T)

        # Term 2: -2 * Tr(W_X L_YZ W_X' K_X'tilde K_tildeX)
        term2 = -2.0 * np.trace(W_Xp @ L_YZ @ W_Xq @ P_Xq @ K @ P_Xp.T)

        # Term 3: Tr(W_X' L_ZZ W_X' K_X'tilde K_tildeX')
        term3 = np.trace(W_Xq @ L_ZZ @ W_Xq @ P_Xq @ K @ P_Xq.T)

        cmmd = (term1 + term2 + term3) / (n + m) ** s

        return float(cmmd)

class CMMD_rho(TestStatistic):
    """CMMD_s with reference covariance rho C_P + (1-rho) C_Q.

    The statistic is Tr(Delta* Delta C_R**s), with uncentered empirical
    covariances C_P and C_Q normalized by their own sample sizes.
    Setting rho=n/(n+m) recovers the pooled CMMDs estimator.
    """

    def __init__(self, rho: float = 0.5, s: float = 1.0):
        if not np.isfinite(rho) or not 0 < rho < 1:
            raise ValueError("rho must lie strictly between 0 and 1")
        if not np.isfinite(s) or s < 0:
            raise ValueError("s must be finite and nonnegative")
        self.rho = float(rho)
        self.s = float(s)

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
        if kernel_y is None:
            kernel_y = kernel_x
        n, m = X_P.shape[0], X_Q.shape[0]
        if n == 0 or m == 0:
            raise ValueError("Both samples must be nonempty")
        X_tilde = np.vstack([X_P, X_Q])
        K = kernel_x(X_tilde, X_tilde, **kwargs)
        L_YY, L_ZZ, L_YZ = _compute_yz_kernels(Y, Z, kernel_y, **kwargs)
        W_p, W_q = _compute_w_matrices(K[:n, :n], K[n:, n:], lam_p, lam_q)

        # For pooled feature columns Phi and D=diag(rho/n, (1-rho)/m),
        # S = Phi* C_R**s Phi = D**(-1/2) (D**(1/2) K D**(1/2))**(s+1) D**(-1/2).
        weights = np.concatenate([np.full(n, self.rho / n),
                                  np.full(m, (1 - self.rho) / m)])
        if self.s == 0:
            S = K
        elif self.s == 1:
            S = (K * weights) @ K
        elif self.s == 2:
            KD = K * weights
            S = KD @ KD @ K
        else:
            root = np.sqrt(weights)
            weighted_gram = root[:, None] * K * root[None, :]
            S = _psd_matrix_power(weighted_gram, self.s + 1)
            S = S / root[:, None] / root[None, :]

        term_p = np.einsum("ij,ji->", W_p @ L_YY @ W_p, S[:n, :n])
        cross = np.einsum("ij,ji->", W_p @ L_YZ @ W_q, S[n:, :n])
        term_q = np.einsum("ij,ji->", W_q @ L_ZZ @ W_q, S[n:, n:])
        return float(term_p - 2 * cross + term_q)


class Algorithm(ABC):
    """
    Abstract base class for hypothesis testing algorithms.
    
    An algorithm takes raw samples and a test statistic, and performs
    the actual hypothesis test (e.g., permutation test, bootstrap, etc.).
    """
    
    @abstractmethod
    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_fn: Callable,
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float]:
        """
        Perform hypothesis test.
        
        Parameters
        ----------
        X_P : np.ndarray, shape (n, d)
            Covariates from distribution P.
        Y : np.ndarray, shape (n, k)
            Outcomes from distribution P (conditional on X_P).
        X_Q : np.ndarray, shape (m, d)
            Covariates from distribution Q.
        Z : np.ndarray, shape (m, k)
            Outcomes from distribution Q (conditional on X_Q).
        test_statistic : TestStatistic
            The test statistic to use.
        kernel_fn : callable
            Kernel function k(X, Y, **kwargs).
        algo_kwargs : dict, optional
            Algorithm-level parameters (alpha, B, lam_p, lam_q, random_state, etc.).
        stat_kwargs : dict, optional
            Test-statistic/kernel parameters (bandwidth, estimator, etc.).
        
        Returns
        -------
        stat : float
            The computed test statistic on the original data.
        p_value : float
            The estimated p-value for the test.
        """
        pass


class Test_Same_Marginal(Algorithm):
    """
    Kernel two-sample test for conditional distribution (P_X = Q_X).
    """

    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float] | Tuple[float, float, np.ndarray]:
        """
        Kernel two-sample test for conditional distribution (P_X = Q_X).

        Set algo_kwargs["return_bootstrap"] to True to also return the
        resampled statistics as a third result. The default returns two values.
        """
        if kernel_y is None:
            kernel_y = kernel_x
        
        algo_kwargs = algo_kwargs or {}
        stat_kwargs = stat_kwargs or {}

        alpha = float(algo_kwargs.get("alpha", 0.05))
        B = int(algo_kwargs.get("B", 1000))
        lam_p = float(algo_kwargs.get("lam_p", algo_kwargs.get("lam", 1e-3)))
        lam_q = float(algo_kwargs.get("lam_q", algo_kwargs.get("lam", 1e-3)))
        rng = np.random.default_rng(algo_kwargs.get("random_state", None))

        # Compute test statistic on original data
        stat = test_statistic.compute(
            X_P,
            Y,
            X_Q,
            Z,
            lam_p,
            lam_q,
            kernel_x,
            kernel_y,
            **stat_kwargs,
        )

        X_all = np.vstack([X_P, X_Q])
        Y_all = np.vstack([Y, Z])
        n = X_P.shape[0]
        total = X_all.shape[0]

        # Bootstrap/permutation under same-marginal null
        stats_boot = np.zeros(B, dtype=float)
        indices = np.arange(total)
        for b in range(B):
            idx_p = rng.choice(indices, size=n, replace=False)
            mask_p = np.zeros(total, dtype=bool)
            mask_p[idx_p] = True
            idx_q = indices[~mask_p]

            X_P_b = X_all[idx_p]
            Y_b = Y_all[idx_p]
            X_Q_b = X_all[idx_q]
            Z_b = Y_all[idx_q]

            stats_boot[b] = test_statistic.compute(
                X_P_b,
                Y_b,
                X_Q_b,
                Z_b,
                lam_p,
                lam_q,
                kernel_x,
                kernel_y,
                **stat_kwargs,
            )

        p_value = (1.0 + np.sum(stats_boot >= stat)) / (1.0 + B)

        # Decision is p_value < alpha (not returned, but computed for callers)
        _ = p_value < alpha

        if algo_kwargs.get("return_bootstrap", False):
            return float(stat), float(p_value), stats_boot

        return float(stat), float(p_value)


class Test_Diff_Marginal(Algorithm):
    """
    Kernel two-sample test for conditional distribution (P_X != Q_X).
    
    Uses propensity score-based bootstrap for the case where marginal distributions
    of X differ between the two groups.
    """
    
    @staticmethod
    def _build_conditional_sampler(e: np.ndarray, n_p: int):
        """
        Precompute quantities needed to sample labels T conditional on

            sum(T) = n_p

        when T_i ~ Bernoulli(e_i) before conditioning.

        Returns
        -------
        log_w : (N,) array
            Log propensity odds.
        log_dp : (N + 1, n_p + 1) array
            Dynamic-programming table.
        """
        e = np.asarray(e, dtype=float).reshape(-1)

        if np.any(e <= 0.0) or np.any(e >= 1.0):
            raise ValueError(
                "Propensity scores must lie strictly between 0 and 1."
            )

        N = len(e)

        if not 0 <= n_p <= N:
            raise ValueError("n_p must lie between 0 and N.")

        # log(e_i / (1 - e_i))
        log_w = np.log(e) - np.log1p(-e)

        # log_dp[i, k] =
        # log(sum of product of odds over all subsets of size k chosen from observations i, ..., N-1)
        log_dp = np.full((N + 1, n_p + 1), -np.inf)
        log_dp[N, 0] = 0.0

        for i in range(N - 1, -1, -1):
            log_dp[i, 0] = 0.0

            max_k = min(n_p, N - i)

            # Vectorised version of
            # dp[i,k] = dp[i+1,k] + w_i dp[i+1,k-1]
            # on the log scale.
            log_dp[i, 1:max_k + 1] = np.logaddexp(
                log_dp[i + 1, 1:max_k + 1],
                log_w[i] + log_dp[i + 1, :max_k],
            )

        return log_w, log_dp

    @staticmethod
    def _sample_conditional_labels_batch(
        log_w: np.ndarray,
        log_dp: np.ndarray,
        n_p: int,
        B: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate B conditional label assignments at once.

        Returns
        -------
        labels : (B, N) boolean array
            Each row contains exactly n_p True entries.
        """
        N = len(log_w)

        labels = np.zeros((B, N), dtype=bool)
        remaining = np.full(B, n_p, dtype=int)

        for i in range(N):

            # Some replicates may have reached the point where every
            # remaining observation must be selected.
            must_take = remaining == (N - i)

            if np.any(must_take):
                labels[must_take, i:] = True
                remaining[must_take] = 0

            active = remaining > 0

            if not np.any(active):
                break

            active_idx = np.flatnonzero(active)
            r = remaining[active_idx]

            # Conditional probability of assigning observation i to P:
            # P(T_i = 1 | need r more assignments)
            log_p = (
                log_w[i]
                + log_dp[i + 1, r - 1]
                - log_dp[i, r]
            )

            p = np.exp(log_p)
            p = np.clip(p, 0.0, 1.0)

            selected = rng.random(len(active_idx)) < p
            selected_idx = active_idx[selected]

            labels[selected_idx, i] = True
            remaining[selected_idx] -= 1

        return labels

    def test(
        self,
        X_P: np.ndarray,
        Y: np.ndarray,
        X_Q: np.ndarray,
        Z: np.ndarray,
        test_statistic: TestStatistic,
        kernel_x: Callable,
        kernel_y: Callable | None = None,
        algo_kwargs: dict | None = None,
        stat_kwargs: dict | None = None
    ) -> Tuple[float, float]:

        if kernel_y is None:
            kernel_y = kernel_x

        algo_kwargs = algo_kwargs or {}
        stat_kwargs = stat_kwargs or {}

        # This alpha is the significance level, not the mixture parameter.
        significance_level = float(algo_kwargs.get("alpha", 0.05))

        B = int(algo_kwargs.get("B", 1000))

        lam_p = float(
            algo_kwargs.get("lam_p", algo_kwargs.get("lam", 1e-3))
        )
        lam_q = float(
            algo_kwargs.get("lam_q", algo_kwargs.get("lam", 1e-3))
        )

        propensity_fn = algo_kwargs.get("propensity_fn", None)

        rng = np.random.default_rng(
            algo_kwargs.get("random_state", None)
        )

        if propensity_fn is None:
            raise ValueError(
                "propensity_fn must be provided for Test_Diff_Marginal"
            )

        # ------------------------------------------------------------
        # Original test statistic
        # ------------------------------------------------------------

        stat = test_statistic.compute(
            X_P,
            Y,
            X_Q,
            Z,
            lam_p,
            lam_q,
            kernel_x,
            kernel_y,
            **stat_kwargs,
        )

        n = X_P.shape[0]
        m = X_Q.shape[0]
        N = n + m

        # ------------------------------------------------------------
        # Pool data
        # ------------------------------------------------------------

        X_all = np.concatenate([X_P, X_Q], axis=0)
        W_all = np.concatenate([Y, Z], axis=0)

        if X_all.ndim == 2 and X_all.shape[1] == 1:
            propensity_input = X_all[:, 0]
        else:
            propensity_input = X_all

        e_all = np.asarray(
            propensity_fn(propensity_input),
            dtype=float,
        ).reshape(-1)

        if len(e_all) != N:
            raise ValueError(
                "propensity_fn must return one propensity per observation."
            )

        # ------------------------------------------------------------
        # PRECOMPUTE conditional resampling distribution ONCE
        # ------------------------------------------------------------

        log_w, log_dp = self._build_conditional_sampler(
            e=e_all,
            n_p=n,
        )

        # ------------------------------------------------------------
        # Generate all B label assignments
        # ------------------------------------------------------------

        labels_boot = self._sample_conditional_labels_batch(
            log_w=log_w,
            log_dp=log_dp,
            n_p=n,
            B=B,
            rng=rng,
        )

        # ------------------------------------------------------------
        # Calculate test statistics
        # ------------------------------------------------------------

        stats_boot = np.empty(B, dtype=float)

        for b in range(B):

            labels_p = labels_boot[b]
            labels_q = ~labels_p

            X_P_b = X_all[labels_p]
            Y_b = W_all[labels_p]

            X_Q_b = X_all[labels_q]
            Z_b = W_all[labels_q]

            stats_boot[b] = test_statistic.compute(
                X_P_b,
                Y_b,
                X_Q_b,
                Z_b,
                lam_p,
                lam_q,
                kernel_x,
                kernel_y,
                **stat_kwargs,
            )

        p_value = (
            1.0 + np.sum(stats_boot >= stat)
        ) / (B + 1.0)

        _ = p_value < significance_level

        return float(stat), float(p_value)


def run_experiment(
    n_trials: int,
    n_samples: int,
    algorithm: Algorithm,
    test_statistic: TestStatistic,
    data_generator: Callable,
    kernel_fn: Callable,
    algo_kwargs: dict | None = None,
    stat_kwargs: dict | None = None
) -> np.ndarray:
    """
    Run hypothesis test multiple times and collect results.
    
    Parameters
    ----------
    n_trials : int
        Number of independent trials to run.
    n_samples : int
        Number of samples in each trial.
    algorithm : Algorithm
        The hypothesis testing algorithm to use.
    test_statistic : TestStatistic
        The test statistic to compute.
    data_generator : callable
        Function that generates (X_P, Y, X_Q, Z) samples.
    kernel_fn : callable
        Kernel function to use.
    algo_kwargs : dict, optional
        Algorithm-level parameters passed to algorithm.test().
    stat_kwargs : dict, optional
        Test-statistic/kernel parameters passed to algorithm.test().
    
    Returns
    -------
    results : np.ndarray, shape (n_trials, 2)
        Array where each row is [test_statistic, p_value] for each trial.
    """
    results = np.zeros((n_trials, 2))
    
    for trial in range(n_trials):
        # Generate data for this trial
        X_P, Y, X_Q, Z = data_generator(n_samples)
        
        # Run test
        stat, pval = algorithm.test(
            X_P, Y, X_Q, Z,
            test_statistic,
            kernel_fn,
            algo_kwargs=algo_kwargs,
            stat_kwargs=stat_kwargs,
        )
        
        results[trial] = [stat, pval]
    
    return results
