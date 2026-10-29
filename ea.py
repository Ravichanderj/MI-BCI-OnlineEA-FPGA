"""
Euclidean Alignment (EA) — offline (training) and online/recursive (deployment).

Offline EA (He & Wu, IEEE TBME 2020):
    R_bar = mean_i  X_i X_i^T / T          (per subject)
    X~_i  = R_bar^{-1/2} X_i

Online EA (this work) — hardware-friendly, calibration-free (no labels):
    P_n   = X_n X_n^T / T                                    (trial covariance, symmetric)
    R_n   = R_{n-1} + a_n (P_n - R_{n-1}),  a_n = max(1/n, 2^-k)   (running mean -> EMA)
    S_n   = R_n + eps * tr(R_n)/C * I                         (shrinkage: bounds condition no.)
    r_n   = tr(S_n)^{-1/2}      scalar Newton, seeded by a leading-one detector (no sqrt unit)
    A_n   = r_n^2 * S_n         (eigenvalues in (0,1], no divider)
    Z_n   = A_n^{-1/2}          coupled Newton-Schulz, K fixed iterations (matmul/add only)
    W_n   = r_n * Z_n ;   X~_n = W_n X_n

Coupled Newton-Schulz (Higham, "Functions of Matrices", 2008) — numerically stable,
converges for ||I - A|| < 1, which trace normalisation guarantees:
    Y_0 = A, Z_0 = I
    T_k = 0.5 (3I - Z_k Y_k);  Y_{k+1} = Y_k T_k;  Z_{k+1} = T_k Z_k
    Y_k -> A^{1/2},  Z_k -> A^{-1/2}
Cost per iteration: 3 CxC matmuls (C=22 -> ~32k MAC). K=18 -> ~0.6 M MAC per trial,
i.e. well under 1 ms on the Zynq-7020 PL; latency is fixed and data-independent.

The same operation order is used in hls/online_ea.cpp, so SW accuracy == HW accuracy
(up to fixed-point rounding, which the HLS testbench quantifies).
"""
import numpy as np


# ---------------------------------------------------------------- offline EA
def trial_covariances(X):
    """X: (n, C, T) -> (n, C, C)"""
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]


def shrink(R, eps):
    C = R.shape[-1]
    return R + eps * np.trace(R) / C * np.eye(C)


def inv_sqrtm_eigh(R):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-12, None)
    return (V * w ** -0.5) @ V.T


def offline_ea(X, eps=1e-2):
    """Per-subject EA used for source subjects during training (and as the 'oracle' baseline)."""
    R = shrink(trial_covariances(X).mean(0), eps)
    W = inv_sqrtm_eigh(R)
    return np.einsum("cd,ndt->nct", W, X).astype(np.float32), W


# ---------------------------------------------------------------- Newton iterations
def coupled_ns_inv_sqrt(A, iters=18):
    """Z -> A^{-1/2} for SPD A with eigenvalues in (0, 1]."""
    C = A.shape[0]
    I = np.eye(C)
    Y, Z = A.copy(), I.copy()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return 0.5 * (Z + Z.T)   # symmetric by construction; averaging removes rounding asymmetry


def scalar_inv_sqrt(s, iters=6):
    """s^{-1/2}. Seed: s = m*2^e (leading-one detector) -> r0 = 2^-ceil(e/2), r0^2 s in [0.5,2)."""
    e = np.floor(np.log2(s))
    r = 2.0 ** (-np.ceil(e / 2))
    for _ in range(iters):
        r = r * (1.5 - 0.5 * s * r * r)
    return r


def ns_iters_needed(A, tol=1e-4, max_iter=60):
    """Diagnostic: iterations until ||Z A Z - I||_F / C < tol."""
    C = A.shape[0]
    for k in range(1, max_iter + 1):
        Z = coupled_ns_inv_sqrt(A, k)
        if np.linalg.norm(Z @ A @ Z - np.eye(C)) / C < tol:
            return k
    return max_iter


# ---------------------------------------------------------------- online EA
class OnlineEA:
    """
    Streaming, label-free EA for a new (target) user.

    k_ema : steady-state forgetting factor a = 2^-k_ema (k_ema=5 -> ~32-trial memory);
            for n <= 2^k_ema the update is an exact running mean (1/n from a 2^k-entry LUT)
    eps   : shrinkage (use the same value as offline EA in training)
    ns_iters : fixed number of coupled Newton-Schulz iterations per trial
    include_current : update the reference with the current (unlabelled) trial before
            whitening it. Causal: the trial is fully observed before it is classified,
            and no label is used.
    """

    def __init__(self, C, k_ema=5, eps=1e-2, ns_iters=18, include_current=True):
        self.C, self.k, self.eps, self.K = C, k_ema, eps, ns_iters
        self.include_current = include_current
        self.reset()

    def reset(self):
        self.n = 0
        self.R = np.zeros((self.C, self.C))

    def _push(self, P):
        self.n += 1
        a = max(1.0 / self.n, 2.0 ** -self.k)
        self.R = self.R + a * (P - self.R)

    def whitener(self):
        S = shrink(self.R, self.eps)
        r = scalar_inv_sqrt(np.trace(S))
        Z = coupled_ns_inv_sqrt(r * r * S, self.K)
        return r * Z

    def step(self, x):
        """x: (C, T) one trial -> whitened (C, T)"""
        P = x @ x.T / x.shape[1]
        if self.include_current or self.n == 0:
            self._push(P)
            W = self.whitener()
        else:
            W = self.whitener()
            self._push(P)
        return (W @ x).astype(np.float32)

    def run(self, X):
        """X: (n, C, T) in natural trial order -> aligned (n, C, T)"""
        self.reset()
        return np.stack([self.step(x) for x in X])
