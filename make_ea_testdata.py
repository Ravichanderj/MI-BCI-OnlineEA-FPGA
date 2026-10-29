"""Writes ea_in.bin (n x C x T float32) and ea_ref_W.bin (C x C float64, final whitener
from python/ea.py::OnlineEA) so the HLS testbench can check SW == HW."""
import sys, numpy as np
sys.path.insert(0, "../python")
from ea import OnlineEA

rng = np.random.default_rng(1)
C, T, n = 22, 500, 60
M = np.eye(C) + 0.6 * rng.standard_normal((C, C)); M[:, :3] *= 4
X = (np.einsum("cd,ndt->nct", M, rng.standard_normal((n, C, T))) * 8).astype(np.float32)
X.tofile("ea_in.bin")
oea = OnlineEA(C, k_ema=5, eps=1e-2, ns_iters=18)
oea.run(X)
oea.whitener().astype(np.float64).tofile("ea_ref_W.bin")
print("max |x| =", np.abs(X).max(), " trials:", n)
