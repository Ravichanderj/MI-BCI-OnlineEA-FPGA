"""
Hardware-friendly approximations used INSIDE training (hardware-aware training),
so the network learns to be accurate with exactly the functions the FPGA computes.

1) Base-2 softmax
   softmax(z)_i = e^{z_i} / sum_j e^{z_j} = 2^{u_i} / sum_j 2^{u_j},  u = z * log2(e)
   The constant log2(e)/sqrt(d_h) is folded into W_q at export -> zero runtime cost.

2) 2^u for u <= 0 (after max subtraction):
   u = i + f,  i = floor(u) (integer),  f in [0,1)
   2^u = 2^f >> (-i)             -> barrel shift
   2^f ~ PWL with N_EXP knots     -> small LUT (slope, intercept) indexed by top bits of f

3) Reciprocal of the row sum S >= 1 (the max element contributes 2^0 = 1):
   S = m * 2^e, m in [1,2)  (leading-one detector gives e)
   1/S = (1/m) >> e,   1/m ~ PWL with N_RECIP knots

4) Normalisation: LayerNorm is replaced by BatchNorm, which is folded into the next
   linear layer at export -> no sqrt / division at inference at all.

Tables are exported to hls/pwl_tables.h by export_hls.py so SW and HW use identical values.
"""
import math
import torch

N_EXP = 8     # segments for 2^f, f in [0,1)   -> max rel. error ~9e-4
N_RECIP = 16  # segments for 1/m, m in [1,2)   -> max rel. error ~1e-3
MIN_EXP = -16  # 2^u underflows to 0 below this (shift >= 16 in HW)

EXP_KNOTS = torch.tensor([2.0 ** (i / N_EXP) for i in range(N_EXP + 1)])
RECIP_KNOTS = torch.tensor([1.0 / (1.0 + i / N_RECIP) for i in range(N_RECIP + 1)])
LOG2E = math.log2(math.e)


def _pwl(t, knots, n):
    """t in [0,1) -> chord interpolation between knots."""
    idx = torch.clamp(torch.floor(t * n), 0, n - 1)
    frac = t * n - idx
    k = knots.to(t.device)
    idx = idx.long()
    y0, y1 = k[idx], k[idx + 1]
    return y0 + (y1 - y0) * frac


def pow2_pwl(u):
    """2^u for u <= 0."""
    u = torch.clamp(u, min=float(MIN_EXP))
    i = torch.floor(u)
    f = u - i
    return _pwl(f, EXP_KNOTS, N_EXP) * torch.exp2(i)


def recip_pwl(s):
    """1/s for s >= 1."""
    e = torch.floor(torch.log2(s)).detach()   # leading-one position (no gradient needed)
    m = s * torch.exp2(-e)                     # mantissa in [1,2)
    return _pwl(m - 1.0, RECIP_KNOTS, N_RECIP) * torch.exp2(-e)


def softmax_pow2(u, dim=-1):
    """Softmax in the log2 domain using only shifts, LUT-PWL and multiply."""
    u = u - u.max(dim=dim, keepdim=True).values
    p = pow2_pwl(u)
    return p * recip_pwl(p.sum(dim=dim, keepdim=True))


if __name__ == "__main__":
    # accuracy check of the approximations
    x = torch.linspace(-12, 0, 100001)
    e1 = ((pow2_pwl(x) - torch.exp2(x)) / torch.exp2(x)).abs().max()
    s = torch.linspace(1, 64, 100001)
    e2 = ((recip_pwl(s) - 1 / s) * s).abs().max()
    z = torch.randn(1000, 15) * 3
    e3 = (softmax_pow2(z * LOG2E) - z.softmax(-1)).abs().max()
    print(f"max rel err 2^x: {e1:.2e} | 1/s: {e2:.2e} | softmax abs err: {e3:.2e}")
