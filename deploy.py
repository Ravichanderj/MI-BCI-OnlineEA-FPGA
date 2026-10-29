"""
Deployment transform + 8-bit quantisation simulation + HLS export.

Steps (all verified numerically in __main__ / smoke_test.py):
 1. Fold every BatchNorm into the adjacent linear op (no normalisation at inference).
 2. Reorder temporal & spatial convolutions (both linear, BN in between is affine):
        original : temporal (F1 x C x T x K1) -> spatial     ~2.99 M MAC
        reordered: spatial (F2 x C x T)      -> temporal on F2 channels   ~0.43 M MAC  (-86%)
 3. Fold the attention scale log2(e)/sqrt(d_h) into W_q.
 4. Post-training int8 quantisation (per-channel weights, per-tensor activations, int32
    accumulators, Q1.15 attention probabilities) simulated with exact rounding/clipping.
 5. Export int8 weights, int32 biases and fixed-point requantisation multipliers to
    hls/hanet_weights.h plus a golden test vector for C/RTL co-simulation.

Usage:
  python deploy.py --ckpt runs/2a/hanet_target1.pt --dataset 2a --target 1
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from approx import softmax_pow2, N_EXP, N_RECIP, EXP_KNOTS, RECIP_KNOTS
from model import HANet, hw_elu


# ----------------------------------------------------------------- BN folding
def bn_affine(bn):
    a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return a.detach(), (bn.bias - bn.running_mean * a).detach()


@torch.no_grad()
def fold(model):
    m, c = model.eval(), model.cfg
    F1, D = c["F1"], c["D"]
    F2 = F1 * D
    g = torch.arange(F2) // D                           # group of each spatial output
    a1, b1 = bn_affine(m.bn1); a2, b2 = bn_affine(m.bn2); a3, b3 = bn_affine(m.bn3)
    w1 = m.temporal.weight[:, 0, 0, :]                  # (F1,K1)
    ws = m.spatial.weight[:, 0, :, 0]                   # (F2,C)
    P = {"cfg": c}
    P["Ws"] = ws.clone()                                # spatial mix, no bias
    P["Wt"] = (a2 * a1[g])[:, None] * w1[g]             # temporal per F2 channel (F2,K1)
    P["bt"] = a2 * b1[g] * ws.sum(1) + b2
    P["Wdw"] = m.sep_dw.weight[:, 0, 0, :].clone()      # (F2,K2)
    P["Wpw"] = a3[:, None] * m.sep_pw.weight[:, :, 0, 0]
    P["bpw"] = b3.clone()
    if c["attention"]:
        aa, ba = bn_affine(m.bn_a); af, bf = bn_affine(m.bn_f)
        at = m.attn
        for n, lin in (("q", at.q), ("k", at.k), ("v", at.v)):
            W, b = lin.weight * aa[None, :], lin.weight @ ba + lin.bias
            if n == "q":
                W, b = W * at.scale, b * at.scale
            P["W" + n], P["b" + n] = W, b
        P["Wo"], P["bo"] = at.o.weight.clone(), at.o.bias.clone()
        P["Wf1"], P["bf1"] = m.ffn1.weight * af[None, :], m.ffn1.weight @ bf + m.ffn1.bias
        P["Wf2"], P["bf2"] = m.ffn2.weight.clone(), m.ffn2.bias.clone()
        P["heads"], P["approx"] = c["heads"], c["approx"]
    P["Wh"], P["bh"] = m.head.weight.clone(), m.head.bias.clone()
    return P


# ----------------------------------------------------------------- quantisers
def qw(W, bits=8):
    """per-output-channel symmetric weight quantisation -> (fake-quant W, scales, int W)"""
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().reshape(W.shape[0], -1).max(1).values.clamp(min=1e-12) / qmax
    shape = (-1,) + (1,) * (W.dim() - 1)
    Wi = torch.clamp(torch.round(W / s.view(shape)), -qmax, qmax)
    return Wi * s.view(shape), s, Wi.to(torch.int32)


class ActQ:
    """Per-tensor symmetric activation quantiser with calibration mode."""
    def __init__(self, bits=8, pct=99.99):
        self.bits, self.pct, self.calib, self.scale, self.obs = bits, pct, True, {}, {}

    def __call__(self, x, name):
        if self.calib:
            v = torch.quantile(x.detach().abs().flatten().float()[:2_000_000], self.pct / 100)
            self.obs[name] = max(self.obs.get(name, 0.0), float(v))
            return x
        qmax = 2 ** (self.bits - 1) - 1
        s = self.scale[name]
        return torch.clamp(torch.round(x / s), -qmax, qmax) * s

    def finalize(self):
        qmax = 2 ** (self.bits - 1) - 1
        self.scale = {k: max(v, 1e-8) / qmax for k, v in self.obs.items()}
        self.calib = False


def q_grid(x, frac_bits, lo, hi):
    return torch.clamp(torch.round(x * 2 ** frac_bits), lo, hi) / 2 ** frac_bits


# ----------------------------------------------------------------- deployed forward
def forward_deploy(P, x, aq=None, wbits=8):
    """x: (B, C, T) EA-aligned. aq=None -> float reference; aq=ActQ -> int8 simulation."""
    c = P["cfg"]; Q = (lambda t, n: t) if aq is None else aq
    W = (lambda k: P[k]) if aq is None else (lambda k: qw(P[k], wbits)[0])
    K1, K2, P1, P2 = c["K1"], c["K2"], c["P1"], c["P2"]
    cact = F.relu if c.get("act", "relu") == "relu" else (lambda t: hw_elu(t, c["approx"]))
    x = Q(x, "in")
    u = Q(torch.einsum("fc,bct->bft", W("Ws"), x), "spat")                       # (B,F2,T)
    u = F.conv1d(F.pad(u, ((K1 - 1) // 2, K1 // 2)), W("Wt").unsqueeze(1), P["bt"],
                 groups=u.shape[1])
    u = Q(F.avg_pool1d(cact(u), P1), "blk1")                                    # (B,F2,T/P1)
    u = F.conv1d(F.pad(u, ((K2 - 1) // 2, K2 // 2)), W("Wdw").unsqueeze(1), groups=u.shape[1])
    u = Q(u, "dw")
    u = torch.einsum("gf,bft->bgt", W("Wpw"), u) + P["bpw"][:, None]
    x0 = Q(F.avg_pool1d(cact(u), P2).transpose(1, 2), "tok")                    # (B,L,F2)
    x = x0
    if "Wq" in P:
        B, L, d = x.shape; h = P["heads"]; dh = d // h
        lin = lambda t, k: t @ W("W" + k).T + P["b" + k]
        sh = lambda t: t.view(B, L, h, dh).transpose(1, 2)
        q, k, v = (sh(Q(lin(x, n), n)) for n in ("q", "k", "v"))
        s = q @ k.transpose(-1, -2)
        if aq is not None:
            s = q_grid(s, 8, -32768, 32767)            # log2-domain scores, Q7.8 int16
        p = softmax_pow2(s) if P["approx"] else s.softmax(-1)
        if aq is not None:
            p = q_grid(p, 15, 0, 32767)                 # probabilities, Q1.15
        o = Q((p @ v).transpose(1, 2).reshape(B, L, d), "pv")
        x = Q(x + Q(lin(o, "o"), "o"), "x1")
        f = Q(F.relu(lin(x, "f1")), "f1")
        x = Q(x + Q(lin(f, "f2"), "f2"), "x2")
    return x.flatten(1) @ W("Wh").T + P["bh"]


# ----------------------------------------------------------------- HLS export
def requant_mult(real, bits=15):
    """real multiplier -> (M, shift) with M in [2^(bits-1), 2^bits): out = (acc*M) >> shift"""
    real = np.asarray(real, dtype=np.float64)
    sh = np.floor(bits - 1 - np.log2(real)).astype(int)
    M = np.round(real * 2.0 ** sh).astype(np.int64)
    return M, sh


def c_array(name, arr, ctype):
    arr = np.asarray(arr)
    dims = "".join(f"[{d}]" for d in arr.shape)
    body = ", ".join(str(int(v)) for v in arr.flatten())
    return f"static const {ctype} {name}{dims} = {{{body}}};\n"


def export_hls(P, aq, path, x_test=None, wbits=8):
    s = aq.scale
    chain = [  # (weight key, bias key, input act, output act)
        ("Ws", None, "in", "spat"), ("Wt", "bt", "spat", "blk1"), ("Wdw", None, "blk1", "dw"),
        ("Wpw", "bpw", "dw", "tok")]
    if "Wq" in P:
        chain += [("Wq", "bq", "tok", "q"), ("Wk", "bk", "tok", "k"), ("Wv", "bv", "tok", "v"),
                  ("Wo", "bo", "pv", "o"), ("Wf1", "bf1", "x1", "f1"), ("Wf2", "bf2", "f1", "f2")]
    chain += [("Wh", "bh", "x2" if "Wq" in P else "tok", None)]
    txt = ["// Auto-generated by deploy.py — HA-Net int8 weights\n#pragma once\n#include <stdint.h>\n"]
    c = P["cfg"]
    for k in ("C", "T", "F1", "D", "K1", "P1", "K2", "P2", "heads", "n_cls"):
        txt.append(f"#define HANET_{k.upper()} {c[k]}\n")
    for wk, bk, a_in, a_out in chain:
        _, sw, Wi = qw(P[wk], wbits)
        txt.append(c_array(f"{wk}_q", Wi.numpy(), "int8_t"))
        acc_scale = s[a_in] * sw.numpy()
        if bk is not None:
            txt.append(c_array(f"{bk}_q", np.round(P[bk].numpy() / acc_scale), "int32_t"))
        if a_out is not None:
            M, sh = requant_mult(acc_scale / s[a_out])
            txt.append(c_array(f"{wk}_M", M, "int32_t") + c_array(f"{wk}_sh", sh, "int8_t"))
        else:
            txt.append(f"// logits scale per class: {list(np.round(acc_scale, 8))}\n")
    # constants for the integer HW-ELU and logit scaling (used by hls/hanet_cnn.cpp)
    import math
    _, sw_h, _ = qw(P["Wh"], wbits)
    ls = s["x2" if "Wq" in P else "tok"] * sw_h.numpy()
    cst = ["// Auto-generated by deploy.py: integer ELU constants and logit scales\n#pragma once\n#include <stdint.h>\n"]
    for k in ("blk1", "tok"):
        cst.append(f"#define ELU_K16_{k.upper()} {round(s[k] * math.log2(math.e) * 2 ** 16)}\n"
                   f"#define ELU_INV16_{k.upper()} {round(2 ** 16 / s[k])}\n")
    cst.append("static const float LOGIT_SCALE[] = {" + ", ".join(f"{v:.9e}f" for v in ls) + "};\n")
    cst.append("static const int32_t LOGIT_W[] = {" + ", ".join(str(int(round(v / ls.max() * 32768))) for v in ls) + "};\n")
    cst.append(f"#define S_IN {s['in']:.9e}f\n")
    open(path.replace("hanet_weights.h", "hanet_consts.h"), "w").write("".join(cst))
    txt.append("// activation scales (real = int * scale):\n")
    txt += [f"//   {k}: {v:.6e}\n" for k, v in s.items()]
    open(path, "w").write("".join(txt))

    # PWL tables in Q1.15 shared with approx.py
    tab = ["// PWL tables (Q1.15) — generated from approx.py\n#pragma once\n#include <stdint.h>\n",
           f"#define N_EXP {N_EXP}\n#define N_RECIP {N_RECIP}\n",
           c_array("EXP_KNOTS_Q14", np.round(EXP_KNOTS.numpy() * 2 ** 14), "int32_t").replace(
               "static", "// knots of 2^f in Q2.14\nstatic"),
           c_array("RECIP_KNOTS_Q15", np.round(RECIP_KNOTS.numpy() * 2 ** 15), "int32_t")]
    open(path.replace("hanet_weights.h", "pwl_tables.h"), "w").write("".join(tab))

    if x_test is not None:   # golden vector: int8 input + expected logits of the int8 simulation
        xi = torch.clamp(torch.round(x_test / s["in"]), -127, 127)
        logits = forward_deploy(P, x_test.unsqueeze(0), aq, wbits)[0]
        open(path.replace("hanet_weights.h", "golden.h"), "w").write(
            "#pragma once\n#include <stdint.h>\n" + c_array("GOLDEN_IN", xi.numpy(), "int8_t")
            + "static const float GOLDEN_LOGITS[] = {" + ", ".join(f"{v:.6f}f" for v in logits.tolist())
            + "};\n")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="2a")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--hls-dir", default="../hls")
    args = ap.parse_args()

    from data import load_moabb, synthetic
    from ea import offline_ea, OnlineEA
    ck = torch.load(args.ckpt, map_location="cpu")
    model = HANet(**ck["cfg"]); model.load_state_dict(ck["state"]); model.eval()
    a = ck["args"]
    data = synthetic() if args.synthetic else load_moabb(args.dataset, classes=a.get("classes", "all"))
    Xs, ys, ss = data[args.target]
    from train_loso import align_online, align_offline
    ns = argparse.Namespace(k_ema=a["k_ema"], eps=a["eps"], ns_iters=a["ns_iters"],
                            reset_session=a.get("reset_session", True))
    Xon = torch.from_numpy(align_online(Xs, ss, ns))
    # calibration set: EA-aligned SOURCE subjects only (target labels/data never used)
    cal = np.concatenate([align_offline(data[s][0], data[s][2], a["eps"], a.get("per_session", True))[:48]
                          for s in data if s != args.target])
    cal = torch.from_numpy(cal)

    P = fold(model)
    with torch.no_grad():
        ref = model(Xon.unsqueeze(1))
        fl = forward_deploy(P, Xon)
        print(f"fold+reorder max |diff| vs PyTorch model: {(ref - fl).abs().max():.2e}")
        aq = ActQ(args.abits); forward_deploy(P, cal, aq, args.wbits); aq.finalize()
        qo = forward_deploy(P, Xon, aq, args.wbits)
    acc = lambda o: (o.argmax(1).numpy() == ys).mean() * 100
    print(f"target S{args.target}: float {acc(ref):.2f}% | int{args.wbits}/{args.abits} {acc(qo):.2f}% "
          f"| agreement {(ref.argmax(1) == qo.argmax(1)).float().mean() * 100:.1f}%")
    import os; os.makedirs(args.hls_dir, exist_ok=True)
    export_hls(P, aq, f"{args.hls_dir}/hanet_weights.h", Xon[0], args.wbits)
    print(f"exported {args.hls_dir}/hanet_weights.h, pwl_tables.h, golden.h")


if __name__ == "__main__":
    main()
