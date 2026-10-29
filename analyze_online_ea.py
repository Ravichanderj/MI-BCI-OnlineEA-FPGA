"""
Inference-only analysis of the online EA on trained LOSO models (no retraining):
  1. warm-up curve: accuracy vs. trial index within a session (online EA starts from nothing)
  2. sensitivity to the hardware parameters: forgetting exponent k, Newton-Schulz iterations K,
     shrinkage eps
Usage (Kaggle):
  python analyze_online_ea.py --runs /kaggle/working/runs/A4_prop_s42 --cache-dir /kaggle/working/cache \
      --out /kaggle/working/online_ea_analysis.json
"""
import argparse, json, os, glob
import numpy as np
import torch

from data import load_moabb
from ea import OnlineEA, offline_ea
from model import HANet
from train_loso import predict


def online(X, sess, k, eps, ns):
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(sess):
        m = sess == s
        out[m] = OnlineEA(X.shape[1], k_ema=k, eps=eps, ns_iters=ns).run(X[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder with S*/hanet_target*.pt")
    ap.add_argument("--dataset", default="2a")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_moabb(a.dataset, cache_dir=a.cache_dir)
    K_SET, NS_SET, EPS_SET = [3, 4, 5, 6, 7], [4, 6, 8, 12, 18], [1e-3, 1e-2, 1e-1]
    res = {"k": {}, "ns": {}, "eps": {}, "oracle": {}, "curve": {}}
    for ck in sorted(glob.glob(os.path.join(a.runs, "S*", "hanet_target*.pt"))):
        s = int(ck.split("hanet_target")[-1].split(".")[0])
        c = torch.load(ck, map_location="cpu")
        m = HANet(**c["cfg"]); m.load_state_dict(c["state"]); m.to(dev).eval()
        X, y, sess = data[s]
        acc = lambda Xa: float((predict(m, Xa, dev) == y).mean())
        for k in K_SET:
            res["k"].setdefault(k, {})[s] = acc(online(X, sess, k, 1e-2, 18))
        for ns in NS_SET:
            res["ns"].setdefault(ns, {})[s] = acc(online(X, sess, 5, 1e-2, ns))
        for e in EPS_SET:
            res["eps"].setdefault(e, {})[s] = acc(online(X, sess, 5, e, 18))
        Xo = np.empty_like(X, dtype=np.float32)
        for ss in np.unique(sess):
            mm = sess == ss
            Xo[mm] = offline_ea(X[mm], 1e-2)[0]
        res["oracle"][s] = acc(Xo)
        p = predict(m, online(X, sess, 5, 1e-2, 18), dev)
        cor = []
        for ss in np.unique(sess):           # per-session correctness sequences
            cor.append((p[sess == ss] == y[sess == ss]).astype(int).tolist())
        res["curve"][s] = cor
        print(f"S{s}: k=5 {res['k'][5][s]*100:.1f} | oracle {res['oracle'][s]*100:.1f}", flush=True)

    mean = lambda d: 100 * np.mean(list(d.values()))
    print("\nforgetting exponent k (memory 2^k trials):")
    for k in K_SET: print(f"  k={k} ({2**k:3d} trials): {mean(res['k'][k]):.2f}%")
    print("Newton-Schulz iterations K:")
    for ns in NS_SET: print(f"  K={ns:2d}: {mean(res['ns'][ns]):.2f}%")
    print("shrinkage eps:")
    for e in EPS_SET: print(f"  eps={e:g}: {mean(res['eps'][e]):.2f}%")
    print(f"oracle (offline EA, whole session): {mean(res['oracle']):.2f}%")
    # warm-up: accuracy in consecutive blocks of 8 trials, averaged over subjects and sessions
    seqs = [np.array(c) for v in res["curve"].values() for c in v]
    L = min(len(q) for q in seqs)
    M = np.stack([q[:L] for q in seqs])
    blocks = M[:, :L - L % 8].reshape(len(M), -1, 8).mean(2).mean(0) * 100
    print("warm-up (accuracy per block of 8 trials from session start):")
    print("  " + " ".join(f"{b:.0f}" for b in blocks[:12]), "...", f"| after 32 trials: {M[:, 32:].mean()*100:.2f}%",
          f"| first 32: {M[:, :32].mean()*100:.2f}%")
    res["warmup_blocks8"] = blocks.tolist()
    json.dump(res, open(a.out, "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
