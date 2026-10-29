"""ARM CPU baseline on the PYNQ-Z2 (no FPGA involved).
Same folder: cpu_baseline.c, board_S1.npz, export_S1/ (hanet_weights.h, pwl_tables.h, hanet_consts.h, golden.h)
Run:  python3 cpu_baseline.py
  1. compiles cpu_baseline.c with gcc -O3 (Cortex-A9, NEON)
  2. self-test of the integer CNN against golden.h
  3. converts board_S1.npz -> board_S1.bin and streams all trials through the C pipeline
  4. also times a NumPy implementation (what a Python-only deployment would cost)
  5. compares with fpga_results_S1.npz if present (prediction agreement, speed-up)
"""
import os, subprocess, time
import numpy as np

CFLAGS = "-O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"gcc {CFLAGS} -o cpu_baseline cpu_baseline.c -lm")
sh("./cpu_baseline --selftest")

d = np.load("board_S1.npz")
X, y, sess = d["X_q10"].astype(np.int32), d["y"].astype(np.int32), d["session"].astype(np.int32)
with open("board_S1.bin", "wb") as f:
    np.array([len(y)], np.int32).tofile(f)
    for k in range(len(y)):
        np.array([sess[k], y[k]], np.int32).tofile(f)
        X[k].astype(np.int32).tofile(f)
sh("./cpu_baseline board_S1.bin cpu_pred_S1.txt")
r = np.loadtxt("cpu_pred_S1.txt")
pred_c, lat_c = r[:, 0].astype(int), r[:, 1] + r[:, 2]

# ---- NumPy version of the online EA (CNN excluded: its integer loops are impractical in NumPy)
def ea_numpy(x, st):
    P = x @ x.T / x.shape[1]
    st["n"] += 1
    a = max(1.0 / st["n"], 2.0 ** -5)
    st["R"] = st["R"] + a * (P - st["R"])
    Cn = x.shape[0]
    S = st["R"] + 0.01 * np.trace(st["R"]) / Cn * np.eye(Cn)
    s = np.trace(S)
    r = 2.0 ** (-np.ceil(np.floor(np.log2(s)) / 2))
    for _ in range(6):
        r = r * (1.5 - 0.5 * s * r * r)
    Y, Z, I = r * r * S, np.eye(Cn), np.eye(Cn)
    for _ in range(18):
        Tm = 0.5 * (3 * I - Z @ Y); Y = Y @ Tm; Z = Tm @ Z
    return (r * 0.5 * (Z + Z.T)) @ x

st, lat_np, prev = None, [], None
for k in range(min(len(y), 100)):
    if sess[k] != prev:
        st = {"n": 0, "R": np.zeros((22, 22))}; prev = sess[k]
    x = X[k] / 1024.0
    t0 = time.perf_counter(); ea_numpy(x, st); lat_np.append((time.perf_counter() - t0) * 1e3)
print(f"NumPy online EA only: {np.mean(lat_np):.2f} ms/trial (first 100 trials)")

out = dict(pred=pred_c, lat_ms=lat_c, ea_ms=r[:, 1], cnn_ms=r[:, 2], numpy_ea_ms=np.array(lat_np), y=y)
if os.path.exists("fpga_results_S1.npz"):
    fp = np.load("fpga_results_S1.npz")
    agree = (fp["pred"] == pred_c).mean() * 100
    print(f"FPGA vs CPU: prediction agreement {agree:.2f}% | accuracy FPGA {(fp['pred'] == y).mean()*100:.2f}% "
          f"vs CPU {(pred_c == y).mean()*100:.2f}% | latency FPGA {fp['lat_ms'].mean():.2f} ms vs CPU "
          f"{lat_c.mean():.2f} ms -> speed-up x{lat_c.mean() / fp['lat_ms'].mean():.2f}")
np.savez("cpu_results_S1.npz", **out)
print("saved cpu_results_S1.npz")
