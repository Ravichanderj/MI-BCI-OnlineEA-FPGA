"""Diagnostic: compare every trial of the board against the C model (class AND the 4 logits).
Same folder: mi_bci.bit, mi_bci.hwh, board_S1.npz, expected_S1.txt
In a notebook cell:   %run diag_board.py
"""
import numpy as np
from pynq import Overlay, allocate, Clocks

ol = Overlay("mi_bci.bit"); Clocks.fclk0_mhz = 50.0
ip = ol.mi_bci_top_0; rm = ip.register_map
def addr(*names):
    for n in names:
        if hasattr(rm, n): return getattr(rm, n).address
    raise RuntimeError(names)
A_X = addr("xin", "xin_V"); A_C = addr("cls", "cls_V"); A_R = addr("reset", "reset_V")
A_L = [addr(f"l{i}", f"l{i}_V") for i in range(4)]
def s32(v): return v - (1 << 32) if v & 0x80000000 else v

d = np.load("board_S1.npz"); X, y, sess = d["X_q10"], d["y"], d["session"]
exp = np.loadtxt("expected_S1.txt", dtype=np.int64)          # cls l0 l1 l2 l3 (C model)
buf = allocate(shape=(22 * 500,), dtype=np.int32)

hw = np.zeros((len(y), 5), dtype=np.int64); prev = None
for k in range(len(y)):
    buf[:] = X[k].reshape(-1); buf.flush()
    ip.write(A_X, buf.device_address)
    ip.write(A_R, 1 if sess[k] != prev else 0); prev = sess[k]
    ip.write(0x00, 1)
    while (ip.read(0x00) & 0x2) == 0: pass
    hw[k] = [ip.read(A_C)] + [s32(ip.read(a)) for a in A_L]

cls_diff = np.where(hw[:, 0] != exp[:, 0])[0]
log_diff = np.where((hw[:, 1:] != exp[:, 1:]).any(1))[0]
print(f"board acc {(hw[:,0]==y).mean()*100:.2f}%   C-model acc {(exp[:,0]==y).mean()*100:.2f}%")
print(f"trials with different CLASS : {len(cls_diff)}")
print(f"trials with different LOGITS: {len(log_diff)}   first at trial {log_diff[0] if len(log_diff) else '-'}")
print("session boundaries at trials:", [0] + list(np.where(np.diff(sess) != 0)[0] + 1))
for k in log_diff[:12]:
    print(f"  trial {k:3d}  board {hw[k].tolist()}   C-model {exp[k].tolist()}")
np.savez("diag_S1.npz", hw=hw, exp=exp)
print("saved diag_S1.npz")
