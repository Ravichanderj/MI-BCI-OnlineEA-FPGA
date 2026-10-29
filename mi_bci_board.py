"""Run on the PYNQ-Z2 (Jupyter terminal):  python3 mi_bci_board.py
Same folder: mi_bci.bit, mi_bci.hwh, board_S1.npz  (exported on Kaggle, see docs)

Streams all 576 trials of target subject S1 (both sessions, recorded order) through the FPGA.
Online EA runs in hardware and is reset at the start of each session. No labels are used
by the hardware; labels are only used here to score accuracy.
"""
import time
import numpy as np
from pynq import Overlay, allocate, Clocks

FCLK_MHZ = 50.0          # post-route Fmax = 57.7 MHz (WNS -7.325 ns at 10 ns) -> run at 50 MHz

ol = Overlay("mi_bci.bit")
Clocks.fclk0_mhz = FCLK_MHZ                 # Overlay() sets 100 MHz from the .hwh; override
print("PL clock:", Clocks.fclk0_mhz, "MHz")
ip = ol.mi_bci_top_0
rm = ip.register_map
print(rm)

def addr(*names):
    for n in names:
        if hasattr(rm, n):
            return getattr(rm, n).address
    raise RuntimeError(f"register not found: {names}; see the register map printed above")

A_X = addr("xin", "xin_V")
A_L = [addr(f"l{i}", f"l{i}_V") for i in range(4)]
A_C = addr("cls", "cls_V")
A_R = addr("reset", "reset_V")

d = np.load("board_S1.npz")
X, y, sess = d["X_q10"], d["y"], d["session"]
buf = allocate(shape=(22 * 500,), dtype=np.int32)

pred, lat = [], []
prev = None
for k in range(len(y)):
    buf[:] = X[k].reshape(-1)
    buf.flush()
    ip.write(A_X, buf.device_address)
    ip.write(A_R, 1 if sess[k] != prev else 0)       # new session -> reset online EA
    prev = sess[k]
    t0 = time.perf_counter()
    ip.write(0x00, 1)                                 # ap_start
    while (ip.read(0x00) & 0x2) == 0:                 # wait for ap_done
        pass
    lat.append(time.perf_counter() - t0)
    pred.append(ip.read(A_C))
    if k % 100 == 0:
        print(f"trial {k}: pred {pred[-1]} true {y[k]}  {lat[-1]*1e3:.2f} ms", flush=True)

pred = np.array(pred); lat = np.array(lat) * 1e3
print(f"\nS1: {len(y)} trials | FPGA accuracy {(pred == y).mean()*100:.2f}% "
      f"(int8 software model: 72.57%) | latency/trial mean {lat.mean():.2f} ms "
      f"[min {lat.min():.2f}, max {lat.max():.2f}] at {FCLK_MHZ:.0f} MHz")
np.savez("fpga_results_S1.npz", pred=pred, lat_ms=lat, y=y)
