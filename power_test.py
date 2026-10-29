"""Power measurement helper for the PYNQ-Z2 (read the DC meter while each phase runs).
Same folder as mi_bci_board.py:  mi_bci.bit, mi_bci.hwh, board_S1.npz, cpu_baseline (compiled), board_S1.bin
In a notebook cell:   %run power_test.py
Phases (60 s each, a message is printed at the start of each):
  1. IDLE  : nothing running            -> write down the meter reading
  2. FPGA  : accelerator processes trials back to back; the ARM sleeps while waiting (no busy polling)
  3. CPU   : the C pipeline runs back to back on one ARM core (FPGA idle)
"""
import time, subprocess
import numpy as np
from pynq import Overlay, allocate, Clocks

PHASE_S = 60
FCLK_MHZ = 50.0                                        # set to 75.0 for the 75-MHz power reading
ol = Overlay("mi_bci.bit"); Clocks.fclk0_mhz = FCLK_MHZ
print(f"PL clock: {Clocks.fclk0_mhz:.2f} MHz")
ip = ol.mi_bci_top_0; rm = ip.register_map
A_X, A_R, A_C = rm.xin.address, rm.reset.address, rm.cls.address
d = np.load("board_S1.npz"); X = d["X_q10"]
buf = allocate(shape=(22 * 500,), dtype=np.int32)

def phase(name):
    print(f"\n=== {name}: running {PHASE_S} s — READ THE METER NOW (note the steady value) ===", flush=True)

phase("1. IDLE")
time.sleep(PHASE_S)

phase("2. FPGA")
t_end, n, k = time.time() + PHASE_S, 0, 0
while time.time() < t_end:
    buf[:] = X[k].reshape(-1); buf.flush()
    ip.write(A_X, buf.device_address); ip.write(A_R, 1 if k == 0 else 0)
    ip.write(0x00, 1)
    time.sleep(0.002)                                  # v5 trial takes ~2.3-3.5 ms: the ARM sleeps most of it
    while (ip.read(0x00) & 0x2) == 0:
        time.sleep(0.0002)
    n += 1; k = (k + 1) % len(X)
print(f"FPGA processed {n} trials in {PHASE_S} s ({1000 * PHASE_S / n:.2f} ms per trial incl. driver)")

phase("3. CPU")
t_end, runs = time.time() + PHASE_S, 0
while time.time() < t_end:
    subprocess.run(["./cpu_baseline", "board_S1.bin", "/tmp/cpu_pred.txt"], stdout=subprocess.DEVNULL, check=True)
    runs += 1
print(f"CPU completed {runs} passes over 576 trials")
print("\nDONE. Send: idle W, FPGA W, CPU W (and the meter's voltage if it shows it).")
