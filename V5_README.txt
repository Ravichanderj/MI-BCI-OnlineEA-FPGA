MI-BCI accelerator v5 (parallel datapaths) — same arithmetic as v4, bit-identical outputs
Verified locally (C model): all 576 S1 trials give identical logits to v4 (72.57%, = board result);
tb_mi_bci_top PASS (59/60), tb_online_ea PASS, tb_hanet_cnn PASS.

What changed (cycle estimates per trial; v4 measured 1.67 M cycles = 33.3 ms @ 50 MHz):
  covariance        253 x 505 serial      -> 22 rows x 1000   (11 multipliers)      ~128k -> ~22k
  Newton-Schulz     3 matmul x 8.2k x 18  -> 3 x ~1.15k x 18   (row-accumulate, 11)  ~471k -> ~64k
  whitening         500 x 22 x 22 serial  -> 500 x 22 x 2      (11 multipliers)      ~259k -> ~22k
  spatial conv      500 x 16 x 22 serial  -> 500 x 16 x 2                            ~189k -> ~16k
  temporal conv     II=4 bank conflicts   -> sliding window, 64 taps / 4 cycles      ~532k -> ~34k
  depthwise / pointwise                   -> sliding window / 16 lanes               ~71k  -> ~8k
  expected total ~0.18 M cycles  =>  ~3.6 ms per trial at 50 MHz (ARM Cortex-A9: 22 ms)
Expected DSP use ~160-170 of 220 (v4: 99).

Build: replace your hls folder with this one, then exactly as before:
  cd <...>\hls      vivado_hls -f run_hls.tcl          (csim must print PASS)
  cd <...>\vivado   vivado -mode batch -source build_pynq_z2.tcl
Board: copy the new mi_bci.bit + mi_bci.hwh, run mi_bci_board.py and power_test.py unchanged.
