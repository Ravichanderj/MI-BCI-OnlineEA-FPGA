"""Export one target subject's filtered (NOT aligned) trials for the board test.
Run on Kaggle after the runs exist:
  python export_board_data.py --target 1 --cache-dir /kaggle/working/cache --out /kaggle/working/board_S1.npz
X_q10 = round(uV * 1024) as int32 (the accelerator's input format, din_t Q.10)."""
import argparse, numpy as np
from data import load_moabb

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=1)
ap.add_argument("--cache-dir", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
data = load_moabb("2a", cache_dir=a.cache_dir)
X, y, ss = data[a.target]
Xq = np.clip(np.round(X.astype(np.float64) * 1024), -(2 ** 23), 2 ** 23 - 1).astype(np.int32)
_, sess = np.unique(ss, return_inverse=True)
np.savez_compressed(a.out, X_q10=Xq, y=y.astype(np.int32), session=sess.astype(np.int32),
                    logit_scale=np.zeros(4, np.float32))
print("saved", a.out, Xq.shape, "max |uV|", np.abs(X).max())
