"""
Data loading for BCI Competition IV-2a (MOABB: BNCI2014_001), IV-2b (BNCI2014_004),
OpenBMI (Lee2019_MI) and PhysioNet MI (PhysionetMI, 109 subjects, used for pretraining),
with a preprocessing chain that the FPGA reproduces exactly:

   raw (250 Hz, uV) -> causal 4th-order Butterworth band-pass 4-40 Hz (2 biquads)
                    -> decimate x2 (125 Hz) -> crop [0.5, 4.5] s after cue (500 samples)

Causal (not zero-phase) filtering is used on purpose: the hardware filter is causal,
so training on zero-phase-filtered data would create a train/deploy mismatch.
The epoch is taken from -1.0 s so the IIR transient dies out before the crop.

First run downloads the data (~1 GB for 2a) to ~/mne_data. Requires: pip install moabb
"""
import numpy as np
from scipy.signal import butter, sosfilt

FS_RAW, DECIM = 250, 2
FS = FS_RAW // DECIM
T_PRE, T_START, T_END = -1.0, 0.5, 4.5


def design_bandpass(lo=4.0, hi=40.0, fs=FS_RAW, order=2):
    """order=2 per band edge -> 4th-order band-pass = 2 second-order sections (biquads)."""
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def preprocess(X, sos=None):
    """X: (n, C, samples @250 Hz, starting at T_PRE) -> (n, C, 500) float32 in uV."""
    sos = design_bandpass() if sos is None else sos
    Xf = sosfilt(sos, X, axis=-1)[..., ::DECIM]
    s0 = int(round((T_START - T_PRE) * FS))
    return Xf[..., s0:s0 + int((T_END - T_START) * FS)].astype(np.float32)


# BCI IV-2a montage; PhysioNet is reduced to exactly these channels, in this order,
# so a model pretrained on PhysioNet plugs straight into the 22-channel hardware.
CH22 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
PHYSIONET_EXCLUDE = [88, 89, 92, 100, 104]   # irregular sampling / annotations (common practice)


def _pick22(epochs):
    """Reorder/select the 22 IV-2a channels by name (case-insensitive)."""
    lut = {c.lower().rstrip("."): c for c in epochs.ch_names}
    missing = [c for c in CH22 if c.lower() not in lut]
    if missing:
        raise RuntimeError(f"channels missing in dataset: {missing}; available: {epochs.ch_names}")
    return epochs.copy().pick([lut[c.lower()] for c in CH22])


def load_moabb(dataset="2a", subjects=None, classes="all", cache_dir=None):
    """Returns dict subject -> (X (n,22,500), y (n,), session labels (n,)) in natural trial order.
    dataset: "2a" | "2b" | "openbmi" | "physionet"
    classes="lr" keeps only left- vs right-hand imagery (2-class task).
    cache_dir: if given, the preprocessed arrays are cached there as .npz (fast reloads)."""
    import os
    tag = f"{dataset}_{classes}_{'all' if subjects is None else '-'.join(map(str, subjects))}_v2"
    if cache_dir:
        fn = os.path.join(cache_dir, f"hanet_cache_{tag}.npz")
        if os.path.exists(fn):
            z = np.load(fn, allow_pickle=True)
            return {int(s): (z[f"X{s}"].astype(np.float32), z[f"y{s}"], z[f"s{s}"]) for s in z["subjects"]}

    from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery
    kw = dict(fmin=0.5, fmax=100, tmin=T_PRE, tmax=T_END, resample=FS_RAW)
    if dataset == "physionet":
        kw["fmax"] = 60   # PhysioNet is sampled at 160 Hz (Nyquist 80 Hz); our 4-40 Hz band-pass follows
        ds = PhysionetMI(imagined=True, executed=False)
        ev = ["left_hand", "right_hand"] if classes == "lr" else ["left_hand", "right_hand", "feet", "hands"]
        paradigm = MotorImagery(events=ev, n_classes=len(ev), **kw)
        subjects = subjects or [s for s in ds.subject_list if s not in PHYSIONET_EXCLUDE]
    else:
        ds, ncls = {"2a": (BNCI2014_001(), 4), "2b": (BNCI2014_004(), 2),
                    "openbmi": (Lee2019_MI(), 2)}[dataset]
        if classes == "lr":
            paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2, **kw)
        else:
            paradigm = MotorImagery(n_classes=ncls, **kw)
        subjects = subjects or ds.subject_list
    out = {}
    for s in subjects:
        try:
            if dataset in ("2a", "physionet"):
                try:
                    ep, y, meta = paradigm.get_data(dataset=ds, subjects=[s], return_epochs=True)
                    X = _pick22(ep).get_data()
                except Exception as e:
                    if dataset == "physionet":
                        raise
                    # IV-2a raw order already equals CH22 -> fall back to the plain array path
                    print(f"[data] epochs path failed for S{s} ({e}); using array path")
                    X, y, meta = paradigm.get_data(dataset=ds, subjects=[s])
            else:
                X, y, meta = paradigm.get_data(dataset=ds, subjects=[s])
        except Exception as e:                         # skip a broken subject, keep going
            print(f"[data] subject {s} skipped: {e}")
            continue
        X = X * 1e6                                   # V -> uV (ADC-like scale)
        names = sorted(set(y))
        y = np.array([names.index(c) for c in y])
        out[s] = (preprocess(X), y, np.asarray(meta["session"].values).astype(str))
        print(f"[data] {dataset} subject {s}: {len(y)} trials ({len(out)}/{len(subjects)})", flush=True)
    if cache_dir and out:                       # never cache an empty/failed load
        os.makedirs(cache_dir, exist_ok=True)
        arrs = {"subjects": np.array(list(out))}
        for s, (X, y, ss) in out.items():
            arrs[f"X{s}"], arrs[f"y{s}"], arrs[f"s{s}"] = X.astype(np.float16), y, ss
        np.savez(fn, **arrs)
    return out


def synthetic(n_subj=9, n_trials=144, C=22, n_cls=4, seed=0):
    """Smoke-test data with subject-specific mixing (mimics inter-subject shift)."""
    rng = np.random.default_rng(seed)
    t = np.arange(500) / FS
    out = {}
    for s in range(1, n_subj + 1):
        mix = np.eye(C) + 0.6 * rng.standard_normal((C, C))      # subject "head model"
        gain = rng.uniform(5, 30)
        y = rng.integers(0, n_cls, n_trials)
        src = rng.standard_normal((n_trials, C, 500))
        for i, c in enumerate(y):   # class-specific mu-rhythm modulation on a few sources
            src[i, c * 3:(c * 3) + 3] += 2.0 * np.sin(2 * np.pi * (10 + c) * t)
        X = gain * np.einsum("cd,ndt->nct", mix, src)
        out[s] = (X.astype(np.float32), y, np.array(["0train"] * n_trials))
    return out
