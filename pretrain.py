"""
Cross-dataset pretraining on PhysioNet MI (≈104 subjects, 22 channels matched to IV-2a).

The pretrained feature extractor is then fine-tuned on the IV-2a SOURCE subjects inside each
LOSO fold (train_loso.py --init <ckpt>). The IV-2a TARGET subject is never seen and needs no
labels, so the whole pipeline stays calibration-free. PhysioNet and IV-2a share no subjects.

Same preprocessing (causal 4-40 Hz, 125 Hz, 4 s) and the same per-subject Euclidean alignment
as the deployment pipeline, so the hardware front end is identical.

Usage (Colab):
  python pretrain.py --out $BASE/pretrain/cnn.pt  --no-attention --K1 64 --P1 4 --P2 8 --act elu
  python pretrain.py --out $BASE/pretrain/hanet.pt --K1 64
  python pretrain.py --synthetic --epochs 2 --out /tmp/pt.pt          # smoke test
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import load_moabb, synthetic
from train_loso import align_offline, augment, predict, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="cache preprocessed PhysioNet (.npz)")
    ap.add_argument("--classes", default="all", choices=["all", "lr"])
    ap.add_argument("--n-subjects", type=int, default=0,
                    help="use only the first N PhysioNet subjects (0 = all); faster download")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    # model options (must match what you later fine-tune)
    ap.add_argument("--model", default="hanet", choices=["hanet", "eegnet", "shallow"])
    ap.add_argument("--F1", type=int, default=8); ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--K1", type=int, default=64); ap.add_argument("--K2", type=int, default=16)
    ap.add_argument("--P1", type=int, default=8); ap.add_argument("--P2", type=int, default=4)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--act", default="relu", choices=["relu", "elu"])
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--exact-softmax", action="store_true")
    args, unknown = ap.parse_known_args()      # tolerate LOSO-only flags such as --dg none
    if unknown:
        print(f"[pretrain] ignoring options not used in pretraining: {unknown}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    subs = None
    if args.n_subjects:
        from data import PHYSIONET_EXCLUDE
        subs = [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDE][:args.n_subjects]
    data = synthetic(n_subj=12) if args.synthetic else load_moabb("physionet", subjects=subs,
                                                                  classes=args.classes,
                                                                  cache_dir=args.cache_dir)
    print(f"loaded {len(data)} subjects in {time.time() - t0:.0f}s")
    if len(data) < 2:
        raise SystemExit("[pretrain] error: fewer than 2 subjects loaded - see the [data] messages above")
    rng = np.random.default_rng(args.seed)
    Xtr, ytr, Xva, yva = [], [], [], []
    for s, (X, y, ss) in data.items():             # 10% of every subject for validation
        Xa = align_offline(X, ss, args.eps, True)
        idx = rng.permutation(len(y)); k = max(1, len(y) // 10)
        Xva.append(Xa[idx[:k]]); yva.append(y[idx[:k]])
        Xtr.append(Xa[idx[k:]]); ytr.append(y[idx[k:]])
    Xtr, ytr, Xva, yva = (np.concatenate(a) for a in (Xtr, ytr, Xva, yva))
    n_cls = int(ytr.max() + 1)
    print(f"train {len(ytr)} trials, val {len(yva)}, classes {n_cls}")

    torch.manual_seed(args.seed)
    model = build_model(Xtr.shape[1], Xtr.shape[2], n_cls, args, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    best, best_state, pat = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = augment(Xt[idx].unsqueeze(1).to(dev)), yt[idx].to(dev)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        va = (predict(model, Xva, dev) == yva).mean()
        print(f"ep {ep + 1:3d} val {va:.3f}", flush=True)
        if va > best:
            best, pat = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= args.patience:
                break
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state": best_state, "cfg": model.cfg, "args": vars(args), "val_acc": float(best)}, args.out)
    print(f"saved {args.out} | best val {best:.3f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
