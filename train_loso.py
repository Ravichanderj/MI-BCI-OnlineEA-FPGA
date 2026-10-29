"""
Leave-one-subject-out (LOSO), calibration-free training/evaluation of HA-Net.

For each target subject s:
  sources   = all other subjects; one source is held out for early stopping (validation)
  training  = offline EA per source subject + domain-generalisation loss (optional CORAL)
  testing   = target subject, NO labels, NO calibration:
               (a) online EA (hardware algorithm, streamed in natural trial order)  <- main result
               (b) offline EA on all target trials (oracle upper bound)
               (c) no alignment (lower bound)

Usage:
  python train_loso.py --dataset 2a --epochs 200            # real data (needs moabb)
  python train_loso.py --synthetic --epochs 5               # smoke test
  python train_loso.py --dataset 2a --no-attention          # ablation: CNN only
  python train_loso.py --dataset 2a --exact-softmax         # ablation: float softmax
  python train_loso.py --dataset 2a --dg none               # ablation: no CORAL
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_moabb, synthetic
from ea import offline_ea, OnlineEA
from model import HANet, EEGNet, ShallowConvNet, count_params


def kappa(y, p, k):
    cm = np.zeros((k, k))
    for a, b in zip(y, p):
        cm[a, b] += 1
    n = cm.sum(); po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe + 1e-12)


def coral_loss(feat, dom):
    """Deep CORAL across source subjects in the batch: aligns 2nd-order feature statistics."""
    covs = []
    for d in dom.unique():
        f = feat[dom == d]
        if f.shape[0] < 4:
            continue
        f = f - f.mean(0, keepdim=True)
        covs.append(f.T @ f / (f.shape[0] - 1))
    if len(covs) < 2:
        return feat.new_zeros(())
    dim = feat.shape[1]
    loss = sum(((covs[i] - covs[j]) ** 2).sum() for i in range(len(covs)) for j in range(i + 1, len(covs)))
    return loss / (4 * dim * dim) / (len(covs) * (len(covs) - 1) / 2)


def augment(x):
    """Label-preserving augmentation: small circular time-shift + amplitude jitter + noise."""
    shift = int(np.random.randint(-12, 13))
    x = torch.roll(x, shift, dims=-1)
    x = x * (1 + 0.1 * torch.randn(x.shape[0], 1, 1, 1, device=x.device))
    return x + 0.05 * torch.randn_like(x)


@torch.no_grad()
def predict(model, X, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


def align_offline(X, sess, eps, per_session=True):
    """Offline EA per subject, or per (subject, session) when per_session=True."""
    if not per_session:
        return offline_ea(X, eps)[0]
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(sess):
        m = sess == s
        out[m] = offline_ea(X[m], eps)[0]
    return out


def align_online(X, sess, args):
    """Streaming online EA in natural trial order; reset at each new session if requested
    (new session = new day / headset re-fitted / device power-cycled)."""
    C = X.shape[1]
    if not args.reset_session:
        return OnlineEA(C, k_ema=args.k_ema, eps=args.eps, ns_iters=args.ns_iters).run(X)
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(sess):          # sessions are contiguous and in order in MOABB output
        m = sess == s
        out[m] = OnlineEA(C, k_ema=args.k_ema, eps=args.eps, ns_iters=args.ns_iters).run(X[m])
    return out


def build_model(C, T, n_cls, args, dev):
    if args.model == "eegnet":
        return EEGNet(C=C, T=T, n_cls=n_cls, dropout=args.dropout).to(dev)
    if args.model == "shallow":
        return ShallowConvNet(C=C, T=T, n_cls=n_cls).to(dev)
    return HANet(C=C, T=T, n_cls=n_cls, F1=args.F1, D=args.D, K1=args.K1, K2=args.K2,
                 heads=args.heads, dropout=args.dropout, attention=not args.no_attention,
                 P1=args.P1, P2=args.P2, act=args.act,
                 approx=not args.exact_softmax).to(dev)


def load_pretrained(model, path):
    """Load a pretrained checkpoint (e.g. PhysioNet) into model, skipping the classifier head
    and any tensor whose shape differs (e.g. different number of classes)."""
    ck = torch.load(path, map_location="cpu")
    own = model.state_dict()
    keep = {k: v for k, v in ck["state"].items()
            if k in own and v.shape == own[k].shape and not k.startswith("head.")}
    model.load_state_dict(keep, strict=False)
    print(f"[init] loaded {len(keep)}/{len(own)} tensors from {path} (head re-initialised)")


def train_model(Xtr, ytr, dtr, Xva, yva, n_cls, args, dev):
    C, T = Xtr.shape[1:]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_model(C, T, n_cls, args, dev)
    if getattr(args, "init", None):
        load_pretrained(model, args.init)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt, yt, dt = (torch.from_numpy(a) for a in (Xtr, ytr, dtr))
    best, best_state, patience = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = Xt[idx].unsqueeze(1).to(dev)
            if not args.no_aug:
                xb = augment(xb)
            yb, db = yt[idx].to(dev), dt[idx].to(dev)
            feat = model.features(xb)
            loss = F.cross_entropy(model.head(feat), yb, label_smoothing=0.1)
            if args.dg == "coral":
                loss = loss + args.lam * coral_loss(feat, db)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        va = (predict(model, Xva, dev) == yva).mean()
        if va >= best:
            best, patience = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, float(best), ep + 1


def evaluate(model, Xs, ys, sess, n_cls, args, dev):
    """Calibration-free test: online EA (HW), offline EA on the test data (oracle), none."""
    res = {}
    variants = {
        "online_ea": align_online(Xs, sess, args),
        "offline_ea_oracle": align_offline(Xs, sess, args.eps, args.per_session),
        "no_ea": (Xs / Xs.std()).astype(np.float32),
    }
    for name, Xa in variants.items():
        p = predict(model, Xa, dev)
        res[name] = {"acc": float((p == ys).mean()), "kappa": float(kappa(ys, p, n_cls)),
                     "pred_counts": np.bincount(p, minlength=n_cls).tolist()}
        if name == "online_ea":   # per-trial correctness (warm-up curves) + accuracy after warm-up
            res[name]["correct"] = (p == ys).astype(int).tolist()
            w = 2 ** args.k_ema
            keep = np.ones(len(ys), bool)
            for s in np.unique(sess):
                idx = np.where(sess == s)[0]
                keep[idx[:w]] = False
            res[name]["acc_after_warmup"] = float((p[keep] == ys[keep]).mean())
    return res


def split_val(n, frac, rng):
    idx = rng.permutation(n)
    k = max(1, int(round(n * frac)))
    return idx[k:], idx[:k]


def run_fold(data, target, args, dev):
    """LOSO: train on all other subjects, test on the unseen target without calibration."""
    subs = sorted(data)
    sources = [s for s in subs if s != target]
    rng = np.random.default_rng(args.seed)
    al = {s: align_offline(data[s][0], data[s][2], args.eps, args.per_session) for s in sources}
    Xtr, ytr, dtr, Xva, yva = [], [], [], [], []
    if args.val == "subject":        # hold out one whole source subject
        val_s = sources[subs.index(target) % len(sources)]
        for i, s in enumerate(s for s in sources if s != val_s):
            Xtr.append(al[s]); ytr.append(data[s][1]); dtr.append(np.full(len(al[s]), i))
        Xva, yva = [al[val_s]], [data[val_s][1]]
    else:                            # pooled: random 10% of every source subject
        for i, s in enumerate(sources):
            tr, va = split_val(len(al[s]), 0.1, rng)
            Xtr.append(al[s][tr]); ytr.append(data[s][1][tr]); dtr.append(np.full(len(tr), i))
            Xva.append(al[s][va]); yva.append(data[s][1][va])
    Xtr, ytr, dtr, Xva, yva = (np.concatenate(a) for a in (Xtr, ytr, dtr, Xva, yva))
    n_cls = int(max(ytr.max(), yva.max()) + 1)
    model, best, eps_run = train_model(Xtr, ytr, dtr, Xva, yva, n_cls, args, dev)
    Xs, ys, ss = data[target]
    res = {"val_acc": best, "epochs_run": eps_run}
    res.update(evaluate(model, Xs, ys, ss, n_cls, args, dev))
    return model, res


def subject_folds(subs, k, seed=0):
    """Subject-exclusive k-fold split (fixed seed -> identical folds for every model/seed)."""
    perm = np.random.default_rng(seed).permutation(sorted(subs))
    return [sorted(int(x) for x in perm[i::k]) for i in range(k)]


def run_kfold(data, fold, args, dev):
    """Subject-exclusive k-fold (for large cohorts, e.g. PhysioNet): train once on the other
    folds' subjects, test every held-out subject separately with its own online EA stream."""
    folds = subject_folds(list(data), args.folds)
    targets = folds[fold]
    sources = [s for s in sorted(data) if s not in targets]
    rng = np.random.default_rng(args.seed)
    Xtr, ytr, dtr, Xva, yva = [], [], [], [], []
    for i, s in enumerate(sources):
        a = align_offline(data[s][0], data[s][2], args.eps, args.per_session)
        tr, va = split_val(len(a), 0.1, rng)
        Xtr.append(a[tr]); ytr.append(data[s][1][tr]); dtr.append(np.full(len(tr), i))
        Xva.append(a[va]); yva.append(data[s][1][va])
    Xtr, ytr, dtr, Xva, yva = (np.concatenate(a) for a in (Xtr, ytr, dtr, Xva, yva))
    n_cls = int(max(ytr.max(), yva.max()) + 1)
    model, best, eps_run = train_model(Xtr, ytr, dtr, Xva, yva, n_cls, args, dev)
    out = {}
    for t in targets:
        Xs, ys, ss = data[t]
        r = {"val_acc": best, "epochs_run": eps_run, "fold": fold}
        r.update(evaluate(model, Xs, ys, ss, n_cls, args, dev))
        out[t] = r
    return model, out


def run_within(data, target, args, dev):
    """Sanity check (subject-dependent): train on session 1, test on session 2 of the SAME subject.
    A healthy pipeline gives roughly 65-75% on 4-class IV-2a."""
    X, y, sess = data[target]
    sessions = sorted(set(sess))
    if len(sessions) < 2:
        raise SystemExit("within protocol needs 2 sessions (use real data, not --synthetic)")
    tr_m, te_m = sess == sessions[0], sess == sessions[1]
    Xa = offline_ea(X[tr_m], args.eps)[0]
    rng = np.random.default_rng(args.seed)
    tr, va = split_val(len(Xa), 0.2, rng)
    n_cls = int(y.max() + 1)
    model, best, eps_run = train_model(Xa[tr], y[tr_m][tr], np.zeros(len(tr), dtype=np.int64),
                                       Xa[va], y[tr_m][va], n_cls, args, dev)
    res = {"val_acc": best, "epochs_run": eps_run}
    res.update(evaluate(model, X[te_m], y[te_m], sess[te_m], n_cls, args, dev))
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2a", choices=["2a", "2b", "openbmi", "physionet"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--init", default=None, help="pretrained checkpoint (pretrain.py) to fine-tune from")
    ap.add_argument("--cache-dir", default=None, help="cache preprocessed data as .npz")
    ap.add_argument("--classes", default="all", choices=["all", "lr"],
                    help="lr = left- vs right-hand only (2-class)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--k-ema", type=int, default=5)
    ap.add_argument("--ns-iters", type=int, default=18)
    ap.add_argument("--dg", default="coral", choices=["coral", "none"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--exact-softmax", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="hanet", choices=["hanet", "eegnet", "shallow"])
    ap.add_argument("--protocol", default="loso", choices=["loso", "within", "kfold"])
    ap.add_argument("--folds", type=int, default=5, help="kfold: number of subject-exclusive folds")
    ap.add_argument("--fold", type=int, nargs="*", default=None, help="kfold: which folds to run")
    ap.add_argument("--val", default="pooled", choices=["pooled", "subject"])
    ap.add_argument("--F1", type=int, default=8)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--K1", type=int, default=32)
    ap.add_argument("--K2", type=int, default=16)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--P1", type=int, default=8, help="first pooling (EEGNet uses 4)")
    ap.add_argument("--P2", type=int, default=4, help="second pooling (EEGNet uses 8)")
    ap.add_argument("--act", default="relu", choices=["relu", "elu"],
                    help="conv activation; elu uses the shared base-2 exp unit")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-per-session", dest="per_session", action="store_false",
                    help="align each subject over all sessions pooled (old behaviour)")
    ap.add_argument("--no-reset-session", dest="reset_session", action="store_false",
                    help="do not reset online EA at a new session (old behaviour)")
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="runs/default")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.synthetic:
        data = synthetic()
    elif args.dataset == "physionet" and args.classes == "lr":
        # reuse the 4-class cache: labels sorted = [feet, hands, left_hand, right_hand]
        full = load_moabb("physionet", classes="all", cache_dir=args.cache_dir)
        data = {}
        for s, (X, y, ss) in full.items():
            m = (y == 2) | (y == 3)
            data[s] = (X[m], (y[m] - 2).astype(np.int64), ss[m])
    else:
        data = load_moabb(args.dataset, classes=args.classes, cache_dir=args.cache_dir)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    if args.protocol == "kfold":
        for f in (args.fold if args.fold is not None else range(args.folds)):
            t0 = time.time()
            model, rs = run_kfold(data, f, args, dev)
            torch.save({"state": model.state_dict(), "cfg": model.cfg, "args": vars(args)},
                       f"{args.out}/model_fold{f}.pt")
            results.update(rs)
            a = np.mean([r["online_ea"]["acc"] for r in rs.values()])
            o = np.mean([r["offline_ea_oracle"]["acc"] for r in rs.values()])
            n = np.mean([r["no_ea"]["acc"] for r in rs.values()])
            print(f"F{f} [kfold] {len(rs)} subjects | val {list(rs.values())[0]['val_acc']:.3f} | "
                  f"online-EA {a:.3f} | oracle {o:.3f} | no-EA {n:.3f} | {time.time() - t0:.0f}s", flush=True)
    for s in ([] if args.protocol == "kfold" else (args.targets or sorted(data))):
        t0 = time.time()
        model, r = (run_fold if args.protocol == "loso" else run_within)(data, s, args, dev)
        torch.save({"state": model.state_dict(), "cfg": model.cfg, "args": vars(args)},
                   f"{args.out}/hanet_target{s}.pt")
        results[s] = r
        print(f"S{s:02d} [{args.protocol}] val {r['val_acc']:.3f} ep {r['epochs_run']} | online-EA {r['online_ea']['acc']:.3f} (k={r['online_ea']['kappa']:.2f}) | "
              f"oracle {r['offline_ea_oracle']['acc']:.3f} | no-EA {r['no_ea']['acc']:.3f} | "
              f"{time.time() - t0:.0f}s | predicted-class counts {r['online_ea']['pred_counts']}", flush=True)
    for key in ("online_ea", "offline_ea_oracle", "no_ea"):
        a = np.array([results[s][key]["acc"] for s in results])
        k = np.array([results[s][key]["kappa"] for s in results])
        print(f"{key:18s} acc {a.mean() * 100:.2f} ± {a.std() * 100:.2f} | kappa {k.mean():.3f}")
    print("params:", count_params(model))
    json.dump({str(k): v for k, v in results.items()}, open(f"{args.out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
