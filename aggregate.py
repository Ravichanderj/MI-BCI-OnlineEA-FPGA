"""
Aggregate LOSO runs into paper-ready numbers.

 * each GROUP = one method, given as several runs (= random seeds)
 * reports: mean accuracy / kappa over subjects, ± std across seeds, per-subject means
 * Wilcoxon signed-rank test (paired over subjects) of every group vs the first group

Usage (Colab):
  python aggregate.py --base $BASE/runs --metric online_ea \
      --group "Proposed=2a_cnn_s42,2a_cnn_s7,2a_cnn_s123" \
      --group "EEGNet=2a_eegnet,2a_eegnet_s7,2a_eegnet_s123" --csv $BASE/table_2a.csv
metric: online_ea | offline_ea_oracle | no_ea
"""
import argparse, glob, json, os
import numpy as np


def load_run(base, run, metric):
    acc, kap = {}, {}
    for f in glob.glob(os.path.join(base, run, "*", "results.json")) + \
             glob.glob(os.path.join(base, run, "results.json")):
        for s, r in json.load(open(f)).items():
            acc[int(s)] = r[metric]["acc"] * 100
            kap[int(s)] = r[metric]["kappa"]
    return acc, kap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--group", action="append", required=True, help='"Name=run1,run2,..."')
    ap.add_argument("--metric", default="online_ea")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    from scipy.stats import wilcoxon

    groups = []
    for g in a.group:
        name, runs = g.split("=", 1)
        accs, kaps = [], []
        for r in runs.split(","):
            acc, kap = load_run(a.base, r.strip(), a.metric)
            if not acc:
                print(f"!! no results for run '{r}' – skipped"); continue
            accs.append(acc); kaps.append(kap)
        if not accs:
            continue
        subs = sorted(set.intersection(*[set(x) for x in accs]))
        A = np.array([[x[s] for s in subs] for x in accs])      # seeds x subjects
        K = np.array([[x[s] for s in subs] for x in kaps])
        groups.append(dict(name=name, subs=subs, A=A, K=K))

    print(f"\nmetric: {a.metric}")
    hdr = f"{'method':<14}{'seeds':>6}{'acc mean':>10}{'± seed-std':>12}{'± subj-std':>12}{'kappa':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for g in groups:
        seed_means = g["A"].mean(1)
        subj_mean = g["A"].mean(0)
        print(f"{g['name']:<14}{len(seed_means):>6}{seed_means.mean():>10.2f}{seed_means.std():>12.2f}"
              f"{subj_mean.std():>12.2f}{g['K'].mean():>8.3f}")
        rows.append([g["name"]] + [f"{v:.2f}" for v in subj_mean] +
                    [f"{seed_means.mean():.2f}", f"{seed_means.std():.2f}", f"{g['K'].mean():.3f}"])

    if len(groups) > 1:
        ref = groups[0]
        print(f"\nWilcoxon signed-rank vs '{ref['name']}' (paired over subjects, seed-averaged):")
        for g in groups[1:]:
            common = [s for s in ref["subs"] if s in g["subs"]]
            x = np.array([ref["A"].mean(0)[ref["subs"].index(s)] for s in common])
            y = np.array([g["A"].mean(0)[g["subs"].index(s)] for s in common])
            try:
                p = wilcoxon(x, y).pvalue
            except ValueError:
                p = float("nan")
            print(f"  {g['name']:<14} diff {np.mean(x - y):+6.2f} pts   p = {p:.4f}"
                  f"{'  (n=' + str(len(common)) + ' subjects; p<0.05 needs consistent wins)' if len(common) < 10 else ''}")

    if a.csv and groups:
        subs = groups[0]["subs"]
        with open(a.csv, "w") as f:
            f.write(",".join(["method"] + [f"S{s}" for s in subs] + ["mean", "seed_std", "kappa"]) + "\n")
            for r in rows:
                f.write(",".join(r) + "\n")
        print(f"\nsaved {a.csv}")


if __name__ == "__main__":
    main()
