"""
Similarity analysis: does the test->train structural similarity (max Tanimoto) track prediction quality?

For each split (random / scaffold) and each model (RF / GINE, 3 seeds each):
  1. per-molecule Brier error vs max Tanimoto: Spearman rho only (descriptive, no p-value)
  2. macro ROC-AUC inside FIXED similarity bins (<0.4, 0.4-0.6, >=0.6; defined before looking at results)
     a label counts only if the bin has >=3 positives AND >=3 negatives; a bin cell is reported only if
     >=10 labels are usable (otherwise 'n/a').
Caution: the bins hold different molecules in different splits, bins are small, and the usable label set
differs per bin, so compare cells descriptively only.

Usage:
    python analyze_similarity.py --dataset sider
Outputs:
    results/similarity_per_molecule.csv, results/similarity_bins.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

BINS = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0001)]
MIN_PER_CLASS, MIN_LABELS = 3, 10


def max_sim(train_smiles, query_smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = lambda s: gen.GetFingerprint(Chem.MolFromSmiles(s))  # noqa: E731
    train_fps = [fp(s) for s in train_smiles]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), train_fps)) for s in query_smiles])


def brier_per_molecule(y, p):
    err = (y - p) ** 2
    return np.nanmean(np.where(np.isnan(y), np.nan, err), axis=1)


def bin_macro_auc(y, p, mask):
    aucs = []
    for j in range(y.shape[1]):
        yt, ys = y[mask, j], p[mask, j]
        ok = ~np.isnan(yt)
        yt, ys = yt[ok], ys[ok]
        if (yt == 1).sum() >= MIN_PER_CLASS and (yt == 0).sum() >= MIN_PER_CLASS:
            aucs.append(roc_auc_score(yt, ys))
    return (float(np.mean(aucs)) if len(aucs) >= MIN_LABELS else float("nan")), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args, _ = ap.parse_known_args()

    with open(os.path.join(args.data_dir, f"{args.dataset}_eligible_tasks.txt")) as f:
        tasks = [l.rstrip("\n") for l in f if l.strip()]

    mol_rows, bin_rows = [], []
    for split in ("random", "scaffold"):
        tr = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_{split}_train.csv"))
        te = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_{split}_test.csv"))
        y = te[tasks].to_numpy(dtype=float)
        sim = max_sim(tr["smiles"].tolist(), te["smiles"].tolist())
        preds = {}
        for model, prefix in (("RF", "rf"), ("GINE", "gnn")):
            preds[model] = [np.load(os.path.join(args.out_dir, f"{prefix}_{args.dataset}_{split}_seed{s}_test_pred.npy"))
                            for s in args.seeds]
            for p in preds[model]:
                assert p.shape == y.shape, f"{model}/{split}: prediction shape {p.shape} != labels {y.shape}"

        brier = {m: np.mean([brier_per_molecule(y, p) for p in ps], axis=0) for m, ps in preds.items()}
        for i in range(len(te)):
            mol_rows.append(dict(split=split, smiles=te["smiles"].iloc[i], max_sim=sim[i],
                                 brier_rf=brier["RF"][i], brier_gine=brier["GINE"][i]))
        print(f"\n=== {split}: n_test={len(te)}, median max-Tanimoto={np.median(sim):.3f} ===")
        for m in preds:
            rho = spearmanr(sim, brier[m]).correlation
            print(f"  Spearman rho (max-sim vs Brier error), {m}: {rho:+.3f}  (descriptive only)")

        for lo, hi in BINS:
            mask = (sim >= lo) & (sim < hi)
            label = f"[{lo:.1f},{min(hi, 1.0):.1f}{']' if hi > 1 else ')'}"
            for m, ps in preds.items():
                if mask.sum() == 0:
                    auc, n_lab = float("nan"), 0
                else:
                    res = [bin_macro_auc(y, p, mask) for p in ps]
                    auc, n_lab = float(np.mean([r[0] for r in res])), res[0][1]
                bin_rows.append(dict(split=split, bin=label, model=m, n_molecules=int(mask.sum()),
                                     n_labels_usable=n_lab, macro_auc=auc))

    pd.DataFrame(mol_rows).to_csv(os.path.join(args.out_dir, "similarity_per_molecule.csv"), index=False)
    bins_df = pd.DataFrame(bin_rows)
    bins_df.to_csv(os.path.join(args.out_dir, "similarity_bins.csv"), index=False)
    print("\n=== macro ROC-AUC inside fixed similarity bins (mean over 3 seeds; n/a = too few usable labels) ===")
    print(bins_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()