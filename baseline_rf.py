"""
Baseline: Morgan fingerprint (radius 2, 2048 bits) + Random Forest, jointly fit on all tasks.

The forest uses scikit-learn's native multi-output mode (one forest, 25 outputs). Hyper-parameters are fixed
(no tuning), so the validation set is only reported, not used for model selection.

Usage:
    python src/baseline_rf.py --dataset sider --split both
Outputs:
    results/rf_{dataset}_{split}_seed{seed}.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

from eval_utils import summarize

RDLogger.DisableLog("rdApp.*")


def morgan_matrix(smiles_list, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smi))
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


def load_split(data_dir, dataset, split, part, tasks):
    df = pd.read_csv(os.path.join(data_dir, f"{dataset}_{split}_{part}.csv"))
    return morgan_matrix(df["smiles"].tolist()), df[tasks].to_numpy(dtype=np.float32)


def positive_proba(model, X, n_tasks):
    """P(label = 1) for every task, shape [n, n_tasks]; robust to a task with a single class in train."""
    probs = model.predict_proba(X)
    out = np.zeros((X.shape[0], n_tasks))
    for j, p in enumerate(probs):
        classes = list(model.classes_[j])
        out[:, j] = p[:, classes.index(1.0)] if 1.0 in classes else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--split", choices=["random", "scaffold", "both"], default="both")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_estimators", type=int, default=500)
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, f"{args.dataset}_eligible_tasks.txt")) as f:
        tasks = [l.rstrip("\n") for l in f if l.strip()]
    splits = ["random", "scaffold"] if args.split == "both" else [args.split]

    for split in splits:
        X_tr, y_tr = load_split(args.data_dir, args.dataset, split, "train", tasks)
        X_va, y_va = load_split(args.data_dir, args.dataset, split, "valid", tasks)
        X_te, y_te = load_split(args.data_dir, args.dataset, split, "test", tasks)
        for seed in args.seeds:
            rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=seed)
            rf.fit(X_tr, y_tr.astype(int))
            p_va = positive_proba(rf, X_va, len(tasks))
            p_te = positive_proba(rf, X_te, len(tasks))
            np.save(os.path.join(args.out_dir, f"rf_{args.dataset}_{split}_seed{seed}_test_pred.npy"), p_te)
            res = {
                "model": "morgan_rf", "dataset": args.dataset, "split": split, "seed": seed,
                "n_train": len(X_tr), "n_valid": len(X_va), "n_test": len(X_te), "n_tasks": len(tasks),
                "valid": summarize(y_va, p_va, tasks),
                "test": summarize(y_te, p_te, tasks, with_ci=True),
            }
            with open(os.path.join(args.out_dir, f"rf_{args.dataset}_{split}_seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)
            t, v = res["test"], res["valid"]
            print(f"[RF | {split:8s} | seed {seed}] "
                  f"valid macro-AUC all={v['macro_auc_all']:.3f} organ={v['macro_auc_organ']:.3f} | "
                  f"test macro-AUC all={t['macro_auc_all']:.3f} "
                  f"(95% CI {t['macro_auc_all_ci95'][0]:.3f}-{t['macro_auc_all_ci95'][1]:.3f}) "
                  f"organ={t['macro_auc_organ']:.3f} "
                  f"(95% CI {t['macro_auc_organ_ci95'][0]:.3f}-{t['macro_auc_organ_ci95'][1]:.3f})")


if __name__ == "__main__":
    main()
