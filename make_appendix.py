"""
Appendix table: per-label test ROC-AUC (RF and GINE, random and scaffold splits), from saved results.

For each of the eligible labels: number of positives / negatives in the test set of each split, and the
test ROC-AUC averaged over the 3 model seeds. Labels are listed alphabetically (NOT sorted by AUC) because the
values are reported descriptively only.

Usage:
    python make_appendix.py --dataset sider
Outputs:
    results/per_label_auc.csv
    results/per_label_auc.md      (markdown table to paste into the README appendix)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PREFIX = {"RF": "rf", "GINE": "gnn"}
SPLITS = ["random", "scaffold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args, _ = ap.parse_known_args()

    with open(os.path.join(args.data_dir, f"{args.dataset}_eligible_tasks.txt")) as f:
        tasks = sorted(l.rstrip("\n") for l in f if l.strip())

    rows = {t: {"label": t} for t in tasks}
    for split in SPLITS:
        te = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_{split}_test.csv"))
        for t in tasks:
            rows[t][f"{split}_n_pos"] = int((te[t] == 1).sum())
            rows[t][f"{split}_n_neg"] = int((te[t] == 0).sum())
        for model, prefix in PREFIX.items():
            per_seed = []
            for s in args.seeds:
                path = os.path.join(args.out_dir, f"{prefix}_{args.dataset}_{split}_seed{s}.json")
                with open(path) as f:
                    per_seed.append(json.load(f)["test"]["per_task_auc"])
            for t in tasks:
                rows[t][f"{split}_{model}_auc"] = float(np.mean([d[t] for d in per_seed]))

    df = pd.DataFrame([rows[t] for t in tasks])
    df.to_csv(os.path.join(args.out_dir, "per_label_auc.csv"), index=False)

    lines = [
        "| Label | Random test (pos/neg) | RF | GINE | Scaffold test (pos/neg) | RF | GINE |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['label']} | {r['random_n_pos']}/{r['random_n_neg']} | "
            f"{r['random_RF_auc']:.3f} | {r['random_GINE_auc']:.3f} | "
            f"{r['scaffold_n_pos']}/{r['scaffold_n_neg']} | "
            f"{r['scaffold_RF_auc']:.3f} | {r['scaffold_GINE_auc']:.3f} |"
        )
    md = "\n".join(lines)
    with open(os.path.join(args.out_dir, "per_label_auc.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nsaved {args.out_dir}/per_label_auc.csv and {args.out_dir}/per_label_auc.md")


if __name__ == "__main__":
    main()
