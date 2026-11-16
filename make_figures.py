"""
Make the main figure from saved results (no training).
  Panel A: test macro ROC-AUC (25 labels) per model x split. Bar = mean of 3 seeds, dots = individual seeds,
           whisker = average of the 3 seeds' bootstrap 95% CI bounds (reflects test-set size only).
  Panel B: macro ROC-AUC inside fixed similarity bins (from similarity_bins.csv; mean of 3 seeds).

Usage:
    python make_figures.py --dataset sider
Output:
    results/figure_main.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"RF": "#1f77b4", "GINE": "#d95f02"}
PREFIX = {"RF": "rf", "GINE": "gnn"}
SPLITS = ["random", "scaffold"]
SPLIT_LABEL = {"random": "Random", "scaffold": "Scaffold"}


def load(out_dir, dataset, model, split, seeds):
    vals, los, his = [], [], []
    for s in seeds:
        path = os.path.join(out_dir, f"{PREFIX[model]}_{dataset}_{split}_seed{s}.json")
        with open(path) as f:
            t = json.load(f)["test"]
        vals.append(t["macro_auc_all"])
        los.append(t["macro_auc_all_ci95"][0])
        his.append(t["macro_auc_all_ci95"][1])
    return np.array(vals), float(np.mean(los)), float(np.mean(his))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args, _ = ap.parse_known_args()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---------------- Panel A ----------------
    width = 0.34
    for mi, model in enumerate(["RF", "GINE"]):
        for si, split in enumerate(SPLITS):
            vals, lo, hi = load(args.out_dir, args.dataset, model, split, args.seeds)
            x = si + (mi - 0.5) * width
            m = vals.mean()
            axA.bar(x, m, width * 0.92, color=COLORS[model], alpha=0.75,
                    label=model if si == 0 else None)
            axA.plot([x, x], [lo, hi], color="black", lw=1.4)
            axA.plot([x - 0.04, x + 0.04], [lo, lo], color="black", lw=1.4)
            axA.plot([x - 0.04, x + 0.04], [hi, hi], color="black", lw=1.4)
            axA.scatter([x] * len(vals), vals, color="black", s=14, zorder=3)
            axA.text(x, 0.455, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color="white",
                     fontweight="bold")
    axA.axhline(0.5, color="gray", ls="--", lw=1)
    axA.text(1.52, 0.502, "chance (0.5)", color="gray", fontsize=8, ha="right", va="bottom")
    axA.set_xticks(range(len(SPLITS)))
    axA.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS])
    axA.set_ylim(0.45, 0.80)
    axA.set_xlim(-0.6, 1.6)
    axA.set_ylabel("Test macro ROC-AUC (25 labels)")
    axA.set_title("A. Test performance by split", fontsize=11)
    axA.legend(loc="upper right", frameon=False)
    axA.text(0.0, -0.17, "Bar: mean of 3 seeds. Dots: seeds. Whisker: avg. of per-seed bootstrap 95% CI\n"
             "(test-set size only; split-to-split variation not measured).",
             transform=axA.transAxes, fontsize=7.5, va="top")

    # ---------------- Panel B ----------------
    bins = pd.read_csv(os.path.join(args.out_dir, "similarity_bins.csv"))
    order = ["[0.0,0.4)", "[0.4,0.6)", "[0.6,1.0]"]
    xs = np.arange(len(order))
    linestyle = {"random": "-", "scaffold": "--"}
    marker = {"random": "o", "scaffold": "s"}
    for model in ["RF", "GINE"]:
        for split in SPLITS:
            sub = bins[(bins["model"] == model) & (bins["split"] == split)].set_index("bin").loc[order]
            axB.plot(xs, sub["macro_auc"].values, color=COLORS[model], ls=linestyle[split],
                     marker=marker[split], lw=1.8, label=f"{model} / {SPLIT_LABEL[split]}")
    axB.axhline(0.5, color="gray", ls="--", lw=1)
    n_rand = bins[(bins["model"] == "RF") & (bins["split"] == "random")].set_index("bin").loc[order, "n_molecules"]
    n_scaf = bins[(bins["model"] == "RF") & (bins["split"] == "scaffold")].set_index("bin").loc[order, "n_molecules"]
    axB.set_xticks(xs)
    axB.set_xticklabels([f"{b}\nRandom n={a}\nScaffold n={c}"
                         for b, a, c in zip(order, n_rand, n_scaf)], fontsize=8)
    axB.set_xlabel("Max Tanimoto similarity of test molecule to training set")
    axB.set_ylim(0.45, 0.80)
    axB.set_ylabel("Macro ROC-AUC within bin")
    axB.set_title("B. Performance by structural similarity", fontsize=11)
    axB.legend(frameon=False, fontsize=8, loc="upper left")
    axB.text(0.0, -0.30, "Bins fixed before looking at results. No CIs; bins hold 30-80 molecules, and the number of\n"
             "usable labels differs per bin (19-25). Descriptive only.",
             transform=axB.transAxes, fontsize=7.5, va="top")

    fig.tight_layout()
    out = os.path.join(args.out_dir, "figure_main.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()