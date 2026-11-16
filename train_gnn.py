"""
Train + evaluate the GINE model (multi-label, masked BCE) on random and/or scaffold splits.

Protocol (fixed in advance, no hyper-parameter search):
  * Adam, lr 1e-3, weight decay 1e-5, batch 32, hidden 128, 3 GINE layers, dropout 0.2
  * up to --epochs epochs; model selection = best validation macro-AUC (all tasks); early stop after --patience
    epochs without improvement. The test set is never used for any decision.
  * 3 model seeds (0,1,2); the SPLIT is the same for all seeds (split variance is not measured).

Usage:
    python train_gnn.py --dataset sider --split both
Outputs:
    results/gnn_{dataset}_{split}_seed{seed}.json        same format as the RF results
    results/gnn_{dataset}_{split}_seed{seed}_test_pred.npy   test-set probabilities [n_test, n_tasks]
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from eval_utils import macro_auc, per_task_auc, summarize
from featurize import ATOM_DIM, BOND_DIM, MoleculeDataset, read_task_list
from gine import GINEModel, masked_bce_with_logits


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = b.to(device)
        ps.append(torch.sigmoid(model(b)).cpu().numpy())
        ys.append(b.y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(args, split, seed, tasks, loaders, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, valid_loader, test_loader = loaders
    model = GINEModel(ATOM_DIM, BOND_DIM, args.hidden, 3, len(tasks), args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = masked_bce_with_logits(model(batch), batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        yv, pv = predict(model, valid_loader, device)
        v_auc = macro_auc(per_task_auc(yv, pv, tasks))
        if v_auc > best_auc:
            best_auc, best_epoch, bad = v_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"   [{split} seed {seed}] epoch {epoch:3d} train loss {total / n:.4f} | "
                  f"valid macro-AUC {v_auc:.3f} | best {best_auc:.3f} @ {best_epoch}")
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    yv, pv = predict(model, valid_loader, device)
    yt, pt = predict(model, test_loader, device)
    res = {
        "model": "gine", "dataset": args.dataset, "split": split, "seed": seed,
        "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset),
        "n_test": len(test_loader.dataset), "n_tasks": len(tasks),
        "best_epoch": best_epoch, "epochs_run": epoch, "train_seconds": round(time.time() - t0, 1),
        "hparams": {k: getattr(args, k) for k in ("lr", "weight_decay", "hidden", "dropout", "batch_size",
                                                   "epochs", "patience")},
        "valid": summarize(yv, pv, tasks),
        "test": summarize(yt, pt, tasks, with_ci=True),
    }
    stem = os.path.join(args.out_dir, f"gnn_{args.dataset}_{split}_seed{seed}")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=2)
    np.save(stem + "_test_pred.npy", pt)
    t, v = res["test"], res["valid"]
    print(f"[GINE | {split:8s} | seed {seed}] best epoch {best_epoch} | "
          f"valid macro-AUC all={v['macro_auc_all']:.3f} organ={v['macro_auc_organ']:.3f} | "
          f"test macro-AUC all={t['macro_auc_all']:.3f} "
          f"(95% CI {t['macro_auc_all_ci95'][0]:.3f}-{t['macro_auc_all_ci95'][1]:.3f}) "
          f"organ={t['macro_auc_organ']:.3f} "
          f"(95% CI {t['macro_auc_organ_ci95'][0]:.3f}-{t['macro_auc_organ_ci95'][1]:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--split", choices=["random", "scaffold", "both"], default="both")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = read_task_list(os.path.join(args.data_dir, f"{args.dataset}_eligible_tasks.txt"))
    splits = ["random", "scaffold"] if args.split == "both" else [args.split]
    print(f"device={device} | tasks={len(tasks)} | splits={splits} | seeds={args.seeds}")

    for split in splits:
        path = lambda part: os.path.join(args.data_dir, f"{args.dataset}_{split}_{part}.csv")  # noqa: E731
        ds = {p: MoleculeDataset(path(p), tasks) for p in ("train", "valid", "test")}
        for p, d in ds.items():   # no molecule may be silently dropped: test predictions must align with the CSV rows
            assert len(d) == len(d.df), f"{split}/{p}: some molecules failed featurization"
        for seed in args.seeds:
            g = torch.Generator().manual_seed(seed)
            loaders = (DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, generator=g),
                       DataLoader(ds["valid"], batch_size=args.batch_size),
                       DataLoader(ds["test"], batch_size=args.batch_size))
            train_one(args, split, seed, tasks, loaders, device)


if __name__ == "__main__":
    main()
