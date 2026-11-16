"""
Smoke test for the GINE training loop (run this BEFORE any real training).

Checks, in order:
  1. one forward pass: output / label shapes agree, loss is finite
  2. one backward pass: every parameter gets a finite gradient
  3. overfit a single batch: the loss must drop clearly (proves gradients actually train the model)
  4. three full epochs: loss stays finite, memory (process RSS, GPU allocated) does not keep growing
  5. eval forward under no_grad + metric code integration (random-init AUC is ~0.5, not a result)

Usage:
    python smoke_test_gine.py --dataset sider --split scaffold
"""
import argparse
import os
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from eval_utils import summarize
from featurize import ATOM_DIM, BOND_DIM, MoleculeDataset, read_task_list
from gine import GINEModel, masked_bce_with_logits


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) / 1024
        except Exception:  # noqa: BLE001
            pass
    return float("nan")


def gpu_mb(device):
    return torch.cuda.memory_allocated() / 1e6 if device.type == "cuda" else 0.0


def run_epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        loss = masked_bce_with_logits(model(batch), batch.y)
        check(torch.isfinite(loss), "non-finite loss")
        loss.backward()
        opt.step()
        total += loss.item() * batch.num_graphs   # .item(): do NOT keep the graph alive across steps
        n += batch.num_graphs
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = read_task_list(os.path.join(args.data_dir, f"{args.dataset}_eligible_tasks.txt"))
    path = lambda part: os.path.join(args.data_dir, f"{args.dataset}_{args.split}_{part}.csv")  # noqa: E731
    train_ds, valid_ds = MoleculeDataset(path("train"), tasks), MoleculeDataset(path("valid"), tasks)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    print(f"device={device} | train={len(train_ds)} valid={len(valid_ds)} tasks={len(tasks)}")

    def new_model():
        torch.manual_seed(args.seed)
        return GINEModel(ATOM_DIM, BOND_DIM, args.hidden, 3, len(tasks)).to(device)

    # ---- [1] forward
    model = new_model()
    print(f"[1/5] forward | parameters: {sum(p.numel() for p in model.parameters()):,}")
    batch = next(iter(train_loader)).to(device)
    logits = model(batch)
    print(f"      logits {tuple(logits.shape)} | y {tuple(batch.y.shape)} | graphs in batch {batch.num_graphs}")
    check(logits.shape == (batch.num_graphs, len(tasks)), "logit shape wrong")
    check(batch.y.shape == logits.shape, "label shape != logit shape")
    loss = masked_bce_with_logits(logits, batch.y)
    check(torch.isfinite(loss), "loss not finite")
    print(f"      initial loss = {loss.item():.4f} (expected roughly 0.6-0.8 for random init)")

    # ---- [2] backward
    loss.backward()
    zero_grad = []
    for name, p in model.named_parameters():
        check(p.grad is not None, f"no gradient reached parameter {name}")
        check(torch.isfinite(p.grad).all(), f"non-finite gradient in {name}")
        if float(p.grad.abs().sum()) == 0.0:
            zero_grad.append(name)
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())).item()
    print(f"[2/5] backward | all params have finite grads | global grad norm = {gnorm:.4f} "
          f"| params with all-zero grad: {len(zero_grad)} {zero_grad[:3]}")

    # ---- [3] overfit one batch (fresh model, dropout switched off so the test is not noisy)
    m3 = new_model()
    opt3 = torch.optim.Adam(m3.parameters(), lr=args.lr)
    m3.train()
    m3.dropout = 0.0                                    # dropout inside the conv stack
    for mod in m3.modules():
        if isinstance(mod, torch.nn.Dropout):           # dropout inside the head
            mod.p = 0.0
    first = None
    for step in range(150):
        opt3.zero_grad()
        l = masked_bce_with_logits(m3(batch), batch.y)
        l.backward()
        opt3.step()
        first = l.item() if first is None else first
    last = l.item()
    print(f"[3/5] overfit one batch | loss {first:.4f} -> {last:.4f} (150 steps)")
    check(last < 0.7 * first, "loss did not drop on a single batch: gradients may not be training the model")

    # ---- [4] full epochs + memory trend
    model = new_model()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rss, gpu = [], []
    for ep in range(3):
        t0 = time.time()
        ep_loss = run_epoch(model, train_loader, opt, device)
        rss.append(rss_mb())
        gpu.append(gpu_mb(device))
        print(f"[4/5] epoch {ep + 1}: train loss {ep_loss:.4f} | {time.time() - t0:.1f}s | "
              f"RSS {rss[-1]:.0f} MB | GPU allocated {gpu[-1]:.1f} MB")
    growth = rss[2] - rss[1]
    check(abs(gpu[2] - gpu[1]) < 1.0, "GPU memory keeps growing between epochs")
    flag = "OK" if not (growth == growth) or growth < 100 else "SUSPICIOUS"
    print(f"      RSS growth epoch 2 -> 3: {growth:+.0f} MB ({flag}; heuristic limit 100 MB)")
    check(flag == "OK", "process memory grows steadily across epochs: possible leak")

    # ---- [5] eval forward + metric integration
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in valid_loader:
            b = b.to(device)
            out = torch.sigmoid(model(b))
            check(not out.requires_grad, "eval forward still builds an autograd graph")
            ys.append(b.y.cpu().numpy())
            ps.append(out.cpu().numpy())
    res = summarize(np.concatenate(ys), np.concatenate(ps), tasks)
    print(f"[5/5] eval | valid macro-AUC (all)={res['macro_auc_all']:.3f} "
          f"(after 3 epochs; only checks that the metric pipeline runs)")
    print(">> GINE smoke test: PASSED")


if __name__ == "__main__":
    main()
