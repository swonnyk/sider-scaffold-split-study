
"""
Step 1: dataset download, validation audit, and Random / Bemis-Murcko scaffold splits.

Usage (Colab or local):
    python src/prepare_data.py --dataset sider
    python src/prepare_data.py --dataset tox21            # fallback dataset
    python src/prepare_data.py --dataset sider --csv path/to/sider.csv   # if download is blocked

Outputs (in --out_dir, default: data/processed):
    {dataset}_clean.csv                     standardized molecules + labels + scaffold
    {dataset}_task_stats.csv                per-task positives / negatives / missing / flags
    {dataset}_{random|scaffold}_{train|valid|test}.csv
    {dataset}_audit_report.json             all numbers printed below, machine-readable
"""
import argparse
import gzip
import io
import json
import os
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

URLS = {
    "sider": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
}
NON_TASK_COLS = {"mol_id", "id", "name", "drug", "smiles"}

# Heuristic thresholds for "is this task usable for evaluation?" (project decision, not a standard)
MIN_POSITIVES_TOTAL = 50      # fewer positives overall -> too noisy to evaluate
MIN_RATE, MAX_RATE = 0.05, 0.95  # extreme imbalance flag
MIN_POS_PER_SPLIT = 3         # need some positives AND negatives in valid/test for ROC-AUC
GO_MIN_ELIGIBLE_TASKS = 10    # go/no-go: need at least this many usable tasks


# ------------------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------------------
def load_raw(dataset: str, csv_path: str = None, raw_dir: str = "data/raw") -> pd.DataFrame:
    os.makedirs(raw_dir, exist_ok=True)
    if csv_path:
        print(f"[load] reading local file: {csv_path}")
        return pd.read_csv(csv_path)
    local = os.path.join(raw_dir, f"{dataset}.csv.gz")
    if not os.path.exists(local):
        url = URLS[dataset]
        print(f"[load] downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = r.read()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Download failed ({e}).\nDownload the file manually from\n  {url}\n"
                f"and rerun with  --csv <path-to-file>"
            )
        with open(local, "wb") as f:
            f.write(payload)
    return pd.read_csv(local)


# ------------------------------------------------------------------------------
# 2. Standardize + validate molecules
# ------------------------------------------------------------------------------
_chooser = rdMolStandardize.LargestFragmentChooser()


def standardize(smiles: str):
    """Parse, keep the largest fragment (drops counter-ions/salts), return (canonical smiles, n_fragments)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None, 0
    n_frag = len(Chem.GetMolFrags(mol))
    mol = _chooser.choose(mol)
    return Chem.MolToSmiles(mol), n_frag


def scaffold_of(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES ('' for acyclic molecules)."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)


def clean_dataframe(raw: pd.DataFrame):
    smiles_col = next(c for c in raw.columns if c.lower() == "smiles")
    task_cols = [c for c in raw.columns if c.lower() not in NON_TASK_COLS]
    report = {"n_raw_rows": int(len(raw)), "n_tasks": len(task_cols)}

    std = raw[smiles_col].map(standardize)
    raw = raw.assign(_smiles=[s for s, _ in std], _n_frag=[n for _, n in std])
    n_invalid = int(raw["_smiles"].isna().sum())
    raw = raw[raw["_smiles"].notna()].copy()
    for c in task_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    report["n_invalid_smiles"] = n_invalid
    report["n_multi_fragment_before_cleanup"] = int((raw["_n_frag"] > 1).sum())

    # duplicates after standardization (e.g. the same parent drug listed as different salts)
    dup_mask = raw.duplicated("_smiles", keep=False)
    conflicts = 0
    if dup_mask.any():
        g = raw[dup_mask].groupby("_smiles")[task_cols].nunique(dropna=True)
        conflicts = int((g.max(axis=1) > 1).sum())
    report["n_duplicate_rows_merged"] = int(raw.duplicated("_smiles").sum())
    report["n_duplicate_groups_with_conflicting_labels"] = conflicts

    # Merge duplicates by logical OR: a class is positive if ANY duplicate entry reports it
    # (an association reported for any salt form still belongs to the parent compound).
    # max() ignores NaN, so a label missing in every duplicate stays missing.
    merged = raw.groupby("_smiles", sort=False)[task_cols].max()
    df = merged.reset_index().rename(columns={"_smiles": "smiles"})
    df["scaffold"] = df["smiles"].map(scaffold_of)
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    df["n_heavy_atoms"] = [m.GetNumHeavyAtoms() for m in mols]
    report["n_molecules_clean"] = int(len(df))
    return df.reset_index(drop=True), task_cols, report


# ------------------------------------------------------------------------------
# 3. Splits
# ------------------------------------------------------------------------------
def random_split(n, seed, frac=(0.8, 0.1, 0.1)):
    idx = np.random.RandomState(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def scaffold_split(scaffolds, frac=(0.8, 0.1, 0.1)):
    """Deterministic DeepChem-style: biggest scaffold groups go to train, rarest/smallest end up in test."""
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(scaffolds)
    n_train, n_valid = frac[0] * n, (frac[0] + frac[1]) * n
    train, valid, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(train) + len(valid) + len(g) <= n_valid:
            valid += g
        else:
            test += g
    return np.array(train), np.array(valid), np.array(test)


# ------------------------------------------------------------------------------
# 4. Audit helpers
# ------------------------------------------------------------------------------
def task_statistics(df, task_cols):
    rows = []
    for c in task_cols:
        y = df[c]
        n_obs = int(y.notna().sum())
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        rate = pos / n_obs if n_obs else np.nan
        flags = []
        if pos < MIN_POSITIVES_TOTAL:
            flags.append("few_positives")
        if n_obs and (rate < MIN_RATE or rate > MAX_RATE):
            flags.append("extreme_imbalance")
        rows.append(dict(task=c, n_observed=n_obs, n_missing=len(df) - n_obs,
                         positives=pos, negatives=neg, positive_rate=round(rate, 4), flags=";".join(flags)))
    return pd.DataFrame(rows)


def split_task_counts(df, task_cols, idx_sets, names):
    """Per task: positives/negatives in each of valid/test; True if ROC-AUC is computable & not tiny."""
    out = {}
    for c in task_cols:
        ok = True
        for nm, idx in zip(names, idx_sets):
            if nm == "train":
                continue
            y = df.loc[idx, c].dropna()
            if (y == 1).sum() < MIN_POS_PER_SPLIT or (y == 0).sum() < MIN_POS_PER_SPLIT:
                ok = False
        out[c] = ok
    return out


def max_sim_to_train(smiles, train_idx, query_idx):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    train_fps = [fps[i] for i in train_idx]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in query_idx])


def scaffold_overlap(df, train_idx, other_idx):
    tr = set(df.loc[train_idx, "scaffold"])
    return float(np.mean([s in tr for s in df.loc[other_idx, "scaffold"]]))


# ------------------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(URLS), default="sider")
    ap.add_argument("--csv", default=None, help="local csv/csv.gz (skips download)")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    # Modified to run cleanly in Jupyter environment by ignoring internal runtime flags
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = args.dataset

    # ---- load + clean
    raw = load_raw(ds, args.csv)
    df, task_cols, report = clean_dataframe(raw)
    print("\n=== 1. Molecule validity ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # ---- label audit
    stats = task_statistics(df, task_cols)
    stats.to_csv(os.path.join(args.out_dir, f"{ds}_task_stats.csv"), index=False)
    print("\n=== 2. Per-task label statistics ===")
    print(stats.to_string(index=False))
    print(f"\n  total missing label cells: {int(df[task_cols].isna().sum().sum())} "
          f"of {df[task_cols].size}")
    print(f"  mean #positive labels per molecule: {df[task_cols].eq(1).sum(axis=1).mean():.2f}")
    print(f"  molecules with zero positive labels: {int((df[task_cols].eq(1).sum(axis=1) == 0).sum())}")

    # ---- scaffold distribution
    sc = df["scaffold"].value_counts()
    scaf = dict(
        n_unique_scaffolds=int(len(sc)),
        n_singleton_scaffolds=int((sc == 1).sum()),
        frac_acyclic_no_scaffold=round(float((df["scaffold"] == "").mean()), 4),
        largest_scaffold_group=int(sc.iloc[0]),
        top5_scaffold_groups_share=round(float(sc.iloc[:5].sum() / len(df)), 4),
    )
    report["scaffold_distribution"] = scaf
    print("\n=== 3. Scaffold distribution ===")
    for k, v in scaf.items():
        print(f"  {k}: {v}")
    print("  molecule size (heavy atoms): "
          f"median={df['n_heavy_atoms'].median():.0f}, min={df['n_heavy_atoms'].min()}, "
          f"max={df['n_heavy_atoms'].max()}")

    # ---- splits
    split_defs = {
        "random": random_split(len(df), args.seed),
        "scaffold": scaffold_split(df["scaffold"].tolist()),
    }
    names = ["train", "valid", "test"]
    eligible = {}
    report["splits"] = {}
    print("\n=== 4. Splits ===")
    for sp, idx_sets in split_defs.items():
        for nm, idx in zip(names, idx_sets):
            df.iloc[idx].to_csv(os.path.join(args.out_dir, f"{ds}_{sp}_{nm}.csv"), index=False)
        sizes = {nm: int(len(i)) for nm, i in zip(names, idx_sets)}
        ms = {nm: max_sim_to_train(df["smiles"].tolist(), idx_sets[0], idx)
              for nm, idx in zip(names[1:], idx_sets[1:])}
        info = dict(
            sizes=sizes,
            test_scaffold_seen_in_train=round(scaffold_overlap(df, idx_sets[0], idx_sets[2]), 4),
            test_max_tanimoto_to_train_median=round(float(np.median(ms["test"])), 3),
            test_max_tanimoto_to_train_q25_q75=[round(float(np.percentile(ms["test"], q)), 3) for q in (25, 75)],
            test_frac_with_max_sim_above_0p7=round(float((ms["test"] > 0.7).mean()), 3),
        )
        report["splits"][sp] = info
        eligible[sp] = split_task_counts(df, task_cols, idx_sets, names)
        print(f"  [{sp}] {json.dumps(info)}")

    # ---- eligible tasks (must be usable under BOTH splits)
    base_ok = set(stats.loc[stats["flags"] == "", "task"])
    final = [c for c in task_cols
             if c in base_ok and eligible["random"][c] and eligible["scaffold"][c]]
    report["eligible_tasks"] = final
    report["excluded_tasks"] = [c for c in task_cols if c not in final]
    with open(os.path.join(args.out_dir, f"{ds}_eligible_tasks.txt"), "w") as f:
        f.write("\n".join(final))
    print(f"\n=== 5. Eligible tasks (usable under BOTH splits): {len(final)} / {len(task_cols)} ===")
    for c in final:
        print(f"   + {c}")
    for c in report["excluded_tasks"]:
        print(f"   - {c}  (excluded)")

    verdict = "GO" if len(final) >= GO_MIN_ELIGIBLE_TASKS else "NO-GO (consider --dataset tox21)"
    report["go_no_go"] = verdict
    print(f"\n>> Heuristic verdict for {ds}: {verdict} "
          f"(rule: >= {GO_MIN_ELIGIBLE_TASKS} eligible tasks)")

    df.to_csv(os.path.join(args.out_dir, f"{ds}_clean.csv"), index=False)
    with open(os.path.join(args.out_dir, f"{ds}_audit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved all outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
