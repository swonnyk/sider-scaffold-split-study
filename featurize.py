
"""
Step 2: molecular graph featurization (SMILES -> PyG Data), multi-label version.

Reused from the Day 2 Lipophilicity code (atom 34-d, bond 10-d features unchanged), with these changes:
  * `y` is now a [1, n_tasks] vector (one binary label per side-effect class), NaN = missing label.
  * bond features no longer alias an unknown bond type / stereo to the last category
    (previously an unknown bond type would silently be encoded as AROMATIC).
  * The dataset reads the task list from {dataset}_eligible_tasks.txt written by prepare_data.py.

Usage:
    python src/featurize.py --dataset sider --split scaffold      # runs the sanity check
"""
import argparse
import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ATOM_DIM, BOND_DIM = 34, 10


# ------------------------------------------------------------------------------
# Atom / bond featurizer (from Day 2)
# ------------------------------------------------------------------------------
def one_hot_encoding(value: Any, choices: List[Any], allow_unknown: bool = True) -> List[float]:
    """One-hot vector over `choices`. If value is not in choices: the last slot is activated when
    allow_unknown=True (the last choice is an 'Other'/overflow bucket), otherwise all zeros."""
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    elif allow_unknown:
        encoding[-1] = 1.0
    return encoding


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """34-d: atomic number (10) + total degree (6) + formal charge (6) + hybridization (6)
    + aromatic (1) + total H count (5)."""
    allowed_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53, "Other"]  # C N O F P S Cl Br I
    allowed_degrees = [0, 1, 2, 3, 4, 5]
    allowed_formal_charges = [-2, -1, 0, 1, 2, "Other"]
    allowed_hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "Other",
    ]
    allowed_total_hs = [0, 1, 2, 3, 4]

    features: List[float] = []
    features.extend(one_hot_encoding(atom.GetAtomicNum(), allowed_atomic_nums))
    features.extend(one_hot_encoding(atom.GetTotalDegree(), allowed_degrees))
    features.extend(one_hot_encoding(atom.GetFormalCharge(), allowed_formal_charges))
    features.extend(one_hot_encoding(atom.GetHybridization(), allowed_hybridizations))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.extend(one_hot_encoding(atom.GetTotalNumHs(), allowed_total_hs))
    return features


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """10-d: bond type (4) + conjugated (1) + in ring (1) + stereo (4). Unknown types -> all zeros."""
    allowed_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    allowed_stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    features: List[float] = []
    features.extend(one_hot_encoding(bond.GetBondType(), allowed_bond_types, allow_unknown=False))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    features.extend(one_hot_encoding(bond.GetStereo(), allowed_stereos, allow_unknown=False))
    return features


def smiles_to_pyg_data(smiles: str, y_vec: Optional[np.ndarray] = None) -> Optional[Data]:
    """SMILES -> PyG Data. Every bond is stored in both directions (undirected graph).
    y_vec: array of shape [n_tasks] (float, NaN = missing label) or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices, edge_feats = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = get_bond_features(bond)
        edge_indices += [[u, v], [v, u]]
        edge_feats += [f, f]

    if not edge_indices:  # single-atom molecule / ion: no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    y = None
    if y_vec is not None:
        y = torch.tensor(np.asarray(y_vec, dtype=np.float32), dtype=torch.float).unsqueeze(0)  # [1, T]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
def read_task_list(path: str) -> List[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


class MoleculeDataset(Dataset):
    """Reads a split CSV (columns: smiles, <task columns>, ...) and builds PyG Data objects in memory."""

    def __init__(self, csv_file: str, task_cols: List[str], smiles_col: str = "smiles"):
        self.df = pd.read_csv(csv_file)
        self.task_cols = task_cols
        self.smiles_col = smiles_col
        self.data_list: List[Data] = []
        self._process()

    def _process(self):
        labels = self.df[self.task_cols].to_numpy(dtype=np.float32)
        dropped = 0
        for smi, y in zip(self.df[self.smiles_col], labels):
            data = smiles_to_pyg_data(smi, y)
            if data is None:
                dropped += 1
            else:
                self.data_list.append(data)
        if dropped:
            print(f"[Warning] dropped {dropped} unparsable SMILES from {len(self.df)} rows")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


# ------------------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sider")
    ap.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    ap.add_argument("--data_dir", default="data/processed")
    args, _ = ap.parse_known_args()

    tasks = read_task_list(os.path.join(args.data_dir, f"{args.dataset}_eligible_tasks.txt"))
    print(f"[1/3] {len(tasks)} eligible tasks loaded")

    sets = {}
    for part in ("train", "valid", "test"):
        path = os.path.join(args.data_dir, f"{args.dataset}_{args.split}_{part}.csv")
        sets[part] = MoleculeDataset(path, tasks)
        print(f"   - {part}: {len(sets[part])} graphs")

    print("\n[2/3] One sample graph")
    s = sets["train"][0]
    print(f"   smiles={s.smiles}\n   x={tuple(s.x.shape)}  edge_index={tuple(s.edge_index.shape)}  "
          f"edge_attr={tuple(s.edge_attr.shape)}  y={tuple(s.y.shape)}")

    print("\n[3/3] DataLoader batch check")
    batch = next(iter(DataLoader(sets["train"], batch_size=32, shuffle=True)))
    print(f"   nodes={batch.x.shape[0]}  edges={batch.edge_index.shape[1]}  y={tuple(batch.y.shape)}")
    assert batch.x.shape[1] == ATOM_DIM, "atom feature dim mismatch"
    assert batch.edge_attr.shape[1] == BOND_DIM, "bond feature dim mismatch"
    assert batch.y.shape == (32, len(tasks)), "label batch shape mismatch"

    # every molecule in every split must produce a graph (no silent drops) and have no NaN features
    for part, ds in sets.items():
        assert len(ds) == len(ds.df), f"{part}: some molecules were dropped"
        assert all(torch.isfinite(d.x).all() and torch.isfinite(d.edge_attr).all() for d in ds.data_list)
    n_nan = sum(int(torch.isnan(d.y).sum()) for d in sets["train"].data_list)
    print(f"   missing label cells in train: {n_nan}")
    print(">> Featurization integrity check: PASSED")


if __name__ == "__main__":
    main()
