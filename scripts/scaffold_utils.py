# -*- coding: utf-8 -*-
"""Scaffold-based data splitting utilities."""
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_split(data_list, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Scaffold-based train/val/test split.
    Same scaffold's molecules always go to the same set.

    Returns: (train_data, val_data, test_data)
    """
    # Group by Bemis-Murcko scaffold
    scaffold_to_indices = defaultdict(list)
    for i, d in enumerate(data_list):
        smi = d.get('smiles', '')
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            else:
                scaffold = f'_unknown_{i}'
        except Exception:
            scaffold = f'_unknown_{i}'
        scaffold_to_indices[scaffold].append(i)

    scaffold_groups = list(scaffold_to_indices.values())
    # Shuffle scaffold groups for reproducible random assignment
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)

    n = len(data_list)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx, val_idx, train_idx = [], [], []
    test_size, val_size = 0, 0

    for group in scaffold_groups:
        if test_size < n_test:
            test_idx.extend(group)
            test_size += len(group)
        elif val_size < n_val:
            val_idx.extend(group)
            val_size += len(group)
        else:
            train_idx.extend(group)

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    n_scaffolds = len(scaffold_groups)
    print(f"  Scaffold split: {n_scaffolds} scaffolds -> "
          f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Verify no scaffold leakage
    train_smiles = {data_list[i].get('smiles', '') for i in train_idx}
    val_smiles = {data_list[i].get('smiles', '') for i in val_idx}
    test_smiles = {data_list[i].get('smiles', '') for i in test_idx}
    overlap_tv = train_smiles & val_smiles
    overlap_tt = train_smiles & test_smiles
    if overlap_tv or overlap_tt:
        print(f"  WARNING: SMILES overlap train-val={len(overlap_tv)}, train-test={len(overlap_tt)}")
    else:
        print(f"  No SMILES overlap between sets (scaffold split clean)")

    return train_data, val_data, test_data
