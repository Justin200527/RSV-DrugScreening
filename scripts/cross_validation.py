# -*- coding: utf-8 -*-
"""5-Fold Scaffold Cross-Validation"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MoleculeDataset, collate_fn, trim_size_features
from features import compute_feature_weights, extract_autocorr_features, KEY_2D_FEATURES
from model import RSVDrugScreeningModel, FocalLoss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
SEED = 42
N_FOLDS = 5


def get_scaffold(smiles):
    """Get Bemis-Murcko scaffold from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return smiles


def scaffold_split_cv(data_list, n_folds=5, seed=42):
    """
    Group molecules by scaffold, then distribute scaffold groups to folds
    using greedy balancing. Returns list of fold indices.
    """
    # Group by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, d in enumerate(data_list):
        smi = d.get('smiles', '')
        scaffold = get_scaffold(smi) if smi else f'_unknown_{i}'
        scaffold_to_indices[scaffold].append(i)

    # Shuffle scaffold groups for reproducible random fold assignment
    scaffold_groups = list(scaffold_to_indices.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)

    # Greedy assignment: assign each scaffold group to the smallest fold
    fold_indices = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for group in scaffold_groups:
        smallest_fold = np.argmin(fold_sizes)
        fold_indices[smallest_fold].extend(group)
        fold_sizes[smallest_fold] += len(group)

    print(f"  Scaffold groups: {len(scaffold_groups)}")
    print(f"  Fold sizes: {fold_sizes}")

    return fold_indices


def compute_fold_active_stats(train_data):
    active_autocorr = []
    for d in train_data:
        if d['y'] == 1:
            ac = d['autocorr'].numpy() if hasattr(d['autocorr'], 'numpy') else np.array(d['autocorr'])
            if not np.any(np.isnan(ac)):
                active_autocorr.append(ac)
            if len(active_autocorr) >= 200:
                break

    if not active_autocorr:
        return [0.0] * 19, [1.0] * 19

    active_autocorr = np.array(active_autocorr)
    means = active_autocorr.mean(axis=0).tolist()
    stds = active_autocorr.std(axis=0).tolist()
    stds = [s if s > 1e-8 else 1.0 for s in stds]
    return means, stds


def train_one_fold(fold_idx, train_data, val_data, active_means, active_stds,
                   feature_weights, device):
    print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} ---")
    n_pos_train = sum(1 for d in train_data if d['y'] == 1)
    n_pos_val = sum(1 for d in val_data if d['y'] == 1)
    print(f"  Train: {len(train_data)} ({n_pos_train} pos), Val: {len(val_data)} ({n_pos_val} pos)")

    train_loader = DataLoader(MoleculeDataset(train_data), batch_size=48, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(MoleculeDataset(val_data), batch_size=48, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = RSVDrugScreeningModel(
        active_means=active_means,
        active_stds=active_stds,
        feature_weights=feature_weights,
        size_features=3,
        large_mol_adapt=False,
    ).to(device)

    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    model_save_path = os.path.join(BASE_DIR, 'models', f'cv_fold{fold_idx}.pt')
    best_auc = 0
    best_epoch = 0
    patience = 30
    no_improve = 0
    t0 = time.time()

    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_data in train_loader:
            z = batch_data['z'].to(device)
            pos = batch_data['pos'].to(device)
            batch_idx = batch_data['batch'].to(device)
            pharm = batch_data['pharm'].to(device)
            size = batch_data['size'].to(device)
            autocorr = batch_data['autocorr'].to(device)
            shape3d = batch_data['shape3d'].to(device)
            y = batch_data['y'].to(device)
            atom_feat = batch_data['atom_feat'].to(device) if 'atom_feat' in batch_data else None
            bond_ei = batch_data['bond_edge_index'].to(device) if 'bond_edge_index' in batch_data else None
            bond_ea = batch_data['bond_edge_attr'].to(device) if 'bond_edge_attr' in batch_data else None

            optimizer.zero_grad()
            logits = model(z, pos, batch_idx, pharm, size, autocorr, shape3d,
                           atom_feat, bond_ei, bond_ea)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                z = batch_data['z'].to(device)
                pos = batch_data['pos'].to(device)
                batch_idx = batch_data['batch'].to(device)
                pharm = batch_data['pharm'].to(device)
                size = batch_data['size'].to(device)
                autocorr = batch_data['autocorr'].to(device)
                shape3d = batch_data['shape3d'].to(device)
                atom_feat = batch_data['atom_feat'].to(device) if 'atom_feat' in batch_data else None
                bond_ei = batch_data['bond_edge_index'].to(device) if 'bond_edge_index' in batch_data else None
                bond_ea = batch_data['bond_edge_attr'].to(device) if 'bond_edge_attr' in batch_data else None
                logits = model(z, pos, batch_idx, pharm, size, autocorr, shape3d,
                               atom_feat, bond_ei, bond_ea)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch_data['y'].numpy())

        val_auc = roc_auc_score(val_labels, val_preds)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

    # Final evaluation on val (which is the held-out fold)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch_data in val_loader:
            z = batch_data['z'].to(device)
            pos = batch_data['pos'].to(device)
            batch_idx = batch_data['batch'].to(device)
            pharm = batch_data['pharm'].to(device)
            size = batch_data['size'].to(device)
            autocorr = batch_data['autocorr'].to(device)
            shape3d = batch_data['shape3d'].to(device)
            atom_feat = batch_data['atom_feat'].to(device) if 'atom_feat' in batch_data else None
            bond_ei = batch_data['bond_edge_index'].to(device) if 'bond_edge_index' in batch_data else None
            bond_ea = batch_data['bond_edge_attr'].to(device) if 'bond_edge_attr' in batch_data else None
            logits = model(z, pos, batch_idx, pharm, size, autocorr, shape3d,
                           atom_feat, bond_ei, bond_ea)
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(batch_data['y'].numpy())

    val_auc = roc_auc_score(val_labels, val_preds)
    val_ap = average_precision_score(val_labels, val_preds)
    train_time = time.time() - t0

    print(f"  Fold {fold_idx+1}: AUC={val_auc:.4f}, AP={val_ap:.4f}, Ep={best_epoch}, Time={train_time:.0f}s")

    return {
        'fold': fold_idx,
        'auc': float(val_auc),
        'ap': float(val_ap),
        'best_epoch': best_epoch,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'train_time_seconds': train_time,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("5-Fold Scaffold Cross-Validation")
    print(f"Device: {device}")

    # Load cached data
    cache_path = os.path.join(BASE_DIR, 'data', 'train_augmented_cache.pkl')
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found: {cache_path}")
        print("Please run train.py first to generate the training data cache.")
        sys.exit(1)
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    data_list = cache['data_list']
    print(f"  Total: {len(data_list)} samples")

    trim_size_features(data_list)
    print(f"  Size features -> 3d (shape: {data_list[0]['size'].shape})")

    if 'smiles' not in data_list[0]:
        print("ERROR: Cache does not contain SMILES. Run train.py first to generate the training data cache.")
        sys.exit(1)

    feature_weights = compute_feature_weights()

    # Scaffold split into 5 folds
    print("\nComputing scaffold splits...")
    fold_indices = scaffold_split_cv(data_list, n_folds=N_FOLDS, seed=SEED)

    # Run CV
    all_results = []
    for fold_idx in range(N_FOLDS):
        # Train = all other folds, Val = current fold
        val_indices = fold_indices[fold_idx]
        train_indices = []
        for j in range(N_FOLDS):
            if j != fold_idx:
                train_indices.extend(fold_indices[j])

        val_data = [data_list[i] for i in val_indices]
        train_data = [data_list[i] for i in train_indices]

        # Compute active stats from training fold only (avoid leakage)
        active_means, active_stds = compute_fold_active_stats(train_data)

        result = train_one_fold(fold_idx, train_data, val_data,
                                active_means, active_stds, feature_weights, device)
        all_results.append(result)

        # Save incrementally
        results_path = os.path.join(BASE_DIR, 'results', 'cv_results.json')
        aucs = [r['auc'] for r in all_results]
        aps = [r['ap'] for r in all_results]
        summary = {
            'n_folds': N_FOLDS,
            'split_type': 'scaffold',
            'seed': SEED,
            'per_fold': all_results,
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs, ddof=1)),
            'ap_mean': float(np.mean(aps)),
            'ap_std': float(np.std(aps, ddof=1)),
        }
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

    # Print summary
    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]

    print("\n5-Fold Scaffold CV Results")
    print(f"{'Fold':>6s} {'AUC':>10s} {'AP':>10s}")
    print("-" * 28)
    for r in all_results:
        print(f"{r['fold']+1:>6d} {r['auc']:>10.4f} {r['ap']:>10.4f}")
    print("-" * 28)
    print(f"{'Mean':>6s} {np.mean(aucs):>10.4f} {np.mean(aps):>10.4f}")
    print(f"{'Std':>6s} {np.std(aucs, ddof=1):>10.4f} {np.std(aps, ddof=1):>10.4f}")
    print(f"\nScaffold CV AUC: {np.mean(aucs):.4f} +/- {np.std(aucs, ddof=1):.4f}")
    print(f"Scaffold CV AP:  {np.mean(aps):.4f} +/- {np.std(aps, ddof=1):.4f}")
    print(f"\nResults saved to {os.path.join(BASE_DIR, 'results', 'cv_results.json')}")


if __name__ == '__main__':
    main()
