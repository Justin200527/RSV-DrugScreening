# -*- coding: utf-8 -*-
"""Ablation Study"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import json
import numpy as np
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MoleculeDataset, collate_fn, trim_size_features
from scaffold_utils import scaffold_split
from features import compute_active_statistics, compute_feature_weights
from model import RSVDrugScreeningModel, FocalLoss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
SEED = 42

ABLATION_CONFIGS = {
    'full':         {'disabled': [], 'name': 'Full Model'},
    'wo_graph':     {'disabled': ['graph'], 'name': 'w/o Graph'},
    'wo_dist':      {'disabled': ['dist'], 'name': 'w/o Distance'},
    'wo_pharm':     {'disabled': ['pharm'], 'name': 'w/o Pharmacophore'},
    'wo_size':      {'disabled': ['size'], 'name': 'w/o Size'},
    'wo_autocorr':  {'disabled': ['autocorr'], 'name': 'w/o Autocorrelation'},
    'wo_shape3d':   {'disabled': ['shape3d'], 'name': 'w/o Shape3D'},
}


def train_one_config(config_key, disabled_modules, data_splits, active_means,
                     active_stds, feature_weights, device):
    train_data, val_data, test_data = data_splits
    config_name = ABLATION_CONFIGS[config_key]['name']
    print(f"\n--- {config_name} (disabled={disabled_modules or 'none'}) ---")

    train_loader = DataLoader(MoleculeDataset(train_data), batch_size=48, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(MoleculeDataset(val_data), batch_size=48, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(MoleculeDataset(test_data), batch_size=48, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = RSVDrugScreeningModel(
        active_means=active_means,
        active_stds=active_stds,
        feature_weights=feature_weights,
        size_features=3,
        large_mol_adapt=False,
        disabled_modules=disabled_modules if disabled_modules else None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    model_save_path = os.path.join(BASE_DIR, 'models', f'ablation_{config_key}.pt')
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

        # Validate
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
        avg_loss = total_loss / len(train_loader)

        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            no_improve = 0
            marker = " ***"
            torch.save(model.state_dict(), model_save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0 or marker:
            print(f"  Ep {epoch+1:02d} | Loss: {avg_loss:.4f} | AUC: {val_auc:.4f} | AP: {val_ap:.4f}{marker}")

    # Test evaluation
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_data in test_loader:
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
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels.extend(batch_data['y'].numpy())

    test_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    train_time = time.time() - t0

    print(f"\n  {config_name}: Val AUC={best_auc:.4f}, Test AUC={test_auc:.4f}, "
          f"AP={test_ap:.4f}, Time={train_time:.0f}s")

    return {
        'config': config_key,
        'name': config_name,
        'disabled_modules': disabled_modules,
        'val_auc': float(best_auc),
        'test_auc': float(test_auc),
        'test_ap': float(test_ap),
        'best_epoch': best_epoch,
        'n_params': n_params,
        'train_time_seconds': train_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=list(ABLATION_CONFIGS.keys()),
                        choices=list(ABLATION_CONFIGS.keys()))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Ablation Study")
    print(f"Device: {device}")
    print(f"Configs: {args.configs}")

    # Load cached data
    cache_path = os.path.join(BASE_DIR, 'data', 'train_augmented_cache.pkl')
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found: {cache_path}")
        print("Please run train.py first to generate the training data cache.")
        sys.exit(1)
    print(f"\nLoading cache from {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    data_list = cache['data_list']
    print(f"  Total: {len(data_list)} samples")

    trim_size_features(data_list)
    print(f"  Size features -> 3d (shape: {data_list[0]['size'].shape})")

    # Compute active statistics
    import pandas as pd
    data_path = os.path.join(BASE_DIR, 'data', 'rsv_training_inhibition_22p3.csv')
    train_df = pd.read_csv(data_path)
    orig_pos = train_df[train_df['label'] == 1]
    active_means, active_stds = compute_active_statistics(orig_pos['SMILES'].tolist())
    feature_weights = compute_feature_weights()

    # Scaffold-based split (no data leakage)
    train_data, val_data, test_data = scaffold_split(data_list, seed=SEED)

    data_splits = (train_data, val_data, test_data)

    # Run ablations
    all_results = {}
    for config_key in args.configs:
        cfg = ABLATION_CONFIGS[config_key]
        result = train_one_config(
            config_key, cfg['disabled'], data_splits,
            active_means, active_stds, feature_weights, device)
        all_results[config_key] = result

        # Save incrementally
        results_path = os.path.join(BASE_DIR, 'results', 'ablation_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print("\nAblation Study Summary")
    print(f"{'Config':<25s} {'Val AUC':>10s} {'Test AUC':>10s} {'Test AP':>10s} {'Params':>10s}")
    print("-" * 65)

    full_auc = all_results.get('full', {}).get('test_auc', 0)
    for key in args.configs:
        r = all_results[key]
        delta = r['test_auc'] - full_auc if key != 'full' else 0
        delta_str = f" ({delta:+.4f})" if key != 'full' else ""
        print(f"{r['name']:<25s} {r['val_auc']:>10.4f} {r['test_auc']:>10.4f}{delta_str:>8s} "
              f"{r['test_ap']:>10.4f} {r['n_params']:>10,}")

    print(f"\nResults saved to {os.path.join(BASE_DIR, 'results', 'ablation_results.json')}")


if __name__ == '__main__':
    main()
