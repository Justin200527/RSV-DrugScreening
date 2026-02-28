# -*- coding: utf-8 -*-
"""GNN Multi-Seed Statistical Comparison"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import json
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import time
import gc
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaffold_utils import scaffold_split
from gnn_baselines import (
    MolGraphDataset, pyg_collate, set_seed, train_and_evaluate,
    GCNModel, MPNNModel, ATOM_FEAT_DIM, BOND_FEAT_DIM,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
DATA_SEED = 42
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def welch_ttest(x, y):
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
    return float(t_stat), float(p_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['mpnn', 'gcn'],
                        choices=['mpnn', 'gcn'])
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("GNN Multi-Seed Statistical Comparison")
    print(f"Device: {device}, Models: {args.models}, Seeds: {args.seeds}")

    # Load cached data (same data as all other scripts)
    cache_path = os.path.join(BASE_DIR, 'data', 'train_augmented_cache.pkl')
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found: {cache_path}")
        print("Please run train.py first to generate the training data cache.")
        sys.exit(1)
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    data_list = cache['data_list']
    print(f"  Total: {len(data_list)} samples")

    # Fixed data split (same seed as multi_seed.py)
    train_data, val_data, test_data = scaffold_split(data_list, seed=DATA_SEED)

    # Load proposed model's multi-seed results for comparison
    ms_path = os.path.join(BASE_DIR, 'results', 'multi_seed_results.json')
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            proposed_ms = json.load(f)
        proposed_aucs = [r['test_auc'] for r in proposed_ms['per_seed']]
        proposed_mean = proposed_ms['test_auc_mean']
        proposed_std = proposed_ms['test_auc_std']
    else:
        print("WARNING: multi_seed_results.json not found. Run multi_seed.py first for comparison.")
        proposed_aucs = None
        proposed_mean = None
        proposed_std = None

    model_configs = {
        'mpnn': {
            'name': 'MPNN', 'use_edge_attr': True, 'lr': 5e-4,
            'factory': lambda: MPNNModel(in_dim=ATOM_FEAT_DIM, hidden_dim=128,
                                         edge_dim=BOND_FEAT_DIM, num_layers=4),
        },
        'gcn': {
            'name': 'GCN', 'use_edge_attr': False, 'lr': 1e-3,
            'factory': lambda: GCNModel(in_dim=ATOM_FEAT_DIM, hidden_dim=128, num_layers=4),
        },
    }

    for model_key in args.models:
        cfg = model_configs[model_key]
        model_name = cfg['name']
        print(f"\n{model_name} Multi-Seed Evaluation ({len(args.seeds)} seeds)")

        seed_results = []
        for seed in args.seeds:
            print(f"\n  Seed {seed}:")
            set_seed(seed)
            model = cfg['factory']().to(device)

            use_ea = cfg['use_edge_attr']
            train_ds = MolGraphDataset(train_data)
            val_ds = MolGraphDataset(val_data)
            test_ds = MolGraphDataset(test_data)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=pyg_collate, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=pyg_collate, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=pyg_collate, num_workers=0)

            t0 = time.time()
            result = train_and_evaluate(
                model, train_loader, val_loader, test_loader, device,
                use_edge_attr=use_ea, num_epochs=args.epochs, lr=cfg['lr'],
                desc=f"{model_name}-{seed}")
            elapsed = time.time() - t0

            seed_results.append({
                'seed': seed,
                'test_auc': float(result['test_auc']),
                'test_ap': float(result['test_ap']),
                'val_auc': float(result['val_auc']),
                'time': elapsed,
            })
            print(f"    Test AUC: {result['test_auc']:.4f}, AP: {result['test_ap']:.4f} ({elapsed:.0f}s)")

            del model, train_ds, val_ds, test_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Statistics
        test_aucs = [r['test_auc'] for r in seed_results]
        model_mean = float(np.mean(test_aucs))
        model_std = float(np.std(test_aucs, ddof=1))

        output = {
            'seeds': args.seeds,
            'per_seed': seed_results,
            f'{model_key}_mean': model_mean,
            f'{model_key}_std': round(model_std, 6),
        }

        # Welch's t-test comparison with proposed model
        if proposed_aucs is not None:
            t_stat, p_val = welch_ttest(test_aucs, proposed_aucs)
            bonferroni_n = 4  # 4 GNN baselines
            bonferroni_threshold = 0.05 / bonferroni_n

            output['comparison'] = {
                'proposed_mean': proposed_mean,
                'proposed_std': proposed_std,
                f'{model_key}_mean': model_mean,
                f'{model_key}_std': round(model_std, 6),
                'test': 'Welch two-sample t-test (unequal variances)',
                't_statistic': round(t_stat, 4),
                'p_value': round(p_val, 4),
                'bonferroni_threshold': bonferroni_threshold,
                'bonferroni_n_comparisons': bonferroni_n,
                'significant_p005_uncorrected': bool(p_val < 0.05),
                'significant_bonferroni': bool(p_val < bonferroni_threshold),
                'note': f'Sample std (ddof=1) used. '
                        f'{"Significant" if p_val < bonferroni_threshold else "Not significant"}'
                        f' after Bonferroni correction for {bonferroni_n} GNN baseline comparisons.',
            }
            print(f"\n  {model_name} vs Proposed (Welch's t-test):")
            print(f"    {model_name}: {model_mean:.4f} +/- {model_std:.4f}")
            print(f"    Proposed:  {proposed_mean:.4f} +/- {proposed_std:.4f}")
            print(f"    t={t_stat:.4f}, p={p_val:.4f}")
            print(f"    Bonferroni (alpha={bonferroni_threshold}): "
                  f"{'Significant' if p_val < bonferroni_threshold else 'Not significant'}")

        # Save
        out_path = os.path.join(BASE_DIR, 'results', f'{model_key}_multi_seed.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
