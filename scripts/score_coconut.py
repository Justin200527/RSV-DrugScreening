# -*- coding: utf-8 -*-
"""COCONUT Virtual Screening — Score and rank natural products for RSV inhibition."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os, json, argparse, pickle, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import process_molecule, compute_active_statistics, compute_feature_weights
from model import RSVDrugScreeningModel
from utils import MoleculeDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
COCONUT_PATH = os.path.join(BASE_DIR, 'data', 'coconut_all_features.pkl')


def collate_fn(batch):
    z_list, pos_list, pharm_list, size_list = [], [], [], []
    autocorr_list, shape3d_list, batch_idx = [], [], []
    atom_feat_list, bond_edge_index_list, bond_edge_attr_list = [], [], []
    atom_offset = 0
    has_graph_feats = 'atom_feat' in batch[0] and batch[0]['atom_feat'] is not None
    for i, d in enumerate(batch):
        z = torch.tensor(d['z'], dtype=torch.long) if not isinstance(d['z'], torch.Tensor) else d['z']
        pos = torch.tensor(d['pos'], dtype=torch.float32) if not isinstance(d['pos'], torch.Tensor) else d['pos']
        n_atoms = len(z)
        z_list.append(z); pos_list.append(pos)
        pharm_list.append(torch.tensor(d['pharm'], dtype=torch.float32) if not isinstance(d['pharm'], torch.Tensor) else d['pharm'])
        size_list.append(torch.tensor(d['size'], dtype=torch.float32) if not isinstance(d['size'], torch.Tensor) else d['size'])
        autocorr_list.append(torch.tensor(d['autocorr'], dtype=torch.float32) if not isinstance(d['autocorr'], torch.Tensor) else d['autocorr'])
        shape3d_list.append(torch.tensor(d['shape3d'], dtype=torch.float32) if not isinstance(d['shape3d'], torch.Tensor) else d['shape3d'])
        batch_idx.extend([i] * n_atoms)
        if has_graph_feats:
            af = d['atom_feat']
            if not isinstance(af, torch.Tensor):
                af = torch.tensor(af, dtype=torch.float32)
            atom_feat_list.append(af)
            bei = d.get('bond_edge_index')
            bea = d.get('bond_edge_attr')
            if bei is not None:
                if not isinstance(bei, torch.Tensor):
                    bei = torch.tensor(bei, dtype=torch.long)
                if bei.size(1) > 0:
                    bond_edge_index_list.append(bei + atom_offset)
                if not isinstance(bea, torch.Tensor):
                    bea = torch.tensor(bea, dtype=torch.float32)
                bond_edge_attr_list.append(bea)
        atom_offset += n_atoms
    result = {
        'z': torch.cat(z_list), 'pos': torch.cat(pos_list),
        'batch': torch.tensor(batch_idx, dtype=torch.long),
        'pharm': torch.stack(pharm_list), 'size': torch.stack(size_list),
        'autocorr': torch.stack(autocorr_list), 'shape3d': torch.stack(shape3d_list),
    }
    if has_graph_feats:
        result['atom_feat'] = torch.cat(atom_feat_list)
        if bond_edge_index_list:
            result['bond_edge_index'] = torch.cat(bond_edge_index_list, dim=1)
            result['bond_edge_attr'] = torch.cat(bond_edge_attr_list, dim=0)
        else:
            result['bond_edge_index'] = torch.zeros((2, 0), dtype=torch.long)
            result['bond_edge_attr'] = torch.zeros((0, 6), dtype=torch.float32)
    return result


def main():
    parser = argparse.ArgumentParser(description='Screen COCONUT database for RSV inhibitor candidates')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--coconut-path', default=COCONUT_PATH,
                        help='Path to COCONUT features pkl (default: auto-detected)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--disabled-modules', nargs='*', default=None,
                        help='Modules to disable (for ablation models like ablation_wo_size)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    # Active stats
    import pandas as pd
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'rsv_training_inhibition_22p3.csv'))
    active_means, active_stds = compute_active_statistics(
        train_df[train_df['label'] == 1]['SMILES'].tolist())
    feature_weights = compute_feature_weights()

    # Auto-detect disabled modules from filename
    disabled = args.disabled_modules
    if disabled is None:
        fname = os.path.basename(args.model)
        if 'wo_graph' in fname: disabled = ['graph']
        elif 'wo_dist' in fname: disabled = ['dist']
        elif 'wo_pharm' in fname: disabled = ['pharm']
        elif 'wo_size' in fname: disabled = ['size']
        elif 'wo_autocorr' in fname: disabled = ['autocorr']
        elif 'wo_shape3d' in fname: disabled = ['shape3d']

    # Load model
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        config = ckpt.get('config', {})
        state_dict = ckpt['model_state_dict']
        active_means = ckpt.get('active_means', active_means)
        active_stds = ckpt.get('active_stds', active_stds)
        feature_weights = ckpt.get('feature_weights', feature_weights)
    else:
        config = {}
        state_dict = ckpt
    model = RSVDrugScreeningModel(
        active_means=active_means, active_stds=active_stds,
        feature_weights=feature_weights,
        disabled_modules=disabled,
        size_features=config.get('size_features', 3),
        large_mol_adapt=False,
        atom_feat_dim=config.get('atom_feat_dim', 153),
        bond_feat_dim=config.get('bond_feat_dim', 6),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    if disabled:
        print(f"Disabled modules: {disabled}")

    # Load COCONUT
    coconut_path = args.coconut_path
    if not os.path.exists(coconut_path):
        print(f"\nERROR: COCONUT feature file not found: {coconut_path}")
        print("The COCONUT features (~10 GB) are not included in the repository.")
        print("Use --coconut-path to specify the path to your local COCONUT features.")
        print("See https://coconut.naturalproducts.net/ for the source database.")
        sys.exit(1)
    print(f"\nLoading COCONUT features from {coconut_path}...")
    t0 = time.time()
    with open(coconut_path, 'rb') as f:
        coconut_data = pickle.load(f)
    valid = [d for d in coconut_data
             if d.get('z') and d.get('pos') and d.get('pharm')
             and d.get('autocorr') and d.get('shape3d') and d.get('size')]
    # Trim size: 5d -> 3d
    for d in valid:
        if hasattr(d.get('size', None), '__len__') and len(d['size']) == 5:
            d['size'] = d['size'][2:]
    print(f"  {len(valid)} valid molecules ({time.time()-t0:.1f}s)")

    # Score all
    loader = DataLoader(MoleculeDataset(valid), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    scores = []
    with torch.no_grad():
        for bd in tqdm(loader, desc="Scoring"):
            af = bd.get('atom_feat')
            bei = bd.get('bond_edge_index')
            bea = bd.get('bond_edge_attr')
            if af is not None: af = af.to(device)
            if bei is not None: bei = bei.to(device)
            if bea is not None: bea = bea.to(device)
            logits = model(bd['z'].to(device), bd['pos'].to(device),
                           bd['batch'].to(device), bd['pharm'].to(device),
                           bd['size'].to(device), bd['autocorr'].to(device),
                           bd['shape3d'].to(device), af, bei, bea)
            scores.extend(torch.sigmoid(logits).cpu().numpy())
    scores = np.array(scores)

    # Handle NaN
    nan_count = int(np.isnan(scores).sum())
    if nan_count:
        print(f"  WARNING: {nan_count} NaN scores (set to 0)")
        scores = np.nan_to_num(scores, nan=0.0)

    print(f"\nCOCONUT Screening Complete: {len(scores)} compounds scored")
    print(f"Score distribution: mean={np.mean(scores):.4f}, median={np.median(scores):.4f}")
    print(f"Top 0.1% threshold: {np.percentile(scores, 99.9):.4f}")
    print(f"Top 1% threshold: {np.percentile(scores, 99):.4f}")

    result = {
        'model': os.path.basename(args.model),
        'total': len(scores),
        'nan_count': nan_count,
        'score_mean': float(np.mean(scores)),
        'score_median': float(np.median(scores)),
        'top_0_1pct_threshold': float(np.percentile(scores, 99.9)),
        'top_1pct_threshold': float(np.percentile(scores, 99)),
    }
    out = os.path.join(BASE_DIR, 'results', 'coconut_screening.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")

    # Save full scores for downstream analysis
    scores_path = os.path.join(BASE_DIR, 'results', 'coconut_scores.npy')
    np.save(scores_path, scores)
    print(f"Full scores saved: {scores_path}")


if __name__ == '__main__':
    main()
