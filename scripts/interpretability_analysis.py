# -*- coding: utf-8 -*-
"""Interpretability Analysis"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

from model import RSVDrugScreeningModel
from scaffold_utils import scaffold_split
from features import process_molecule
from utils import MoleculeDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CASE_STUDY_SMILES = 'O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'  # quercetin


def collate_fn(batch):
    z_list, pos_list, batch_idx = [], [], []
    pharm_list, size_list, autocorr_list, shape3d_list, y_list = [], [], [], [], []
    atom_feat_list, bond_ei_list, bond_ea_list = [], [], []
    offset = 0
    for i, d in enumerate(batch):
        n = d['z'].shape[0] if isinstance(d['z'], torch.Tensor) else len(d['z'])
        z_list.append(d['z'] if isinstance(d['z'], torch.Tensor) else torch.tensor(d['z'], dtype=torch.long))
        pos_list.append(d['pos'] if isinstance(d['pos'], torch.Tensor) else torch.tensor(d['pos'], dtype=torch.float32))
        batch_idx.extend([i] * n)
        pharm_list.append(d['pharm'] if isinstance(d['pharm'], torch.Tensor) else torch.tensor(d['pharm'], dtype=torch.float32))
        s = d['size']
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        if s.shape[0] != 3: s = s[-3:]  # keep last 3: rot, rings, aromatic
        size_list.append(s)
        autocorr_list.append(d['autocorr'] if isinstance(d['autocorr'], torch.Tensor) else torch.tensor(d['autocorr'], dtype=torch.float32))
        shape3d_list.append(d['shape3d'] if isinstance(d['shape3d'], torch.Tensor) else torch.tensor(d['shape3d'], dtype=torch.float32))
        y_list.append(d.get('y', 0))
        if 'atom_feat' in d and d['atom_feat'] is not None:
            atom_feat_list.append(d['atom_feat'] if isinstance(d['atom_feat'], torch.Tensor) else torch.tensor(d['atom_feat'], dtype=torch.float32))
            bei = d['bond_edge_index'] if isinstance(d['bond_edge_index'], torch.Tensor) else torch.tensor(d['bond_edge_index'], dtype=torch.long)
            bea = d['bond_edge_attr'] if isinstance(d['bond_edge_attr'], torch.Tensor) else torch.tensor(d['bond_edge_attr'], dtype=torch.float32)
            if bei.size(1) > 0:
                bond_ei_list.append(bei + offset)
            bond_ea_list.append(bea)
        offset += n
    result = {
        'z': torch.cat(z_list), 'pos': torch.cat(pos_list),
        'batch': torch.tensor(batch_idx, dtype=torch.long),
        'pharm': torch.stack(pharm_list), 'size': torch.stack(size_list),
        'autocorr': torch.stack(autocorr_list), 'shape3d': torch.stack(shape3d_list),
        'y': torch.tensor(y_list, dtype=torch.float32),
    }
    if atom_feat_list:
        result['atom_feat'] = torch.cat(atom_feat_list)
        if bond_ei_list:
            result['bond_edge_index'] = torch.cat(bond_ei_list, dim=1)
            result['bond_edge_attr'] = torch.cat(bond_ea_list)
        else:
            result['bond_edge_index'] = torch.zeros((2, 0), dtype=torch.long)
            result['bond_edge_attr'] = torch.zeros((0, 6), dtype=torch.float32)
    return result


def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'rsv_best.pt')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        print("Run train.py first.")
        sys.exit(1)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = RSVDrugScreeningModel(
        active_means=ckpt['active_means'],
        active_stds=ckpt['active_stds'],
        feature_weights=ckpt['feature_weights'],
        size_features=3, large_mol_adapt=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def analyze_module_contributions(model, test_data, n_samples=2000):
    """Capture each module's output magnitude using forward hooks.
    This analysis is UNIQUE to the multi-modal architecture — GCN cannot provide it."""

    print("\nModule Contribution Analysis")
    print(f"  Analyzing {min(n_samples, len(test_data))} test samples...")

    # The model's forward builds: combined = cat([h_size*w, h_graph*w, h_dist*w, h_pharm*w, h_autocorr*w, h_shape3d*w])
    # Each is 128d, total 768d. We hook readout's input to capture the fused vector.
    module_names_order = ['size', 'graph', 'dist', 'pharm', 'autocorr', 'shape3d']
    module_weights = {
        'graph': 0.5, 'dist': 0.5, 'pharm': 1.0,
        'size': 1.0, 'autocorr': 1.5, 'shape3d': 1.5,
    }
    hidden_dim = 128
    fusion_inputs = []

    def readout_hook(module, input, output):
        # input[0] is the 768d combined vector (already weight-scaled)
        fusion_inputs.append(input[0].detach().cpu())

    hook = model.readout.register_forward_hook(readout_hook)

    # Run inference
    subset = test_data[:n_samples]
    loader = DataLoader(MoleculeDataset(subset), batch_size=48, shuffle=False, collate_fn=collate_fn)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data in loader:
            z = batch_data['z'].to(device)
            pos = batch_data['pos'].to(device)
            bi = batch_data['batch'].to(device)
            pharm = batch_data['pharm'].to(device)
            size = batch_data['size'].to(device)
            autocorr = batch_data['autocorr'].to(device)
            shape3d = batch_data['shape3d'].to(device)
            af = batch_data['atom_feat'].to(device) if 'atom_feat' in batch_data else None
            bei = batch_data['bond_edge_index'].to(device) if 'bond_edge_index' in batch_data else None
            bea = batch_data['bond_edge_attr'].to(device) if 'bond_edge_attr' in batch_data else None
            logits = model(z, pos, bi, pharm, size, autocorr, shape3d, af, bei, bea)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(batch_data['y'].numpy().tolist())

    # Remove hook
    hook.remove()

    # Split fusion vector into 6 modules (each 128d, already weight-scaled)
    all_fusion = torch.cat(fusion_inputs, dim=0).numpy()  # [N, 768]
    labels_arr = np.array(all_labels[:len(all_fusion)])
    active_mask = labels_arr == 1
    inactive_mask = labels_arr == 0

    results = {}
    for i, name in enumerate(module_names_order):
        # Extract this module's 128d weighted output
        start = i * hidden_dim
        end = (i + 1) * hidden_dim
        module_out = all_fusion[:, start:end]  # [N, 128] (already scaled by weight)
        l2_norms = np.linalg.norm(module_out, axis=1)  # [N]
        # Unscale to get raw L2
        raw_l2 = l2_norms / module_weights[name] if module_weights[name] > 0 else l2_norms

        results[name] = {
            'weight': module_weights[name],
            'mean_l2_raw': float(np.mean(raw_l2)),
            'mean_l2_weighted': float(np.mean(l2_norms)),
            'active_mean': float(np.mean(raw_l2[active_mask])) if active_mask.any() else 0,
            'inactive_mean': float(np.mean(raw_l2[inactive_mask])) if inactive_mask.any() else 0,
            'active_inactive_ratio': float(
                np.mean(raw_l2[active_mask]) / np.mean(raw_l2[inactive_mask])
            ) if inactive_mask.any() and active_mask.any() and np.mean(raw_l2[inactive_mask]) > 0 else 0,
        }

    # Relative contribution (weighted L2 as percentage)
    total_weighted = sum(r['mean_l2_weighted'] for r in results.values())
    for name in results:
        results[name]['relative_contribution'] = float(
            results[name]['mean_l2_weighted'] / total_weighted * 100
        ) if total_weighted > 0 else 0

    print(f"\n  {'Module':<15s} {'Weight':>6s} {'Raw L2':>8s} {'Weighted':>8s} {'Contrib%':>8s} {'Act/Inact':>10s}")
    print("  " + "-" * 60)
    for name in ['graph', 'dist', 'pharm', 'autocorr', 'shape3d', 'size']:
        if name in results:
            r = results[name]
            print(f"  {name:<15s} {r['weight']:>6.1f} {r['mean_l2_raw']:>8.3f} "
                  f"{r['mean_l2_weighted']:>8.3f} {r['relative_contribution']:>7.1f}% "
                  f"{r['active_inactive_ratio']:>9.3f}")

    return results


def gradient_atom_attribution(model):
    print("\nGradient-based Atom Attribution (case study)")

    case_data = process_molecule(CASE_STUDY_SMILES, label=1, seed=42)
    if case_data is None:
        print("  Failed to process compound")
        return {}

    # Fix size dimension
    s = case_data['size']
    if isinstance(s, torch.Tensor):
        if s.shape[0] != 3: case_data['size'] = s[-3:]  # keep last 3: rot, rings, aromatic

    batch = collate_fn([case_data])
    z = batch['z'].to(device)
    pos = batch['pos'].to(device).requires_grad_(True)
    bi = batch['batch'].to(device)
    pharm = batch['pharm'].to(device)
    size = batch['size'].to(device)
    autocorr = batch['autocorr'].to(device)
    shape3d = batch['shape3d'].to(device)
    af = batch['atom_feat'].to(device) if 'atom_feat' in batch else None
    bei = batch['bond_edge_index'].to(device) if 'bond_edge_index' in batch else None
    bea = batch['bond_edge_attr'].to(device) if 'bond_edge_attr' in batch else None

    model.zero_grad()
    logit = model(z, pos, bi, pharm, size, autocorr, shape3d, af, bei, bea)
    score = torch.sigmoid(logit)
    score.backward()

    grad = pos.grad.detach().cpu().numpy()
    atom_importance = np.linalg.norm(grad, axis=1)
    if atom_importance.max() > 0:
        atom_importance = atom_importance / atom_importance.max()

    case_score = float(score.item())
    print(f"  Predicted score: {case_score:.6f}")

    # Get atom info from RDKit
    mol = Chem.MolFromSmiles(CASE_STUDY_SMILES)
    atom_info = []
    for i, atom in enumerate(mol.GetAtoms()):
        if i < len(atom_importance):
            atom_info.append({
                'atom_idx': i,
                'element': atom.GetSymbol(),
                'importance': float(atom_importance[i]),
                'is_aromatic': atom.GetIsAromatic(),
                'degree': atom.GetDegree(),
            })

    # Sort by importance
    atom_info.sort(key=lambda x: x['importance'], reverse=True)

    # Summarize by element
    element_imp = {}
    for a in atom_info:
        elem = a['element']
        if elem not in element_imp:
            element_imp[elem] = []
        element_imp[elem].append(a['importance'])
    element_summary = {elem: {'mean': float(np.mean(vals)), 'count': len(vals)}
                       for elem, vals in element_imp.items()}

    # Summarize by functional group
    aromatic_imp = [a['importance'] for a in atom_info if a['is_aromatic']]
    nonaromatic_imp = [a['importance'] for a in atom_info if not a['is_aromatic']]

    print(f"  Total atoms: {len(atom_info)}")
    top5 = [(a['atom_idx'], a['element'], round(a['importance'], 3)) for a in atom_info[:5]]
    print(f"  Top-5 atoms: {top5}")
    elem_strs = [f"{e}: {v['mean']:.3f} (n={v['count']})" for e, v in
                  sorted(element_summary.items(), key=lambda x: x[1]['mean'], reverse=True)]
    print(f"  Element importance: {', '.join(elem_strs)}")
    print(f"  Aromatic atoms mean: {np.mean(aromatic_imp):.3f} vs Non-aromatic: {np.mean(nonaromatic_imp):.3f}")

    return {
        'case_score': case_score,
        'n_atoms': len(atom_info),
        'element_summary': element_summary,
        'aromatic_mean_importance': float(np.mean(aromatic_imp)),
        'nonaromatic_mean_importance': float(np.mean(nonaromatic_imp)),
        'top_10_atoms': atom_info[:10],
        'all_atoms': atom_info,
    }


def active_vs_inactive_response(model, test_data, n_samples=2000):
    print("\nActive vs Inactive Module Response")

    module_names_order = ['size', 'graph', 'dist', 'pharm', 'autocorr', 'shape3d']
    module_weights = {'graph': 0.5, 'dist': 0.5, 'pharm': 1.0, 'size': 1.0, 'autocorr': 1.5, 'shape3d': 1.5}
    hidden_dim = 128
    fusion_inputs = []

    def readout_hook(module, input, output):
        fusion_inputs.append(input[0].detach().cpu())

    hook = model.readout.register_forward_hook(readout_hook)

    subset = test_data[:n_samples]
    loader = DataLoader(MoleculeDataset(subset), batch_size=48, shuffle=False, collate_fn=collate_fn)

    all_labels = []
    with torch.no_grad():
        for batch_data in loader:
            z = batch_data['z'].to(device)
            pos = batch_data['pos'].to(device)
            bi = batch_data['batch'].to(device)
            pharm = batch_data['pharm'].to(device)
            size = batch_data['size'].to(device)
            autocorr = batch_data['autocorr'].to(device)
            shape3d = batch_data['shape3d'].to(device)
            af = batch_data['atom_feat'].to(device) if 'atom_feat' in batch_data else None
            bei = batch_data['bond_edge_index'].to(device) if 'bond_edge_index' in batch_data else None
            bea = batch_data['bond_edge_attr'].to(device) if 'bond_edge_attr' in batch_data else None
            model(z, pos, bi, pharm, size, autocorr, shape3d, af, bei, bea)
            all_labels.extend(batch_data['y'].numpy().tolist())

    hook.remove()

    all_fusion = torch.cat(fusion_inputs, dim=0).numpy()
    labels = np.array(all_labels[:len(all_fusion)])
    active_mask = labels == 1
    inactive_mask = labels == 0

    results = {}
    print(f"\n  {'Module':<15s} {'Active L2':>10s} {'Inactive L2':>12s} {'Ratio':>8s} {'Cohen d':>8s}")
    print("  " + "-" * 55)

    for idx, name in enumerate(module_names_order):
        start = idx * hidden_dim
        end = (idx + 1) * hidden_dim
        module_out = all_fusion[:, start:end]
        # Unscale by weight to get raw output
        norms = np.linalg.norm(module_out, axis=1) / module_weights[name] if module_weights[name] > 0 else np.linalg.norm(module_out, axis=1)

        active_norms = norms[active_mask]
        inactive_norms = norms[inactive_mask]

        if len(active_norms) > 0 and len(inactive_norms) > 0:
            ratio = np.mean(active_norms) / np.mean(inactive_norms)
            pooled_std = np.sqrt((np.var(active_norms) + np.var(inactive_norms)) / 2)
            cohen_d = (np.mean(active_norms) - np.mean(inactive_norms)) / pooled_std if pooled_std > 0 else 0
        else:
            ratio = 0
            cohen_d = 0

        results[name] = {
            'active_mean': float(np.mean(active_norms)) if len(active_norms) > 0 else 0,
            'inactive_mean': float(np.mean(inactive_norms)) if len(inactive_norms) > 0 else 0,
            'ratio': float(ratio),
            'cohen_d': float(cohen_d),
        }

    for name in ['graph', 'dist', 'pharm', 'autocorr', 'shape3d', 'size']:
        if name in results:
            r = results[name]
            print(f"  {name:<15s} {r['active_mean']:>10.3f} "
                  f"{r['inactive_mean']:>12.3f} {r['ratio']:>8.3f} {r['cohen_d']:>+8.3f}")

    return results


def main():
    print("Interpretability Analysis")
    print(f"Device: {device}")

    model = load_model()

    # Load test data
    cache_path = os.path.join(BASE_DIR, 'data', 'train_augmented_cache.pkl')
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}. Run train.py first.")
        sys.exit(1)
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    data_list = cache['data_list']
    _, _, test_data = scaffold_split(data_list, seed=42)
    print(f"Test set: {len(test_data)} samples")

    results = {}

    # 1. Module contributions
    results['module_contributions'] = analyze_module_contributions(model, test_data)

    # 2. Atom attribution (case study)
    results['atom_attribution'] = gradient_atom_attribution(model)

    # 3. Active vs Inactive response
    results['active_vs_inactive'] = active_vs_inactive_response(model, test_data)

    # Save
    out_path = os.path.join(BASE_DIR, 'results', 'interpretability_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
