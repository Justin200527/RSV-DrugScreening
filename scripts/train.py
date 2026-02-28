# -*- coding: utf-8 -*-
"""RSV Drug Screening — Training Script"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import warnings
import time
import pickle
warnings.filterwarnings('ignore')

# Import shared modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (
    process_molecule, process_molecule_multiconf,
    extract_autocorr_features, compute_active_statistics, compute_feature_weights,
    KEY_2D_FEATURES
)
from model import RSVDrugScreeningModel, FocalLoss
from utils import MoleculeDataset, collate_fn, trim_size_features
from scaffold_utils import scaffold_split

# ---
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--np-conformers', type=int, default=10)
parser.add_argument('--active-conformers', type=int, default=3)
args = parser.parse_args()

SEED = args.seed
NP_CONFORMERS = args.np_conformers
ACTIVE_CONFORMERS = args.active_conformers

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("RSV Drug Screening — Training")
print(f"Seed: {SEED}, NP conformers: {NP_CONFORMERS}, Active conformers: {ACTIVE_CONFORMERS}")
print(f"Device: {device}")

# --- Data preparation + feature extraction (with caching)

cache_path = os.path.join(BASE_DIR, 'data', 'train_augmented_cache.pkl')
cache_version = 1
_cache_loaded = False

if os.path.exists(cache_path):
    print(f"\nLoading cache from {cache_path}...")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    if cache.get('version', 1) >= cache_version:
        data_list = cache['data_list']
        supp_active_count = cache.get('supp_active_count', 0)
        supp_inactive_count = cache.get('supp_inactive_count', 0)
        print(f"  Loaded {len(data_list)} samples from cache (v{cache.get('version',1)})")
        _cache_loaded = True
        del cache
    else:
        print(f"  Old cache (v{cache.get('version',1)} < v{cache_version}), regenerating...")
        del cache

if not _cache_loaded:
    print("\nData preparation (no cache found)...")

    data_path = os.path.join(BASE_DIR, 'data', 'rsv_training_inhibition_22p3.csv')
    train_df = pd.read_csv(data_path)
    print(f"Original data: {len(train_df)} samples")

    supplement_path = os.path.join(BASE_DIR, 'data', 'chembl_rsv_supplement.csv')
    supp_df = pd.read_csv(supplement_path)
    print(f"Supplement data: {len(supp_df)} samples")

    # Deduplicate
    original_smiles_set = set()
    for smi in train_df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            original_smiles_set.add(Chem.MolToSmiles(mol))

    supp_canonical = []
    for smi in supp_df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        supp_canonical.append(Chem.MolToSmiles(mol) if mol else None)
    supp_df['canonical'] = supp_canonical
    overlap = supp_df['canonical'].isin(original_smiles_set)
    supp_df = supp_df[~overlap].reset_index(drop=True)
    supp_actives = supp_df[supp_df['label'] == 1]
    supp_inactives = supp_df[supp_df['label'] == 0]
    literature_smiles = set(supp_df[supp_df['source'] == 'Literature']['SMILES'])

    # MW-stratified hard negative sampling
    train_df['MW'] = train_df['SMILES'].apply(
        lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else 0)
    orig_pos = train_df[train_df['label'] == 1].copy()
    orig_neg = train_df[train_df['label'] == 0].copy()

    MW_BINS = [0, 200, 300, 400, 500, 600, 800, 1000, float('inf')]
    NEG_RATIO = 3
    HARD_NEG_FRACTION = 0.5
    orig_pos['mw_bin'] = pd.cut(orig_pos['MW'], bins=MW_BINS, right=False)
    orig_neg['mw_bin'] = pd.cut(orig_neg['MW'], bins=MW_BINS, right=False)

    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    print("Computing active fingerprints for hard negative mining...")
    active_fps = []
    for smi in tqdm(orig_pos['SMILES'], desc="Active FPs"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            active_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    print("Computing negative-to-active similarity...")
    neg_sims = []
    for smi in tqdm(orig_neg['SMILES'], desc="Neg similarity"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, active_fps))
            neg_sims.append(max_sim)
        else:
            neg_sims.append(0.0)
    orig_neg['active_sim'] = neg_sims

    sampled_neg_dfs = []
    for mw_bin, pos_count in orig_pos['mw_bin'].value_counts().items():
        if pos_count == 0:
            continue
        target_neg = pos_count * NEG_RATIO
        neg_in_bin = orig_neg[orig_neg['mw_bin'] == mw_bin]
        if len(neg_in_bin) == 0:
            continue
        n_hard = min(int(target_neg * HARD_NEG_FRACTION), len(neg_in_bin))
        n_random = min(target_neg - n_hard, len(neg_in_bin) - n_hard)
        sorted_neg = neg_in_bin.sort_values('active_sim', ascending=False)
        hard_neg = sorted_neg.head(n_hard)
        remaining = sorted_neg.iloc[n_hard:]
        random_neg = remaining.sample(n=min(n_random, len(remaining)), random_state=SEED)
        sampled_neg_dfs.append(pd.concat([hard_neg, random_neg]))

    orig_neg = pd.concat(sampled_neg_dfs).reset_index(drop=True) if sampled_neg_dfs else orig_neg.head(0)
    train_df = pd.concat([orig_pos, orig_neg]).reset_index(drop=True)
    print(f"After sampling: {len(train_df)} (pos={len(orig_pos)}, neg={len(orig_neg)})")

    # Feature extraction
    print("Feature extraction...")
    data_list = []
    failed = 0
    t0 = time.time()
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Original data"):
        result = process_molecule(row['SMILES'], row['label'], seed=42)
        if result is not None:
            data_list.append(result)
        else:
            failed += 1
    orig_count = len(data_list)

    supp_active_count = 0
    NP_CONFORMERS_local = NP_CONFORMERS
    ACTIVE_CONFORMERS_local = ACTIVE_CONFORMERS
    for idx, row in tqdm(supp_actives.iterrows(), total=len(supp_actives), desc="Supp actives"):
        smi = row['SMILES']
        n_conf = NP_CONFORMERS_local if smi in literature_smiles else ACTIVE_CONFORMERS_local
        results = process_molecule_multiconf(smi, label=1, n_conformers=n_conf, base_seed=42)
        if results:
            data_list.extend(results)
            supp_active_count += len(results)
        else:
            failed += 1

    supp_inactive_count = 0
    for idx, row in tqdm(supp_inactives.iterrows(), total=len(supp_inactives), desc="Supp inactives"):
        result = process_molecule(row['SMILES'], row['label'], seed=42)
        if result is not None:
            data_list.append(result)
            supp_inactive_count += 1
        else:
            failed += 1
    print(f"  Total: {len(data_list)} samples, {failed} failed, {time.time()-t0:.1f}s")

    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'data_list': data_list,
            'supp_active_count': supp_active_count,
            'supp_inactive_count': supp_inactive_count,
            'orig_count': orig_count,
            'version': cache_version,
        }, f)
    print(f"  Cache saved")

# Active compound statistics (fast, always recompute)
print("Computing active compound feature statistics...")
data_path = os.path.join(BASE_DIR, 'data', 'rsv_training_inhibition_22p3.csv')
_df = pd.read_csv(data_path)
_active_smiles = _df[_df['label'] == 1]['SMILES'].tolist()
active_means, active_stds = compute_active_statistics(_active_smiles)
feature_weights = compute_feature_weights()
del _df, _active_smiles
print(f"  Active means (first 3): {active_means[:3]}")

trim_size_features(data_list)
print(f"  Size features -> 3d (shape: {data_list[0]['size'].shape})")

# Label statistics
labels = [d['y'] for d in data_list]
n_pos = sum(1 for l in labels if l == 1)
n_neg = sum(1 for l in labels if l == 0)
print(f"  Label distribution: pos={n_pos}, neg={n_neg}, ratio=1:{n_neg/n_pos:.1f}")

# Scaffold split (no data leakage)
print("\nScaffold-based split...")
train_data, val_data, test_data = scaffold_split(data_list, seed=SEED)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
n_pos_train = sum(1 for d in train_data if d['y'] == 1)
n_pos_val = sum(1 for d in val_data if d['y'] == 1)
n_pos_test = sum(1 for d in test_data if d['y'] == 1)
print(f"  Train pos: {n_pos_train}, Val pos: {n_pos_val}, Test pos: {n_pos_test}")

BATCH_SIZE = 48
train_loader = DataLoader(MoleculeDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(MoleculeDataset(val_data), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(MoleculeDataset(test_data), batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)

# --- Training

model = RSVDrugScreeningModel(
    active_means=active_means,
    active_stds=active_stds,
    feature_weights=feature_weights,
    size_features=3,
    large_mol_adapt=False,
).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = FocalLoss(alpha=0.5, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

model_save_path = os.path.join(output_dir, 'rsv_best.pt')
best_auc = 0
best_epoch = 0
patience = 30
no_improve = 0

print(f"Model selection: best Val AUC, patience={patience}")
print("Training...\n")
train_start = time.time()
training_history = []

for epoch in range(100):
    model.train()
    total_loss = 0
    train_preds, train_labels = [], []
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
        # Collect training predictions for train AUC
        with torch.no_grad():
            train_preds.extend(torch.sigmoid(logits).cpu().numpy())
            train_labels.extend(batch_data['y'].numpy())

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_auc = roc_auc_score(train_labels, train_preds)

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    val_loss_total = 0
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
            val_loss_total += criterion(logits, batch_data['y'].to(device)).item()
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(batch_data['y'].numpy())

    val_auc = roc_auc_score(val_labels, val_preds)
    val_ap = average_precision_score(val_labels, val_preds)
    avg_val_loss = val_loss_total / len(val_loader)

    marker = ""
    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1
        no_improve = 0
        marker = " *** BEST"
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_auc': val_auc, 'val_ap': val_ap,
            'epoch': epoch + 1,
            'active_means': active_means, 'active_stds': active_stds,
            'feature_weights': feature_weights,
            'config': {
                'hidden_dim': 128, 'num_layers': 4, 'num_rbf': 64,
                'cutoffs': [5.0, 10.0, 15.0, 20.0, 25.0], 'dropout': 0.15,
                'pharm_features': 10, 'size_features': 3,
                'autocorr_features': 19, 'shape3d_features': 5,
                'module_weights': model.module_weights,
                'atom_feat_dim': 153, 'bond_feat_dim': 6,
            },
            'seed': SEED,
            'num_train': len(train_data), 'num_val': len(val_data), 'num_test': len(test_data),
            'augmentation': {
                'np_conformers': NP_CONFORMERS, 'active_conformers': ACTIVE_CONFORMERS,
                'supplement_actives': supp_active_count, 'supplement_inactives': supp_inactive_count,
            },
        }, model_save_path)
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | ValLoss: {avg_val_loss:.4f} | TrainAUC: {train_auc:.4f} | ValAUC: {val_auc:.4f} | AP: {val_ap:.4f}{marker}")
    training_history.append({
        'epoch': epoch + 1,
        'loss': round(avg_loss, 4), 'val_loss': round(avg_val_loss, 4),
        'train_auc': round(train_auc, 4), 'val_auc': round(val_auc, 4),
        'val_ap': round(val_ap, 4),
    })

train_time = time.time() - train_start
print(f"\nTraining complete in {train_time:.1f}s")
print(f"Best val AUC: {best_auc:.4f} at epoch {best_epoch}")

# --- Test set evaluation
print("\nEvaluating on test set...")

ckpt = torch.load(model_save_path, map_location=device, weights_only=False)
model_eval = RSVDrugScreeningModel(
    active_means=ckpt['active_means'],
    active_stds=ckpt['active_stds'],
    feature_weights=ckpt['feature_weights'],
    size_features=3,
    large_mol_adapt=False,
).to(device)
model_eval.load_state_dict(ckpt['model_state_dict'])
model_eval.eval()

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
        logits = model_eval(z, pos, batch_idx, pharm, size, autocorr, shape3d,
                            atom_feat, bond_ei, bond_ea)
        test_preds.extend(torch.sigmoid(logits).cpu().numpy())
        test_labels.extend(batch_data['y'].numpy())

test_auc = roc_auc_score(test_labels, test_preds)
test_ap = average_precision_score(test_labels, test_preds)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test AP:  {test_ap:.4f}")

test_binary = [1 if p > 0.5 else 0 for p in test_preds]
cm = confusion_matrix(test_labels, test_binary)
print(f"Confusion matrix:\n{cm}")

# Save training history
history_path = os.path.join(BASE_DIR, 'results', 'training_history.json')
with open(history_path, 'w') as f:
    json.dump({'epochs': training_history, 'best_epoch': best_epoch,
               'total_epochs': len(training_history)}, f, indent=2)
print(f"Training history saved: {history_path}")

# Save ROC curve data
fpr, tpr, _ = roc_curve(test_labels, test_preds)
precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_preds)
roc_path = os.path.join(BASE_DIR, 'results', 'roc_data.json')
with open(roc_path, 'w') as f:
    json.dump({
        'test_auc': float(test_auc), 'test_ap': float(test_ap),
        'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        'precision': precision_vals.tolist(), 'recall': recall_vals.tolist(),
    }, f)
print(f"ROC data saved: {roc_path}")

metrics = {
    'seed': SEED,
    'best_epoch': best_epoch,
    'val_auc': float(best_auc),
    'test_auc': float(test_auc),
    'test_ap': float(test_ap),
    'confusion_matrix': cm.tolist(),
    'num_train': len(train_data),
    'num_val': len(val_data),
    'num_test': len(test_data),
    'num_params': sum(p.numel() for p in model.parameters()),
    'train_time_seconds': train_time,
}

metrics_path = os.path.join(BASE_DIR, 'results', f'train_metrics_seed{SEED}.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved: {metrics_path}")
print(f"\nDone. Model: {model_save_path}")
print(f"Best Val AUC: {best_auc:.4f} | Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")
