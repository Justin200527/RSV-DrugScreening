# -*- coding: utf-8 -*-
"""GNN Baseline Comparisons"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import json
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem
import time
import gc
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaffold_utils import scaffold_split
from model import FocalLoss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
SEED = 42

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'is_aromatic': [False, True],
}


def one_hot(val, allowed):
    vec = [0] * (len(allowed) + 1)
    if val in allowed:
        vec[allowed.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_to_feature_vector(atom):
    features = []
    features.extend(one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    features.append(atom.GetTotalNumHs())
    features.append(atom.GetNumRadicalElectrons())
    return features


ATOM_FEAT_DIM = (118 + 1) + (6 + 1) + (5 + 1) + (5 + 1) + (2 + 1) + 1 + 1  # 143


def bond_to_feature_vector(bond):
    bt = bond.GetBondType()
    features = [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]
    return features


BOND_FEAT_DIM = 6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MolGraphDataset(Dataset):
    def __init__(self, data_list, use_edge_attr=False):
        self.data_list = data_list
        self.use_edge_attr = use_edge_attr

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        try:
            smi = d.get('smiles', '')
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            label = float(d['y'])

            # Node features
            x = torch.tensor([atom_to_feature_vector(a) for a in mol.GetAtoms()],
                             dtype=torch.float)

            # Edge index + edge features
            edge_list, edge_feat_list = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bf = bond_to_feature_vector(bond)
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_feat_list.append(bf)
                edge_feat_list.append(bf)

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_feat_list, dtype=torch.float)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=torch.tensor([label], dtype=torch.float))
            return data
        except Exception:
            return None


def pyg_collate(data_list):
    data_list = [d for d in data_list if d is not None]
    if not data_list:
        return None
    return Batch.from_data_list(data_list)


class GCNModel(nn.Module):
    def __init__(self, in_dim=ATOM_FEAT_DIM, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


class GATModel(nn.Module):
    def __init__(self, in_dim=ATOM_FEAT_DIM, hidden_dim=128, num_layers=4, heads=4, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(GATConv(in_ch, hidden_dim // heads, heads=heads,
                                       dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


class AttentiveFPModel(nn.Module):
    def __init__(self, in_channels=ATOM_FEAT_DIM, hidden_channels=128,
                 edge_dim=BOND_FEAT_DIM, num_layers=3, num_timesteps=2, dropout=0.2):
        super().__init__()
        self.afp = AttentiveFP(
            in_channels=in_channels, hidden_channels=hidden_channels,
            out_channels=hidden_channels, edge_dim=edge_dim,
            num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.afp(x, edge_index, edge_attr, batch)
        return self.classifier(h).squeeze(-1)


class MPNNModel(nn.Module):
    def __init__(self, in_dim=ATOM_FEAT_DIM, hidden_dim=128, edge_dim=BOND_FEAT_DIM,
                 num_layers=4, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim))
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.relu(self.lin_in(x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index, edge_attr)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)


def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       use_edge_attr=False, num_epochs=30, lr=1e-3, patience=8, desc=""):
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_auc = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                ea = batch.edge_attr if use_edge_attr else None
                logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
                if torch.isnan(logits).any():
                    continue
                loss = criterion(logits, batch.y.squeeze())
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except Exception:
                continue
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                try:
                    batch = batch.to(device)
                    ea = batch.edge_attr if use_edge_attr else None
                    logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
                    preds = torch.sigmoid(logits).cpu().numpy()
                    labs = batch.y.squeeze().cpu().numpy()
                    valid = ~(np.isnan(preds) | np.isinf(preds))
                    if valid.any():
                        val_preds.extend(preds[valid])
                        val_labels.extend(labs[valid])
                except Exception:
                    continue

        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0
        if (epoch + 1) % 5 == 0:
            print(f"  {desc} Ep {epoch+1} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {desc} Early stop at epoch {epoch+1}")
                break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            try:
                batch = batch.to(device)
                ea = batch.edge_attr if use_edge_attr else None
                logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
                preds = torch.sigmoid(logits).cpu().numpy()
                labs = batch.y.squeeze().cpu().numpy()
                valid = ~(np.isnan(preds) | np.isinf(preds))
                if valid.any():
                    test_preds.extend(preds[valid])
                    test_labels.extend(labs[valid])
            except Exception:
                continue

    test_auc = roc_auc_score(test_labels, test_preds) if len(set(test_labels)) > 1 else 0
    test_ap = average_precision_score(test_labels, test_preds) if len(set(test_labels)) > 1 else 0

    return {'val_auc': best_val_auc, 'test_auc': test_auc, 'test_ap': test_ap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gcn', 'gat', 'attentivefp', 'mpnn'],
                        choices=['gcn', 'gat', 'attentivefp', 'mpnn'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("GNN Baseline Comparisons")
    print(f"Device: {device}, Models: {args.models}")

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

    # Scaffold-based split (no data leakage)
    train_data, val_data, test_data = scaffold_split(data_list, seed=SEED)

    model_configs = {
        'gcn': {
            'name': 'GCN', 'use_edge_attr': False,
            'factory': lambda: GCNModel(in_dim=ATOM_FEAT_DIM, hidden_dim=128, num_layers=4),
            'lr': 1e-3,
        },
        'gat': {
            'name': 'GAT', 'use_edge_attr': False,
            'factory': lambda: GATModel(in_dim=ATOM_FEAT_DIM, hidden_dim=128, num_layers=4, heads=4),
            'lr': 5e-4,
        },
        'attentivefp': {
            'name': 'AttentiveFP', 'use_edge_attr': True,
            'factory': lambda: AttentiveFPModel(
                in_channels=ATOM_FEAT_DIM, hidden_channels=128,
                edge_dim=BOND_FEAT_DIM, num_layers=3, num_timesteps=2),
            'lr': 5e-4,
        },
        'mpnn': {
            'name': 'MPNN', 'use_edge_attr': True,
            'factory': lambda: MPNNModel(
                in_dim=ATOM_FEAT_DIM, hidden_dim=128, edge_dim=BOND_FEAT_DIM, num_layers=4),
            'lr': 5e-4,
        },
    }

    results = {}
    for model_key in args.models:
        cfg = model_configs[model_key]
        print(f"\nTraining {cfg['name']}")

        set_seed(SEED)
        model = cfg['factory']().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        use_ea = cfg['use_edge_attr']
        train_ds = MolGraphDataset(train_data, use_edge_attr=use_ea)
        val_ds = MolGraphDataset(val_data, use_edge_attr=use_ea)
        test_ds = MolGraphDataset(test_data, use_edge_attr=use_ea)

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
            desc=cfg['name'])
        elapsed = time.time() - t0

        results[cfg['name']] = {
            'val_auc': float(result['val_auc']),
            'test_auc': float(result['test_auc']),
            'test_ap': float(result['test_ap']),
            'n_params': n_params,
            'train_time_seconds': elapsed,
        }
        print(f"  {cfg['name']}: Val AUC={result['val_auc']:.4f}, "
              f"Test AUC={result['test_auc']:.4f}, AP={result['test_ap']:.4f}")

        # Save model
        torch.save(model.state_dict(),
                    os.path.join(BASE_DIR, 'models', f'gnn_{model_key}.pt'))

        del model, train_ds, val_ds, test_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save incrementally
        results_path = os.path.join(BASE_DIR, 'results', 'gnn_baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\nGNN Baseline Summary")
    print(f"{'Model':<15s} {'Params':>10s} {'Val AUC':>10s} {'Test AUC':>10s} {'Test AP':>10s}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<15s} {r['n_params']:>10,} {r['val_auc']:>10.4f} "
              f"{r['test_auc']:>10.4f} {r['test_ap']:>10.4f}")

    print(f"\nResults saved to {os.path.join(BASE_DIR, 'results', 'gnn_baseline_results.json')}")


if __name__ == '__main__':
    main()
