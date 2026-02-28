# Shared data utilities for training/evaluation scripts
import torch
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    """Collate molecule dicts into batched tensors for the model."""
    z_list, pos_list, pharm_list, size_list = [], [], [], []
    autocorr_list, shape3d_list, y_list = [], [], []
    atom_feat_list, bond_edge_index_list, bond_edge_attr_list = [], [], []
    batch_idx = []
    atom_offset = 0

    has_graph_feats = 'atom_feat' in batch[0]

    for i, data in enumerate(batch):
        n_atoms = len(data['z'])
        z_list.append(data['z'])
        pos_list.append(data['pos'])
        pharm_list.append(data['pharm'])
        size_list.append(data['size'])
        autocorr_list.append(data['autocorr'])
        shape3d_list.append(data['shape3d'])
        y_list.append(data['y'])
        batch_idx.extend([i] * n_atoms)

        if has_graph_feats:
            atom_feat_list.append(data['atom_feat'])
            bond_ei = data['bond_edge_index']
            if bond_ei.size(1) > 0:
                bond_edge_index_list.append(bond_ei + atom_offset)
            bond_edge_attr_list.append(data['bond_edge_attr'])
        atom_offset += n_atoms

    result = {
        'z': torch.cat(z_list),
        'pos': torch.cat(pos_list),
        'batch': torch.tensor(batch_idx, dtype=torch.long),
        'pharm': torch.stack(pharm_list),
        'size': torch.stack(size_list),
        'autocorr': torch.stack(autocorr_list),
        'shape3d': torch.stack(shape3d_list),
        'y': torch.tensor(y_list, dtype=torch.float32),
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


def trim_size_features(data_list):
    """Trim size features to 3d (rot, rings, aromatic).

    Handles the different cache versions:
      5d = [mw, ha, rot, rings, aromatic] -> take last 3
      3d = already trimmed
      8d = [rot, rings, aromatic, ...] -> take first 3
    """
    for d in data_list:
        s = d['size']
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32)
        if s.shape[0] == 5:
            d['size'] = s[2:]
        elif s.shape[0] == 3:
            d['size'] = s
        else:
            d['size'] = s[:3]
    return data_list
