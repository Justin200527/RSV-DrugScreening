# -*- coding: utf-8 -*-
"""
Model Definition
E(3)-invariant 3D GNN + multi-modal fusion (6 modules, 768d)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, radius_graph
from torch_scatter import scatter


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class RadialBasisFunction(nn.Module):
    def __init__(self, num_rbf=64, cutoff=5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.register_buffer('centers', torch.linspace(0, cutoff, num_rbf))
        self.width = (cutoff / num_rbf) * 0.5

    def forward(self, dist):
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width ** 2))


class CosineCutoff(nn.Module):
    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist):
        return 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0) * (dist < self.cutoff).float()


class MultiScaleE3Layer(nn.Module):
    def __init__(self, hidden_dim=128, num_rbf=32, cutoffs=[5.0, 10.0, 15.0, 20.0, 25.0],
                 bond_feat_dim=6):
        super().__init__()
        self.cutoffs = cutoffs
        self.num_spatial_scales = len(cutoffs)
        self.num_scales = len(cutoffs) + 1  # +1 for bond topology path
        self.rbfs = nn.ModuleList([RadialBasisFunction(num_rbf, c) for c in cutoffs])
        self.cutoff_fns = nn.ModuleList([CosineCutoff(c) for c in cutoffs])
        self.message_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in cutoffs
        ])
        # Bond topology message path
        self.bond_message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + bond_feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Attention over all paths (spatial + bond)
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.num_scales), nn.Softmax(dim=-1)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, pos, batch, bond_edge_index=None, bond_edge_attr=None):
        scale_outputs = []
        # Path 1-5: spatial distance edges
        for i, cutoff in enumerate(self.cutoffs):
            edge_index = radius_graph(pos, r=cutoff, batch=batch, max_num_neighbors=32)
            if edge_index.size(1) == 0:
                scale_outputs.append(torch.zeros_like(x))
                continue
            row, col = edge_index
            dist = (pos[col] - pos[row]).norm(dim=-1)
            rbf_feat = self.rbfs[i](dist)
            cutoff_weight = self.cutoff_fns[i](dist)
            msg_input = torch.cat([x[row], x[col], rbf_feat], dim=-1)
            msg = self.message_mlps[i](msg_input) * cutoff_weight.unsqueeze(-1)
            agg = scatter(msg, row, dim=0, dim_size=x.size(0), reduce='mean')
            scale_outputs.append(agg)
        # Path 6: bond topology edges
        if bond_edge_index is not None and bond_edge_index.size(1) > 0:
            row_b, col_b = bond_edge_index
            msg_input_b = torch.cat([x[row_b], x[col_b], bond_edge_attr], dim=-1)
            msg_b = self.bond_message_mlp(msg_input_b)
            agg_b = scatter(msg_b, row_b, dim=0, dim_size=x.size(0), reduce='mean')
            scale_outputs.append(agg_b)
        else:
            scale_outputs.append(torch.zeros_like(x))

        stacked = torch.stack(scale_outputs, dim=-1)
        concat_scales = torch.cat(scale_outputs, dim=-1)
        attn = self.scale_attention(concat_scales)
        weighted = (stacked * attn.unsqueeze(1)).sum(dim=-1)
        return x + self.update_mlp(torch.cat([x, weighted], dim=-1))


class VirtualNode(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.node_to_global = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.global_to_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, batch, vn_embedding):
        batch_size = batch.max().item() + 1
        node_agg = scatter(x, batch, dim=0, dim_size=batch_size, reduce='mean')
        global_update = self.node_to_global(node_agg)
        vn_embedding = self.gru(global_update, vn_embedding)
        node_update = self.global_to_node(vn_embedding)[batch]
        return x + node_update, vn_embedding


class LongRangeDistanceModule(nn.Module):
    def __init__(self, hidden_dim, num_bins=64, max_distance=25.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_distance = max_distance
        self.encoder = nn.Sequential(
            nn.Linear(num_bins, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, pos, batch):
        batch_size = batch.max().item() + 1
        histograms = []
        for b in range(batch_size):
            mask = (batch == b)
            pos_b = pos[mask]
            if pos_b.size(0) < 2:
                histograms.append(torch.zeros(self.num_bins, device=pos.device))
                continue
            dist_matrix = torch.cdist(pos_b, pos_b)
            triu_idx = torch.triu_indices(dist_matrix.size(0), dist_matrix.size(1), offset=1)
            distances = dist_matrix[triu_idx[0], triu_idx[1]]
            hist = torch.histc(distances, bins=self.num_bins, min=0, max=self.max_distance)
            hist = hist / (hist.sum() + 1e-8)
            histograms.append(hist)
        return self.encoder(torch.stack(histograms))


class LargeMoleculeAdaptation(nn.Module):
    def __init__(self, hidden_dim, size_features=3, adapt_graph=True):
        super().__init__()
        self.adapt_graph = adapt_graph
        self.size_encoder = nn.Sequential(
            nn.Linear(size_features, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim))
        if adapt_graph:
            self.scale_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, size_feat, h):
        size_emb = self.size_encoder(size_feat)
        if self.adapt_graph:
            scale = self.scale_predictor(h)
            return h * (1 + scale * 0.5), size_emb
        return h, size_emb


class PharmacophoreModule(nn.Module):
    def __init__(self, hidden_dim, pharm_features=9):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(pharm_features + 1, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim))

    def forward(self, pharm_feat):
        return self.attention(pharm_feat)


class AutocorrelationModule(nn.Module):
    def __init__(self, hidden_dim, num_features=19,
                 active_means=None, active_stds=None, feature_weights=None):
        super().__init__()
        if active_means is not None:
            self.register_buffer('means', torch.tensor(active_means, dtype=torch.float32))
        else:
            self.register_buffer('means', torch.zeros(num_features))
        if active_stds is not None:
            self.register_buffer('stds', torch.tensor(active_stds, dtype=torch.float32))
        else:
            self.register_buffer('stds', torch.ones(num_features))
        if feature_weights is not None:
            self.register_buffer('weights', torch.tensor(feature_weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.ones(num_features))
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim // 2), nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim))

    def forward(self, x):
        x_norm = (x - self.means) / (self.stds + 1e-8)
        x_weighted = x_norm * self.weights
        return self.encoder(x_weighted)


class Shape3DModule(nn.Module):
    def __init__(self, hidden_dim, num_features=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim))

    def forward(self, x):
        return self.encoder(x)


class RSVDrugScreeningModel(nn.Module):
    """
    RSV Drug Screening Model
    6-module fusion: graph + distance + pharmacophore + size + autocorrelation + shape3D
    Input: atomic numbers z, 3D coords pos, batch index, pharm(10d), size(3d), autocorr(19d), shape3d(5d)
    Output: logit (scalar, pre-sigmoid)

    disabled_modules: set of module names to disable for ablation study.
        Valid names: 'graph', 'dist', 'pharm', 'size', 'autocorr', 'shape3d'
        When None, all modules are enabled (default, compatible with existing checkpoints).
    """
    MODULE_NAMES = ['graph', 'dist', 'pharm', 'size', 'autocorr', 'shape3d']

    def __init__(self, num_elements=100, hidden_dim=128, num_layers=4, num_rbf=64,
                 cutoffs=[5.0, 10.0, 15.0, 20.0, 25.0], dropout=0.15,
                 pharm_features=10, size_features=3,
                 autocorr_features=19, shape3d_features=5,
                 active_means=None, active_stds=None, feature_weights=None,
                 module_weights=None, disabled_modules=None,
                 large_mol_adapt=False, atom_feat_dim=153, bond_feat_dim=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.disabled_modules = set(disabled_modules) if disabled_modules else set()
        self.large_mol_adapt = large_mol_adapt
        self._return_embedding = False  # set True to also return 768d fusion vector
        self.module_weights = module_weights or {
            'graph': 0.5, 'dist': 0.5, 'pharm': 1.0,
            'size': 1.0, 'autocorr': 1.5, 'shape3d': 1.5
        }

        # Always create atom_embedding (needed for graph module and as fallback)
        self.atom_embedding = nn.Embedding(num_elements, hidden_dim)
        # Rich atom feature projection (153d → hidden_dim)
        self.atom_projection = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim), nn.SiLU()
        )

        if 'graph' not in self.disabled_modules:
            self.equi_layers = nn.ModuleList(
                [MultiScaleE3Layer(hidden_dim, num_rbf, cutoffs, bond_feat_dim)
                 for _ in range(num_layers)])
            self.vn_layers = nn.ModuleList([VirtualNode(hidden_dim) for _ in range(num_layers)])
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        if 'dist' not in self.disabled_modules:
            self.dist_module = LongRangeDistanceModule(hidden_dim, num_bins=64, max_distance=25.0)

        if 'size' not in self.disabled_modules:
            self.large_mol_module = LargeMoleculeAdaptation(
                hidden_dim, size_features, adapt_graph=large_mol_adapt)

        if 'pharm' not in self.disabled_modules:
            # pharm_features=10 (9 substructure + MolLogP); module adds +1 internally
            self.pharm_module = PharmacophoreModule(hidden_dim, pharm_features - 1)

        if 'autocorr' not in self.disabled_modules:
            self.autocorr_module = AutocorrelationModule(
                hidden_dim, autocorr_features, active_means, active_stds, feature_weights)

        if 'shape3d' not in self.disabled_modules:
            self.shape3d_module = Shape3DModule(hidden_dim, shape3d_features)

        n_active = len(self.MODULE_NAMES) - len(self.disabled_modules)
        total_dim = hidden_dim * n_active
        self.readout = nn.Sequential(
            nn.Linear(total_dim, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 1))

        # Learnable temperature for anti-saturation (init=1.0, learns to spread scores)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, z, pos, batch, pharm_feat, size_feat, autocorr_feat, shape3d_feat,
                atom_feat=None, bond_edge_index=None, bond_edge_attr=None):
        w = self.module_weights
        parts = []
        batch_size = batch.max().item() + 1

        # Atom representation: use rich 153d features if available, else fallback to embedding
        if atom_feat is not None:
            x = self.atom_projection(atom_feat)
        else:
            x = self.atom_embedding(z)

        if 'graph' not in self.disabled_modules:
            vn_emb = torch.zeros(batch_size, self.hidden_dim, device=z.device)
            for equi, vn, ln in zip(self.equi_layers, self.vn_layers, self.layer_norms):
                x_res = x
                x = equi(x, pos, batch, bond_edge_index, bond_edge_attr)
                x, vn_emb = vn(x, batch, vn_emb)
                x = ln(x + x_res)
            h_graph = global_mean_pool(x, batch)
        else:
            # Fallback: simple pooling (no message passing)
            h_graph = global_mean_pool(x, batch)

        if 'size' not in self.disabled_modules:
            h_graph, h_size = self.large_mol_module(size_feat, h_graph)
            parts.append(h_size * w['size'])
        # h_graph is always produced (graph or fallback)
        if 'graph' not in self.disabled_modules:
            parts.append(h_graph * w['graph'])

        if 'dist' not in self.disabled_modules:
            h_dist = self.dist_module(pos, batch)
            parts.append(h_dist * w['dist'])

        if 'pharm' not in self.disabled_modules:
            h_pharm = self.pharm_module(pharm_feat)
            parts.append(h_pharm * w['pharm'])

        if 'autocorr' not in self.disabled_modules:
            h_autocorr = self.autocorr_module(autocorr_feat)
            parts.append(h_autocorr * w['autocorr'])

        if 'shape3d' not in self.disabled_modules:
            h_shape3d = self.shape3d_module(shape3d_feat)
            parts.append(h_shape3d * w['shape3d'])

        combined = torch.cat(parts, dim=-1)
        logits = self.readout(combined).squeeze(-1)
        if self._return_embedding:
            return logits / self.temperature.clamp(min=0.1), combined
        return logits / self.temperature.clamp(min=0.1)
