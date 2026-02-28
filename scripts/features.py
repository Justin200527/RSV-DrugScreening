# -*- coding: utf-8 -*-
"""Shared feature extraction for training and prediction."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Descriptors3D
from mordred import Calculator, descriptors

KEY_2D_FEATURES = [
    'AATS7Z', 'AATS8Z', 'AATS7m', 'AATS8m', 'AATS8v',
    'AATS2Z', 'AATS2m', 'AATS2v', 'AATS2dv',
    'AATSC5Z', 'AATSC5m', 'AATSC8are', 'AATSC8pe',
    'MATS7pe', 'MATS7are', 'MATS7dv', 'MATS8se',
    'GATS4Z', 'GATS4m',
]

KEY_3D_FEATURES = [
    'Eccentricity', 'NPR1', 'NPR2', 'SpherocityIndex', 'InertialShapeFactor'
]

PHARMACOPHORE_PATTERNS = [
    '[OH]c1ccccc1', '[OH]C=C', 'c1ccc2c(c1)ccc1ccccc12',
    'O=C(O)c1ccccc1', 'OC(=O)C=C', 'c1ccc(O)c(O)c1',
    'O=C1CC(c2ccccc2)Oc2ccccc21', 'c1cc(O)cc(O)c1', '[OH]c1cc(O)cc(O)c1'
]

FEATURE_COCONUT_PERCENTILES = {
    'AATS7Z': 95.6, 'AATS8Z': 94.6, 'AATS7m': 95.4, 'AATS8m': 94.6, 'AATS8v': 95.2,
    'AATS2Z': 92.2, 'AATS2m': 92.0, 'AATS2v': 90.2, 'AATS2dv': 92.0,
    'AATSC5Z': 97.2, 'AATSC5m': 97.2, 'AATSC8are': 96.6, 'AATSC8pe': 95.6,
    'MATS7pe': 96.2, 'MATS7are': 96.2, 'MATS7dv': 95.4, 'MATS8se': 93.8,
    'GATS4Z': 92.6, 'GATS4m': 92.4,
}

# Mordred calculator (module-level singleton)
mordred_calc = Calculator(descriptors, ignore_3D=True)

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

BASIC_ATOM_FEAT_DIM = (118 + 1) + (6 + 1) + (5 + 1) + (5 + 1) + (2 + 1) + 1 + 1  # 143
FUNC_GROUP_DIM = 10  # functional group node annotations
ATOM_FEAT_DIM = BASIC_ATOM_FEAT_DIM + FUNC_GROUP_DIM  # 153
BOND_FEAT_DIM = 6

# Functional group SMARTS for node-level annotations
FUNC_GROUP_SMARTS = [
    ('catechol', 'c1cc(O)c(O)cc1'),          # 1,2-dihydroxybenzene
    ('ester', '[#6](=O)[O][#6]'),             # ester linkage
    ('michael_acceptor', '[#6]=[#6][#6]=O'),  # α,β-unsaturated carbonyl
    ('hydroxyl', '[OX2H]'),                   # hydroxyl group
    ('carboxyl', '[CX3](=O)[OX2H]'),         # carboxylic acid
    ('resorcinol', 'c1cc(O)cc(O)c1'),        # 1,3-dihydroxybenzene
]


def _one_hot(val, allowed):
    vec = [0] * (len(allowed) + 1)
    if val in allowed:
        vec[allowed.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_to_feature_vector(atom):
    """143d one-hot atom features."""
    features = []
    features.extend(_one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
    features.extend(_one_hot(atom.GetDegree(), ATOM_FEATURES['degree']))
    features.extend(_one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
    features.extend(_one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
    features.extend(_one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic']))
    features.append(atom.GetTotalNumHs())
    features.append(atom.GetNumRadicalElectrons())
    return features


def bond_to_feature_vector(bond):
    """6d bond features."""
    bt = bond.GetBondType()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


def extract_functional_group_node_features(mol):
    """Per-atom functional group annotations. Returns np.array [N, FUNC_GROUP_DIM=10]."""
    n_atoms = mol.GetNumAtoms()
    feats = np.zeros((n_atoms, FUNC_GROUP_DIM), dtype=np.float32)

    # Features 0-5: SMARTS-based functional group membership
    for feat_idx, (name, smarts) in enumerate(FUNC_GROUP_SMARTS):
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            for match in mol.GetSubstructMatches(patt):
                for atom_idx in match:
                    if atom_idx < n_atoms:
                        feats[atom_idx, feat_idx] = 1.0

    # Feature 6: vinyl carbon (non-aromatic C=C)
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                and not bond.GetIsAromatic()):
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                    and mol.GetAtomWithIdx(j).GetAtomicNum() == 6):
                feats[i, 6] = 1.0
                feats[j, 6] = 1.0

    # Feature 7: ring membership count (normalized by 3)
    ri = mol.GetRingInfo()
    for i in range(n_atoms):
        feats[i, 7] = ri.NumAtomRings(i) / 3.0

    # Feature 8: linker atom (non-ring atom connecting ring systems)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        if not atom.IsInRing():
            ring_neighbors = sum(1 for nb in atom.GetNeighbors() if nb.IsInRing())
            feats[i, 8] = min(ring_neighbors / 2.0, 1.0)

    # Feature 9: conjugated non-aromatic
    for bond in mol.GetBonds():
        if bond.GetIsConjugated() and not bond.GetIsAromatic():
            feats[bond.GetBeginAtomIdx(), 9] = 1.0
            feats[bond.GetEndAtomIdx(), 9] = 1.0

    return feats


def extract_graph_features(mol):
    """Extract atom features, bond edge_index, bond edge_attr from RDKit mol (heavy atoms only).
    Returns (atom_feat [N,153], bond_edge_index [2,E], bond_edge_attr [E,6]) as tensors.
    """
    # Basic one-hot features (143d)
    atom_feats = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    atom_feat_basic = torch.tensor(atom_feats, dtype=torch.float32)

    # Functional group annotations (10d)
    fg_feats = torch.tensor(extract_functional_group_node_features(mol), dtype=torch.float32)

    # Concatenate: [N, 143] + [N, 10] = [N, 153]
    atom_feat = torch.cat([atom_feat_basic, fg_feats], dim=-1)

    edge_list = []
    edge_feat_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature_vector(bond)
        edge_list.append([i, j])
        edge_list.append([j, i])
        edge_feat_list.append(feat)
        edge_feat_list.append(feat)

    if edge_list:
        bond_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        bond_edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)
    else:
        bond_edge_index = torch.zeros((2, 0), dtype=torch.long)
        bond_edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)

    return atom_feat, bond_edge_index, bond_edge_attr


def extract_pharmacophore_features(mol):
    features = []
    for pattern in PHARMACOPHORE_PATTERNS:
        patt = Chem.MolFromSmarts(pattern)
        features.append(len(mol.GetSubstructMatches(patt)) if patt else 0)
    features.append(Descriptors.MolLogP(mol))
    return features


def extract_size_features(mol):
    mw = Descriptors.MolWt(mol)
    ha = mol.GetNumHeavyAtoms()
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return [mw / 500.0, ha / 50.0, rot / 10.0, rings / 5.0, aromatic / 20.0]


def extract_autocorr_features(mol):
    try:
        result = mordred_calc(mol)
        features = []
        for feat in KEY_2D_FEATURES:
            try:
                val = result[feat]
                if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'Missing'):
                    val = 0.0
                else:
                    val = float(val)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
            except Exception:
                val = 0.0
            features.append(val)
        return features
    except Exception:
        return [0.0] * len(KEY_2D_FEATURES)


def extract_shape3d_features(mol, conf_id=-1):
    try:
        features = [
            Descriptors3D.Eccentricity(mol, confId=conf_id),
            Descriptors3D.NPR1(mol, confId=conf_id),
            Descriptors3D.NPR2(mol, confId=conf_id),
            Descriptors3D.SpherocityIndex(mol, confId=conf_id),
            Descriptors3D.InertialShapeFactor(mol, confId=conf_id),
        ]
        return features
    except Exception:
        return [0.0] * len(KEY_3D_FEATURES)



def generate_conformer(mol_h, seed=42):
    if AllChem.EmbedMolecule(mol_h, randomSeed=seed, maxAttempts=50) != 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    return mol_h


def process_molecule(smi, label, seed=42):
    """Extract all features from SMILES (single conformer). Returns feature dict or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    mol_h = generate_conformer(mol_h, seed=seed)
    if mol_h is None:
        return None

    # 3D shape computed from molecule with hydrogens
    shape3d = torch.tensor(extract_shape3d_features(mol_h), dtype=torch.float32)
    # GNN input uses heavy atoms (H removed)
    mol_no_h = Chem.RemoveHs(mol_h)
    conf = mol_no_h.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    z = torch.tensor([atom.GetAtomicNum() for atom in mol_no_h.GetAtoms()], dtype=torch.long)
    # 2D features
    pharm = torch.tensor(extract_pharmacophore_features(mol_no_h), dtype=torch.float32)
    size = torch.tensor(extract_size_features(mol_no_h), dtype=torch.float32)
    autocorr = torch.tensor(extract_autocorr_features(mol_no_h), dtype=torch.float32)

    # Graph features (atom 143d + bond topology)
    atom_feat, bond_edge_index, bond_edge_attr = extract_graph_features(mol_no_h)

    canonical = Chem.MolToSmiles(mol)
    return {
        'z': z, 'pos': pos, 'pharm': pharm, 'size': size,
        'autocorr': autocorr, 'shape3d': shape3d, 'y': label,
        'smiles': canonical,
        'atom_feat': atom_feat,
        'bond_edge_index': bond_edge_index,
        'bond_edge_attr': bond_edge_attr,
    }


def process_molecule_multiconf(smi, label, n_conformers=10, base_seed=42):
    """Extract multi-conformer features from SMILES (data augmentation). 2D features shared, 3D features vary."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    canonical = Chem.MolToSmiles(mol)
    pharm = torch.tensor(extract_pharmacophore_features(mol), dtype=torch.float32)
    size = torch.tensor(extract_size_features(mol), dtype=torch.float32)
    autocorr = torch.tensor(extract_autocorr_features(mol), dtype=torch.float32)

    results = []
    for i in range(n_conformers):
        seed = base_seed + i * 137
        mol_h = Chem.AddHs(mol)
        mol_h = generate_conformer(mol_h, seed=seed)
        if mol_h is None:
            continue

        shape3d = torch.tensor(extract_shape3d_features(mol_h), dtype=torch.float32)
        mol_no_h = Chem.RemoveHs(mol_h)
        conf = mol_no_h.GetConformer()
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        z = torch.tensor([atom.GetAtomicNum() for atom in mol_no_h.GetAtoms()], dtype=torch.long)

        # Graph features (same topology across conformers, but extract per-conf for consistency)
        atom_feat, bond_edge_index, bond_edge_attr = extract_graph_features(mol_no_h)

        results.append({
            'z': z, 'pos': pos, 'pharm': pharm, 'size': size,
            'autocorr': autocorr, 'shape3d': shape3d, 'y': label,
            'smiles': canonical,
            'atom_feat': atom_feat,
            'bond_edge_index': bond_edge_index,
            'bond_edge_attr': bond_edge_attr,
        })

    return results


def process_molecule_from_mol(mol, mol_3d=None):
    """Extract features from RDKit Mol object (for SDF with existing 3D coords). Returns feature dict or None."""
    if mol is None:
        return None

    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None

    # 2D features (from H-removed molecule)
    mol_no_h = Chem.RemoveHs(mol) if mol.GetNumAtoms() != mol.GetNumHeavyAtoms() else mol
    pharm = extract_pharmacophore_features(mol_no_h)
    size = extract_size_features(mol_no_h)
    autocorr = extract_autocorr_features(mol_no_h)

    # 3D features
    mol_for_3d = mol_3d if mol_3d is not None else mol
    if mol_for_3d.GetNumConformers() > 0:
        shape3d = extract_shape3d_features(mol_for_3d)
        conf = Chem.RemoveHs(mol_for_3d).GetConformer() if mol_for_3d.GetNumAtoms() != mol_for_3d.GetNumHeavyAtoms() else mol_for_3d.GetConformer()
        pos = conf.GetPositions().tolist()
        z = [atom.GetAtomicNum() for atom in Chem.RemoveHs(mol_for_3d).GetAtoms()]
    else:
        # No 3D coords, generate conformer
        mol_h = Chem.AddHs(mol)
        mol_h = generate_conformer(mol_h, seed=42)
        if mol_h is None:
            return None
        shape3d = extract_shape3d_features(mol_h)
        mol_no_h2 = Chem.RemoveHs(mol_h)
        conf = mol_no_h2.GetConformer()
        pos = conf.GetPositions().tolist()
        z = [atom.GetAtomicNum() for atom in mol_no_h2.GetAtoms()]

    # Graph features
    atom_feat, bond_edge_index, bond_edge_attr = extract_graph_features(mol_no_h)

    return {
        'smiles': smiles,
        'z': z, 'pos': pos, 'pharm': pharm, 'size': size,
        'autocorr': autocorr, 'shape3d': shape3d,
        'mw': Descriptors.MolWt(mol_no_h),
        'atom_feat': atom_feat,
        'bond_edge_index': bond_edge_index,
        'bond_edge_attr': bond_edge_attr,
    }


def compute_active_statistics(active_smiles_list, max_samples=200):
    """Compute mean/std of autocorrelation features from active compounds."""
    from tqdm import tqdm
    active_2d_values = {feat: [] for feat in KEY_2D_FEATURES}
    for smi in tqdm(active_smiles_list[:max_samples], desc="Active stats"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats = extract_autocorr_features(mol)
        for i, feat in enumerate(KEY_2D_FEATURES):
            if not np.isnan(feats[i]) and not np.isinf(feats[i]) and feats[i] != 0.0:
                active_2d_values[feat].append(feats[i])

    active_means = [np.mean(active_2d_values[f]) if active_2d_values[f] else 0.0
                    for f in KEY_2D_FEATURES]
    active_stds = [np.std(active_2d_values[f]) if active_2d_values[f] else 1.0
                   for f in KEY_2D_FEATURES]
    active_stds = [s if s > 1e-8 else 1.0 for s in active_stds]
    return active_means, active_stds


def compute_feature_weights():
    """Compute feature weights based on COCONUT percentiles."""
    weights = []
    for feat in KEY_2D_FEATURES:
        pct = FEATURE_COCONUT_PERCENTILES.get(feat, 50)
        w = 1.0 + abs(pct - 50) / 50
        weights.append(w)
    return weights
