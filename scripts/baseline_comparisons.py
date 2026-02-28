# -*- coding: utf-8 -*-
"""Baseline Comparisons — Traditional ML baselines."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import pickle
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaffold_utils import scaffold_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
SEED = 42


def extract_37d_features(data_list):
    """Extract 37d descriptor features from cached data:
    pharm(10d) + size(3d, no MW/HA) + autocorr(19d) + shape3d(5d)"""
    X = np.zeros((len(data_list), 37), dtype=np.float32)
    y = np.zeros(len(data_list), dtype=np.float32)
    for i, d in enumerate(data_list):
        pharm = d['pharm'].numpy() if hasattr(d['pharm'], 'numpy') else np.array(d['pharm'])
        size_raw = d['size'].numpy() if hasattr(d['size'], 'numpy') else np.array(d['size'])
        size = size_raw[2:] if len(size_raw) == 5 else size_raw
        autocorr = d['autocorr'].numpy() if hasattr(d['autocorr'], 'numpy') else np.array(d['autocorr'])
        shape3d = d['shape3d'].numpy() if hasattr(d['shape3d'], 'numpy') else np.array(d['shape3d'])
        X[i] = np.concatenate([pharm, size, autocorr, shape3d])
        y[i] = d['y']
    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def extract_morgan_fp(data_list, radius=2, n_bits=2048):
    """Extract Morgan fingerprints from SMILES in cached data."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    X = np.zeros((len(data_list), n_bits), dtype=np.float32)
    y = np.zeros(len(data_list), dtype=np.float32)
    for i, d in enumerate(data_list):
        smi = d.get('smiles', '')
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            X[i] = np.array(fp)
        y[i] = d['y']
    return X, y


def run_rf(X_train, y_train, X_test, y_test, desc="RF"):
    print(f"\n  Training {desc}...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight='balanced',
        random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    ap = average_precision_score(y_test, preds)
    elapsed = time.time() - t0
    print(f"    {desc}: AUC={auc:.4f}, AP={ap:.4f} ({elapsed:.1f}s)")
    return {'auc': float(auc), 'ap': float(ap), 'time': elapsed}


def run_xgboost(X_train, y_train, X_test, y_test, desc="XGBoost"):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print(f"  {desc}: xgboost not installed, skipping")
        return None

    print(f"\n  Training {desc}...")
    t0 = time.time()
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED, n_jobs=-1, verbosity=0,
        eval_metric='logloss')
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    ap = average_precision_score(y_test, preds)
    elapsed = time.time() - t0
    print(f"    {desc}: AUC={auc:.4f}, AP={ap:.4f} ({elapsed:.1f}s)")
    return {'auc': float(auc), 'ap': float(ap), 'time': elapsed}


def run_svm(X_train, y_train, X_test, y_test, max_train=20000, desc="SVM"):
    from sklearn.svm import SVC

    print(f"\n  Training {desc} (max {max_train} samples)...")
    t0 = time.time()

    if len(X_train) > max_train:
        idx = np.random.RandomState(SEED).choice(len(X_train), max_train, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=SEED)
    clf.fit(X_sub_s, y_sub)
    preds = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, preds)
    ap = average_precision_score(y_test, preds)
    elapsed = time.time() - t0
    print(f"    {desc}: AUC={auc:.4f}, AP={ap:.4f} ({elapsed:.1f}s)")
    return {'auc': float(auc), 'ap': float(ap), 'time': elapsed}


def main():
    print("Baseline Comparisons")

    # Load cached data
    cache_path = os.path.join(BASE_DIR, 'data', 'train_augmented_cache.pkl')
    if not os.path.exists(cache_path):
        print(f"Cache not found. Run train.py first.")
        sys.exit(1)
    print(f"\nLoading cache from {cache_path}")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    data_list = cache['data_list']
    print(f"  Total: {len(data_list)} samples")

    # Check that SMILES are available (cache v2)
    if 'smiles' not in data_list[0]:
        print("Cache missing SMILES field. Regenerate with train.py.")
        sys.exit(1)

    # Scaffold-based split (same as ablation/multi_seed for fair comparison)
    train_data, val_data, test_data = scaffold_split(data_list, seed=SEED)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    results = {}

    print("\n--- 37d Descriptor Features (pharm+size+autocorr+shape3d) ---")

    X_train_37, y_train = extract_37d_features(train_data)
    X_test_37, y_test = extract_37d_features(test_data)
    print(f"  Feature shape: {X_train_37.shape}")

    results['RF_37d'] = run_rf(X_train_37, y_train, X_test_37, y_test, "RF (37d)")
    results['XGBoost_37d'] = run_xgboost(X_train_37, y_train, X_test_37, y_test, "XGBoost (37d)")
    results['SVM_37d'] = run_svm(X_train_37, y_train, X_test_37, y_test, desc="SVM (37d)")

    print("\n--- Morgan Fingerprint (2048-bit) ---")

    X_train_fp, _ = extract_morgan_fp(train_data)
    X_test_fp, _ = extract_morgan_fp(test_data)
    print(f"  Feature shape: {X_train_fp.shape}")

    results['RF_Morgan'] = run_rf(X_train_fp, y_train, X_test_fp, y_test, "RF (Morgan)")
    results['XGBoost_Morgan'] = run_xgboost(X_train_fp, y_train, X_test_fp, y_test, "XGBoost (Morgan)")

    # Remove None results
    results = {k: v for k, v in results.items() if v is not None}

    # Save results
    results_path = os.path.join(BASE_DIR, 'results', 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\nBaseline Comparison Summary")
    print(f"{'Method':<25s} {'Test AUC':>10s} {'Test AP':>10s} {'Time(s)':>10s}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25s} {r['auc']:>10.4f} {r['ap']:>10.4f} {r['time']:>10.1f}")

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
