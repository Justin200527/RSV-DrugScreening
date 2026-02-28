# RSV-DrugScreen

Multi-modal deep learning framework combining E(3)-invariant 3D graph neural networks with molecular descriptor modules for virtual screening of RSV inhibitors from natural products.

## Results (Scaffold Split)

| Metric | Value |
|--------|-------|
| Test AUC (seed 789) | 0.775 |
| Multi-seed AUC (5 seeds) | 0.767 +/- 0.006 |
| 5-Fold CV AUC | 0.773 +/- 0.020 |

## Installation

```bash
pip install -r requirements.txt
```

**Note**: PyTorch Geometric requires platform-specific installation. See [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

```bash
# Training
python scripts/train.py

# Validation
python scripts/multi_seed.py
python scripts/cross_validation.py
python scripts/ablation_study.py
python scripts/gnn_baselines.py
python scripts/baseline_comparisons.py

# COCONUT screening (requires pre-extracted features)
python scripts/score_coconut.py --model models/rsv_best.pt
```

**Note**: Run `train.py` first — other scripts depend on the cached training data it generates.

## License

MIT License — see [LICENSE](LICENSE) for details.
