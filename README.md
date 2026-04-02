# Math4AI Final Capstone — From Linear Scores to a Single Hidden Layer

> **A Mathematical Study of Simple Learning Systems**
> National AI Center — AI Academy

## Project Objective

This project investigates when a **one-hidden-layer nonlinear classifier** genuinely improves on a **linear decision rule**, and when additional model complexity is unnecessary. We implement two models entirely from scratch using NumPy:

1. **Softmax Regression** — multiclass linear baseline (s(x) = Wx + b → softmax)
2. **One-Hidden-Layer Neural Network** — tanh activation + softmax output (h = tanh(W₁x + b₁), s = W₂h + b₂)

We evaluate both models across three tasks (linear Gaussian, moons, digits), perform capacity ablation, optimizer comparison, failure analysis, and PCA/SVD analysis (Track A).

## Team & Contributions

| Member | Tasks | Key Files |
|--------|-------|-----------|
| **Sharaf** | 1 (Softmax Regression), 2 (Neural Network), 5 (Moons + Capacity Ablation) | `models.py`, `data_utils.py`, `plotting.py` |
| **Samir** | 3 (Sanity Checks), 6 (Digits Optimizer Study) | `optimizers.py`, `train.py`, `sanity_checks.py` |
| **Nicat** | 4 (Linear Gaussian), 7 (Track A PCA + Repeated Seeds) | `run_experiments.py` (Track A, failure analysis) |

## Environment Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Murad-Huseynli/math4ai_capstone.git
cd math4ai_capstone
pip install -r requirements.txt
```

### Dependencies
```
numpy
matplotlib
```

## Reproducing Experiments

### Run All Experiments
```bash
python -m starter_pack.src.run_experiments
```

This executes the full experiment suite:
1. **Sanity Checks** — softmax-sum, loss-decrease, gradient check, NaN/Inf, overfit
2. **Synthetic Tasks** — linear Gaussian + moons (Softmax vs NN comparison)
3. **Digits Benchmark** — Softmax vs NN (h=32), SGD, 200 epochs
4. **Capacity Ablation** — moons with h ∈ {2, 8, 32}
5. **Optimizer Study** — SGD vs Momentum vs Adam on digits (h=32)
6. **Repeated-Seed Evaluation** — 5 seeds, 95% CI on digits
7. **Track A: PCA/SVD** — scree plot, 2D projection, softmax at m ∈ {10, 20, 40, 64}
8. **Failure Analysis** — under-capacity NN (h=2) on 10-class digits

### Run Sanity Checks Only
```bash
python -m starter_pack.src.sanity_checks
```

### Digits Protocol (Fixed)
| Parameter | Value |
|-----------|-------|
| Hidden width | 32 |
| L₂ regularization | λ = 10⁻⁴ |
| Batch size | 64 |
| Softmax optimizer | SGD (lr=0.05) |
| NN optimizers | SGD (0.05), Momentum (0.05, μ=0.9), Adam (0.001) |
| Epochs | 200 (best val-CE checkpoint) |

## Repository Structure

```
math4ai_capstone/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── deliverables/
│   └── math4ai_capstone_assignment.tex   # Official assignment handout
│
└── starter_pack/
    ├── README.md                # Starter pack overview
    ├── CHECKLIST.md             # Pre-implementation checklist
    │
    ├── data/                    # Fixed datasets (DO NOT MODIFY)
    │   ├── linear_gaussian.npz  # 2D Gaussian blobs (binary)
    │   ├── moons.npz            # 2D moons (binary, nonlinear)
    │   ├── digits_data.npz      # 64-dim digit features + labels
    │   └── digits_split_indices.npz  # Fixed train/val/test split
    │
    ├── scripts/                 # Data generation utilities
    │   ├── generate_synthetic.py
    │   └── make_digits_split.py
    │
    ├── src/                     # Core implementation (all from scratch)
    │   ├── __init__.py          # Package initializer
    │   ├── models.py            # SoftmaxRegression + NeuralNetwork
    │   ├── optimizers.py        # SGD, Momentum, Adam
    │   ├── train.py             # Mini-batch training loop + checkpointing
    │   ├── data_utils.py        # Data loaders + mini-batch generator
    │   ├── plotting.py          # Decision boundaries, loss curves, PCA plots
    │   ├── sanity_checks.py     # Automated implementation verification
    │   └── run_experiments.py   # Full experiment suite orchestration
    │
    ├── figures/                 # Generated experiment figures
    │   ├── comparison_lineargaussian.png
    │   ├── comparison_moons.png
    │   ├── capacity_ablation_boundaries.png
    │   ├── capacity_ablation_curves.png
    │   ├── optimizer_study_digits.png
    │   ├── repeated_seed_digits.png
    │   ├── failure_analysis.png
    │   ├── pca_scree_digits.png
    │   ├── pca_2d_digits.png
    │   ├── pca_softmax_comparison.png
    │   ├── loss_curves_*.png
    │   └── decision_boundary_*.png
    │
    ├── results/                 # Experiment logs and metrics
    │   ├── optimizer_study.txt
    │   ├── repeated_seed.txt
    │   ├── track_a_pca.txt
    │   └── failure_analysis.txt
    │
    ├── report/                  # LaTeX report
    │   ├── template.tex         # TA-provided template
    │   └── main.tex             # Final report source
    │
    └── slides/                  # Presentation
        └── presentation.tex     # Beamer slides source
```

## Key Results Summary

### Synthetic Tasks
- **Linear Gaussian:** Softmax regression achieves near-perfect separation — the hidden layer provides no benefit (as expected for linearly separable data)
- **Moons:** The neural network's nonlinear decision boundary captures the crescent structure, while softmax regression fails (linear boundary cannot separate interleaving moons)

### Capacity Ablation (Moons)
- **h=2:** Insufficient capacity — boundary is too simple
- **h=8:** Captures the general shape of the moons
- **h=32:** Best fit with smooth probability gradients

### Optimizer Study (Digits, h=32)
Results saved in `starter_pack/results/optimizer_study.txt`

### Repeated-Seed Evaluation (Digits)
5-seed evaluation with 95% CI (t=2.776, df=4) — results in `starter_pack/results/repeated_seed.txt`

### Track A: PCA/SVD
Scree plot shows ~90% variance captured by top ~20 PCs. Softmax accuracy at m=40 is comparable to full 64-dim input.

## Implementation Notes

- **No ML frameworks** — all models, optimizers, and training loops are implemented from scratch using NumPy
- **Numerically stable softmax** — uses max-subtraction trick to prevent overflow
- **Validation checkpointing** — saves best parameters at lowest validation cross-entropy
- **Xavier initialization** — scale = √(2 / (fan_in + fan_out))
- **Reproducibility** — all experiments use fixed seeds

## License

This project is an academic capstone for the National AI Center — AI Academy Math4AI course.