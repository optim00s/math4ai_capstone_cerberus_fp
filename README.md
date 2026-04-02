# Math4AI Final Capstone — From Linear Scores to a Single Hidden Layer

> **A Mathematical Study of Simple Learning Systems** National AI Center — AI Academy, April 2026

## Project Objective

This project investigates when a **one-hidden-layer nonlinear classifier** genuinely improves on a **linear decision rule**, and when additional model complexity is unnecessary. Both models are implemented entirely from scratch using NumPy:

1. **Softmax Regression** — multiclass linear baseline (`s = Wx + b → softmax`)
2. **One-Hidden-Layer Neural Network** — `tanh` hidden activation + softmax output (`h = tanh(W₁x + b₁)`, `s = W₂h + b₂`)

We evaluate both models across three tasks of increasing geometric difficulty — linearly separable Gaussian blobs, nonlinear moons, and a 10-class digits benchmark — and perform capacity ablation, optimizer comparison, failure-case analysis, repeated-seed evaluation, and PCA/SVD analysis (Track A).

## Team & Contributions

| Member | Responsibilities | Key Files |
|--------|-----------------|-----------|
| **Sharaf Feyzullayev** | Core models (Softmax Regression & Neural Network), data loading & mini-batch utilities, decision-boundary & loss-curve plotting, mathematical derivations, moons capacity-ablation experiment | `models.py`, `data_utils.py`, `plotting.py` |
| **Nijat Samadov** | Full experiment orchestration, synthetic task evaluation (Linear Gaussian), Track A (PCA/SVD analysis), repeated-seed evaluation with 95% CIs, failure-case analysis (H=2 bottleneck) | `run_experiments.py`, `plotting.py` |
| **Samir Abdullazade** | Three optimizers (SGD, Momentum, Adam), mini-batch training loop with validation-checkpoint saving, five automated sanity checks, digits optimizer comparison study | `optimizers.py`, `train.py`, `sanity_checks.py` |

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

This executes the full experiment suite in order:

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Sanity Checks | Softmax-sum, loss-decrease, gradient check, NaN/Inf, overfit |
| 2 | Synthetic Tasks | Linear Gaussian + Moons — Softmax vs NN (h=32) with decision boundaries |
| 3 | Digits Benchmark | Softmax (SGD lr=0.05) vs NN h=32 (Adam lr=0.001), 200 epochs |
| 4 | Capacity Ablation | Moons with h ∈ {2, 8, 32}, Adam lr=0.01, 500 epochs |
| 5 | Optimizer Study | SGD vs Momentum vs Adam on Digits (NN h=32, 200 epochs) |
| 6 | Repeated-Seed Eval | 5 seeds, 95% CI (t₀.₀₂₅,₄ = 2.776) on Digits |
| 7 | Track A: PCA/SVD | Scree plot, 2D projection, softmax at m ∈ {10, 20, 40, 64} |
| 8 | Failure Analysis | Under-capacity NN (h=2) on 10-class Digits |

All figures are saved to `starter_pack/figures/` and all result logs to `starter_pack/results/`.

### Run Sanity Checks Only
```bash
python -m starter_pack.src.sanity_checks
```

### Training Protocols

**Synthetic tasks** (Linear Gaussian, Moons):
| Parameter | Value |
|-----------|-------|
| Epochs | 300 |
| Batch size | Full-batch |
| Softmax optimizer | SGD (lr=0.1) |
| NN optimizer | Adam (lr=0.01) |
| L₂ regularization | λ = 10⁻⁵ |

**Digits benchmark**:
| Parameter | Value |
|-----------|-------|
| Epochs | 200 (best val-CE checkpoint) |
| Batch size | 64 |
| Hidden width | 32 |
| Softmax optimizer | SGD (lr=0.05) |
| NN optimizer | Adam (lr=0.001) |
| L₂ regularization | λ = 10⁻⁴ |

## Repository Structure

```
math4ai_capstone/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies (numpy, matplotlib)
├── .gitignore
│
├── deliverables/
│   └── math4ai_capstone_assignment.tex  # Assignment source (LaTeX)
│
└── starter_pack/
    ├── README.md                    # Starter pack overview
    ├── CHECKLIST.md                 # Pre-implementation checklist
    │
    ├── data/                        # Fixed datasets (DO NOT MODIFY)
    │   ├── linear_gaussian.npz      # 400 points, 2D Gaussian blobs (binary)
    │   ├── moons.npz                # 400 points, 2D moons (binary, nonlinear)
    │   ├── digits_data.npz          # 1,797 samples, 64-dim pixel features, 10 classes
    │   └── digits_split_indices.npz # Fixed train/val/test split (1258/270/269)
    │
    ├── scripts/                     # Data generation utilities
    │   ├── generate_synthetic.py    # Regenerate linear_gaussian + moons
    │   └── make_digits_split.py     # Reproduce the digits split indices
    │
    ├── src/                         # Core implementation 
    │   ├── __init__.py              # Package initializer
    │   ├── models.py                # SoftmaxRegression + NeuralNetwork classes
    │   ├── optimizers.py            # SGD, Momentum, Adam optimizers
    │   ├── train.py                 # Mini-batch training loop + val-CE checkpointing
    │   ├── data_utils.py            # Data loaders + mini-batch generator
    │   ├── plotting.py              # Decision boundaries, loss curves, PCA plots
    │   ├── sanity_checks.py         # 5 automated implementation checks
    │   └── run_experiments.py       # Full experiment suite orchestration
    │
    ├── figures/                     # Generated experiment figures (21 PNG files)
    │   ├── comparison_lineargaussian.png    # Side-by-side: SR vs NN on Gaussian
    │   ├── comparison_moons.png             # Side-by-side: SR vs NN on Moons
    │   ├── capacity_ablation_boundaries.png # h={2,8,32} boundaries on Moons
    │   ├── capacity_ablation_curves.png     # Loss/acc curves for capacity ablation
    │   ├── loss_curves_digits.png           # Training dynamics on Digits
    │   ├── optimizer_study_digits.png       # SGD vs Momentum vs Adam
    │   ...
    │
    ├── results/                     # Experiment logs and metrics
    │   ├── sanity_checks.txt        # All 5 sanity checks
    │   ├── synthetic_results.txt    # Gaussian & Moons test acc/CE
    │   ├── optimizer_study.txt      # SGD/Momentum/Adam comparison table
    │   ├── repeated_seed.txt        # 5-seed mean ± 95% CI + raw values
    │   ├── track_a_pca.txt          # PCA cumulative variance + softmax at m dims
    │   └── failure_analysis.txt     # H=2 bottleneck results + explanation
    │
    ├── report/                      # LaTeX report (ACM sigconf format)
    │   ├── main.tex                 # Final report source
    │   ├── math4ai_final_project_report.pdf  # Compiled report
    │
    └── slides/                    # Presentation (Beamer)
        ├── presentation.tex       # Beamer slides source 
        └── math4ai_final_project_presentation.pdf  # Compiled presentation
```

## Key Results

### Synthetic Tasks
| Dataset | Softmax Acc | NN (h=32) Acc | Takeaway |
|---------|------------|---------------|----------|
| **Linear Gaussian** | 95.00% | 95.00% | Linear model suffices — Bayes boundary is a hyperplane |
| **Moons** | 83.75% | 97.50% | Nonlinearity essential — hidden layer "untwists" the arcs |

### Digits Benchmark (5-seed, 95% CI)
| Model | Test Accuracy | Test CE |
|-------|--------------|---------|
| Softmax Regression | 93.26% ± 0.73% | 0.2692 ± 0.0067 |
| Neural Network (h=32) | **95.71% ± 0.91%** | **0.1566 ± 0.0162** |

### Optimizer Study (Digits, NN h=32)
| Optimizer | Test Acc | Test CE | Best Epoch |
|-----------|---------|---------|------------|
| SGD (lr=0.05) | 94.84% | 0.1700 | 197 |
| Momentum (lr=0.05, μ=0.9) | **95.38%** | 0.1653 | 197 |
| Adam (lr=0.001) | 95.11% | **0.1507** | 197 |

### Failure Case (h=2 on Digits)
NN with h=2 achieves only **58.97%** — a 2D bottleneck destroys 64D information, demonstrating that **representational capacity must match task complexity**.

### Track A: PCA/SVD
- Top 20 PCs capture **89.6%** of variance
- Softmax at m=20 achieves **94.08%** val accuracy (only 1.4 ppt below full 64-dim)
- Neural network at full 64 dims (**95.71%**) still outperforms any PCA level → nonlinear features add value

## Implementation Notes

- **No ML frameworks** — all models, optimizers, and training loops implemented from scratch with NumPy
- **Numerically stable softmax** — row-max subtraction exploiting shift-invariance
- **Validation checkpointing** — saves best parameters at lowest validation cross-entropy
- **Xavier initialization** — scale = √(2 / (fan_in + fan_out)) for gradient stability
- **Reproducibility** — all experiments use fixed random seeds
- **5 Sanity Checks** — probability normalization, monotonic loss decrease, numerical gradient check (max error < 10⁻¹¹), NaN/Inf detection, micro-dataset overfit

## License

This project is an academic capstone for the National AI Center — AI Academy Math4AI course, April 2026.
