# Math4AI Starter Pack

This starter pack is intentionally minimal. Its job is to remove setup noise, not to reduce the intellectual core of the capstone.

## Included

- `data/digits_data.npz`, containing the fixed digits feature matrix and label vector
- `data/digits_split_indices.npz`, containing the fixed train/validation/test indices
- `scripts/make_digits_split.py`, a deterministic split-generation script
- `data/linear_gaussian.npz` for the linear synthetic task
- `data/moons.npz` for the nonlinear synthetic task
- `scripts/generate_synthetic.py`, which regenerates the two synthetic datasets
- a minimal repository skeleton
- `starter_pack/README.md` and `starter_pack/CHECKLIST.md`
- an optional LaTeX report template at `report/template.tex`

## Not Included

- softmax regression code
- neural network code
- model skeletons or placeholder methods
- training-loop scaffolding
- optimizer implementations
- optimization utilities beyond basic data preparation
- plotting code for experiments
- any hidden implementation of the core methods

Students are expected to implement the intellectually central parts of the project themselves.

## Team

Replace this section with your team members and high-level responsibilities.

- Name 1: role
- Name 2: role
- Name 3: role

## Suggested Repository Layout

- `data/`: provided data, split indices, and any team-generated derived arrays
- `scripts/`: deterministic starter-pack utilities
- `src/`: model, training, and evaluation code written by the team
- `figures/`: plots for the report and slides
- `results/`: saved experiment outputs and summary tables
- `report/`: final PDF report and source files
- `slides/`: presentation materials

The starter pack leaves `src/` intentionally empty except for guidance text. Begin by reading `starter_pack/README.md` and `starter_pack/CHECKLIST.md`.
