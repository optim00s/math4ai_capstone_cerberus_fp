# Starter Pack Checklist

Use this checklist before you begin substantive work.

## Starter Pack Integrity Check

- Verify that `data/digits_data.npz` is present.
- Verify that `data/digits_split_indices.npz` is present.
- Verify that `data/linear_gaussian.npz` and `data/moons.npz` can be regenerated with `scripts/generate_synthetic.py`.
- Verify that `scripts/make_digits_split.py` reproduces the published split indices.
- Verify that `report/template.tex` compiles if you plan to use LaTeX.
- Verify that there is no model implementation code anywhere in the starter pack.

## Student Setup Checklist

- Read the assignment handout carefully before writing code.
- Clone the repository and confirm the folder structure.
- Open `starter_pack/README.md` and add team names and roles.
- Read `starter_pack/CHECKLIST.md` once before writing code.
- Create branches before beginning implementation work.
- Inspect the provided data files and confirm shapes.
- Decide which team member owns:
  - softmax regression
  - neural network
  - experiment orchestration and plots
  - report and presentation integration
- Choose one advanced analysis track only.
- Plan a short implementation sanity-check section for the report.
- Decide which verification checks you will document, such as a gradient check, probability-sum check, tiny-subset loss decrease, tiny-subset overfitting, or NaN/Inf checks.

## Submission Checklist

- Repository contains meaningful commit history.
- Every team member has at least one merged feature branch.
- README explains setup and reproduction clearly.
- Report includes a limitations subsection.
- Slides are aligned with the report's main claim.
- All team members are prepared to speak during the presentation.
