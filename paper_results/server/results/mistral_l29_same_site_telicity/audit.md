# Mistral L29 same-site Telicity audit

## AUDIT = PASS

The Telicity and frozen causal decision directions occupy the same zero-based L29 decoder-block output at the answer position.

## Checks

- [x] five_folds
- [x] all_fold_audits_PASS
- [x] rows_1200
- [x] missing_jobs_0
- [x] same_site_L29

## Results

See `magnitude_summary.csv`, `cosine_summary.csv`, `model_summary.csv`, `decision_telicity_comparison.csv`, and `publication_row.csv`.

Telicity direction: outer-training C/D-only class-balanced dual ridge (alpha=.01), C-positive, unit-L2. Projection SD uses ddof=0.

- Held-out Telicity balanced accuracy by fold: 0.975; 0.975; 0.975; 0.925; 0.925.
- Group C magnitude ratio (Decision/Telicity): 0.907521; Group D: 0.808503.
- Mean fold signed cosine: -0.001486.
- Group C Telicity G: -0.058750 [-0.063594, -0.053906]; random-adjusted -0.058438 [-0.065469, -0.051406].
- Group D Telicity G: -0.047969 [-0.053125, -0.042656]; random-adjusted -0.055000 [-0.062344, -0.047188].
- Same-site absolute Decision/Telicity G ratio: C 35.787, D 40.765.
- Legacy table-definition pooled C/D max-one-sided ratio: 45.868.
