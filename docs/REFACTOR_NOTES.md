# Public-code refactor notes

The supplied workstation script was reorganized into reusable command-line modules without changing the scientific workflow categories. The public package retains:

- modlAMP + ProPy peptide descriptor calculation;
- PLS regression and VIP > 1 feature selection;
- Random Forest, SVM, XGBoost and soft-voting AMP classification;
- Kennard-Stone splitting for pMIC regression;
- Random Forest, SVR, XGBoost and stacking pMIC models;
- Optuna/TPE hyperparameter optimization for pMIC regression;
- PLS-space Mahalanobis distance for prediction-confidence / applicability-domain assessment;
- batch virtual screening of peptide sequences.

## Changes made for a public repository

1. All hard-coded local `E:/...` paths were removed and replaced by command-line arguments.
2. Repeated helper functions were consolidated.
3. Output files are written to user-specified directories instead of personal workstation folders.
4. Model bundles include the feature-column order needed at prediction time.
5. Cross-validation uses pipeline-local scaling, which prevents validation folds from contributing to fitted scaling parameters. Because the original workstation script sometimes scaled the whole training set before internal CV, refactored CV metrics can differ slightly from earlier exploratory runs.
6. The unused RDKit dependency was removed. The original classification code imported `rdkit.Chem.BRICS.labels` for confusion-matrix labels; the public implementation uses the binary class labels directly.
7. Descriptor failures preserve row alignment rather than silently shifting ProPy rows relative to modlAMP rows.
8. No source databases or experimental data are included.

## Important publication check

Before citing a release, confirm that the final manuscript's preprocessing rules, model settings, target-specific datasets, and reported numerical metrics correspond to the released code version. Tag the verified state (for example, `v1.0.0`) and archive that release if a permanent DOI is required.
