# KL-25-ML-screening

Machine-learning-assisted QSAR workflow for screening antimicrobial peptides targeting ammonia-associated bacteria and identifying KL-25 for ammonia mitigation in laying hens.

## Overview

This repository contains the **machine-learning source code only** for the KL-25 antimicrobial-peptide discovery workflow. It does not redistribute the original AMP databases, target-specific pMIC datasets, UniProt screening sequences, animal data, fermentation data, or microbiome data.

The computational workflow is organized into five stages:

1. peptide sequence descriptor calculation with **modlAMP** and **ProPy**;
2. **PLS-VIP** feature selection;
3. antimicrobial-peptide classification with **Random Forest, SVM, XGBoost, and a soft-voting ensemble**;
4. target-specific pMIC modeling using a **Kennard-Stone** train/test split and **Random Forest, SVR, XGBoost, and stacking regression**;
5. virtual screening with model-based predictions plus a **PLS/Mahalanobis-distance** confidence assessment.

The large-scale screening described in the associated study evaluated 3,050,242 UniProt sequences and yielded 22,436 computational candidates before experimental prioritization.

## Repository structure

```text
KL-25-ML-screening/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── REFACTOR_NOTES.md
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── src/
    ├── common.py
    ├── descriptor_calculation.py
    ├── feature_selection.py
    ├── amp_classification.py
    ├── pmic_regression.py
    └── virtual_screening.py
```

## Installation

Python 3.9-3.11 is recommended for compatibility with the peptide-descriptor packages.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Descriptor calculation

Input: a CSV containing a `sequence` column with peptide sequences represented by standard amino-acid one-letter codes.

```bash
python src/descriptor_calculation.py \
  --input path/to/sequences.csv \
  --output results/descriptors.csv
```

The script calculates the same descriptor families represented in the original analysis code: modlAMP global descriptors and ProPy composition, dipeptide/tripeptide composition, autocorrelation, CTD, sequence-order, QSO, PAAC, and APAAC descriptors.

## 2. PLS-VIP feature selection

For AMP classification:

```bash
python src/feature_selection.py \
  --input results/classification_descriptors.csv \
  --output-dir results/classification_vip \
  --target Antibacterial \
  --vip-threshold 1.0
```

For a target-specific pMIC dataset:

```bash
python src/feature_selection.py \
  --input results/target_pmic_descriptors.csv \
  --output-dir results/target_pmic_vip \
  --target pMIC \
  --vip-threshold 1.0
```

The selected dataset is written as `selected_features_dataset.csv`.

## 3. AMP classification

The classification stage expects **separate training and independent test CSVs**, each containing `sequence`, numeric selected features, and a binary `Antibacterial` column.

```bash
python src/amp_classification.py \
  --train path/to/classification_train.csv \
  --test path/to/classification_test.csv \
  --output-dir results/classification_models \
  --search-profile paper
```

`--search-profile paper` uses the parameter grids carried over from the original analysis script. Because the XGBoost grid is large, a reduced `--search-profile quick` is provided only for installation checks and smoke tests; it should not be used to represent manuscript results unless explicitly intended.

Main outputs include:

- `rf_classifier.joblib`
- `svm_classifier.joblib`
- `xgb_classifier.joblib`
- `ensemble_classifier.joblib`
- `classification_bundle.joblib`
- `classification_metrics.json`
- `model_comparison.csv`

## 4. Target-specific pMIC regression

Run this stage independently for each ammonia-associated bacterial target.

```bash
python src/pmic_regression.py \
  --input results/target_pmic_vip/selected_features_dataset.csv \
  --output-dir results/target_name \
  --target pMIC \
  --models rf svr xgb stacking \
  --n-trials 100
```

The script uses an 80:20 Kennard-Stone split by default, tunes RF/SVR/XGBoost with Optuna, evaluates R²/RMSE/MAE, builds an optional stacking model, and trains a PLS-space Mahalanobis-distance confidence model.

Prediction-ready files include:

- `rf_pmic_bundle.joblib`
- `svr_pmic_bundle.joblib`
- `xgb_pmic_bundle.joblib`
- `stacking_pmic_bundle.joblib`

## 5. Virtual screening

Virtual screening can first apply the AMP classifier and then predict pMIC for one or more bacterial targets.

```bash
python src/virtual_screening.py \
  --input path/to/uniprot_peptides.csv \
  --output results/virtual_screening.csv \
  --classification-bundle results/classification_models/classification_bundle.joblib \
  --pmic-bundle Target_1=results/target_1/xgb_pmic_bundle.joblib \
  --pmic-bundle Target_2=results/target_2/xgb_pmic_bundle.joblib \
  --chunksize 1000
```

Add another `--pmic-bundle NAME=PATH` for each bacterial target. The output contains AMP probabilities/predictions and, when pMIC bundles are supplied, target-specific predicted pMIC values and confidence levels.

## Data sources

AMP source records used to construct the study dataset were derived from **CAMP, DBAASP, and DRAMP**. Original database exports are not redistributed in this code repository. See `data/README.md` for the expected input schemas.

## Reproducibility note

This public version replaces workstation-specific paths and consolidates repeated analysis blocks. It also uses fold-local preprocessing during cross-validation to avoid validation-fold leakage. These engineering changes can produce small numerical differences from exploratory workstation runs. See `docs/REFACTOR_NOTES.md` before matching a manuscript table or figure to a release.

## Citation

If you use this code, please cite the associated article. Add the final article citation here after acceptance/publication, and optionally add a `CITATION.cff` file before creating the archived release.

## License

MIT License.
