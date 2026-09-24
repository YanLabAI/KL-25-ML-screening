# Data availability and input formats

This repository intentionally contains **machine-learning code only**. The original AMP database exports, target-specific pMIC datasets, UniProt screening sequences, and experimental datasets are not redistributed here.

The model-development workflow used AMP records derived from the CAMP, DBAASP, and DRAMP resources supplied for the study. Users should obtain source records from the respective database providers and apply the same study-specific inclusion, cleaning, deduplication, labeling, and train/test preparation steps described in the associated manuscript.

## Required CSV schemas

### Descriptor calculation input
At minimum:

```text
sequence
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK
...
```

Additional columns are preserved by row position.

### Classification input
After descriptor calculation and feature selection, the training and independent test CSVs must contain:

```text
sequence,<numeric feature columns...>,Antibacterial
```

`Antibacterial` is expected to be binary (0/1).

### pMIC regression input
Each bacterial target should be processed as a separate dataset:

```text
sequence,<numeric feature columns...>,pMIC
```

Run `pmic_regression.py` independently for each bacterial target.
