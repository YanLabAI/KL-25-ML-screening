"""Calculate sequence-derived peptide descriptors using modlAMP and ProPy."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from modlamp.descriptors import GlobalDescriptor
from propy.PyPro import GetProDes
from tqdm.auto import tqdm

from common import clean_sequence, ensure_dir, read_csv_checked, save_json, sequence_is_standard

# Compatibility aliases for older descriptor packages when NumPy removes legacy names.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


def _propy_descriptor(sequence: str) -> dict:
    try:
        obj = GetProDes(sequence)
        features = {"sequence": sequence}
        features.update(obj.GetAAComp())
        features.update(obj.GetDPComp())
        features.update(obj.GetTPComp())
        features.update(obj.GetMoreauBrotoAuto())
        features.update(obj.GetMoranAuto())
        features.update(obj.GetGearyAuto())
        features.update(obj.GetCTD())
        features.update(obj.GetSOCN())
        features.update(obj.GetQSO())
        features.update(obj.GetPAAC(lamda=10, weight=0.05))
        features.update(obj.GetAPAAC(lamda=10, weight=0.05))
        features["descriptor_error"] = ""
        return features
    except Exception as exc:  # preserve the row rather than shifting downstream data
        return {"sequence": sequence, "descriptor_error": str(exc)}


def compute_descriptors(sequences: Sequence[str], n_jobs: int | None = None) -> pd.DataFrame:
    """Return the modlAMP + ProPy descriptor table for peptide sequences."""
    seqs = [clean_sequence(seq) for seq in sequences]
    if not seqs:
        return pd.DataFrame(columns=["sequence"])

    invalid = [i for i, seq in enumerate(seqs) if not sequence_is_standard(seq)]
    if invalid:
        preview = ", ".join(str(i) for i in invalid[:10])
        raise ValueError(
            f"Found {len(invalid)} empty/non-standard sequence(s) at zero-based row(s): {preview}. "
            "Preprocess sequences to the 20 standard amino-acid one-letter codes before descriptor calculation."
        )

    global_desc = GlobalDescriptor(seqs)
    global_desc.calculate_all()
    modlamp_df = pd.DataFrame(global_desc.descriptor, columns=global_desc.featurenames)
    modlamp_df.insert(0, "sequence", seqs)
    modlamp_df = modlamp_df.drop(columns=["Antifungal"], errors="ignore")

    if n_jobs is None:
        n_jobs = max(1, int((os.cpu_count() or 1) * 0.8))
    propy_rows = Parallel(n_jobs=n_jobs)(
        delayed(_propy_descriptor)(seq)
        for seq in tqdm(seqs, desc="ProPy descriptors", ncols=90)
    )
    propy_df = pd.DataFrame(propy_rows)

    combined = pd.concat(
        [modlamp_df.reset_index(drop=True), propy_df.drop(columns=["sequence"], errors="ignore").reset_index(drop=True)],
        axis=1,
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing peptide sequences")
    parser.add_argument("--output", required=True, help="Output descriptor CSV")
    parser.add_argument("--sequence-col", default="sequence", help="Sequence column name")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers for ProPy")
    args = parser.parse_args()

    df = read_csv_checked(args.input, [args.sequence_col])
    sequences = df[args.sequence_col].map(clean_sequence)
    descriptors = compute_descriptors(sequences.tolist(), n_jobs=args.n_jobs)

    # Preserve non-sequence metadata columns by row position.
    metadata = df.drop(columns=[args.sequence_col]).reset_index(drop=True)
    out = pd.concat([descriptors.reset_index(drop=True), metadata], axis=1)
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)

    errors = int(out.get("descriptor_error", pd.Series(dtype=str)).fillna("").astype(str).ne("").sum())
    save_json(
        {
            "input": str(Path(args.input)),
            "output": str(out_path),
            "n_sequences": int(len(out)),
            "descriptor_errors": errors,
        },
        out_path.with_suffix(".metadata.json"),
    )
    print(f"Saved {len(out):,} descriptor rows to {out_path}")


if __name__ == "__main__":
    main()
