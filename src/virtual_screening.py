"""Batch virtual screening from peptide sequences using saved classification and pMIC bundles."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import clean_sequence, ensure_dir
from descriptor_calculation import compute_descriptors


def confidence_from_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    scaled = bundle["confidence_scaler"].transform(X)
    projected = bundle["confidence_pls"].transform(scaled)
    mean = np.asarray(bundle["confidence_mean"])
    cov_inv = np.asarray(bundle["confidence_cov_inv"])
    diff = projected - mean
    md = np.sqrt(np.clip(np.sum((diff @ cov_inv) * diff, axis=1), 0, None))
    denom = max(float(bundle["confidence_max_mdist"]), np.finfo(float).eps)
    relative = md / denom
    return np.round((1.0 - np.tanh(relative)) * 100.0, 1)


def confidence_level(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 80, "high", np.where(values >= 50, "medium", "low"))


def parse_named_bundles(values: list[str]) -> dict[str, dict]:
    bundles = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --pmic-bundle must be NAME=PATH")
        name, path = value.split("=", 1)
        bundles[name] = joblib.load(path)
    return bundles


def align_features(descriptors: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in descriptors.columns]
    if missing:
        raise ValueError(f"Descriptor table is missing model feature(s): {missing[:20]}")
    return descriptors.reindex(columns=feature_columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing a sequence column")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--classification-bundle", default=None, help="classification_bundle.joblib; omit to skip AMP filtering")
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--pmic-bundle", action="append", default=[], metavar="NAME=PATH", help="Repeat for each bacterial target")
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    classifier_bundle = joblib.load(args.classification_bundle) if args.classification_bundle else None
    pmic_bundles = parse_named_bundles(args.pmic_bundle)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()

    wrote_header = False
    total_input = 0
    total_retained = 0
    reader = pd.read_csv(args.input, chunksize=args.chunksize)
    for chunk in tqdm(reader, desc="Screening batches"):
        if args.sequence_col not in chunk.columns:
            raise ValueError(f"Missing sequence column: {args.sequence_col}")
        chunk = chunk.copy()
        chunk[args.sequence_col] = chunk[args.sequence_col].map(clean_sequence)
        descriptors = compute_descriptors(chunk[args.sequence_col].tolist(), n_jobs=args.n_jobs)
        total_input += len(chunk)

        result = chunk.reset_index(drop=True).copy()
        keep = np.ones(len(result), dtype=bool)
        if classifier_bundle is not None:
            Xc = align_features(descriptors, classifier_bundle["feature_columns"])
            model = classifier_bundle["model"]
            prob = model.predict_proba(Xc)[:, 1]
            pred = (prob >= args.classification_threshold).astype(int)
            result["AMP_probability"] = prob
            result["AMP_prediction"] = pred
            keep = pred == 1

        result = result.loc[keep].reset_index(drop=True)
        descriptors_kept = descriptors.loc[keep].reset_index(drop=True)
        total_retained += len(result)
        if result.empty:
            continue

        for target_name, bundle in pmic_bundles.items():
            Xp = align_features(descriptors_kept, bundle["feature_columns"])
            result[f"{target_name}_predicted_pMIC"] = np.round(bundle["model"].predict(Xp), 3)
            conf = confidence_from_bundle(bundle, Xp)
            result[f"{target_name}_confidence_percent"] = conf
            result[f"{target_name}_confidence_level"] = confidence_level(conf)

        result.to_csv(output_path, index=False, mode="a", header=not wrote_header, encoding="utf-8")
        wrote_header = True

    if not wrote_header:
        pd.DataFrame(columns=[args.sequence_col]).to_csv(output_path, index=False)
    print(f"Screening complete: {total_input:,} input sequences; {total_retained:,} retained rows; output={output_path}")


if __name__ == "__main__":
    main()
