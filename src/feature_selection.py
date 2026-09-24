"""PLS-VIP feature selection for AMP classification or pMIC regression data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from common import ensure_dir, numeric_feature_frame, read_csv_checked, safe_dump, save_json


def calculate_vip(pls_model: PLSRegression) -> np.ndarray:
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p = w.shape[0]
    h = t.shape[1]
    vip = np.zeros(p, dtype=float)
    for i in range(p):
        numerator = 0.0
        denominator = 0.0
        for j in range(h):
            norm = np.linalg.norm(w[:, j])
            weight = 0.0 if norm == 0 else (w[i, j] / norm) ** 2
            ss = np.sum(t[:, j] ** 2) * q[0, j] ** 2
            numerator += weight * ss
            denominator += ss
        vip[i] = np.sqrt(p * numerator / denominator) if denominator > 0 else 0.0
    return vip


def select_features(
    df: pd.DataFrame,
    target_col: str,
    sequence_col: str = "sequence",
    vip_threshold: float = 1.0,
    max_components: int = 20,
    repeats: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X = numeric_feature_frame(df, exclude=[target_col, sequence_col, "descriptor_error"])
    y = pd.to_numeric(df[target_col], errors="raise")

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing feature values detected. Clean descriptor errors before selection: {bad[:20]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    upper = min(max_components, X_scaled.shape[1], max(1, X_scaled.shape[0] - 1))
    rng = np.random.default_rng(random_state)
    mse_values: list[float] = []

    for n_components in range(1, upper + 1):
        scores = []
        for _ in range(repeats):
            indices = rng.permutation(X_scaled.shape[0])
            split = max(2, min(X_scaled.shape[0] - 1, int(0.8 * X_scaled.shape[0])))
            train_idx, val_idx = indices[:split], indices[split:]
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_scaled[train_idx], y.iloc[train_idx])
            pred = pls.predict(X_scaled[val_idx]).ravel()
            scores.append(float(np.mean((y.iloc[val_idx].to_numpy() - pred) ** 2)))
        mse_values.append(float(np.mean(scores)))

    optimal_components = int(np.argmin(mse_values) + 1)
    pls = PLSRegression(n_components=optimal_components)
    pls.fit(X_scaled, y)
    vip = calculate_vip(pls)

    vip_table = pd.DataFrame({"Feature": X.columns, "VIP_Score": vip}).sort_values("VIP_Score", ascending=False)
    selected_features = vip_table.loc[vip_table["VIP_Score"] > vip_threshold, "Feature"].tolist()
    selected = df[[sequence_col] + selected_features + [target_col]].copy() if sequence_col in df.columns else df[selected_features + [target_col]].copy()

    metadata = {
        "target_column": target_col,
        "sequence_column": sequence_col,
        "vip_threshold": float(vip_threshold),
        "optimal_pls_components": optimal_components,
        "n_input_features": int(X.shape[1]),
        "n_selected_features": int(len(selected_features)),
        "selected_features": selected_features,
        "component_mse": mse_values,
    }
    return selected, vip_table, {"metadata": metadata, "scaler": scaler, "pls": pls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Descriptor CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", required=True, help="Target column, e.g. Antibacterial or pMIC")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--vip-threshold", type=float, default=1.0)
    parser.add_argument("--max-components", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = read_csv_checked(args.input, [args.target])
    out_dir = ensure_dir(args.output_dir)
    selected, vip_table, fitted = select_features(
        df,
        target_col=args.target,
        sequence_col=args.sequence_col,
        vip_threshold=args.vip_threshold,
        max_components=args.max_components,
        repeats=args.repeats,
        random_state=args.random_state,
    )

    vip_table.to_csv(out_dir / "vip_scores.csv", index=False)
    vip_table[vip_table["VIP_Score"] > args.vip_threshold].to_csv(out_dir / "selected_features_vip.csv", index=False)
    selected.to_csv(out_dir / "selected_features_dataset.csv", index=False)
    safe_dump(fitted["scaler"], out_dir / "vip_scaler.joblib")
    safe_dump(fitted["pls"], out_dir / "vip_pls_model.joblib")
    save_json(fitted["metadata"], out_dir / "feature_selection_metadata.json")

    top = vip_table.head(min(30, len(vip_table))).sort_values("VIP_Score")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["Feature"], top["VIP_Score"])
    ax.axvline(args.vip_threshold, linestyle="--", linewidth=1)
    ax.set_xlabel("VIP score")
    ax.set_title("Top PLS-VIP features")
    fig.tight_layout()
    fig.savefig(out_dir / "vip_scores_top30.png", dpi=300)
    plt.close(fig)

    print(f"Selected {len(fitted['metadata']['selected_features'])} features; results saved to {out_dir}")


if __name__ == "__main__":
    main()
