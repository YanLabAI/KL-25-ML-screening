"""Train RF, SVR, XGBoost and stacking models for target-specific pMIC prediction."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from scipy.linalg import pinv
from scipy.stats import chi2, probplot
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from common import ensure_dir, kennard_stone, numeric_feature_frame, read_csv_checked, regression_metrics, safe_dump, save_json, usable_cores

optuna.logging.set_verbosity(optuna.logging.WARNING)


def rf_objective(trial, X, y, cv, n_jobs):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 200),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_float("max_features", 0.1, 0.9),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }
    if params["bootstrap"]:
        params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
    model = Pipeline([("scale", StandardScaler()), ("model", RandomForestRegressor(**params, random_state=42, n_jobs=n_jobs))])
    return float(cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=1).mean())


def svr_objective(trial, X, y, cv):
    params = {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
    }
    if params["kernel"] == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    model = Pipeline([("scale", StandardScaler()), ("model", SVR(**params, max_iter=50000))])
    return float(cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=1).mean())


def xgb_objective(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = Pipeline([("scale", StandardScaler()), ("model", XGBRegressor(**params, random_state=42, n_jobs=1, objective="reg:squarederror"))])
    return float(cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=1).mean())


def optimize(name, X, y, n_trials, cv, n_jobs, random_state):
    if name == "rf":
        objective = lambda trial: rf_objective(trial, X, y, cv, n_jobs)
    elif name == "svr":
        objective = lambda trial: svr_objective(trial, X, y, cv)
    elif name == "xgb":
        objective = lambda trial: xgb_objective(trial, X, y, cv)
    else:
        raise ValueError(name)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state, n_startup_trials=min(20, max(5, n_trials // 5))))
    study.optimize(objective, n_trials=n_trials, n_jobs=min(4, n_jobs), show_progress_bar=True)
    return study


def make_model(name: str, params: dict, n_jobs: int, random_state: int):
    if name == "rf":
        estimator = RandomForestRegressor(**params, random_state=random_state, n_jobs=n_jobs)
    elif name == "svr":
        estimator = SVR(**params, max_iter=50000)
    elif name == "xgb":
        estimator = XGBRegressor(**params, random_state=random_state, n_jobs=n_jobs, objective="reg:squarederror")
    else:
        raise ValueError(name)
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def train_confidence_model(X_train: pd.DataFrame, y_train: pd.Series, max_components: int = 10) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    n_comp = min(max_components, X_scaled.shape[1], max(1, X_scaled.shape[0] - 1))
    pls = PLSRegression(n_components=n_comp)
    projected = pls.fit_transform(X_scaled, y_train)[0]
    mean = np.mean(projected, axis=0)
    cov = np.atleast_2d(np.cov(projected, rowvar=False))
    eps = 1e-6 * (np.trace(cov) / max(1, cov.shape[0]) if np.trace(cov) else 1.0)
    cov_inv = pinv(cov + eps * np.eye(cov.shape[0]))
    diff = projected - mean
    md = np.sqrt(np.clip(np.sum((diff @ cov_inv) * diff, axis=1), 0, None))
    max_mdist = float(np.percentile(md, 99))
    return {"scaler": scaler, "pls": pls, "mean": mean, "cov_inv": cov_inv, "max_mdist": max_mdist, "n_components": n_comp, "train_mahalanobis": md}


def plot_fit(y_true, y_pred, title, path):
    metrics = regression_metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k")
    lo = min(float(np.min(y_true)), float(np.min(y_pred))); hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed pMIC"); ax.set_ylabel("Predicted pMIC")
    ax.set_title(f"{title}\nR²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Target-specific selected-feature CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="pMIC")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--models", nargs="+", choices=["rf", "svr", "xgb", "stacking"], default=["rf", "svr", "xgb", "stacking"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    n_jobs = args.n_jobs or usable_cores()
    df = read_csv_checked(args.input, [args.target]).dropna().reset_index(drop=True)
    X = numeric_feature_frame(df, [args.target, args.sequence_col, "descriptor_error"])
    y = pd.to_numeric(df[args.target], errors="raise")
    feature_columns = list(X.columns)
    train_idx, test_idx = kennard_stone(X, ratio=args.train_ratio)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    out_dir = ensure_dir(args.output_dir)
    save_json({"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(), "split_method": "Kennard-Stone"}, out_dir / "ks_split.json")
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)

    best_params: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    metrics_summary: dict[str, dict] = {}

    for name in [m for m in ["rf", "svr", "xgb"] if m in args.models or "stacking" in args.models]:
        print(f"Optimizing {name.upper()} ({args.n_trials} trials)...")
        study = optimize(name, X_train, y_train, args.n_trials, cv, n_jobs, args.random_state)
        best_params[name] = dict(study.best_params)
        model = make_model(name, best_params[name], n_jobs, args.random_state)
        model.fit(X_train, y_train)
        fitted[name] = model
        pred_train = model.predict(X_train); pred_test = model.predict(X_test)
        metrics_summary[name] = {
            "best_cv_r2": float(study.best_value),
            "best_params": best_params[name],
            "train": regression_metrics(y_train, pred_train),
            "test": regression_metrics(y_test, pred_test),
        }
        safe_dump(model, out_dir / f"{name}_pmic_model.joblib")
        plot_fit(y_test, pred_test, f"{name.upper()} test set", out_dir / f"{name}_test_fit.png")

    if "stacking" in args.models:
        estimators = [(name, make_model(name, best_params[name], n_jobs, args.random_state)) for name in ["rf", "svr", "xgb"]]
        stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=args.cv, n_jobs=n_jobs, passthrough=False)
        stacking.fit(X_train, y_train)
        fitted["stacking"] = stacking
        pred_train = stacking.predict(X_train); pred_test = stacking.predict(X_test)
        metrics_summary["stacking"] = {"train": regression_metrics(y_train, pred_train), "test": regression_metrics(y_test, pred_test)}
        safe_dump(stacking, out_dir / "stacking_pmic_model.joblib")
        plot_fit(y_test, pred_test, "Stacking test set", out_dir / "stacking_test_fit.png")

    confidence = train_confidence_model(X_train, y_train)
    conf_dir = ensure_dir(out_dir / "confidence")
    safe_dump(confidence["scaler"], conf_dir / "confidence_scaler.joblib")
    safe_dump(confidence["pls"], conf_dir / "pls_confidence_model.joblib")
    safe_dump({k: v for k, v in confidence.items() if k not in {"scaler", "pls", "train_mahalanobis"}}, conf_dir / "pls_confidence_stats.joblib")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(confidence["train_mahalanobis"], bins=30, alpha=0.8)
    ax.axvline(confidence["max_mdist"], linestyle="--", label=f"99th percentile = {confidence['max_mdist']:.3f}")
    ax.set_xlabel("Mahalanobis distance"); ax.set_ylabel("Count"); ax.legend()
    fig.tight_layout(); fig.savefig(conf_dir / "confidence_distribution.png", dpi=300); plt.close(fig)

    # Repeat evaluation on the fixed KS split using different model seeds where applicable.
    rng = np.random.default_rng(args.random_state)
    repeat_rows = []
    for model_name, params in best_params.items():
        for repeat, seed in enumerate(rng.integers(0, 1_000_000, size=args.repeats), start=1):
            model = make_model(model_name, params, n_jobs, int(seed))
            model.fit(X_train, y_train)
            m = regression_metrics(y_test, model.predict(X_test))
            repeat_rows.append({"model": model_name, "repeat": repeat, "seed": int(seed), **m})
    if repeat_rows:
        pd.DataFrame(repeat_rows).to_csv(out_dir / "fixed_KS_repeated_evaluation.csv", index=False)

    # Bundles combine a prediction model with the independent PLS/Mahalanobis confidence model.
    for model_name, model in fitted.items():
        bundle = {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": args.target,
            "confidence_scaler": confidence["scaler"],
            "confidence_pls": confidence["pls"],
            "confidence_mean": confidence["mean"],
            "confidence_cov_inv": confidence["cov_inv"],
            "confidence_max_mdist": confidence["max_mdist"],
        }
        safe_dump(bundle, out_dir / f"{model_name}_pmic_bundle.joblib")

    save_json(metrics_summary, out_dir / "pmic_metrics.json")
    save_json({"feature_columns": feature_columns, "best_params": best_params}, out_dir / "pmic_model_metadata.json")
    print(f"pMIC models and confidence components saved to {out_dir}")


if __name__ == "__main__":
    main()
