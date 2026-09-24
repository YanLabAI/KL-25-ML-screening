"""Train RF, SVM, XGBoost and soft-voting models for AMP classification."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import classification_metrics, ensure_dir, numeric_feature_frame, read_csv_checked, safe_dump, save_json


def parameter_grids(profile: str = "paper") -> dict:
    if profile == "quick":
        return {
            "rf": {"model__n_estimators": [300], "model__max_depth": [17, 19], "model__max_features": ["sqrt"], "model__min_samples_split": [2], "model__min_samples_leaf": [1], "model__bootstrap": [False]},
            "svm": {"model__C": [1, 100], "model__gamma": [1e-4, 1e-2], "model__kernel": ["rbf"]},
            "xgb": {"model__n_estimators": [300], "model__max_depth": [4, 8], "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8], "model__colsample_bytree": [0.8], "model__gamma": [0, 1], "model__min_child_weight": [1, 3], "model__reg_alpha": [0, 0.5], "model__reg_lambda": [1.0]},
        }
    return {
        "rf": {
            "model__n_estimators": [500, 520, 540, 560, 580, 600],
            "model__max_depth": [16, 17, 18, 19],
            "model__max_features": ["sqrt"],
            "model__min_samples_split": [2, 3, 4],
            "model__min_samples_leaf": [1, 2],
            "model__bootstrap": [False],
        },
        "svm": {
            "model__C": [0.1, 1, 10, 100, 1000, 10000, 100000],
            "model__gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
            "model__kernel": ["rbf"],
        },
        "xgb": {
            "model__n_estimators": [200, 300, 400, 500, 600],
            "model__max_depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__gamma": [0, 0.5, 1, 2],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_alpha": [0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 1.5, 2.0],
        },
    }


def base_pipelines(random_state: int) -> dict[str, Pipeline]:
    return {
        "rf": Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=random_state, criterion="entropy"))]),
        "svm": Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, random_state=random_state))]),
        "xgb": Pipeline([("scale", StandardScaler()), ("model", xgb.XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=1))]),
    }


def evaluate_model(model, X, y) -> tuple[dict, np.ndarray, np.ndarray]:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return classification_metrics(y, pred, prob), pred, prob


def save_diagnostics(name: str, y_true, y_pred, y_prob, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(cm, interpolation="nearest")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"{name} confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=300); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {classification_metrics(y_true, y_pred, y_prob)['auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{name} ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_dir / f"{name}_roc.png", dpi=300); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training CSV with selected numeric features")
    parser.add_argument("--test", required=True, help="Independent test CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="Antibacterial")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--search-profile", choices=["paper", "quick"], default="paper", help="paper reproduces original search spaces; quick is a smoke-test profile")
    parser.add_argument("--inner-cv", type=int, default=3)
    parser.add_argument("--outer-cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_df = read_csv_checked(args.train, [args.label_col])
    test_df = read_csv_checked(args.test, [args.label_col])
    X_train = numeric_feature_frame(train_df, [args.label_col, args.sequence_col, "descriptor_error"])
    X_test = numeric_feature_frame(test_df, [args.label_col, args.sequence_col, "descriptor_error"])
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns/order do not match")
    y_train = train_df[args.label_col].astype(int)
    y_test = test_df[args.label_col].astype(int)

    out_dir = ensure_dir(args.output_dir)
    grids = parameter_grids(args.search_profile)
    models = base_pipelines(args.random_state)
    inner_cv = StratifiedKFold(n_splits=args.inner_cv, shuffle=True, random_state=args.random_state)
    outer_cv = StratifiedKFold(n_splits=args.outer_cv, shuffle=True, random_state=args.random_state)

    fitted: dict[str, Pipeline] = {}
    summary: dict[str, dict] = {}
    for name in ["rf", "svm", "xgb"]:
        print(f"Tuning {name.upper()}...")
        search = GridSearchCV(models[name], grids[name], cv=inner_cv, scoring="accuracy", n_jobs=args.n_jobs, verbose=1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best
        test_metrics, pred, prob = evaluate_model(best, X_test, y_test)
        cv = cross_validate(
            clone(best), X_train, y_train, cv=outer_cv,
            scoring={"accuracy": "accuracy", "f1": "f1_weighted", "auc": "roc_auc"},
            n_jobs=args.n_jobs,
        )
        summary[name] = {
            "best_params": search.best_params_,
            "inner_cv_best_accuracy": float(search.best_score_),
            "test": test_metrics,
            "outer_cv_mean": {key.replace("test_", ""): float(np.mean(value)) for key, value in cv.items() if key.startswith("test_")},
            "outer_cv_sd": {key.replace("test_", ""): float(np.std(value)) for key, value in cv.items() if key.startswith("test_")},
        }
        safe_dump(best, out_dir / f"{name}_classifier.joblib")
        save_diagnostics(name.upper(), y_test, pred, prob, out_dir)

    ensemble = VotingClassifier(
        estimators=[("rf", fitted["rf"]), ("svm", fitted["svm"]), ("xgb", fitted["xgb"])],
        voting="soft",
        n_jobs=args.n_jobs,
    )
    ensemble.fit(X_train, y_train)
    ensemble_metrics, pred, prob = evaluate_model(ensemble, X_test, y_test)
    summary["ensemble"] = {"test": ensemble_metrics}
    safe_dump(ensemble, out_dir / "ensemble_classifier.joblib")
    save_diagnostics("Ensemble", y_test, pred, prob, out_dir)

    feature_columns = list(X_train.columns)
    safe_dump(feature_columns, out_dir / "classification_feature_columns.joblib")
    safe_dump({"model": ensemble, "feature_columns": feature_columns, "label_column": args.label_col}, out_dir / "classification_bundle.joblib")
    save_json(summary, out_dir / "classification_metrics.json")
    pd.DataFrame({name: info["test"] for name, info in summary.items() if "test" in info}).T.to_csv(out_dir / "model_comparison.csv")
    print(f"Classification models saved to {out_dir}")


if __name__ == "__main__":
    main()
