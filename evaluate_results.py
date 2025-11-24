#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_common_patients(gt_dir: Path, pred_dir: Path):
    gt_files = {f.name for f in gt_dir.glob("*.json")}
    pred_files = {f.name for f in pred_dir.glob("*.json")}
    common = sorted(gt_files & pred_files)
    if not common:
        raise RuntimeError(f"No overlapping JSON filenames between {gt_dir} and {pred_dir}")
    return common


def load_arrays(
    gt_dir: Path,
    pred_dir: Path,
    common_files: List[str],
    horizon_years: float,
    time_key: str = "event_time_years",
    event_key: str = "event_observed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        y_true_bin   : shape (n,) binary labels for 'event within horizon'
        y_scores     : shape (n,) predicted risk score at horizon
        event_times  : shape (n,) event/censor times (for c-index)
        event_obsv   : shape (n,) event indicators (0/1)
    """
    y_true_bin = []
    y_scores = []
    event_times = []
    event_obsv = []

    horizon_key = f"risk_{int(horizon_years)}y" if float(horizon_years).is_integer() else None

    for fname in common_files:
        gt_path = gt_dir / fname
        pr_path = pred_dir / fname

        try:
            with gt_path.open("r", encoding="utf-8") as f:
                gt = json.load(f)
            with pr_path.open("r", encoding="utf-8") as f:
                pr = json.load(f)
        except Exception as e:
            print(f"Skipping {fname} due to JSON error: {e}")
            continue

        # Ground truth time + indicator
        try:
            t = float(gt[time_key])
            e = int(gt[event_key])
        except Exception as e:
            print(f"Skipping {fname}: missing or bad '{time_key}'/'{event_key}' in GT: {e}")
            continue

        # Predicted risk for this horizon
        try:
            if horizon_key is None:
                raise KeyError
            risk = float(pr["risk_estimates"][horizon_key])
        except Exception:
            print(f"Skipping {fname}: missing or bad predicted risk for key '{horizon_key}'")
            continue

        # Binary outcome: event within horizon?
        y_true = 1 if (e == 1 and t <= horizon_years) else 0

        y_true_bin.append(y_true)
        y_scores.append(risk)
        event_times.append(t)
        event_obsv.append(e)

    if not y_true_bin:
        raise RuntimeError("No valid samples after loading GT + predictions; check keys & schema.")

    return (
        np.array(y_true_bin, dtype=int),
        np.array(y_scores, dtype=float),
        np.array(event_times, dtype=float),
        np.array(event_obsv, dtype=int),
    )


def binary_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5):
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1, y_pred


def concordance_index(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_scores: np.ndarray,
) -> float:
    """
    Harrell's C-index for right-censored survival data.

    event_times     : time to event or censor
    event_observed  : 1 if event, 0 if censored
    predicted_scores: higher score = higher risk (worse outcome)
    """
    n = len(event_times)
    assert len(event_observed) == n and len(predicted_scores) == n

    num_concordant = 0.0
    num_tied = 0.0
    num_pairs = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if event_times[i] == event_times[j] and event_observed[i] == event_observed[j]:
                # same time & same event/censor status -> not a comparable pair
                continue

            # Identify which patient has the shorter time (and if it's an event)
            if event_times[i] < event_times[j]:
                ti, ei, si = event_times[i], event_observed[i], predicted_scores[i]
                tj, ej, sj = event_times[j], event_observed[j], predicted_scores[j]
            else:
                ti, ei, si = event_times[j], event_observed[j], predicted_scores[j]
                tj, ej, sj = event_times[i], event_observed[i], predicted_scores[i]

            # Comparable only if the shorter time is actually an event (not censor)
            if ei == 0:
                continue

            num_pairs += 1.0
            if si == sj:
                num_tied += 1.0
            elif si > sj:
                # higher risk for earlier event -> concordant
                num_concordant += 1.0
            else:
                # discordant
                pass

    if num_pairs == 0:
        return np.nan

    cindex = (num_concordant + 0.5 * num_tied) / num_pairs
    return float(cindex)


def bootstrap_cindex_ci(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = len(event_times)
    if n < 2:
        return np.nan, np.nan

    c_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        c_boot = concordance_index(
            event_times[idx],
            event_observed[idx],
            predicted_scores[idx],
        )
        if not np.isnan(c_boot):
            c_vals.append(c_boot)

    if not c_vals:
        return np.nan, np.nan

    lower = np.percentile(c_vals, 100 * (alpha / 2.0))
    upper = np.percentile(c_vals, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy/precision/recall/F1 and c-index (with 95% CI) for risk predictions."
    )
    parser.add_argument("--gt_dir", required=True, help="Directory with ground-truth JSON files.")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted JSON files.")
    parser.add_argument(
        "--horizon_years",
        type=float,
        default=8.0,
        help="Risk time horizon in years (must match risk_<y>y key in predictions). Default: 8.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on predicted risk for binary classification. Default: 0.5.",
    )
    parser.add_argument(
        "--time_key",
        type=str,
        default="event_time_years",
        help="Key in GT JSON for time-to-event in years. Default: event_time_years",
    )
    parser.add_argument(
        "--event_key",
        type=str,
        default="event_observed",
        help="Key in GT JSON for event indicator (1=event, 0=censor). Default: event_observed",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for c-index CI. Default: 1000.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CI (default 0.05 = 95%% CI).",
    )

    args = parser.parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    common_files = load_common_patients(gt_dir, pred_dir)
    print(f"Found {len(common_files)} common patients to evaluate.")

    y_true, y_scores, event_times, event_obsv = load_arrays(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        common_files=common_files,
        horizon_years=args.horizon_years,
        time_key=args.time_key,
        event_key=args.event_key,
    )

    print(f"Using horizon = {args.horizon_years} years")
    print(f"Binary label: event_observed=1 & event_time <= {args.horizon_years} → 1, else 0")
    print(f"Predicted risk key: risk_{int(args.horizon_years)}y\n")

    # Binary metrics
    acc, prec, rec, f1, y_pred = binary_metrics(y_true, y_scores, threshold=args.threshold)
    print("=== Binary classification metrics ===")
    print(f"Threshold on risk: {args.threshold:.3f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    # c-index + 95% CI
    cidx = concordance_index(event_times, event_obsv, y_scores)
    lower, upper = bootstrap_cindex_ci(
        event_times,
        event_obsv,
        y_scores,
        n_bootstrap=args.bootstrap,
        alpha=args.alpha,
    )

    print("=== Concordance index (Harrell's C) ===")
    print(f"C-index : {cidx:.4f}" if not np.isnan(cidx) else "C-index : NaN (no comparable pairs)")
    print(f"{int((1-args.alpha)*100)}% CI: [{lower:.4f}, {upper:.4f}]"
          if not (np.isnan(lower) or np.isnan(upper)) else "95% CI : NaN (bootstrap failed)")


if __name__ == "__main__":
    main()
