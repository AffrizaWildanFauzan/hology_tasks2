"""Blend the per-model OOF predictions and write a submission.

Uses Caruana-style greedy ensemble selection with replacement: pick the model
that most improves CV MAE, repeat. It is far more robust on 14k rows than
fitting a stacker, and it optimises the competition metric directly (MAE in
dollars) instead of a proxy loss in log space.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ARTIFACTS, from_log, load_data, load_oof, mae,
                    make_submission, price_bin_folds, to_log)


def available_models():
    names = []
    for p in sorted(glob.glob(os.path.join(ARTIFACTS, "oof_*.npy"))):
        name = os.path.basename(p)[4:-4]
        if os.path.exists(os.path.join(ARTIFACTS, f"test_{name}.npy")):
            names.append(name)
    return names


def greedy_blend(oofs: dict, y_raw: np.ndarray, n_iter: int = 40, verbose: bool = True):
    """Caruana greedy selection with replacement.

    It runs the full budget instead of stopping at the first non-improving step
    (a worse single step often unlocks a better combination) and returns the
    best combination seen along the way.
    """
    names = list(oofs)
    singles = {n: mae(y_raw, from_log(oofs[n])) for n in names}
    if verbose:
        for n, v in sorted(singles.items(), key=lambda kv: kv[1]):
            print(f"  single {n:18s} {v:,.0f}")

    best0 = min(singles, key=singles.get)
    current = oofs[best0].copy()
    total = 1
    counts = {n: 0 for n in names}
    counts[best0] = 1
    best_score, best_counts = singles[best0], dict(counts)

    for _ in range(n_iter):
        cand_name, cand_score = None, np.inf
        for n in names:
            s_n = mae(y_raw, from_log((current * total + oofs[n]) / (total + 1)))
            if s_n < cand_score:
                cand_name, cand_score = n, s_n
        current = (current * total + oofs[cand_name]) / (total + 1)
        total += 1
        counts[cand_name] += 1
        if cand_score < best_score - 1e-9:
            best_score, best_counts = cand_score, dict(counts)

    total_best = sum(best_counts.values())
    weights = {n: c / total_best for n, c in best_counts.items() if c}
    return weights, best_score


def stack_lgbm(oofs: dict, tests: dict, y_raw: np.ndarray, splits, extra_train=None, extra_test=None):
    """Level-2 LightGBM on the model predictions (MAE objective, log target).

    Can beat a linear blend because it learns *where* each model is reliable,
    e.g. kNN is strong on listings with near-duplicates and weak elsewhere.
    Scored with an inner CV so we only keep it if it actually wins.
    """
    import lightgbm as lgb

    names = sorted(oofs)
    X = np.column_stack([oofs[n] for n in names])
    X_te = np.column_stack([tests[n] for n in names])
    X = np.column_stack([X, X.std(axis=1), X.mean(axis=1)])
    X_te = np.column_stack([X_te, X_te.std(axis=1), X_te.mean(axis=1)])
    if extra_train is not None:
        X = np.column_stack([X, extra_train])
        X_te = np.column_stack([X_te, extra_test])

    y_log = to_log(y_raw)
    params = dict(objective="mae", metric="mae", learning_rate=0.02, num_leaves=15,
                  min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbosity=-1, seed=0)
    oof = np.zeros(len(y_log))
    test_pred = np.zeros(X_te.shape[0])
    for trn, val in splits:
        dtr = lgb.Dataset(X[trn], label=y_log[trn])
        dva = lgb.Dataset(X[val], label=y_log[val], reference=dtr)
        bst = lgb.train(params, dtr, 3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val] = bst.predict(X[val], num_iteration=bst.best_iteration)
        test_pred += bst.predict(X_te, num_iteration=bst.best_iteration) / len(splits)
    return oof, test_pred, mae(y_raw, from_log(oof))


def fit_calibration(pred_log: np.ndarray, y_raw: np.ndarray, n_bins: int = 12):
    """Per-decile multiplicative correction in log space.

    A blend of mean-ish fits is biased against the MAE-optimal conditional
    median, and the bias differs between cheap and expensive listings. We learn
    one additive log-offset per predicted-price bin (an offset in log space is a
    multiplier in dollars) using the median of the log residuals.
    """
    edges = np.quantile(pred_log, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    offsets = np.zeros(n_bins)
    for b in range(n_bins):
        m = (pred_log >= edges[b]) & (pred_log < edges[b + 1])
        if m.sum() >= 50:
            offsets[b] = np.median(to_log(y_raw[m]) - pred_log[m])
    return edges, offsets


def apply_calibration(pred_log, edges, offsets):
    idx = np.clip(np.searchsorted(edges, pred_log, side="right") - 1, 0, len(offsets) - 1)
    # smooth: average the bin offset with its neighbours to avoid step artefacts
    smooth = np.convolve(offsets, np.array([0.25, 0.5, 0.25]), mode="same")
    smooth[0], smooth[-1] = offsets[0], offsets[-1]
    return pred_log + smooth[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--out", default="submission_blend.csv")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--stack", action="store_true", help="also try a level-2 LightGBM")
    args = ap.parse_args()

    train, test = load_data()
    y_raw = train["listPrice"].values.astype(float)
    names = args.models.split(",") if args.models else available_models()
    oofs, tests = {}, {}
    for n in names:
        o, t = load_oof(n)
        oofs[n], tests[n] = o, t
    print(f"blending {len(oofs)} models: {', '.join(oofs)}")

    weights, score = greedy_blend(oofs, y_raw)
    print("\nweights:", {k: round(v, 3) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])})
    print(f"blend CV MAE = {score:,.0f}")

    blend_oof = sum(w * oofs[n] for n, w in weights.items())
    blend_test = sum(w * tests[n] for n, w in weights.items())
    splits = price_bin_folds(y_raw)

    if args.stack and len(oofs) >= 3:
        from features import build_features
        F_tr = np.nan_to_num(build_features(train["text"]).values, nan=-999)
        F_te = np.nan_to_num(build_features(test["text"]).values, nan=-999)
        st_oof, st_test, st_score = stack_lgbm(oofs, tests, y_raw, splits, F_tr, F_te)
        print(f"stack CV MAE  = {st_score:,.0f}")
        # blending the stack back with the greedy average is usually safer than
        # trusting either one alone
        mixes = {w: mae(y_raw, from_log(w * st_oof + (1 - w) * blend_oof))
                 for w in np.arange(0, 1.01, 0.1)}
        w_best = min(mixes, key=mixes.get)
        print(f"greedy/stack mix w={w_best:.1f} -> {mixes[w_best]:,.0f}")
        if mixes[w_best] < score:
            blend_oof = w_best * st_oof + (1 - w_best) * blend_oof
            blend_test = w_best * st_test + (1 - w_best) * blend_test
            score = mixes[w_best]

    if args.calibrate:
        # honest check: calibration is fit inside CV folds before being trusted
        cal_oof = np.zeros_like(blend_oof)
        for trn, val in splits:
            e, o = fit_calibration(blend_oof[trn], y_raw[trn])
            cal_oof[val] = apply_calibration(blend_oof[val], e, o)
        cal_score = mae(y_raw, from_log(cal_oof))
        print(f"calibrated CV MAE = {cal_score:,.0f} (delta {cal_score - score:+,.0f})")
        if cal_score < score:
            e, o = fit_calibration(blend_oof, y_raw)
            blend_test = apply_calibration(blend_test, e, o)
            score = cal_score
        else:
            print("  -> calibration rejected, keeping raw blend")

    path = make_submission(test["id"], from_log(blend_test), args.out)
    print(f"\nwrote {path}  (CV MAE {score:,.0f})")
    print("prediction summary:", np.percentile(from_log(blend_test), [1, 25, 50, 75, 99]).round(0))


if __name__ == "__main__":
    main()
