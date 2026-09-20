"""CPU models on top of TF-IDF + extracted features.

Every model is trained with the *same* folds and writes out-of-fold / test
predictions in LOG-price space to artifacts/, so blend.py can combine them.

Why log space + L1-ish losses: the metric is MAE in dollars, whose optimum is
the conditional MEDIAN. exp() of a median fit in log space is exactly that
median, and log-space also stops the $80M tail from dominating the fit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ARTIFACTS, SEED, from_log, load_data, mae, make_submission,
                    price_bin_folds, save_oof, to_log)
from features import build_features


def word_tfidf(train_texts, test_texts, **kw):
    params = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                  strip_accents="unicode", lowercase=True, dtype=np.float32)
    params.update(kw)
    vec = TfidfVectorizer(**params)
    return vec.fit_transform(train_texts), vec.transform(test_texts), vec


def char_tfidf(train_texts, test_texts, **kw):
    params = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                  max_features=300000, lowercase=True, dtype=np.float32)
    params.update(kw)
    vec = TfidfVectorizer(**params)
    return vec.fit_transform(train_texts), vec.transform(test_texts), vec


def run_cv(name, fit_predict, X, Xte, y_log, y_raw, splits, verbose=True):
    oof = np.zeros(len(y_log))
    test_pred = np.zeros(Xte.shape[0] if hasattr(Xte, "shape") else len(Xte))
    t0 = time.time()
    for k, (trn, val) in enumerate(splits):
        p_val, p_test = fit_predict(X, y_log, trn, val, Xte, k)
        oof[val] = p_val
        test_pred += p_test / len(splits)
        if verbose:
            print(f"  fold {k}: MAE={mae(y_raw[val], from_log(p_val)):,.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    score = mae(y_raw, from_log(oof))
    print(f"[{name}] CV MAE = {score:,.0f}   (median AE {np.median(np.abs(y_raw - from_log(oof))):,.0f})")
    save_oof(name, oof, test_pred)
    return oof, test_pred, score


# ----------------------------------------------------------------- models ----
def ridge_fp(alpha):
    def fp(X, y, trn, val, Xte, k):
        m = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-4, random_state=SEED)
        m.fit(X[trn], y[trn])
        return m.predict(X[val]), m.predict(Xte)
    return fp


def sgd_l1_fp(alpha=1e-6, epsilon=0.05):
    """Epsilon-insensitive SGD. Kept for reference only: it needs careful LR
    tuning and LinearSVR fits the same loss far better out of the box."""
    def fp(X, y, trn, val, Xte, k):
        m = SGDRegressor(loss="epsilon_insensitive", epsilon=epsilon, penalty="l2",
                         alpha=alpha, max_iter=60, tol=1e-4, learning_rate="invscaling",
                         eta0=0.05, power_t=0.25, random_state=SEED + k, average=True)
        m.fit(X[trn], y[trn])
        return m.predict(X[val]), m.predict(Xte)
    return fp


def linsvr_fp(C=0.5, epsilon=0.05):
    """LinearSVR = epsilon-insensitive (L1) loss -> fits the conditional MEDIAN
    of log-price, which is exactly the MAE optimum. Complements the L2 ridge."""
    from sklearn.svm import LinearSVR

    def fp(X, y, trn, val, Xte, k):
        m = LinearSVR(C=C, epsilon=epsilon, loss="epsilon_insensitive", dual=True,
                      max_iter=3000, tol=1e-4, random_state=SEED + k)
        m.fit(X[trn], y[trn])
        return m.predict(X[val]), m.predict(Xte)
    return fp


def knn_fp(k_neighbors=25):
    """Cosine-similarity neighbours on TF-IDF: listings for similar homes in the
    same market share phrasing, so a similarity-weighted MEDIAN of neighbour
    prices is a genuinely different view from the linear/tree models."""
    from sklearn.preprocessing import normalize

    def weighted_median(values, weights):
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cw = np.cumsum(weights)
        return values[np.searchsorted(cw, 0.5 * cw[-1])]

    def fp(X, y, trn, val, Xte, k):
        Xn = normalize(X)
        ref = Xn[trn]
        out = []
        for target in (Xn[val], normalize(Xte)):
            preds = np.empty(target.shape[0])
            step = 512
            for s in range(0, target.shape[0], step):
                sim = (target[s:s + step] @ ref.T).toarray()
                idx = np.argpartition(-sim, k_neighbors, axis=1)[:, :k_neighbors]
                for i in range(sim.shape[0]):
                    w = np.clip(sim[i, idx[i]], 0, None) ** 3
                    if w.sum() <= 0:
                        preds[s + i] = np.median(y[trn])
                    else:
                        preds[s + i] = weighted_median(y[trn][idx[i]], w)
            out.append(preds)
        return out[0], out[1]
    return fp


def lgbm_fp(params=None, num_round=4000):
    import lightgbm as lgb

    base = dict(objective="mae", metric="mae", learning_rate=0.03, num_leaves=63,
                min_data_in_leaf=20, feature_fraction=0.35, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=4, seed=SEED)
    if params:
        base.update(params)

    def fp(X, y, trn, val, Xte, k):
        dtr = lgb.Dataset(X[trn], label=y[trn])
        dva = lgb.Dataset(X[val], label=y[val], reference=dtr)
        m = lgb.train(base, dtr, num_boost_round=num_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.predict(X[val], num_iteration=m.best_iteration), m.predict(Xte, num_iteration=m.best_iteration)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge_word,ridge_char,linsvr_word,knn_word,lgbm_dense")
    ap.add_argument("--svd", type=int, default=250)
    args = ap.parse_args()
    want = set(args.models.split(","))

    train, test = load_data()
    y_raw = train["listPrice"].values.astype(float)
    y_log = to_log(y_raw)
    splits = price_bin_folds(y_raw)
    os.makedirs(ARTIFACTS, exist_ok=True)

    print("building features ...", flush=True)
    F_tr = build_features(train["text"]).values
    F_te = build_features(test["text"]).values
    Xw, Xw_te, _ = word_tfidf(train["text"], test["text"])
    print("word tfidf:", Xw.shape, flush=True)

    scores = {}
    if "ridge_word" in want:
        scores["ridge_word"] = run_cv("ridge_word", ridge_fp(1.0), Xw, Xw_te, y_log, y_raw, splits)[2]
    if "sgd_word" in want:
        scores["sgd_word"] = run_cv("sgd_word", sgd_l1_fp(), Xw, Xw_te, y_log, y_raw, splits)[2]
    if "linsvr_word" in want:
        scores["linsvr_word"] = run_cv("linsvr_word", linsvr_fp(), Xw, Xw_te, y_log, y_raw, splits)[2]
    if "knn_word" in want:
        scores["knn_word"] = run_cv("knn_word", knn_fp(), Xw, Xw_te, y_log, y_raw, splits)[2]
    if "ridge_char" in want:
        Xc, Xc_te, _ = char_tfidf(train["text"], test["text"])
        print("char tfidf:", Xc.shape, flush=True)
        scores["ridge_char"] = run_cv("ridge_char", ridge_fp(1.0), Xc, Xc_te, y_log, y_raw, splits)[2]
        del Xc, Xc_te

    if "lgbm_dense" in want:
        print(f"SVD({args.svd}) on word tfidf ...", flush=True)
        svd = TruncatedSVD(n_components=args.svd, random_state=SEED)
        Z = svd.fit_transform(Xw).astype(np.float32)
        Z_te = svd.transform(Xw_te).astype(np.float32)
        print("  explained var:", svd.explained_variance_ratio_.sum().round(3), flush=True)
        D = np.hstack([Z, np.nan_to_num(F_tr, nan=-999)])
        D_te = np.hstack([Z_te, np.nan_to_num(F_te, nan=-999)])
        np.save(os.path.join(ARTIFACTS, "svd_train.npy"), Z)
        np.save(os.path.join(ARTIFACTS, "svd_test.npy"), Z_te)
        scores["lgbm_dense"] = run_cv("lgbm_dense", lgbm_fp(), D, D_te, y_log, y_raw, splits)[2]

    print("\n== summary ==")
    for k, v in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"{k:14s} {v:,.0f}")


if __name__ == "__main__":
    main()
