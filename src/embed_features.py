"""Frozen sentence-embeddings -> Ridge / LightGBM head.

Much cheaper than fine-tuning (one forward pass over 18k texts) and it usually
lands between TF-IDF and a fine-tuned encoder in accuracy, while being very
diverse from both, which is what makes it pull its weight in the blend.

    python src/embed_features.py --model BAAI/bge-small-en-v1.5 --tag bgesmall
    python src/embed_features.py --model Alibaba-NLP/gte-modernbert-base --tag gteemb --max-len 1024

Embeddings are cached in artifacts/emb_<tag>_{train,test}.npy, so re-running the
heads is instant.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ARTIFACTS, SEED, explain_hub_error, from_log, load_data, mae,
                    price_bin_folds, save_oof, to_log)


def encode(model_name, texts, max_len, batch_size, prefix=""):
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as exc:
        print(explain_hub_error(exc, model_name), file=sys.stderr)
        raise SystemExit(1)
    model.max_seq_length = max_len
    payload = [prefix + t for t in texts] if prefix else list(texts)
    return model.encode(payload, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def get_embeddings(args, train_texts, test_texts):
    tr_path = os.path.join(ARTIFACTS, f"emb_{args.tag}_train.npy")
    te_path = os.path.join(ARTIFACTS, f"emb_{args.tag}_test.npy")
    if os.path.exists(tr_path) and os.path.exists(te_path) and not args.refresh:
        print("using cached embeddings")
        return np.load(tr_path), np.load(te_path)
    os.makedirs(ARTIFACTS, exist_ok=True)
    E_tr = encode(args.model, train_texts, args.max_len, args.batch_size, args.prefix)
    E_te = encode(args.model, test_texts, args.max_len, args.batch_size, args.prefix)
    np.save(tr_path, E_tr)
    np.save(te_path, E_te)
    return E_tr, E_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prefix", default="", help='e.g. "query: " for e5 models')
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--with-features", action="store_true",
                    help="append the regex-extracted numeric features to the LGBM head")
    args = ap.parse_args()
    args.tag = args.tag or args.model.split("/")[-1].replace(".", "").lower()

    train, test = load_data()
    y_raw = train["listPrice"].values.astype(float)
    y_log = to_log(y_raw)
    splits = price_bin_folds(y_raw)

    E_tr, E_te = get_embeddings(args, train["text"].values, test["text"].values)
    print("embeddings:", E_tr.shape)

    X_tr, X_te = E_tr, E_te
    if args.with_features:
        from features import build_features
        F_tr = np.nan_to_num(build_features(train["text"]).values, nan=-999)
        F_te = np.nan_to_num(build_features(test["text"]).values, nan=-999)
        X_tr = np.hstack([E_tr, F_tr])
        X_te = np.hstack([E_te, F_te])

    import lightgbm as lgb
    from sklearn.linear_model import RidgeCV

    for head in ("ridge", "lgbm"):
        oof = np.zeros(len(y_log))
        test_pred = np.zeros(len(E_te))
        for k, (trn, val) in enumerate(splits):
            if head == "ridge":
                m = RidgeCV(alphas=np.logspace(-2, 3, 12))
                m.fit(E_tr[trn], y_log[trn])
                oof[val], p = m.predict(E_tr[val]), m.predict(E_te)
            else:
                params = dict(objective="mae", metric="mae", learning_rate=0.03, num_leaves=63,
                              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1,
                              min_data_in_leaf=20, lambda_l2=1.0, verbosity=-1, seed=SEED)
                dtr = lgb.Dataset(X_tr[trn], label=y_log[trn])
                dva = lgb.Dataset(X_tr[val], label=y_log[val], reference=dtr)
                bst = lgb.train(params, dtr, 4000, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(150, verbose=False)])
                oof[val] = bst.predict(X_tr[val], num_iteration=bst.best_iteration)
                p = bst.predict(X_te, num_iteration=bst.best_iteration)
            test_pred += p / len(splits)
        name = f"emb_{args.tag}_{head}"
        print(f"[{name}] CV MAE = {mae(y_raw, from_log(oof)):,.0f}")
        save_oof(name, oof, test_pred)


if __name__ == "__main__":
    main()
