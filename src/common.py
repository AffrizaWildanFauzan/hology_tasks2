"""Shared utilities: data loading, CV splits, metric, target transforms."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.environ.get("HOLOMINE_ARTIFACTS", os.path.join(ROOT, "artifacts"))
SUBMISSIONS = os.environ.get("HOLOMINE_SUBMISSIONS", os.path.join(ROOT, "submissions"))
N_SPLITS = 5
SEED = 42


def data_dir() -> str:
    """Repo root locally; the competition folder when running on Kaggle."""
    env = os.environ.get("HOLOMINE_DATA")
    if env and os.path.exists(os.path.join(env, "train.csv")):
        return env
    if os.path.exists(os.path.join(ROOT, "train.csv")):
        return ROOT
    import glob as _glob
    for cand in sorted(_glob.glob("/kaggle/input/*")):
        if os.path.exists(os.path.join(cand, "train.csv")):
            return cand
    raise FileNotFoundError("train.csv not found; set HOLOMINE_DATA to its folder")


def load_data():
    d = data_dir()
    train = pd.read_csv(os.path.join(d, "train.csv"))
    test = pd.read_csv(os.path.join(d, "test.csv"))
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")
    return train, test


def folds(n_rows: int, n_splits: int = N_SPLITS, seed: int = SEED):
    """Plain KFold. Prices are continuous, so we stratify on log-price bins."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n_rows))


def price_bin_folds(y: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED):
    """Stratified folds over log-price deciles: keeps the heavy tail balanced."""
    from sklearn.model_selection import StratifiedKFold

    bins = pd.qcut(np.log(y), q=20, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), bins))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def to_log(y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(y, dtype=float), 1.0, None))


def from_log(z: np.ndarray) -> np.ndarray:
    """Back to dollars. L1-fit in log space -> exp() is the conditional median,
    which is exactly what MAE rewards."""
    return np.exp(np.clip(np.asarray(z, dtype=float), np.log(1.0), np.log(3e8)))


def save_oof(name: str, oof: np.ndarray, test_pred: np.ndarray) -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)
    np.save(os.path.join(ARTIFACTS, f"oof_{name}.npy"), oof)
    np.save(os.path.join(ARTIFACTS, f"test_{name}.npy"), test_pred)


def load_oof(name: str):
    oof = np.load(os.path.join(ARTIFACTS, f"oof_{name}.npy"))
    test_pred = np.load(os.path.join(ARTIFACTS, f"test_{name}.npy"))
    return oof, test_pred


def make_submission(test_ids, prices, filename: str) -> str:
    os.makedirs(SUBMISSIONS, exist_ok=True)
    path = os.path.join(SUBMISSIONS, filename)
    pd.DataFrame({"id": test_ids, "listPrice": np.round(np.asarray(prices), 2)}).to_csv(
        path, index=False
    )
    return path


def explain_hub_error(exc: Exception, model_name: str) -> str:
    """Terjemahkan kegagalan unduh HuggingFace jadi langkah yang bisa dikerjakan."""
    text = f"{type(exc).__name__}: {exc}"
    hints = [f"Gagal memuat '{model_name}'.", f"  penyebab: {text.splitlines()[0][:200]}"]
    low = text.lower()
    if "403" in low or "proxy" in low or "connect" in low or "resolve" in low or "timed out" in low:
        hints += [
            "  -> Sepertinya jaringan memblokir huggingface.co.",
            "     Kaggle: Notebook Settings -> Internet: ON.",
            "     Offline: unggah bobot sebagai Kaggle Dataset, ganti 'model' di",
            "     src/model_zoo.py ke path lokalnya, lalu set HF_HUB_OFFLINE=1.",
        ]
    elif "401" in low or "gated" in low or "authoriz" in low:
        hints += ["  -> Model gated: terima lisensinya di halaman model lalu login `huggingface-cli login`."]
    elif "not a local folder" in low or "404" in low or "repositorynotfound" in low:
        hints += ["  -> Nama model salah ketik, atau repo privat."]
    elif "sentencepiece" in low or "protobuf" in low:
        hints += ["  -> Tokenizer deberta-v3 butuh: pip install sentencepiece protobuf"]
    elif "modernbert" in low or "unrecognized" in low or "trust_remote_code" in low:
        hints += ["  -> ModernBERT butuh transformers>=4.48: pip install -U 'transformers>=4.48'"]
    return "\n".join(hints)
