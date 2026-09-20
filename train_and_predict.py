#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
 HoloMine Task 2 - LATIH train.csv, PREDIKSI test.csv, TULIS submission.csv
 Dirancang untuk Kaggle GPU T4.
===============================================================================

Alurnya lurus dari atas ke bawah, tidak ada cache yang bisa melewati training:

    1. Baca train.csv (14.640 baris) + test.csv (3.659 baris)
    2. LATIH model teks cepat di CPU      (TF-IDF + kNN + LightGBM, ~8 menit)
    3. LATIH transformer di GPU T4        (fine-tune DeBERTa-v3, ~60 menit)
    4. Gabungkan prediksi semua model
    5. Isi kolom listPrice di sample_submission.csv -> submission.csv

Jalankan:
    python train_and_predict.py                 # semua, butuh GPU
    python train_and_predict.py --quick         # 1 fold 1 epoch, untuk uji cepat
    python train_and_predict.py --no-transformer  # CPU saja, tanpa GPU

Di sel notebook Kaggle/Colab:
    !python train_and_predict.py
atau kalau isinya ditempel ke sel: panggil  main()

KENAPA TARGETNYA log(harga)
---------------------------
Metrik lomba MAE diminimalkan oleh MEDIAN bersyarat, bukan rata-rata. Karena
exp() monoton, median di ruang log = log dari median di ruang dolar. Jadi semua
model dilatih pada log(listPrice) dengan loss L1/Huber lalu di-exp(). Ini juga
mencegah satu rumah $80 juta mendominasi gradien.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# =============================================================================
# KONFIGURASI - ubah di sini kalau perlu
# =============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"   # juara de-facto regresi teks di Kaggle
MAX_LEN    = 512      # p99 teks = 521 token; di 512 cuma 1,1% listing terpotong
BATCH_SIZE = 16       # muat di T4 16GB dengan fp16
EPOCHS     = 3
LR         = 2e-5     # untuk backbone
HEAD_LR    = 1e-4     # untuk kepala regresi (dilatih dari nol, perlu lebih besar)
LLRD       = 0.9      # layer-wise LR decay
N_FOLDS    = 5
SEED       = 42

SUBMISSION_NAME = "submission.csv"


# =============================================================================
# 1. DATA
# =============================================================================

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:          # kalau isi file ditempel ke sel notebook
    HERE = os.getcwd()


def find_data_dir() -> str:
    """Cari folder berisi train.csv (lokal, /kaggle/input, atau /content)."""
    cands = [os.environ.get("HOLOMINE_DATA"), HERE, os.getcwd()]
    for pattern in ("/kaggle/input/*", "/kaggle/input/*/*", "/content/*", "/content/*/*"):
        cands += sorted(glob.glob(pattern))
    for c in cands:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    raise FileNotFoundError(
        "train.csv tidak ketemu. Set dulu:\n"
        "  os.environ['HOLOMINE_DATA'] = '/kaggle/input/nama-kompetisi'")


def output_dir() -> str:
    """Tempat menulis submission. Di Kaggle WAJIB /kaggle/working -- hanya isi
    folder itu yang muncul sebagai Output notebook setelah Save Version."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    return HERE if os.access(HERE, os.W_OK) else os.getcwd()


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def to_log(y):
    return np.log(np.clip(np.asarray(y, dtype=float), 1.0, None))


def from_log(z):
    return np.exp(np.clip(np.asarray(z, dtype=float), 0.0, np.log(3e8)))


def make_folds(y):
    """5-fold distratifikasi atas desil log-harga, supaya rumah mahal terbagi
    rata. Semua model memakai fold yang sama agar prediksinya bisa digabung."""
    from sklearn.model_selection import StratifiedKFold
    bins = pd.qcut(np.log(y), q=20, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), bins))


# =============================================================================
# 2. FITUR DARI TEKS (regex)
# =============================================================================
# Lomba hanya memberi teks, jadi kolom yang biasanya dimiliki model properti
# (kamar, kamar mandi, luas, tanah, tahun bangun) harus digali balik dari prosa,
# ditambah sinyal fasilitas/kondisi yang menggerakkan harga.

WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "a": 1, "an": 1, "single": 1, "double": 2, "triple": 3}
NUM_WORD_RE = "|".join(WORD_NUM)
NUMTOK = rf"(\d+(?:[.,]\d+)?|\d+/\d+|{NUM_WORD_RE})"

RE_BED = re.compile(rf"{NUMTOK}[\s-]*(?:bed(?:room)?s?|br\b|bdrm)", re.I)
RE_BED2 = re.compile(rf"(?:bed(?:room)?s?)[:\s-]*{NUMTOK}", re.I)
RE_BATH = re.compile(rf"{NUMTOK}[\s-]*(?:full[\s-]*|half[\s-]*)?(?:bath(?:room)?s?|ba\b)", re.I)
RE_FULLBATH = re.compile(rf"{NUMTOK}[\s-]*full[\s-]*bath", re.I)
RE_HALFBATH = re.compile(rf"{NUMTOK}[\s-]*half[\s-]*bath", re.I)
RE_SQFT = re.compile(r"([\d,]{3,9}(?:\.\d+)?)\s*(?:\+/-\s*)?(?:sq\.?\s*(?:ft|feet)|square\s*(?:foot|feet)|sf\b)", re.I)
RE_ACRE = re.compile(rf"{NUMTOK}(?:\s*\+/-)?\s*(?:\+)?\s*acres?", re.I)
RE_ACRE_OF = re.compile(rf"{NUMTOK}\s*of\s*an?\s*acre", re.I)
RE_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
RE_BUILT = re.compile(r"(?:built|constructed|circa|est\.?)\s*(?:in\s*)?(1[89]\d{2}|20[0-2]\d)", re.I)
RE_GARAGE = re.compile(rf"{NUMTOK}[\s-]*(?:car|bay)[\s-]*(?:attached\s*|detached\s*)?(?:garage|carport)", re.I)
RE_STORY = re.compile(rf"{NUMTOK}[\s-]*(?:story|storey|stories|level)", re.I)
RE_HOA = re.compile(r"hoa[^.]{0,40}?\$\s*([\d,]+)", re.I)
RE_TAX = re.compile(r"tax(?:es)?[^.]{0,40}?\$\s*([\d,]+)", re.I)
RE_RENT = re.compile(r"(?:rent|rents|rental|income)[^.]{0,40}?\$\s*([\d,]+)", re.I)
RE_DOLLAR = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k|m|million)?", re.I)
RE_UNITS = re.compile(rf"{NUMTOK}[\s-]*(?:unit|units|apartment|plex)", re.I)
RE_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")

KEYWORDS = {
    "kw_luxury": r"luxur|prestigious|exquisite|opulent|magnificent|estate\b|bespoke|world-class",
    "kw_custom": r"custom[\s-]?(?:built|home|design)|architect",
    "kw_waterfront": r"waterfront|water\s*front|oceanfront|lakefront|riverfront|beachfront|bayfront",
    "kw_water_view": r"ocean|lake|river|beach|bay\b|waterview|water view|canal|dock|marina|boat\s*(?:slip|lift|house)",
    "kw_view": r"\bview(?:s)?\b|panoramic|vista|skyline|mountain view",
    "kw_pool": r"\bpool\b|spa\b|hot tub|jacuzzi",
    "kw_highend_finish": r"granite|quartz|marble|stainless|hardwood|travertine|chef'?s kitchen|gourmet|sub[\s-]?zero|wolf\b|viking\b|thermador|cathedral ceiling|coffered|crown moulding|crown molding",
    "kw_new_build": r"new construction|newly built|brand new|to be built|never lived|builder",
    "kw_renovated": r"renovat|remodel|updated|upgraded|refinish|restored|new roof|new furnace|new hvac|new windows|move[\s-]in ready|turn[\s-]?key",
    "kw_fixer": r"fixer|handyman|tlc\b|as[\s-]is|needs work|investor special|rehab|distressed|foreclos|short sale|auction|sold as|cash only|gut\b",
    "kw_condo": r"\bcondo|condominium|co-?op\b",
    "kw_townhouse": r"townh(?:ouse|ome)|row house|duplex|end[\s-]unit",
    "kw_mobile": r"mobile home|manufactur(?:ed)? home|doublewide|double[\s-]wide|singlewide|trailer|modular",
    "kw_land": r"\bland\b|vacant lot|buildable|raw land|parcel|acreage|timber|pasture|farmland|ranch land",
    "kw_multifamily": r"multi[\s-]?family|duplex|triplex|fourplex|apartment building|income property|cap rate|noi\b|tenant",
    "kw_commercial": r"commercial|retail|office space|warehouse|industrial|zoned c|business",
    "kw_hoa": r"\bhoa\b|association fee|condo fee|maintenance fee",
    "kw_gated": r"gated|guard[\s-]?house|concierge|doorman|private community|country club|golf",
    "kw_garage": r"garage|carport",
    "kw_basement": r"basement|cellar|lower level",
    "kw_fireplace": r"fireplace|wood stove|pellet stove",
    "kw_deck": r"\bdeck\b|patio|porch|lanai|veranda|terrace|courtyard",
    "kw_elevator": r"elevator|dumbwaiter",
    "kw_guest": r"guest (?:house|suite|cottage)|casita|in[\s-]?law|adu\b|carriage house|accessory dwelling",
    "kw_wine": r"wine (?:cellar|room)|theater|theatre|media room|home gym|sauna|bowling",
    "kw_solar": r"solar|geothermal|energy efficient|net zero|ev charg",
    "kw_barn": r"\bbarn\b|stable|equestrian|paddock|corral|silo|chicken",
    "kw_school": r"school|district|university|college|campus",
    "kw_commute": r"commut|highway|freeway|interstate|metro|subway|train station|airport|minutes (?:from|to)",
    "kw_downtown": r"downtown|city center|walk(?:able|ing distance)|shops|dining|restaurants",
    "kw_historic": r"historic|victorian|colonial|craftsman|tudor|antique|century|heritage|landmark",
    "kw_modern": r"modern|contemporary|mid[\s-]century|minimalist|sleek|open concept|smart home",
    "kw_ranch_style": r"ranch[\s-]?style|\branch\b|bungalow|cape cod|split[\s-]level|raised ranch|cottage|farmhouse",
    "kw_penthouse": r"penthouse|high[\s-]rise|doorman|loft\b",
    "kw_acre_word": r"acre",
    "kw_privacy": r"private|seclu|tranquil|serene|retreat|oasis|sanctuary|wooded|cul[\s-]de[\s-]sac",
    "kw_rare": r"rare(?:ly)?|one of a kind|unique|opportunity|potential|must see|won'?t last|priced to sell|motivated",
    "kw_starter": r"starter home|first[\s-]time|affordable|budget|value|bargain",
    "kw_sqft_word": r"square (?:foot|feet)|sq\.? ?ft",
}
KEYWORD_RE = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}

US_STATES = ("alabama alaska arizona arkansas california colorado connecticut delaware florida georgia "
             "hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts "
             "michigan minnesota mississippi missouri montana nebraska nevada hampshire jersey mexico "
             "york carolina dakota ohio oklahoma oregon pennsylvania rhode tennessee texas utah vermont "
             "virginia washington wisconsin wyoming").split()
STATE_RE = re.compile(r"\b(" + "|".join(US_STATES) + r")\b", re.I)


def _num(tok: str) -> float:
    tok = tok.strip().lower().replace(",", "")
    if tok in WORD_NUM:
        return float(WORD_NUM[tok])
    if "/" in tok:
        try:
            a, b = tok.split("/")
            return float(a) / float(b)
        except Exception:
            return np.nan
    try:
        return float(tok)
    except Exception:
        return np.nan


def _first(pattern, text: str) -> float:
    m = pattern.search(text)
    return _num(m.group(1)) if m else np.nan


def _dollar_amounts(text: str) -> list:
    out = []
    for amt, suffix in RE_DOLLAR.findall(text):
        v = _num(amt)
        if np.isnan(v):
            continue
        s = (suffix or "").lower()
        if s == "k":
            v *= 1e3
        elif s in ("m", "million"):
            v *= 1e6
        out.append(v)
    return out


def extract_row(text: str) -> dict:
    t = text if isinstance(text, str) else ""
    low = t.lower()
    f = {}

    beds = _first(RE_BED, t)
    if np.isnan(beds):
        beds = _first(RE_BED2, t)
    f["beds"] = beds if (not np.isnan(beds) and 0 < beds <= 30) else np.nan
    baths = _first(RE_BATH, t)
    f["baths"] = baths if (not np.isnan(baths) and 0 < baths <= 30) else np.nan
    f["full_baths"] = _first(RE_FULLBATH, t)
    f["half_baths"] = _first(RE_HALFBATH, t)

    sqft_vals = [_num(x) for x in RE_SQFT.findall(t)]
    sqft_vals = [v for v in sqft_vals if not np.isnan(v) and 100 <= v <= 60000]
    f["sqft"] = max(sqft_vals) if sqft_vals else np.nan
    f["sqft_min"] = min(sqft_vals) if sqft_vals else np.nan
    f["n_sqft_mentions"] = len(sqft_vals)

    acres = [_num(x) for x in RE_ACRE.findall(t)] + [_num(x) for x in RE_ACRE_OF.findall(t)]
    acres = [v for v in acres if not np.isnan(v) and 0 < v <= 100000]
    f["acres"] = max(acres) if acres else np.nan

    yb = _first(RE_BUILT, t)
    years = [float(y) for y in RE_YEAR.findall(t)]
    f["year_built"] = yb if not np.isnan(yb) else (min(years) if years else np.nan)
    f["year_recent"] = max(years) if years else np.nan
    f["n_years"] = len(years)

    f["garage_cars"] = _first(RE_GARAGE, t)
    f["stories"] = _first(RE_STORY, t)
    f["units"] = _first(RE_UNITS, t)
    f["hoa_fee"] = _first(RE_HOA, t)
    f["tax_amt"] = _first(RE_TAX, t)
    f["rent_amt"] = _first(RE_RENT, t)

    amounts = _dollar_amounts(t)
    f["n_dollar"] = len(amounts)
    f["max_dollar"] = max(amounts) if amounts else np.nan
    f["min_dollar"] = min(amounts) if amounts else np.nan

    nums = [float(x.replace(",", "")) for x in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", t)[:200]]
    nums = [n for n in nums if n < 1e9]
    f["n_numbers"] = len(nums)
    f["max_number"] = max(nums) if nums else np.nan
    f["median_number"] = float(np.median(nums)) if nums else np.nan

    f["len_chars"] = len(t)
    words = low.split()
    f["len_words"] = len(words)
    f["n_sentences"] = t.count(".") + t.count("!") + t.count("?")
    f["avg_word_len"] = float(np.mean([len(w) for w in words])) if words else 0.0
    f["n_upper_words"] = sum(1 for w in t.split() if w.isupper() and len(w) > 2)
    f["upper_ratio"] = sum(1 for c in t if c.isupper()) / max(len(t), 1)
    f["digit_ratio"] = sum(1 for c in t if c.isdigit()) / max(len(t), 1)
    f["excl_marks"] = t.count("!")
    f["n_commas"] = t.count(",")
    f["n_redacted"] = low.count("[redacted")
    f["has_zip"] = 1.0 if RE_ZIP.search(t) else 0.0
    f["n_states"] = len(set(m.lower() for m in STATE_RE.findall(t)))

    for name, rx in KEYWORD_RE.items():
        f[name] = float(len(rx.findall(t)))

    f["bed_bath"] = f["beds"] * f["baths"]
    f["sqft_per_bed"] = f["sqft"] / f["beds"] if (not np.isnan(f["sqft"]) and not np.isnan(f["beds"]) and f["beds"] > 0) else np.nan
    f["log_sqft"] = np.log1p(f["sqft"]) if not np.isnan(f["sqft"]) else np.nan
    f["log_acres"] = np.log1p(f["acres"]) if not np.isnan(f["acres"]) else np.nan
    f["log_max_dollar"] = np.log1p(f["max_dollar"]) if not np.isnan(f["max_dollar"]) else np.nan
    f["age"] = 2025 - f["year_built"] if not np.isnan(f["year_built"]) else np.nan
    return f


def build_features(texts) -> np.ndarray:
    return pd.DataFrame([extract_row(t) for t in texts]).astype(np.float32).values


# =============================================================================
# 3. LATIH MODEL TEKS CEPAT DI CPU
# =============================================================================
# Empat model, semuanya dilatih ulang tiap dijalankan. Masing-masing menghasilkan
# prediksi out-of-fold (untuk mengukur & menggabung) dan prediksi test.

def train_cpu_models(train, test, y_raw, folds):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import normalize
    from sklearn.svm import LinearSVR
    import lightgbm as lgb

    y_log = to_log(y_raw)
    oof, testp = {}, {}

    def record(name, o, t):
        oof[name], testp[name] = o, t
        print(f"   -> [{name}] MAE = {mae(y_raw, from_log(o)):,.0f}", flush=True)

    print("\n[2/5] LATIH model CPU")
    print("   membangun TF-IDF kata (1-2 gram) ...", flush=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                          strip_accents="unicode", lowercase=True, dtype=np.float32)
    Xw = vec.fit_transform(train["text"])
    Xw_te = vec.transform(test["text"])
    print(f"   {Xw.shape[1]:,} fitur", flush=True)

    # --- Ridge: loss L2, menangkap nama kota & kata kunci langka ---
    print("   melatih ridge ...", flush=True)
    o, t = np.zeros(len(y_log)), np.zeros(Xw_te.shape[0])
    for tr, va in folds:
        m = Ridge(alpha=1.0, solver="sparse_cg", tol=1e-4).fit(Xw[tr], y_log[tr])
        o[va] = m.predict(Xw[va])
        t += m.predict(Xw_te) / len(folds)
    record("ridge_word", o, t)

    # --- LinearSVR: loss epsilon-insensitive = L1 -> mencocokkan MEDIAN ---
    print("   melatih LinearSVR ...", flush=True)
    o, t = np.zeros(len(y_log)), np.zeros(Xw_te.shape[0])
    for k, (tr, va) in enumerate(folds):
        m = LinearSVR(C=0.5, epsilon=0.05, loss="epsilon_insensitive", dual=True,
                      max_iter=3000, tol=1e-4, random_state=SEED + k).fit(Xw[tr], y_log[tr])
        o[va] = m.predict(Xw[va])
        t += m.predict(Xw_te) / len(folds)
    record("linsvr_word", o, t)

    # --- kNN kosinus: median harga tetangga. Listing rumah serupa di pasar yang
    #     sama memakai frasa mirip, jadi ini sudut pandang yang benar-benar beda.
    print("   melatih kNN kosinus ...", flush=True)

    def wmedian(v, w):
        order = np.argsort(v)
        v, w = v[order], w[order]
        return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

    Xn, Xn_te = normalize(Xw), normalize(Xw_te)
    o, t = np.zeros(len(y_log)), np.zeros(Xw_te.shape[0])
    for tr, va in folds:
        ref = Xn[tr]
        for target, out, div in ((Xn[va], o, None), (Xn_te, t, len(folds))):
            buf = np.empty(target.shape[0])
            for s in range(0, target.shape[0], 512):
                sim = (target[s:s + 512] @ ref.T).toarray()
                idx = np.argpartition(-sim, 25, axis=1)[:, :25]
                for i in range(sim.shape[0]):
                    w = np.clip(sim[i, idx[i]], 0, None) ** 3
                    buf[s + i] = np.median(y_log[tr]) if w.sum() <= 0 else wmedian(y_log[tr][idx[i]], w)
            if div is None:
                out[va] = buf
            else:
                out += buf / div
    record("knn_word", o, t)

    # --- Ridge di TF-IDF karakter: tahan salah ketik & variasi penulisan ---
    print("   membangun TF-IDF karakter (3-5 gram) + melatih ridge ...", flush=True)
    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                           max_features=300000, lowercase=True, dtype=np.float32)
    Xc, Xc_te = cvec.fit_transform(train["text"]), cvec.transform(test["text"])
    o, t = np.zeros(len(y_log)), np.zeros(Xc_te.shape[0])
    for tr, va in folds:
        m = Ridge(alpha=1.0, solver="sparse_cg", tol=1e-4).fit(Xc[tr], y_log[tr])
        o[va] = m.predict(Xc[va])
        t += m.predict(Xc_te) / len(folds)
    record("ridge_char", o, t)
    del Xc, Xc_te
    gc.collect()

    # --- LightGBM di SVD + fitur regex: menangkap angka eksplisit (sqft, kamar) ---
    print("   SVD(250) + fitur regex, melatih LightGBM ...", flush=True)
    svd = TruncatedSVD(n_components=250, random_state=SEED)
    D = np.hstack([svd.fit_transform(Xw).astype(np.float32),
                   np.nan_to_num(build_features(train["text"]), nan=-999)])
    D_te = np.hstack([svd.transform(Xw_te).astype(np.float32),
                      np.nan_to_num(build_features(test["text"]), nan=-999)])
    params = dict(objective="mae", metric="mae", learning_rate=0.03, num_leaves=63,
                  min_data_in_leaf=20, feature_fraction=0.35, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=4, seed=SEED)
    o, t = np.zeros(len(y_log)), np.zeros(D_te.shape[0])
    for tr, va in folds:
        dtr = lgb.Dataset(D[tr], label=y_log[tr])
        bst = lgb.train(params, dtr, 4000, valid_sets=[lgb.Dataset(D[va], label=y_log[va], reference=dtr)],
                        callbacks=[lgb.early_stopping(150, verbose=False)])
        o[va] = bst.predict(D[va], num_iteration=bst.best_iteration)
        t += bst.predict(D_te, num_iteration=bst.best_iteration) / len(folds)
    record("lgbm_dense", o, t)

    return oof, testp


# =============================================================================
# 4. LATIH TRANSFORMER DI GPU (T4)
# =============================================================================

def train_transformer(train, test, y_raw, folds, args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                              DataCollatorWithPadding, get_cosine_schedule_with_warmup)

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.cpu_transformer:
        print("\n[3/5] LEWAT: tidak ada GPU. Di Kaggle: Settings -> Accelerator -> GPU T4.")
        print("      (paksa di CPU dengan --cpu-transformer, tapi hitungan jamnya, bukan menit)")
        return {}, {}

    device = torch.device("cuda" if has_cuda else "cpu")
    if has_cuda:
        gp = torch.cuda.get_device_properties(0)
        # T4 (Turing) tidak punya bf16, jadi fp16 + GradScaler.
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"\n[3/5] LATIH transformer di {gp.name} ({gp.total_memory / 1e9:.0f} GB)")
        print(f"   presisi: {'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}")
    else:
        amp_dtype = None
        print("\n[3/5] LATIH transformer di CPU (sangat lambat -- hanya untuk uji)")
    print(f"   model  : {args.model}")
    print(f"   max_len {args.max_len} | batch {args.batch_size} | {args.epochs} epoch "
          f"| {len(folds)} fold")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as exc:
        print(f"   GAGAL memuat '{args.model}': {type(exc).__name__}: {exc}", file=sys.stderr)
        print("   -> Kaggle: Settings -> Internet: ON. deberta-v3 juga butuh "
              "`pip install sentencepiece`.", file=sys.stderr)
        return {}, {}

    class Listings(Dataset):
        def __init__(self, texts, targets):
            self.texts = list(texts)
            self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=args.max_len, padding=False)
            item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")}
            if self.targets is not None:
                item["labels"] = torch.tensor(self.targets[i], dtype=torch.float)
            return item

    mean_log_price = float(np.mean(to_log(y_raw)))

    class Regressor(nn.Module):
        """Encoder + mean pooling + kepala linear -> satu angka (log harga).
        Mean pooling mengalahkan [CLS] untuk teks deskriptif panjang."""
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            cfg.update({"hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0})
            self.backbone = AutoModel.from_pretrained(args.model, config=cfg, trust_remote_code=True)
            self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, 1))
            # Bias awal = rata-rata log harga, jadi tebakan pertama model adalah
            # harga median (~$500rb), bukan exp(0) = $1. Tanpa ini epoch pertama
            # habis hanya untuk merangkak naik ke skala yang benar.
            nn.init.zeros_(self.head[-1].weight)
            nn.init.constant_(self.head[-1].bias, mean_log_price)

        def forward(self, **batch):
            batch.pop("labels", None)
            out = self.backbone(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            return self.head((out * m).sum(1) / m.sum(1).clamp(min=1e-6)).squeeze(-1)

    def param_groups(model):
        """Layer-wise LR decay: lapisan bawah menyimpan pengetahuan bahasa umum,
        kepala & lapisan atas beradaptasi paling cepat."""
        enc = getattr(model.backbone, "encoder", None) or model.backbone
        layers = getattr(enc, "layer", None) or getattr(enc, "layers", None)
        groups = [{"params": model.head.parameters(), "lr": args.head_lr, "weight_decay": 0.01}]
        if layers is None:
            groups.append({"params": model.backbone.parameters(), "lr": args.lr, "weight_decay": 0.01})
            return groups
        n, seen = len(layers), set()
        for i, layer in enumerate(layers):
            groups.append({"params": layer.parameters(), "lr": args.lr * (LLRD ** (n - 1 - i)),
                           "weight_decay": 0.01})
            seen.update(id(p) for p in layer.parameters())
        rest = [p for p in model.backbone.parameters() if id(p) not in seen]
        if rest:
            groups.append({"params": rest, "lr": args.lr * (LLRD ** n), "weight_decay": 0.01})
        return groups

    @torch.no_grad()
    def infer(model, loader):
        model.eval()
        out = []
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out.append(model(**batch).float().cpu())
        return torch.cat(out).numpy()

    texts = train["text"].values
    y_log = to_log(y_raw)
    collate = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    dl_test = DataLoader(Listings(test["text"].values, None), batch_size=args.batch_size * 2,
                         shuffle=False, collate_fn=collate, num_workers=2)

    oof = np.full(len(texts), np.nan)
    test_pred = np.zeros(len(test))
    trained = 0

    for fold, (tr, va) in enumerate(folds):
        t_fold = time.time()
        torch.manual_seed(SEED + fold)
        model = Regressor().to(device)
        dl_tr = DataLoader(Listings(texts[tr], y_log[tr]), batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate, num_workers=2, pin_memory=True, drop_last=True)
        dl_va = DataLoader(Listings(texts[va], y_log[va]), batch_size=args.batch_size * 2,
                           shuffle=False, collate_fn=collate, num_workers=2)

        opt = torch.optim.AdamW(param_groups(model))
        total_steps = len(dl_tr) * args.epochs
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))  # no-op di CPU
        loss_fn = nn.SmoothL1Loss(beta=0.15)   # Huber: L1 yang halus di dekat nol

        best_mae, best_val, best_test = np.inf, None, None
        for epoch in range(args.epochs):
            model.train()
            running, seen, t0 = 0.0, 0, time.time()
            for step, batch in enumerate(dl_tr):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = batch["labels"]
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = loss_fn(model(**batch), labels)
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)      # overflow fp16: buang step
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                running += loss.item() * len(labels)
                seen += len(labels)
                if step % 100 == 0:
                    done = step + 1
                    eta = (time.time() - t0) / done * (len(dl_tr) - done)
                    print(f"      fold {fold} epoch {epoch}  step {done}/{len(dl_tr)}  "
                          f"loss {running / max(seen, 1):.4f}  sisa ~{eta / 60:.1f} menit", flush=True)

            val_pred = infer(model, dl_va)
            val_mae = mae(y_raw[va], from_log(val_pred))
            print(f"   fold {fold} epoch {epoch}: val MAE = {val_mae:,.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if val_mae < best_mae:
                best_mae, best_val = val_mae, val_pred
                best_test = infer(model, dl_test)

        oof[va] = best_val
        test_pred += best_test
        trained += 1
        print(f"   fold {fold} selesai: MAE {best_mae:,.0f} "
              f"({(time.time() - t_fold) / 60:.1f} menit)", flush=True)
        del model, opt
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()

    if trained == 0:
        return {}, {}
    test_pred /= trained
    done = ~np.isnan(oof)
    print(f"   -> [transformer] MAE = {mae(y_raw[done], from_log(oof[done])):,.0f} "
          f"(atas {done.sum():,} baris)", flush=True)
    return {"transformer": oof}, {"transformer": test_pred}


# =============================================================================
# 5. GABUNGKAN PREDIKSI & TULIS SUBMISSION
# =============================================================================

def blend(oof: dict, testp: dict, y_raw, folds, train=None, test=None):
    """Tiga langkah, masing-masing hanya dipakai kalau benar-benar memperbaiki MAE."""
    import lightgbm as lgb

    print("\n[4/5] GABUNGKAN prediksi")
    for n, o in sorted(oof.items(), key=lambda kv: mae(y_raw, from_log(kv[1]))):
        print(f"   {n:14s} MAE {mae(y_raw, from_log(o)):,.0f}")

    # (a) rata-rata berbobot, dicari greedy langsung terhadap MAE
    names = list(oof)
    singles = {n: mae(y_raw, from_log(oof[n])) for n in names}
    best0 = min(singles, key=singles.get)
    cur, total = oof[best0].copy(), 1
    counts = {n: 0 for n in names}
    counts[best0] = 1
    best_score, best_counts = singles[best0], dict(counts)
    for _ in range(40):
        pick, pick_score = None, np.inf
        for n in names:
            s = mae(y_raw, from_log((cur * total + oof[n]) / (total + 1)))
            if s < pick_score:
                pick, pick_score = n, s
        cur = (cur * total + oof[pick]) / (total + 1)
        total += 1
        counts[pick] += 1
        if pick_score < best_score - 1e-9:
            best_score, best_counts = pick_score, dict(counts)
    tb = sum(best_counts.values())
    weights = {n: c / tb for n, c in best_counts.items() if c}
    print(f"   bobot: { {k: round(v, 3) for k, v in weights.items()} }")
    print(f"   rata-rata berbobot MAE = {best_score:,.0f}")
    b_oof = sum(w * oof[n] for n, w in weights.items())
    b_test = sum(w * testp[n] for n, w in weights.items())
    score = best_score

    # (b) stacking: LightGBM belajar DI MANA tiap model bisa dipercaya
    if len(oof) >= 3:
        cols = sorted(oof)
        X = np.column_stack([oof[n] for n in cols])
        X_te = np.column_stack([testp[n] for n in cols])
        X = np.column_stack([X, X.std(1), X.mean(1)])
        X_te = np.column_stack([X_te, X_te.std(1), X_te.mean(1)])
        if train is not None:
            # Beri juga fitur regex mentah: model level-2 jadi bisa belajar
            # bahwa mis. kNN dipercaya untuk rumah biasa tapi tidak untuk tanah
            # kosong. Ini menurunkan MAE sekitar 9.000.
            X = np.column_stack([X, np.nan_to_num(build_features(train["text"]), nan=-999)])
            X_te = np.column_stack([X_te, np.nan_to_num(build_features(test["text"]), nan=-999)])
        y_log = to_log(y_raw)
        params = dict(objective="mae", metric="mae", learning_rate=0.02, num_leaves=15,
                      min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=4, seed=0)
        s_oof, s_test = np.zeros(len(y_log)), np.zeros(X_te.shape[0])
        for tr, va in folds:
            dtr = lgb.Dataset(X[tr], label=y_log[tr])
            bst = lgb.train(params, dtr, 3000,
                            valid_sets=[lgb.Dataset(X[va], label=y_log[va], reference=dtr)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            s_oof[va] = bst.predict(X[va], num_iteration=bst.best_iteration)
            s_test += bst.predict(X_te, num_iteration=bst.best_iteration) / len(folds)
        print(f"   stacking MAE = {mae(y_raw, from_log(s_oof)):,.0f}")
        mixes = {w: mae(y_raw, from_log(w * s_oof + (1 - w) * b_oof)) for w in np.arange(0, 1.01, 0.1)}
        w = min(mixes, key=mixes.get)
        if mixes[w] < score:
            b_oof, b_test, score = w * s_oof + (1 - w) * b_oof, w * s_test + (1 - w) * b_test, mixes[w]
            print(f"   campuran w={w:.1f} -> {score:,.0f}")

    # (c) kalibrasi per-desil, diuji DI DALAM fold sebelum dipercaya
    def fit_cal(p, y):
        edges = np.quantile(p, np.linspace(0, 1, 13))
        edges[0], edges[-1] = -np.inf, np.inf
        off = np.zeros(12)
        for b in range(12):
            m = (p >= edges[b]) & (p < edges[b + 1])
            if m.sum() >= 50:
                off[b] = np.median(to_log(y[m]) - p[m])
        return edges, off

    def apply_cal(p, edges, off):
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(off) - 1)
        sm = np.convolve(off, [0.25, 0.5, 0.25], mode="same")
        sm[0], sm[-1] = off[0], off[-1]
        return p + sm[idx]

    cal = np.zeros_like(b_oof)
    for tr, va in folds:
        e, o = fit_cal(b_oof[tr], y_raw[tr])
        cal[va] = apply_cal(b_oof[va], e, o)
    cal_score = mae(y_raw, from_log(cal))
    if cal_score < score:
        e, o = fit_cal(b_oof, y_raw)
        b_test, score = apply_cal(b_test, e, o), cal_score
        print(f"   kalibrasi per-desil -> {score:,.0f}")
    else:
        print("   kalibrasi tidak membantu, dilewati")
    return b_test, score


def write_submission(test, prices, data_path, out_path):
    """Isi kolom listPrice di sample_submission.csv dengan prediksi."""
    preds = pd.DataFrame({"id": test["id"].values, "listPrice": np.round(prices, 2)})
    sample_file = os.path.join(data_path, "sample_submission.csv")
    if os.path.exists(sample_file):
        sub = pd.read_csv(sample_file)[["id"]].merge(preds, on="id", how="left")
        if sub["listPrice"].isna().any():
            n = int(sub["listPrice"].isna().sum())
            print(f"   !! {n} id tanpa prediksi, diisi median", file=sys.stderr)
            sub["listPrice"] = sub["listPrice"].fillna(preds["listPrice"].median())
    else:
        sub = preds
    sub.to_csv(out_path, index=False)
    return sub


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Latih, prediksi, tulis submission")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--head-lr", type=float, default=HEAD_LR)
    ap.add_argument("--folds", type=int, default=N_FOLDS, help="jumlah fold transformer")
    ap.add_argument("--no-transformer", action="store_true", help="CPU saja")
    ap.add_argument("--no-cpu-models", action="store_true", help="transformer saja")
    ap.add_argument("--quick", action="store_true", help="1 fold 1 epoch, untuk uji cepat")
    ap.add_argument("--cpu-transformer", action="store_true",
                    help="paksa fine-tune di CPU (lambat sekali; untuk uji saja)")
    ap.add_argument("--out", default=SUBMISSION_NAME)

    if argv is None:
        # Di notebook, sys.argv milik kernel ("-f /tmp/xxx.json") -> jangan dibaca.
        try:
            from IPython import get_ipython
            nb = get_ipython() is not None and get_ipython().__class__.__name__ != "TerminalInteractiveShell"
        except Exception:
            nb = False
        argv = [] if nb else sys.argv[1:]
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"argumen diabaikan: {' '.join(unknown)}", file=sys.stderr)
    if args.quick:
        args.folds, args.epochs = 1, 1

    t_start = time.time()
    data_path = find_data_dir()
    out_path = os.path.join(output_dir(), args.out)

    print("=" * 70)
    print(" HoloMine Task 2 - latih, prediksi, tulis submission")
    print("=" * 70)
    print(f"data       : {data_path}")
    print(f"submission : {out_path}")

    print("\n[1/5] BACA data")
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")
    y_raw = train["listPrice"].values.astype(float)
    print(f"   train {train.shape}, test {test.shape}")
    print(f"   harga: median ${np.median(y_raw):,.0f}, maks ${y_raw.max():,.0f}")
    print(f"   baseline tebak-median: MAE {mae(y_raw, np.median(y_raw)):,.0f} "
          f"<- angka yang harus dikalahkan")

    folds = make_folds(y_raw)
    oof, testp = {}, {}

    if not args.no_cpu_models:
        o, t = train_cpu_models(train, test, y_raw, folds)
        oof.update(o)
        testp.update(t)
    if not args.no_transformer:
        o, t = train_transformer(train, test, y_raw, folds[:args.folds], args)
        oof.update(o)
        testp.update(t)

    if not oof:
        print("\nTidak ada model yang berhasil dilatih. Berhenti.", file=sys.stderr)
        return 1

    # Model yang tidak dilatih di SEMUA fold tidak boleh ikut menentukan bobot
    # blend: bobotnya akan dihitung dari baris yang prediksinya tidak ada.
    partial = [n for n, o in oof.items() if np.isnan(o).any()]
    if partial and len(oof) > len(partial):
        for n in partial:
            frac = float(np.mean(~np.isnan(oof[n])))
            print(f"\n   ! '{n}' hanya dilatih di {frac:.0%} baris (mode --quick), "
                  f"dikeluarkan dari blend.")
            oof.pop(n)
            testp.pop(n)

    if len(oof) == 1:
        name = next(iter(oof))
        valid = ~np.isnan(oof[name])
        final_log, score = testp[name], mae(y_raw[valid], from_log(oof[name][valid]))
        if not valid.all():
            print(f"\n   ! skor dihitung atas {valid.sum():,} dari {len(valid):,} baris "
                  f"(fold sisanya tidak dilatih). Jangan pakai --quick untuk submission nyata.")
    else:
        final_log, score = blend(oof, testp, y_raw, folds, train, test)

    print("\n[5/5] TULIS submission")
    sub = write_submission(test, from_log(final_log), data_path, out_path)

    print("\n" + "=" * 70)
    print(f" SELESAI dalam {(time.time() - t_start) / 60:.1f} menit")
    print("=" * 70)
    print(f"File   : {out_path}")
    print(f"Baris  : {len(sub)}  (format sama dengan sample_submission.csv)")
    print(f"CV MAE : {score:,.0f}   vs baseline {mae(y_raw, np.median(y_raw)):,.0f}")
    print("\nContoh isinya:")
    print(sub.head(5).to_string(index=False))
    if os.path.isdir("/kaggle/working"):
        print("\nIsi /kaggle/working (inilah Output notebook setelah Save Version):")
        for f in sorted(os.listdir("/kaggle/working")):
            full = os.path.join("/kaggle/working", f)
            print(f"   {f}" + ("/" if os.path.isdir(full) else f"  {os.path.getsize(full):,} byte"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
