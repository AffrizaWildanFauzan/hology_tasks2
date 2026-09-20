#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
 HoloMine Task 2 - Prediksi listPrice dari teks listing properti  (FILE TUNGGAL)
===============================================================================

Semua ada di file ini: ekstraksi fitur, model CPU, embedding HuggingFace,
fine-tune transformer, blending, sampai submission. Tidak butuh file lain.

CARA PAKAI
----------
    pip install numpy pandas scikit-learn scipy lightgbm
    pip install torch transformers sentence-transformers sentencepiece   # untuk tahap HF

    python holomine_solution.py                    # jalankan semua (fine-tune butuh GPU)
    python holomine_solution.py --dry-run          # lihat rencananya dulu
    python holomine_solution.py --stage cpu        # model CPU saja (~10 menit, tanpa GPU)
    python holomine_solution.py --tier extra       # + model besar kalau kuota GPU cukup

Di Kaggle: Accelerator = GPU, Internet = ON, lalu
    !python holomine_solution.py --tier core

KUNCI SKOR
----------
Metrik MAE diminimalkan oleh MEDIAN bersyarat, bukan rata-rata. Karena exp()
monoton, median di ruang log = log dari median di ruang dolar. Jadi semua model
dilatih pada log(listPrice) dengan loss L1/Huber lalu di-exp(). Ini sekaligus
mencegah satu rumah $80 juta mendominasi gradien. Melatih langsung di skala
dolar dengan MSE adalah kesalahan paling mahal di kompetisi ini.
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
# 1. KONFIGURASI
# =============================================================================

SEED = 42
N_SPLITS = 5

def in_notebook() -> bool:
    """True kalau kode ini ditempel/dijalankan di dalam Jupyter, Colab, atau
    notebook Kaggle. Dipakai untuk dua hal: jangan baca sys.argv (itu milik
    kernel, bukan milik kita), dan jangan auto-run saat sel dieksekusi."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


# __file__ tidak ada kalau file ini ditempel ke sel notebook -> pakai cwd.
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()


def _is_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w"):
            pass
        os.remove(probe)
        return True
    except Exception:
        return False


def _default_out(name: str, flat_on_kaggle: bool = False) -> str:
    """Folder output.

    Di Kaggle semuanya diarahkan ke /kaggle/working: itu satu-satunya folder
    yang bisa ditulis DAN ikut tersimpan sebagai output notebook. /kaggle/input
    bersifat READ-ONLY, jadi mengarahkan ROOT ke sana pasti gagal saat menyimpan.
    """
    base = ROOT
    on_kaggle = os.path.isdir("/kaggle/working")
    if on_kaggle and not base.startswith("/kaggle/working"):
        base = "/kaggle/working"
    elif not _is_writable(base):
        base = os.getcwd()
    # Kaggle mengharapkan submission di /kaggle/working/submission.csv, bukan
    # di dalam subfolder, supaya langsung terdeteksi sebagai output notebook.
    if flat_on_kaggle and base == "/kaggle/working":
        return base
    return os.path.join(base, name)


ARTIFACTS = os.environ.get("HOLOMINE_ARTIFACTS") or _default_out("artifacts")
SUBMISSIONS = os.environ.get("HOLOMINE_SUBMISSIONS") or _default_out("submissions", flat_on_kaggle=True)

# --- kombinasi model HuggingFace yang direkomendasikan ---
# tier "core" = selalu jalan, "extra" = hanya dengan --tier extra,
# "cpu" = dipakai sebagai pengganti kalau tidak ada GPU.
FINETUNE = {
    "deb3base": dict(
        tier="core", model="microsoft/deberta-v3-base",
        max_len=512, batch_size=16, accum=1, epochs=3,
        lr=2e-5, head_lr=1e-4, llrd=0.9, vram_gb=10,
        note="Juara de-facto regresi teks di Kaggle. Titik awal terbaik.",
    ),
    "mbert": dict(
        tier="core", model="answerdotai/ModernBERT-base",
        # Diukur di train.csv: p99 = 521 token, dan pada 512 cuma 1,1% listing
        # terpotong. Jadi 1024 hanya menggandakan waktu latih tanpa informasi
        # tambahan. Nilai model ini ada di arsitektur & tokenizer-nya yang
        # berbeda dari DeBERTa, bukan di konteks panjangnya.
        max_len=512, batch_size=16, accum=1, epochs=3,
        lr=3e-5, head_lr=1e-4, llrd=0.9, vram_gb=10,
        note="Arsitektur & tokenizer beda -> error tidak berkorelasi.",
    ),
    "deb3large": dict(
        tier="extra", model="microsoft/deberta-v3-large",
        max_len=512, batch_size=4, accum=4, epochs=3,
        lr=8e-6, head_lr=5e-5, llrd=0.85, grad_checkpoint=True, vram_gb=16,
        note="Biasanya lebih baik dari base, tapi rawan divergen di fp16.",
    ),
    "mbertlarge": dict(
        tier="extra", model="answerdotai/ModernBERT-large",
        max_len=512, batch_size=4, accum=4, epochs=3,
        lr=1e-5, head_lr=5e-5, llrd=0.85, grad_checkpoint=True, vram_gb=16,
        note="Pasangan besar yang beragam dari DeBERTa.",
    ),
}

EMBED = {
    "gte":    dict(tier="core",  model="Alibaba-NLP/gte-modernbert-base", max_len=512, batch_size=32,
                   note="Embedding kuat berbasis ModernBERT, tanpa training."),
    "bge":    dict(tier="extra", model="BAAI/bge-base-en-v1.5", max_len=512, batch_size=64,
                   note="Standar industri, sangat stabil."),
    "e5":     dict(tier="extra", model="intfloat/e5-base-v2", max_len=512, batch_size=64,
                   prefix="query: ", note="Wajib prefiks 'query: '."),
    "qwen":   dict(tier="extra", model="Qwen/Qwen3-Embedding-0.6B", max_len=512, batch_size=16,
                   note="Peringkat MTEB teratas di kelas kecil."),
    "minilm": dict(tier="cpu",   model="sentence-transformers/all-MiniLM-L6-v2", max_len=256,
                   batch_size=64, note="Cukup cepat untuk CPU."),
}

SPARSE_MODELS = ["ridge_word", "ridge_char", "linsvr_word", "knn_word", "lgbm_dense"]


# =============================================================================
# 2. DATA, FOLD, METRIK
# =============================================================================

def data_dir() -> str:
    """Cari folder yang berisi train.csv.

    Kaggle menaruh data di /kaggle/input/<nama-kompetisi>/, tapi lewat mount
    Colab bisa jadi /kaggle/input/competitions/<nama>/ -- jadi dua tingkat
    dicari, bukan satu. Anda TIDAK perlu mengubah ROOT untuk ini; kalau mau
    memaksa, set HOLOMINE_DATA.
    """
    searched = []
    cands = []
    env = os.environ.get("HOLOMINE_DATA")
    if env:
        cands.append(env)
    cands += [ROOT, os.getcwd()]
    for pattern in ("/kaggle/input/*", "/kaggle/input/*/*", "/content/*", "/content/*/*"):
        cands += sorted(glob.glob(pattern))
    for cand in cands:
        if not cand or not os.path.isdir(cand):
            continue
        searched.append(cand)
        if os.path.exists(os.path.join(cand, "train.csv")):
            return cand
    raise FileNotFoundError(
        "train.csv tidak ketemu. Set HOLOMINE_DATA ke folder yang berisi "
        "train.csv/test.csv, mis.\n"
        "  os.environ['HOLOMINE_DATA'] = '/kaggle/input/nama-kompetisi'\n"
        "Folder yang sudah dicari: " + ", ".join(searched[:12]))


def load_data():
    d = data_dir()
    train = pd.read_csv(os.path.join(d, "train.csv"))
    test = pd.read_csv(os.path.join(d, "test.csv"))
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")
    return train, test


def price_bin_folds(y: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED):
    """Fold distratifikasi atas desil log-harga, supaya ekor mahal terbagi rata.
    Semua model memakai fold yang sama -> OOF-nya bisa di-blend dengan jujur."""
    from sklearn.model_selection import StratifiedKFold
    bins = pd.qcut(np.log(y), q=20, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), bins))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def to_log(y) -> np.ndarray:
    return np.log(np.clip(np.asarray(y, dtype=float), 1.0, None))


def from_log(z) -> np.ndarray:
    """Kembali ke dolar. Fit L1 di ruang log -> exp() = median bersyarat,
    persis yang diminta MAE."""
    return np.exp(np.clip(np.asarray(z, dtype=float), 0.0, np.log(3e8)))


def save_oof(name: str, oof: np.ndarray, test_pred: np.ndarray) -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)
    np.save(os.path.join(ARTIFACTS, f"oof_{name}.npy"), oof)
    np.save(os.path.join(ARTIFACTS, f"test_{name}.npy"), test_pred)


def artifact_done(name: str) -> bool:
    return (os.path.exists(os.path.join(ARTIFACTS, f"oof_{name}.npy"))
            and os.path.exists(os.path.join(ARTIFACTS, f"test_{name}.npy")))


def load_all_oof():
    oofs, tests = {}, {}
    for p in sorted(glob.glob(os.path.join(ARTIFACTS, "oof_*.npy"))):
        name = os.path.basename(p)[4:-4]
        tp = os.path.join(ARTIFACTS, f"test_{name}.npy")
        if os.path.exists(tp):
            oofs[name] = np.load(p)
            tests[name] = np.load(tp)
    return oofs, tests


def make_submission(test_ids, prices, filename: str) -> str:
    os.makedirs(SUBMISSIONS, exist_ok=True)
    path = os.path.join(SUBMISSIONS, filename)
    pd.DataFrame({"id": test_ids, "listPrice": np.round(np.asarray(prices), 2)}).to_csv(path, index=False)
    return path


def explain_hub_error(exc: Exception, model_name: str) -> str:
    """Terjemahkan kegagalan unduh HuggingFace jadi langkah yang bisa dikerjakan."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    hints = [f"Gagal memuat '{model_name}'.", f"  penyebab: {text.splitlines()[0][:200]}"]
    if any(k in low for k in ("403", "proxy", "connect", "resolve", "timed out")):
        hints += ["  -> Jaringan memblokir huggingface.co.",
                  "     Kaggle: Notebook Settings -> Internet: ON.",
                  "     Offline: unggah bobot sebagai Kaggle Dataset, ganti 'model' di",
                  "     dict FINETUNE/EMBED ke path lokalnya, lalu set HF_HUB_OFFLINE=1."]
    elif any(k in low for k in ("401", "gated", "authoriz")):
        hints += ["  -> Model gated: terima lisensinya lalu `huggingface-cli login`."]
    elif any(k in low for k in ("404", "not a local folder", "repositorynotfound")):
        hints += ["  -> Nama model salah ketik, atau repo privat."]
    elif "sentencepiece" in low or "protobuf" in low:
        hints += ["  -> Tokenizer deberta-v3 butuh: pip install sentencepiece protobuf"]
    elif any(k in low for k in ("modernbert", "unrecognized", "trust_remote_code")):
        hints += ["  -> ModernBERT butuh: pip install -U 'transformers>=4.48'"]
    return "\n".join(hints)


# =============================================================================
# 3. INFORMATION EXTRACTION: teks bebas -> fitur numerik
# =============================================================================
# Kompetisi hanya memberi teks, jadi semua kolom yang biasanya dimiliki model
# properti (kamar, luas, tanah, tahun bangun) harus digali balik dari prosa.

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
# 4. MODEL CPU: TF-IDF + kNN + LightGBM  (tanpa HuggingFace sama sekali)
# =============================================================================

def run_cv(name, fit_predict, X, Xte, y_log, y_raw, splits):
    """Latih satu model di semua fold, simpan OOF + prediksi test (ruang log)."""
    oof = np.zeros(len(y_log))
    test_pred = np.zeros(Xte.shape[0])
    t0 = time.time()
    for k, (trn, val) in enumerate(splits):
        p_val, p_test = fit_predict(X, y_log, trn, val, Xte, k)
        oof[val] = p_val
        test_pred += p_test / len(splits)
        print(f"  fold {k}: MAE={mae(y_raw[val], from_log(p_val)):,.0f} ({time.time() - t0:.0f}s)", flush=True)
    score = mae(y_raw, from_log(oof))
    print(f"[{name}] CV MAE = {score:,.0f}", flush=True)
    save_oof(name, oof, test_pred)
    return score


def ridge_fp(alpha=1.0):
    from sklearn.linear_model import Ridge

    def fp(X, y, trn, val, Xte, k):
        m = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-4, random_state=SEED)
        m.fit(X[trn], y[trn])
        return m.predict(X[val]), m.predict(Xte)
    return fp


def linsvr_fp(C=0.5, epsilon=0.05):
    """Loss epsilon-insensitive = L1 -> mencocokkan MEDIAN log-harga, yang persis
    merupakan optimum MAE. Pelengkap ridge yang berbasis L2."""
    from sklearn.svm import LinearSVR

    def fp(X, y, trn, val, Xte, k):
        m = LinearSVR(C=C, epsilon=epsilon, loss="epsilon_insensitive", dual=True,
                      max_iter=3000, tol=1e-4, random_state=SEED + k)
        m.fit(X[trn], y[trn])
        return m.predict(X[val]), m.predict(Xte)
    return fp


def knn_fp(k_neighbors=25):
    """Tetangga kosinus di TF-IDF: listing rumah serupa di pasar yang sama memakai
    frasa yang mirip, jadi MEDIAN berbobot harga tetangga adalah sudut pandang
    yang benar-benar berbeda dari model linear/tree."""
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
                    preds[s + i] = np.median(y[trn]) if w.sum() <= 0 else weighted_median(y[trn][idx[i]], w)
            out.append(preds)
        return out[0], out[1]
    return fp


def lgbm_fp(params=None, num_round=4000):
    import lightgbm as lgb

    # Objective L1 memang lebih lambat dari L2 (LightGBM menghitung ulang median
    # tiap daun), tapi early stopping menghentikannya jauh sebelum 4000 ronde:
    # sekitar 45 detik per fold di 4 core.
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
        return (m.predict(X[val], num_iteration=m.best_iteration),
                m.predict(Xte, num_iteration=m.best_iteration))
    return fp


def stage_cpu(train, test, y_raw, splits, wanted, force=False):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    todo = [m for m in wanted if force or not artifact_done(m)]
    if not todo:
        print("[cpu] semua artefak sudah ada, dilewati")
        return {}
    y_log = to_log(y_raw)
    scores = {}

    print("membangun TF-IDF kata ...", flush=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                          strip_accents="unicode", lowercase=True, dtype=np.float32)
    Xw = vec.fit_transform(train["text"])
    Xw_te = vec.transform(test["text"])
    print("  ", Xw.shape, flush=True)

    if "ridge_word" in todo:
        scores["ridge_word"] = run_cv("ridge_word", ridge_fp(1.0), Xw, Xw_te, y_log, y_raw, splits)
    if "linsvr_word" in todo:
        scores["linsvr_word"] = run_cv("linsvr_word", linsvr_fp(), Xw, Xw_te, y_log, y_raw, splits)
    if "knn_word" in todo:
        scores["knn_word"] = run_cv("knn_word", knn_fp(), Xw, Xw_te, y_log, y_raw, splits)

    if "ridge_char" in todo:
        print("membangun TF-IDF karakter ...", flush=True)
        cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                               max_features=300000, lowercase=True, dtype=np.float32)
        Xc = cvec.fit_transform(train["text"])
        Xc_te = cvec.transform(test["text"])
        print("  ", Xc.shape, flush=True)
        scores["ridge_char"] = run_cv("ridge_char", ridge_fp(1.0), Xc, Xc_te, y_log, y_raw, splits)
        del Xc, Xc_te
        gc.collect()

    if "lgbm_dense" in todo:
        print("SVD(250) + fitur regex ...", flush=True)
        svd = TruncatedSVD(n_components=250, random_state=SEED)
        Z = svd.fit_transform(Xw).astype(np.float32)
        Z_te = svd.transform(Xw_te).astype(np.float32)
        D = np.hstack([Z, np.nan_to_num(build_features(train["text"]), nan=-999)])
        D_te = np.hstack([Z_te, np.nan_to_num(build_features(test["text"]), nan=-999)])
        scores["lgbm_dense"] = run_cv("lgbm_dense", lgbm_fp(), D, D_te, y_log, y_raw, splits)
    return scores


# =============================================================================
# 5. EMBEDDING BEKU (HuggingFace, tanpa training)
# =============================================================================
# Jauh lebih murah dari fine-tune (sekali forward pass) dan biasanya berada di
# antara TF-IDF dan encoder yang di-fine-tune, tapi sangat beragam dari keduanya
# -- itulah yang membuatnya berguna di blend.

def encode_texts(model_name, texts, max_len, batch_size, prefix=""):
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


def stage_embed(tag, cfg, train, test, y_raw, splits, force=False):
    import lightgbm as lgb
    from sklearn.linear_model import RidgeCV

    if not force and artifact_done(f"emb_{tag}_lgbm"):
        print(f"[embed] {tag} sudah ada, dilewati")
        return {}

    os.makedirs(ARTIFACTS, exist_ok=True)
    tr_path = os.path.join(ARTIFACTS, f"emb_{tag}_train.npy")
    te_path = os.path.join(ARTIFACTS, f"emb_{tag}_test.npy")
    if os.path.exists(tr_path) and os.path.exists(te_path) and not force:
        E_tr, E_te = np.load(tr_path), np.load(te_path)
        print(f"[embed] memakai embedding {tag} dari cache")
    else:
        E_tr = encode_texts(cfg["model"], train["text"].values, cfg["max_len"],
                            cfg["batch_size"], cfg.get("prefix", ""))
        E_te = encode_texts(cfg["model"], test["text"].values, cfg["max_len"],
                            cfg["batch_size"], cfg.get("prefix", ""))
        np.save(tr_path, E_tr)
        np.save(te_path, E_te)
    print(f"[embed] {tag}: {E_tr.shape}", flush=True)

    y_log = to_log(y_raw)
    F_tr = np.nan_to_num(build_features(train["text"]), nan=-999)
    F_te = np.nan_to_num(build_features(test["text"]), nan=-999)
    X_tr = np.hstack([E_tr, F_tr])
    X_te = np.hstack([E_te, F_te])

    scores = {}
    for head in ("ridge", "lgbm"):
        oof = np.zeros(len(y_log))
        test_pred = np.zeros(len(E_te))
        for trn, val in splits:
            if head == "ridge":
                m = RidgeCV(alphas=np.logspace(-2, 3, 12))
                m.fit(E_tr[trn], y_log[trn])
                oof[val], p = m.predict(E_tr[val]), m.predict(E_te)
            else:
                params = dict(objective="mae", metric="mae", learning_rate=0.05, num_leaves=31,
                              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1,
                              min_data_in_leaf=20, lambda_l2=1.0, verbosity=-1,
                              num_threads=4, seed=SEED)
                dtr = lgb.Dataset(X_tr[trn], label=y_log[trn])
                dva = lgb.Dataset(X_tr[val], label=y_log[val], reference=dtr)
                bst = lgb.train(params, dtr, 1500, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
                oof[val] = bst.predict(X_tr[val], num_iteration=bst.best_iteration)
                p = bst.predict(X_te, num_iteration=bst.best_iteration)
            test_pred += p / len(splits)
        name = f"emb_{tag}_{head}"
        scores[name] = mae(y_raw, from_log(oof))
        print(f"[{name}] CV MAE = {scores[name]:,.0f}", flush=True)
        save_oof(name, oof, test_pred)
    return scores


# =============================================================================
# 6. FINE-TUNE ENCODER HUGGINGFACE  (butuh GPU untuk praktis)
# =============================================================================
# * Target log(price), loss SmoothL1 (Huber): MAE diminimalkan median bersyarat,
#   dan exp() dari fit L1 di ruang log PERSIS median itu.
# * Mean pooling mengalahkan [CLS] untuk teks deskriptif panjang.
# * Layer-wise LR decay: lapisan bawah menyimpan pengetahuan bahasa umum,
#   head dan lapisan atas beradaptasi paling cepat.

def _torch():
    import torch
    return torch


def resolve_amp_dtype(precision: str, device):
    """None = fp32 penuh. T4/V100 tidak punya bf16, jadi 'auto' turun ke fp16.
    deberta-v3-large dikenal overflow di fp16 -> pakai --precision fp32."""
    torch = _torch()
    if device.type != "cuda" or precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_dataset_classes():
    torch = _torch()
    from torch.utils.data import Dataset

    class ListingDataset(Dataset):
        def __init__(self, texts, targets, tokenizer, max_len):
            self.texts = list(texts)
            self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
            self.tok = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                           padding=False, return_tensors=None)
            item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")}
            if self.targets is not None:
                item["labels"] = torch.tensor(self.targets[i], dtype=torch.float)
            return item

    class MeanPoolRegressor(torch.nn.Module):
        def __init__(self, model_name, dropout=0.0):
            super().__init__()
            from transformers import AutoConfig, AutoModel
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            cfg.update({"hidden_dropout_prob": dropout, "attention_probs_dropout_prob": dropout})
            self.backbone = AutoModel.from_pretrained(model_name, config=cfg, trust_remote_code=True)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(cfg.hidden_size),
                                            torch.nn.Linear(cfg.hidden_size, 1))

        def forward(self, **batch):
            batch.pop("labels", None)
            mask = batch["attention_mask"]
            out = self.backbone(**batch).last_hidden_state
            m = mask.unsqueeze(-1).to(out.dtype)
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return self.head(pooled).squeeze(-1)

    return ListingDataset, MeanPoolRegressor


def layerwise_params(model, base_lr, head_lr, decay=0.9, weight_decay=0.01):
    """Lapisan encoder teratas dapat ~base_lr, tiap lapisan di bawahnya dikali decay."""
    backbone = model.backbone
    layers = None
    for attr in ("encoder", "layers"):
        obj = getattr(backbone, attr, None)
        if obj is not None:
            layers = getattr(obj, "layer", None) or getattr(obj, "layers", None) or obj
            break
    groups = [{"params": model.head.parameters(), "lr": head_lr, "weight_decay": weight_decay}]
    if layers is None or not hasattr(layers, "__len__"):
        groups.append({"params": backbone.parameters(), "lr": base_lr, "weight_decay": weight_decay})
        return groups
    n = len(layers)
    assigned = set()
    for i, layer in enumerate(layers):
        groups.append({"params": layer.parameters(), "lr": base_lr * (decay ** (n - 1 - i)),
                       "weight_decay": weight_decay})
        assigned.update(id(p) for p in layer.parameters())
    rest = [p for p in backbone.parameters() if id(p) not in assigned]
    if rest:
        groups.append({"params": rest, "lr": base_lr * (decay ** n), "weight_decay": weight_decay})
    return groups


def predict_loader(model, loader, device, amp_dtype):
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out.append(model(**batch).float().cpu())
    return torch.cat(out).numpy()


def train_transformer_fold(cfg, args, texts, y_log, y_raw, trn, val, test_texts,
                           fold, tokenizer, collator, device, Dataset, Regressor):
    torch = _torch()
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(SEED + fold)
    model = Regressor(cfg["model"], dropout=args.dropout).to(device)
    if cfg.get("grad_checkpoint"):
        model.backbone.gradient_checkpointing_enable()

    mk = lambda ds, bs, sh, dl: DataLoader(ds, batch_size=bs, shuffle=sh, collate_fn=collator,
                                           num_workers=args.workers, drop_last=dl)
    dl_tr = mk(Dataset(texts[trn], y_log[trn], tokenizer, cfg["max_len"]), cfg["batch_size"], True, True)
    dl_va = mk(Dataset(texts[val], y_log[val], tokenizer, cfg["max_len"]), args.eval_batch_size, False, False)
    dl_te = mk(Dataset(test_texts, None, tokenizer, cfg["max_len"]), args.eval_batch_size, False, False)

    opt = torch.optim.AdamW(layerwise_params(model, cfg["lr"], cfg["head_lr"], cfg["llrd"], args.weight_decay))
    accum = cfg.get("accum", 1)
    steps = max(1, len(dl_tr) // accum) * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    amp_dtype = resolve_amp_dtype(args.precision, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    loss_fn = torch.nn.SmoothL1Loss(beta=args.huber_beta)

    best = (np.inf, None, None)
    bad_steps = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(dl_tr):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model(**batch), labels) / accum
            if not torch.isfinite(loss):
                # overflow fp16 (klasik di deberta-v3-large): buang step ini
                bad_steps += 1
                opt.zero_grad(set_to_none=True)
                if bad_steps == 50:
                    print("  !! banyak loss non-finite -- ulangi dengan --precision fp32", flush=True)
                continue
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        p_val = predict_loader(model, dl_va, device, amp_dtype)
        if not np.isfinite(p_val).all():
            print(f"  fold {fold} epoch {epoch}: prediksi non-finite, epoch dilewati", flush=True)
            continue
        score = mae(y_raw[val], from_log(p_val))
        print(f"  fold {fold} epoch {epoch}: val MAE = {score:,.0f}  ({time.time() - t0:.0f}s)", flush=True)
        if score < best[0]:
            best = (score, p_val, predict_loader(model, dl_te, device, amp_dtype))

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return best


def stage_finetune(tag, cfg, args, train, test, y_raw, splits, force=False):
    torch = _torch()
    from transformers import AutoTokenizer, DataCollatorWithPadding

    if not force and artifact_done(tag):
        print(f"[finetune] {tag} sudah ada, dilewati")
        return {}

    texts = train["text"].values
    test_texts = test["text"].values
    y_log = to_log(y_raw)
    Dataset, Regressor = build_dataset_classes()

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    except Exception as exc:
        print(explain_hub_error(exc, cfg["model"]), file=sys.stderr)
        return {tag: None}

    sample = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts[:2000]]
    cut = float(np.mean(np.asarray(sample) > cfg["max_len"]))
    print(f"[finetune] {tag} ({cfg['model']}): panjang token median {int(np.median(sample))}, "
          f"p95 {int(np.percentile(sample, 95))} -> {cut:.1%} listing terpotong di max_len={cfg['max_len']}",
          flush=True)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want = range(len(splits)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]

    oof = np.full(len(texts), np.nan)
    test_pred = np.zeros(len(test_texts))
    used = 0
    for fold, (trn, val) in enumerate(splits):
        if fold not in want:
            continue
        try:
            score, p_val, p_test = train_transformer_fold(
                cfg, args, texts, y_log, y_raw, trn, val, test_texts, fold,
                tokenizer, collator, device, Dataset, Regressor)
        except Exception as exc:
            print(explain_hub_error(exc, cfg["model"]), file=sys.stderr)
            return {tag: None}
        oof[val] = p_val
        test_pred += p_test
        used += 1
        print(f"fold {fold} MAE terbaik = {score:,.0f}", flush=True)

    test_pred /= max(used, 1)
    done = ~np.isnan(oof)
    score = mae(y_raw[done], from_log(oof[done]))
    print(f"[{tag}] CV MAE (atas {done.sum()} baris) = {score:,.0f}", flush=True)
    if done.all():
        save_oof(tag, oof, test_pred)
    else:
        print("  fold belum lengkap -> tidak disimpan untuk blend "
              "(jalankan semua fold dulu)", flush=True)
    return {tag: score}


# =============================================================================
# 7. BLEND: greedy selection + stacking + kalibrasi
# =============================================================================

def greedy_blend(oofs: dict, y_raw: np.ndarray, n_iter: int = 40):
    """Caruana greedy selection with replacement.

    Menjalankan seluruh budget (bukan berhenti di langkah pertama yang tidak
    membaik -- satu langkah lebih buruk sering membuka kombinasi yang lebih baik)
    dan mengembalikan kombinasi terbaik yang pernah dilihat. Dioptimalkan
    langsung ke MAE dolar, bukan proxy loss di ruang log.
    """
    names = list(oofs)
    singles = {n: mae(y_raw, from_log(oofs[n])) for n in names}
    for n, v in sorted(singles.items(), key=lambda kv: kv[1]):
        print(f"  single {n:18s} {v:,.0f}")

    best0 = min(singles, key=singles.get)
    current, total = oofs[best0].copy(), 1
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

    tb = sum(best_counts.values())
    return {n: c / tb for n, c in best_counts.items() if c}, best_score


def stack_lgbm(oofs, tests, y_raw, splits, extra_train=None, extra_test=None):
    """LightGBM level-2 di atas prediksi tiap model (objective MAE, target log).

    Bisa mengalahkan blend linear karena ia belajar DI MANA tiap model bisa
    dipercaya -- mis. kNN kuat pada listing yang punya kembaran, lemah di luar itu.
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
                  bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=4, seed=0)
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


def fit_calibration(pred_log, y_raw, n_bins: int = 12):
    """Koreksi multiplikatif per-desil di ruang log.

    Blend dari fit yang condong ke rata-rata bias terhadap median optimal MAE,
    dan biasnya berbeda antara rumah murah dan mahal. Kita pelajari satu offset
    log per bin harga prediksi (offset di ruang log = pengali di ruang dolar).
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
    smooth = np.convolve(offsets, np.array([0.25, 0.5, 0.25]), mode="same")
    smooth[0], smooth[-1] = offsets[0], offsets[-1]
    return pred_log + smooth[idx]


def stage_blend(train, test, y_raw, splits, out_name):
    oofs, tests = load_all_oof()
    if not oofs:
        print("[blend] belum ada artefak model; jalankan tahap cpu dulu")
        return None
    print(f"[blend] menggabungkan {len(oofs)} model: {', '.join(sorted(oofs))}")

    weights, score = greedy_blend(oofs, y_raw)
    print("bobot:", {k: round(v, 3) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])})
    print(f"blend CV MAE  = {score:,.0f}")
    blend_oof = sum(w * oofs[n] for n, w in weights.items())
    blend_test = sum(w * tests[n] for n, w in weights.items())

    if len(oofs) >= 3:
        F_tr = np.nan_to_num(build_features(train["text"]), nan=-999)
        F_te = np.nan_to_num(build_features(test["text"]), nan=-999)
        st_oof, st_test, st_score = stack_lgbm(oofs, tests, y_raw, splits, F_tr, F_te)
        print(f"stack CV MAE  = {st_score:,.0f}")
        mixes = {w: mae(y_raw, from_log(w * st_oof + (1 - w) * blend_oof))
                 for w in np.arange(0, 1.01, 0.1)}
        w_best = min(mixes, key=mixes.get)
        print(f"campuran greedy/stack w={w_best:.1f} -> {mixes[w_best]:,.0f}")
        if mixes[w_best] < score:
            blend_oof = w_best * st_oof + (1 - w_best) * blend_oof
            blend_test = w_best * st_test + (1 - w_best) * blend_test
            score = mixes[w_best]

    # kalibrasi hanya dipakai kalau menang saat diuji DI DALAM fold
    cal_oof = np.zeros_like(blend_oof)
    for trn, val in splits:
        e, o = fit_calibration(blend_oof[trn], y_raw[trn])
        cal_oof[val] = apply_calibration(blend_oof[val], e, o)
    cal_score = mae(y_raw, from_log(cal_oof))
    print(f"terkalibrasi  = {cal_score:,.0f} (delta {cal_score - score:+,.0f})")
    if cal_score < score:
        e, o = fit_calibration(blend_oof, y_raw)
        blend_test = apply_calibration(blend_test, e, o)
        score = cal_score
    else:
        print("  -> kalibrasi ditolak, memakai blend mentah")

    path = make_submission(test["id"], from_log(blend_test), out_name)
    print(f"\n==> {path}   (CV MAE {score:,.0f})")
    print("kuantil prediksi:", np.percentile(from_log(blend_test), [1, 25, 50, 75, 99]).round(0))
    return score


# =============================================================================
# 8. RUNNER
# =============================================================================

def have_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} ({p.total_memory / 1e9:.0f} GB)"
    except Exception:
        pass
    return "tidak ada GPU"


def banner(text):
    print("\n" + "=" * 78 + f"\n {text}\n" + "=" * 78, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HoloMine Task 2 - solusi file tunggal")
    ap.add_argument("--stage", default="all", choices=["all", "cpu", "embed", "finetune", "blend"])
    ap.add_argument("--tier", default="core", choices=["core", "extra"])
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--folds", default="all", help="mis. '0,1' untuk uji cepat")
    ap.add_argument("--force", action="store_true", help="latih ulang walau artefak sudah ada")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--assume-gpu", action="store_true", help="dengan --dry-run: tampilkan rencana GPU")
    # knob fine-tune
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--huber-beta", type=float, default=0.15)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    if argv is None:
        # Di notebook, sys.argv milik kernel ("-f /tmp/xxx.json ..."), bukan milik
        # kita -- membacanya bikin argparse mati dengan SystemExit: 2.
        argv = [] if in_notebook() else sys.argv[1:]
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"argumen diabaikan: {' '.join(unknown)}", file=sys.stderr)

    tiers = {"core"} if args.tier == "core" else {"core", "extra"}
    gpu = have_gpu() or (args.assume_gpu and args.dry_run)
    print(f"perangkat : {gpu_name()}" + ("  [--assume-gpu]" if args.assume_gpu else ""))
    print(f"data      : {data_dir()}")
    print(f"artefak   : {ARTIFACTS}")

    embed_picks = ({n: c for n, c in EMBED.items() if c["tier"] in tiers} if gpu
                   else {n: c for n, c in EMBED.items() if c["tier"] == "cpu"})
    ft_picks = {n: c for n, c in FINETUNE.items() if c["tier"] in tiers} if gpu else {}
    if not gpu:
        print("Tanpa GPU: fine-tune dilewati, embedding memakai MiniLM."
              " (--dry-run --assume-gpu untuk melihat rencana lengkap)")

    if args.dry_run:
        banner("RENCANA")
        if args.stage in ("all", "cpu"):
            print(f"  1. model CPU      : {', '.join(SPARSE_MODELS)}")
        if args.stage == "all":
            print("  2. blend sementara -> submission_cpu_only.csv")
        if args.stage in ("all", "embed"):
            for n, c in embed_picks.items():
                print(f"  3. embedding beku : {n:10s} {c['model']}")
        if args.stage in ("all", "finetune"):
            for n, c in ft_picks.items():
                print(f"  4. fine-tune      : {n:10s} {c['model']}  (max_len {c['max_len']}, "
                      f"bs {c['batch_size']}, {c['epochs']} epoch, ~{c['vram_gb']}GB VRAM)")
        if args.stage in ("all", "blend"):
            print(f"  5. blend akhir    -> {args.out}")
        return 0

    train, test = load_data()
    y_raw = train["listPrice"].values.astype(float)
    splits = price_bin_folds(y_raw)
    print(f"train {train.shape} | test {test.shape} | "
          f"baseline tebak-median MAE = {mae(y_raw, np.median(y_raw)):,.0f}")
    failures = []
    final = None

    if args.stage in ("all", "cpu"):
        banner("TAHAP 1/5  model CPU (TF-IDF, kNN, LightGBM)")
        try:
            stage_cpu(train, test, y_raw, splits, SPARSE_MODELS, force=args.force)
        except Exception as exc:
            print(f"GAGAL: {exc}", file=sys.stderr)
            failures.append("cpu")

    if args.stage == "all" and (embed_picks or ft_picks):
        banner("TAHAP 2/5  submission sementara (supaya tidak pernah tangan kosong)")
        stage_blend(train, test, y_raw, splits, "submission_cpu_only.csv")

    if args.stage in ("all", "embed"):
        banner("TAHAP 3/5  embedding beku HuggingFace")
        for tag, cfg in embed_picks.items():
            try:
                stage_embed(tag, cfg, train, test, y_raw, splits, force=args.force)
            except SystemExit:
                failures.append(f"embed:{tag}")
            except Exception as exc:
                print(f"GAGAL {tag}: {exc}", file=sys.stderr)
                failures.append(f"embed:{tag}")

    if args.stage in ("all", "finetune") and gpu:
        banner("TAHAP 4/5  fine-tune encoder HuggingFace")
        for tag, cfg in ft_picks.items():
            res = stage_finetune(tag, cfg, args, train, test, y_raw, splits, force=args.force)
            if res.get(tag) is None and tag in res:
                failures.append(f"finetune:{tag}")

    # Tahap apa pun diakhiri blend, supaya menjalankan satu tahap saja pun
    # tetap menghasilkan file submission -- bukan cuma artefak .npy.
    banner("TAHAP 5/5  blend akhir -> submission")
    final = stage_blend(train, test, y_raw, splits, args.out)

    banner("SELESAI" if not failures else "SELESAI DENGAN KEGAGALAN")
    if final is not None:
        path = os.path.join(SUBMISSIONS, args.out)
        print(f"FILE SUBMISSION ANDA:\n    {path}")
        print(f"    {len(test)} baris, kolom id + listPrice, urutan id sama dengan sample_submission.")
        print(f"    CV MAE {final:,.0f}  (baseline tebak-median {mae(y_raw, np.median(y_raw)):,.0f})")
        print("    -> unggah file ini ke Kaggle.")
    if failures:
        print("tahap gagal:", ", ".join(failures))
        print("Blend tetap memakai model yang berhasil. Perbaiki lalu jalankan ulang -- "
              "artefak yang sudah jadi otomatis dilewati.")
    return 1 if failures else 0


def run(stage="all", tier="core", out="submission.csv", **kwargs):
    """Entry point untuk notebook (Kaggle/Colab):

        run()                      # jalankan semua
        run(stage="cpu")           # model CPU saja
        run(tier="extra")          # + model besar
        run(folds="0,1")           # uji cepat 2 fold
    """
    argv = ["--stage", str(stage), "--tier", str(tier), "--out", str(out)]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return main(argv)


if __name__ == "__main__" and not in_notebook():
    sys.exit(main())
elif in_notebook():
    print("Mode notebook terdeteksi. Jalankan dengan:  run()\n"
          "  run(stage='cpu')   -> model CPU saja (tanpa GPU)\n"
          "  run(tier='extra')  -> tambah model besar\n"
          f"  data      : dicari otomatis (paksa lewat os.environ['HOLOMINE_DATA'])\n"
          f"  artefak   : {ARTIFACTS}\n"
          f"  submission: {SUBMISSIONS}")
