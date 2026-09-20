"""Konfigurasi kombinasi model HuggingFace yang direkomendasikan.

Satu tempat untuk semua hyperparameter per-backbone, dipakai oleh run_all.py
maupun saat memanggil train_transformer.py / embed_features.py manual.

Hyperparameter di sini bukan tebakan: lr 2e-5 dengan layer-wise decay 0.9 dan
3 epoch adalah resep standar fine-tune encoder untuk regresi pada dataset
puluhan ribu baris; deberta-v3-large dipasang lebih konservatif (lr 8e-6,
batch efektif tetap 16 lewat akumulasi) karena model besar mudah divergen.
"""
from __future__ import annotations

# ---------------------------------------------------------------- fine-tune --
# "tier" menentukan apa yang dijalankan oleh run_all.py:
#   core     -> kombinasi inti yang direkomendasikan (selalu jalan)
#   extra    -> tambahan kalau kuota GPU masih ada (--tier extra)
FINETUNE = {
    "deb3base": dict(
        tier="core",
        model="microsoft/deberta-v3-base",
        max_len=512, batch_size=16, accum=1, epochs=3,
        lr=2e-5, head_lr=1e-4, llrd=0.9,
        vram_gb=10,
        note="Juara de-facto regresi teks di Kaggle. Titik awal terbaik.",
    ),
    "mbert": dict(
        tier="core",
        model="answerdotai/ModernBERT-base",
        # Diukur di train.csv: p99 = 521 token, dan pada 512 hanya 1,1% listing
        # terpotong. Jadi 512 sudah cukup -- 1024 hanya menggandakan waktu
        # latih tanpa informasi tambahan yang berarti. Nilai model ini ada di
        # arsitektur & tokenizer-nya yang berbeda, bukan di konteks panjangnya.
        max_len=512, batch_size=16, accum=1, epochs=3,
        lr=3e-5, head_lr=1e-4, llrd=0.9,
        vram_gb=10,
        note="Arsitektur & tokenizer beda dari DeBERTa -> error tidak berkorelasi.",
    ),
    "deb3large": dict(
        tier="extra",
        model="microsoft/deberta-v3-large",
        max_len=512, batch_size=4, accum=4, epochs=3,
        lr=8e-6, head_lr=5e-5, llrd=0.85,
        grad_checkpoint=True, vram_gb=16,
        note="Biasanya lebih baik dari base, tapi rawan divergen di fp16.",
    ),
    "mbertlarge": dict(
        tier="extra",
        model="answerdotai/ModernBERT-large",
        max_len=512, batch_size=4, accum=4, epochs=3,
        lr=1e-5, head_lr=5e-5, llrd=0.85,
        grad_checkpoint=True, vram_gb=16,
        note="Pasangan besar yang beragam dari DeBERTa.",
    ),
}

# ------------------------------------------------------------- embedding ----
EMBED = {
    "gte": dict(
        tier="core",
        model="Alibaba-NLP/gte-modernbert-base",
        max_len=512, batch_size=32,
        note="Embedding kuat berbasis ModernBERT, murah karena tanpa training.",
    ),
    "bge": dict(
        tier="extra",
        model="BAAI/bge-base-en-v1.5",
        max_len=512, batch_size=64,
        note="Standar industri, sangat stabil.",
    ),
    "e5": dict(
        tier="extra",
        model="intfloat/e5-base-v2",
        max_len=512, batch_size=64, prefix="query: ",
        note="Wajib pakai prefiks 'query: '.",
    ),
    "qwen": dict(
        tier="extra",
        model="Qwen/Qwen3-Embedding-0.6B",
        max_len=512, batch_size=16,
        note="Peringkat MTEB teratas di kelas kecil; butuh VRAM lebih.",
    ),
    "minilm": dict(
        tier="cpu",
        model="sentence-transformers/all-MiniLM-L6-v2",
        max_len=256, batch_size=64,
        note="Cukup cepat untuk CPU; dipakai kalau tidak ada GPU.",
    ),
}

# Model CPU yang selalu ikut ke dalam blend (tanpa HuggingFace sama sekali).
SPARSE_MODELS = "ridge_word,ridge_char,linsvr_word,knn_word,lgbm_dense"


def to_cli(name: str, cfg: dict, kind: str) -> list:
    """Ubah satu entri config jadi argumen CLI."""
    args = ["--model", cfg["model"], "--tag", name,
            "--max-len", str(cfg["max_len"]), "--batch-size", str(cfg["batch_size"])]
    if kind == "finetune":
        args += ["--accum", str(cfg["accum"]), "--epochs", str(cfg["epochs"]),
                 "--lr", str(cfg["lr"]), "--head-lr", str(cfg["head_lr"]),
                 "--llrd", str(cfg["llrd"])]
        if cfg.get("grad_checkpoint"):
            args.append("--grad-checkpoint")
    else:
        args.append("--with-features")
        if cfg.get("prefix"):
            args += ["--prefix", cfg["prefix"]]
    return args


def describe():
    lines = ["Fine-tune:"]
    for n, c in FINETUNE.items():
        lines.append(f"  [{c['tier']:5s}] {n:11s} {c['model']:34s} ~{c['vram_gb']}GB VRAM  {c['note']}")
    lines.append("Embedding beku:")
    for n, c in EMBED.items():
        lines.append(f"  [{c['tier']:5s}] {n:11s} {c['model']:34s} {c['note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
