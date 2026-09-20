"""Satu sel Kaggle/Colab: fine-tune transformer + blend -> submission.csv

Cara pakai di Kaggle
--------------------
1. Notebook -> Settings -> Accelerator: GPU T4 x2 (atau P100), Internet: ON.
   (Kalau internet harus OFF, tambahkan model HuggingFace sebagai Kaggle Dataset
   lalu ganti MODELS ke path lokalnya, mis. "/kaggle/input/deberta-v3-base".)
2. Tambahkan dataset kompetisi ini.
3. Clone repo ini (atau upload folder src/ sebagai dataset), lalu jalankan file ini.

    !git clone https://github.com/AffrizaWildanFauzan/hology_tasks2.git /kaggle/working/repo
    %run /kaggle/working/repo/notebooks/kaggle_gpu_pipeline.py
"""
import os
import subprocess
import sys

REPO = os.environ.get("HOLOMINE_REPO", "/kaggle/working/repo")
WORK = "/kaggle/working"
os.environ.setdefault("HOLOMINE_ARTIFACTS", f"{WORK}/artifacts")
os.environ.setdefault("HOLOMINE_SUBMISSIONS", WORK)
sys.path.insert(0, os.path.join(REPO, "src"))

# Backbone yang di-fine-tune. Mulai dari dua teratas; tambahkan sisanya kalau
# masih ada kuota GPU. Semakin beragam arsitekturnya, semakin besar gain blend.
MODELS = [
    # (model_id, tag, argumen tambahan)
    ("microsoft/deberta-v3-base",   "deb3base", ["--max-len", "512",  "--batch-size", "16"]),
    ("answerdotai/ModernBERT-base", "mbert",    ["--max-len", "1024", "--batch-size", "16"]),
    # ("microsoft/deberta-v3-large", "deb3large", ["--max-len", "512", "--batch-size", "4",
    #                                              "--accum", "4", "--lr", "8e-6", "--grad-checkpoint"]),
]
EMBEDDERS = [
    ("Alibaba-NLP/gte-modernbert-base", "gte", ["--max-len", "1024"]),
    # ("BAAI/bge-base-en-v1.5", "bge", []),
    # ("intfloat/e5-base-v2",   "e5",  ["--prefix", "query: "]),
]


def run(script, *args):
    cmd = [sys.executable, os.path.join(REPO, "src", script), *map(str, args)]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    # 1) Model CPU dulu (~20 menit, jalan paralel dengan GPU idle)
    run("run_sparse.py", "--models", "ridge_word,ridge_char,linsvr_word,knn_word,lgbm_dense")

    # 2) Embedding beku
    for model, tag, extra in EMBEDDERS:
        run("embed_features.py", "--model", model, "--tag", tag, "--with-features", *extra)

    # 3) Fine-tune (bagian terberat; ~40-60 menit per model di T4 untuk 5 fold)
    for model, tag, extra in MODELS:
        run("train_transformer.py", "--model", model, "--tag", tag, "--epochs", "3", *extra)

    # 4) Blend semua OOF yang ada -> submission
    run("blend.py", "--calibrate", "--stack", "--out", "submission.csv")
    print("\nselesai ->", os.path.join(os.environ["HOLOMINE_SUBMISSIONS"], "submission.csv"))


if __name__ == "__main__":
    main()
