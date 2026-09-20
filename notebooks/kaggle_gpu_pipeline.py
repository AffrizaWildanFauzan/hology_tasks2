"""Satu sel Kaggle/Colab: jalankan kombinasi HuggingFace yang direkomendasikan.

Setup di Kaggle
---------------
1. Settings -> Accelerator: **GPU T4 x2** (atau P100), **Internet: ON**.
2. Add Data -> dataset kompetisi ini.
3. Jalankan sel di bawah.

Kalau Internet harus OFF: tambahkan bobot model sebagai Kaggle Dataset, lalu
ubah `model=` di src/model_zoo.py ke path lokalnya (mis.
"/kaggle/input/deberta-v3-base") dan set HF_HUB_OFFLINE=1.
"""
import os
import subprocess
import sys

REPO = os.environ.get("HOLOMINE_REPO", "/kaggle/working/repo")
os.environ.setdefault("HOLOMINE_ARTIFACTS", "/kaggle/working/artifacts")
os.environ.setdefault("HOLOMINE_SUBMISSIONS", "/kaggle/working")

if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/AffrizaWildanFauzan/hology_tasks2.git", REPO], check=True)

# ModernBERT butuh transformers >= 4.48; deberta-v3 butuh sentencepiece.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers>=4.48", "sentence-transformers>=3.0", "sentencepiece", "lightgbm"],
               check=False)

# --tier core  = deberta-v3-base + ModernBERT-base + gte-modernbert-base
# --tier extra = tambah deberta-v3-large, ModernBERT-large, bge, e5, qwen
subprocess.run([sys.executable, os.path.join(REPO, "src", "run_all.py"),
                "--tier", "core", "--out", "submission.csv"], check=False)

print("\nsubmission:", os.path.join(os.environ["HOLOMINE_SUBMISSIONS"], "submission.csv"))
