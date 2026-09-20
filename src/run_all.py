"""Jalankan kombinasi model yang direkomendasikan, satu perintah.

    python src/run_all.py                 # kombinasi inti (butuh GPU untuk tahap 3-4)
    python src/run_all.py --tier extra    # + deberta-v3-large, bge, e5, qwen
    python src/run_all.py --stage cpu     # hanya model CPU
    python src/run_all.py --dry-run       # tampilkan rencana tanpa menjalankan

Urutannya disusun supaya selalu ada submission yang valid sedini mungkin:
tahap CPU dulu (10 menit, tanpa HuggingFace), blend sementara, baru model berat.
Setiap tahap melewati pekerjaan yang artefaknya sudah ada, jadi runner ini aman
dijalankan ulang setelah sesi Kaggle terputus.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARTIFACTS
from model_zoo import EMBED, FINETUNE, SPARSE_MODELS, to_cli

SRC = os.path.dirname(os.path.abspath(__file__))


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


def artifact_done(tag: str) -> bool:
    return os.path.exists(os.path.join(ARTIFACTS, f"oof_{tag}.npy"))


def run(script: str, *args, dry: bool = False) -> bool:
    cmd = [sys.executable, os.path.join(SRC, script), *map(str, args)]
    print("\n" + "=" * 78 + f"\n$ {' '.join(cmd)}\n" + "=" * 78, flush=True)
    if dry:
        return True
    t0 = time.time()
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    print(f"--> {'selesai' if ok else 'GAGAL (rc=%d)' % result.returncode} dalam {time.time() - t0:.0f}s",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="core", choices=["core", "extra"],
                    help="'extra' menambah model besar bila kuota GPU masih ada")
    ap.add_argument("--stage", default="all",
                    choices=["all", "cpu", "embed", "finetune", "blend"])
    ap.add_argument("--folds", default="all", help="mis. '0,1' untuk uji cepat")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--assume-gpu", action="store_true",
                    help="dengan --dry-run: tampilkan rencana GPU dari mesin tanpa GPU")
    ap.add_argument("--force", action="store_true", help="latih ulang walau artefak sudah ada")
    args = ap.parse_args()

    tiers = {"core"} if args.tier == "core" else {"core", "extra"}
    gpu = have_gpu() or (args.assume_gpu and args.dry_run)
    print(f"perangkat: {gpu_name()}" + ("  [--assume-gpu]" if args.assume_gpu else ""))
    if not gpu:
        print("Tanpa GPU: tahap fine-tune dilewati, embedding memakai MiniLM (CPU).\n"
              "  (pakai --dry-run --assume-gpu untuk melihat rencana lengkapnya)")

    failures = []

    # --- 1. model CPU (selalu) --------------------------------------------
    if args.stage in ("all", "cpu"):
        todo = [m for m in SPARSE_MODELS.split(",") if args.force or not artifact_done(m)]
        if todo:
            if not run("run_sparse.py", "--models", ",".join(todo), dry=args.dry_run):
                failures.append("run_sparse")
        else:
            print("\n[cpu] semua artefak sudah ada, dilewati")

    # --- 2. submission sementara supaya tidak pernah tangan kosong ---------
    if args.stage == "all":
        run("blend.py", "--calibrate", "--stack", "--out", "submission_cpu_only.csv",
            dry=args.dry_run)

    # --- 3. embedding beku ------------------------------------------------
    if args.stage in ("all", "embed"):
        picks = {n: c for n, c in EMBED.items() if c["tier"] in tiers} if gpu else \
                {n: c for n, c in EMBED.items() if c["tier"] == "cpu"}
        for name, cfg in picks.items():
            if not args.force and artifact_done(f"emb_{name}_lgbm"):
                print(f"\n[embed] {name} sudah ada, dilewati")
                continue
            if not run("embed_features.py", *to_cli(name, cfg, "embed"), dry=args.dry_run):
                failures.append(f"embed:{name}")

    # --- 4. fine-tune (inti dari rekomendasi) -----------------------------
    if args.stage in ("all", "finetune") and gpu:
        picks = {n: c for n, c in FINETUNE.items() if c["tier"] in tiers}
        for name, cfg in picks.items():
            if not args.force and artifact_done(name):
                print(f"\n[finetune] {name} sudah ada, dilewati")
                continue
            cli = to_cli(name, cfg, "finetune") + ["--folds", args.folds]
            if not run("train_transformer.py", *cli, dry=args.dry_run):
                failures.append(f"finetune:{name}")

    # --- 5. blend akhir ---------------------------------------------------
    if args.stage in ("all", "blend"):
        if not run("blend.py", "--calibrate", "--stack", "--out", args.out, dry=args.dry_run):
            failures.append("blend")

    print("\n" + "=" * 78)
    if failures:
        print("TAHAP GAGAL:", ", ".join(failures))
        print("Blend tetap memakai model yang berhasil; perbaiki lalu jalankan ulang "
              "(artefak yang sudah jadi otomatis dilewati).")
    else:
        print("Semua tahap selesai.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
