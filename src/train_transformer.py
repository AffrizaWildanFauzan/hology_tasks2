"""Fine-tune a HuggingFace encoder as a price regressor (GPU recommended).

Run one command per backbone; each run writes artifacts/oof_<tag>.npy and
artifacts/test_<tag>.npy in log-price space, so src/blend.py mixes them with
the TF-IDF / kNN / LightGBM models automatically.

    python src/train_transformer.py --model microsoft/deberta-v3-base --tag deb3base
    python src/train_transformer.py --model answerdotai/ModernBERT-base --tag mbert --max-len 1024
    python src/train_transformer.py --model Alibaba-NLP/gte-modernbert-base --tag gte

Design notes
------------
* Target = log(price); loss = SmoothL1 (Huber). MAE in dollars is minimised by
  the conditional median, and exp() of an L1 fit in log space *is* that median.
  Huber with a small beta keeps the gradient stable early in training.
* Log-target is also what keeps a $80M mansion from dominating the gradient.
* Mean pooling over tokens beats [CLS] for long descriptive text.
* Layer-wise LR decay: lower layers keep general language knowledge, the head
  and top layers adapt fastest.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ARTIFACTS, SEED, from_log, load_data, mae, price_bin_folds,
                    save_oof, to_log)


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


class MeanPoolRegressor(nn.Module):
    def __init__(self, model_name, dropout=0.0):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        cfg.update({"hidden_dropout_prob": dropout, "attention_probs_dropout_prob": dropout})
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg, trust_remote_code=True)
        hidden = cfg.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, **batch):
        batch.pop("labels", None)
        mask = batch["attention_mask"]
        out = self.backbone(**batch).last_hidden_state
        m = mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled).squeeze(-1)


def layerwise_params(model, base_lr, head_lr, decay=0.9, weight_decay=0.01):
    """Top encoder layers get ~base_lr, each lower layer gets decay x less."""
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
        lr = base_lr * (decay ** (n - 1 - i))
        groups.append({"params": layer.parameters(), "lr": lr, "weight_decay": weight_decay})
        assigned.update(id(p) for p in layer.parameters())
    rest = [p for p in backbone.parameters() if id(p) not in assigned]
    if rest:
        groups.append({"params": rest, "lr": base_lr * (decay ** n), "weight_decay": weight_decay})
    return groups


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model(**batch).float().cpu())
    return torch.cat(out).numpy()


def train_fold(args, texts, y_log, y_raw, trn, val, test_texts, fold, tokenizer, collator, device):
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(SEED + fold)
    model = MeanPoolRegressor(args.model, dropout=args.dropout).to(device)
    if args.grad_checkpoint:
        model.backbone.gradient_checkpointing_enable()

    ds_tr = ListingDataset(texts[trn], y_log[trn], tokenizer, args.max_len)
    ds_va = ListingDataset(texts[val], y_log[val], tokenizer, args.max_len)
    ds_te = ListingDataset(test_texts, None, tokenizer, args.max_len)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator,
                       num_workers=args.workers)
    dl_te = DataLoader(ds_te, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator,
                       num_workers=args.workers)

    opt = torch.optim.AdamW(layerwise_params(model, args.lr, args.head_lr, args.llrd, args.weight_decay))
    steps = max(1, len(dl_tr) // args.accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)
    amp_dtype = (torch.bfloat16 if args.bf16 else torch.float16) if device.type == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    best = (np.inf, None, None)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(dl_tr):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model(**batch), labels) / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        p_val = predict(model, dl_va, device, amp_dtype)
        score = mae(y_raw[val], from_log(p_val))
        print(f"  fold {fold} epoch {epoch}: val MAE = {score:,.0f}  ({time.time() - t0:.0f}s)", flush=True)
        if score < best[0]:
            best = (score, p_val, predict(model, dl_te, device, amp_dtype))

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--tag", default=None, help="artifact name; defaults to a slug of --model")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--llrd", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--huber-beta", type=float, default=0.15)
    ap.add_argument("--folds", default="all", help="e.g. '0,1' to train a subset")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", help="use on A100/L4/H100; fp16 otherwise")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample the training rows: for smoke-testing the pipeline")
    args = ap.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace(".", "").lower()

    from transformers import AutoTokenizer, DataCollatorWithPadding

    train, test = load_data()
    texts = train["text"].values
    test_texts = test["text"].values
    y_raw = train["listPrice"].values.astype(float)
    y_log = to_log(y_raw)
    if args.limit:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(texts), size=min(args.limit, len(texts)), replace=False)
        texts, y_raw = texts[keep], y_raw[keep]
        y_log = to_log(y_raw)
        test_texts = test_texts[:args.limit]
        print(f"--limit: using {len(texts)} train rows / {len(test_texts)} test rows")
    splits = price_bin_folds(y_raw)
    want = range(len(splits)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.model} -> tag '{tag}' on {device}", flush=True)

    oof = np.full(len(texts), np.nan)
    test_pred = np.zeros(len(test_texts))
    used = 0
    for fold, (trn, val) in enumerate(splits):
        if fold not in want:
            continue
        score, p_val, p_test = train_fold(args, texts, y_log, y_raw, trn, val,
                                          test_texts, fold, tokenizer, collator, device)
        oof[val] = p_val
        test_pred += p_test
        used += 1
        print(f"fold {fold} best MAE = {score:,.0f}", flush=True)

    test_pred /= max(used, 1)
    done = ~np.isnan(oof)
    print(f"\n[{tag}] CV MAE (on {done.sum()} rows) = {mae(y_raw[done], from_log(oof[done])):,.0f}")
    if args.limit:
        print("--limit was set: predictions are a smoke test, not saved for blending")
    elif done.all():
        save_oof(tag, oof, test_pred)
        print(f"saved artifacts/oof_{tag}.npy + artifacts/test_{tag}.npy")
    else:
        np.save(os.path.join(ARTIFACTS, f"partial_oof_{tag}.npy"), oof)
        np.save(os.path.join(ARTIFACTS, f"partial_test_{tag}.npy"), test_pred)
        print("partial folds only -> saved as partial_* (run all folds before blending)")


if __name__ == "__main__":
    main()
