"""
Step 2 — Train the multi-head classifier.

Run on a GPU (Kaggle/Colab free tier is plenty: ~44k images, ~10 min/epoch on a
T4 at 224px with MobileNetV3-Small).

    ZINTOO_DATASET_DIR=/kaggle/input/fashion-product-images-dataset/fashion-dataset \
    python -m ml.train

Design notes:
  * Loss = sum of per-task class-weighted cross-entropy with label smoothing.
    Primary task (articleType) gets weight 1.0; auxiliaries 0.3 so they
    regularise without hijacking the trunk.
  * Early stopping / model selection on **macro-F1 of articleType**, not
    accuracy. With a long-tailed catalogue, accuracy is dominated by Tshirts
    and tells you nothing about the classes you actually care about.
  * AMP + cosine schedule + short warmup. Nothing exotic.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml import config
from ml.model import FashionDataset, FashionNet, class_weights

AUX_WEIGHT = 0.3


def _loaders(label_maps):
    dfs = {s: pd.read_csv(config.MANIFEST_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    loaders = {}
    for split, df in dfs.items():
        is_train = split == "train"
        loaders[split] = DataLoader(
            FashionDataset(df, train=is_train),
            batch_size=config.BATCH_SIZE,
            shuffle=is_train,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=config.NUM_WORKERS > 0,
        )
    return loaders, dfs


@torch.no_grad()
def evaluate(model, loader, device, tasks):
    model.eval()
    preds = {t: [] for t in tasks}
    trues = {t: [] for t in tasks}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            _, logits = model(x)
        for t in tasks:
            preds[t].append(logits[t].float().argmax(1).cpu().numpy())
            trues[t].append(y[t].numpy())
    out = {}
    for t in tasks:
        p, g = np.concatenate(preds[t]), np.concatenate(trues[t])
        out[t] = {
            "accuracy": float((p == g).mean()),
            "macro_f1": float(f1_score(g, p, average="macro", zero_division=0)),
        }
    return out


def main() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("⚠️  Training on CPU will take many hours. Use a GPU runtime.")

    label_maps = json.loads((config.MANIFEST_DIR / "label_maps.json").read_text())
    num_classes = {t: len(c) for t, c in label_maps.items()}
    loaders, dfs = _loaders(label_maps)
    print(f"train={len(dfs['train']):,} val={len(dfs['val']):,} test={len(dfs['test']):,}")
    print("classes:", num_classes)

    model = FashionNet(num_classes).to(device)

    criteria = {}
    for t in config.TASKS:
        w = class_weights(dfs["train"][t].to_numpy(), num_classes[t]).to(device)
        criteria[t] = nn.CrossEntropyLoss(weight=w, label_smoothing=config.LABEL_SMOOTHING)

    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    steps = max(1, len(loaders["train"])) * config.EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.LR, total_steps=steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1, best_epoch, patience = -1.0, -1, 3
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = config.ARTIFACTS_DIR / "best.pt"
    last = config.ARTIFACTS_DIR / "last.pt"
    start_epoch = 1

    # ─── Resume (essential on Colab: free runtimes disconnect) ────────
    # `last.pt` is written every epoch, so a disconnect costs at most one epoch.
    # Point ZINTOO_MODELS_DIR at Google Drive so it survives the VM dying.
    if os.getenv("ZINTOO_RESUME", "").lower() in {"1", "true", "yes"} and last.exists():
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_f1, best_epoch = state["best_f1"], state["best_epoch"]

        # OneCycleLR bakes `total_steps` into its state. If EPOCHS (or the
        # dataset size) changed since the checkpoint was written, restoring that
        # state makes the scheduler run off the end of its schedule and raise
        # "Tried to step N+1 times". Detect it and rebuild instead of crashing.
        ckpt_epochs = state.get("epochs")
        ckpt_steps = state.get("total_steps")
        compatible = (ckpt_epochs == config.EPOCHS and ckpt_steps == steps)

        if compatible:
            sched.load_state_dict(state["scheduler"])
            print(f"▶ resumed from {last} at epoch {start_epoch} "
                  f"(best macro-F1 so far {best_f1:.4f})")
        else:
            remaining = max(1, len(loaders["train"]) * (config.EPOCHS - start_epoch + 1))
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=config.LR, total_steps=remaining, pct_start=0.1
            )
            print(
                f"▶ resumed weights from {last} at epoch {start_epoch} "
                f"(best macro-F1 so far {best_f1:.4f})\n"
                f"   ⚠ schedule changed since checkpoint "
                f"(epochs {ckpt_epochs} -> {config.EPOCHS}); rebuilt LR schedule "
                f"over the remaining {remaining} steps."
            )

        if start_epoch > config.EPOCHS:
            print(f"   ⏹ checkpoint is already at epoch {state['epoch']} >= EPOCHS="
                  f"{config.EPOCHS}. Raise ZINTOO_EPOCHS or delete {last.name} to restart.")
            return

    for epoch in range(start_epoch, config.EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = {t: v.to(device, non_blocking=True) for t, v in y.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                _, logits = model(x)
                loss = sum(
                    (1.0 if t == config.PRIMARY_TASK else AUX_WEIGHT) * criteria[t](logits[t], y[t])
                    for t in config.TASKS
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        val = evaluate(model, loaders["val"], device, config.TASKS)
        f1 = val[config.PRIMARY_TASK]["macro_f1"]
        print(
            f"epoch {epoch:>2}/{config.EPOCHS}  loss={running/len(loaders['train']):.4f}  "
            f"val_acc={val[config.PRIMARY_TASK]['accuracy']:.4f}  val_macroF1={f1:.4f}  "
            f"({time.time()-t0:.0f}s)"
        )

        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save({"state_dict": model.state_dict(), "num_classes": num_classes,
                        "normalize_input": True}, ckpt)
            print(f"   ✅ new best (macro-F1 {f1:.4f}) -> {ckpt.name}")

        # Always snapshot full training state, so a killed runtime loses <= 1 epoch.
        torch.save({
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            # Recorded so a later resume can tell whether the LR schedule
            # in this checkpoint is still valid (see resume block above).
            "epochs": config.EPOCHS,
            "total_steps": steps,
            "normalize_input": True,
        }, last)

        if epoch - best_epoch >= patience:
            print(f"   ⏹ early stop (no improvement for {patience} epochs)")
            break

    # Final, honest evaluation on the held-out test split.
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    test = evaluate(model, loaders["test"], device, config.TASKS)
    print("\nTest metrics:")
    for t, m in test.items():
        print(f"  {t:<16} acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    (config.ARTIFACTS_DIR / "train_report.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test": test,
        "backbone": config.BACKBONE,
        "embedding_dim": config.EMBEDDING_DIM,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"\n✅ Best checkpoint: {ckpt}\n   Next: python -m ml.export_onnx")


if __name__ == "__main__":
    main()
