"""
maf_siglip.py -- MAF with SigLIP in place of CLIP as the vision encoder.

The paper's MAF fuses a frozen CLIP ViT-B/32 image embedding with BanglaBERT token
embeddings through a multihead attention block. This module keeps that architecture
exactly and swaps only the vision tower for SigLIP (Zhai et al., "Sigmoid Loss for
Language Image Pre-Training", ICCV 2023), so any difference in the numbers is
attributable to the encoder rather than to the fusion, the text side or the recipe.

Why SigLIP is a good swap here
------------------------------
SigLIP replaces CLIP's softmax contrastive loss with a pairwise sigmoid loss, which
removes the global normalisation over the batch and trains better at reachable batch
sizes. Two practical consequences for MAF:

  * `google/siglip-base-patch16-224` has **hidden size 768**, exactly BanglaBERT's width,
    so the visual projection becomes a same-width map instead of the 512 -> 768 widening
    CLIP forced.
  * SigLIP exposes a real **196-token patch sequence**, where CLIP's visual tower gives a
    single pooled vector that MAF then broadcasts 70 times (see `--vision tokens`).

What is kept identical to models.py
-----------------------------------
Frozen vision tower, BanglaBERT text tower, `MultiheadAttention(query=image,
key=text, value=image)`, concat of [attention, image, text] -> mean over the sequence
-> the same 2304 -> 128 -> 5 classifier, AdamW(lr, wd=0.01), linear schedule stepped
once per epoch, best checkpoint by validation accuracy, 5 epochs / batch 16 / 16 heads /
max_len 70. The defaults reproduce the paper's run with SigLIP dropped in.

Deliberate differences, each behind a flag
------------------------------------------
  * Image normalisation is SigLIP's own (mean/std 0.5), not the ImageNet constants in
    dataset.py. `--imagenet_norm` restores the original, wrong-for-the-encoder values so
    you can measure how much of any gain is just correct preprocessing.
  * `--vision tokens` feeds the real patch sequence instead of a broadcast pooled vector.
  * `--scheduler per_batch` fixes the once-per-epoch LR decay noted in the guide.
  * `--select_on macro_f1` selects the checkpoint on macro F1 rather than accuracy.

Usage
-----
    python maf_siglip.py                                  # paper defaults, SigLIP swapped in
    python maf_siglip.py --vision tokens --run_name tok   # use the 196-token sequence
    python maf_siglip.py --imagenet_norm --run_name inorm # normalisation control
    python maf_siglip.py --max_batches 3 --n_iter 2 --run_name smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import experiment_logger as el
from experiment_logger import CLASS_FULL, CLASS_IDS, CLASS_SHORT, ROOT_DIR

TEXT_MODEL = "sagorsarker/bangla-bert-base"
SPLIT_FILES = {
    "train": "training_set.csv",
    "validation": "validation_set.csv",
    "test": "testing_set.csv",
}


# ===========================================================================
# Model -- MAF with the vision tower swapped
# ===========================================================================
def build_maf_siglip(vision_name, num_classes, num_heads, seq_len, vision_mode, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, CLIPVisionModelWithProjection, SiglipVisionModel

    is_clip = "clip" in vision_name.lower()

    class MAFSigLIP(nn.Module):
        def __init__(self):
            super().__init__()
            # ---- vision tower, frozen (MAF freezes CLIP the same way) ------
            # CLIPVisionModelWithProjection is used for CLIP so that `pooled`
            # yields image_embeds -- the same projected vector clip_model.visual
            # hands the original MAF -- making the two encoders directly comparable.
            if is_clip:
                self.vision = CLIPVisionModelWithProjection.from_pretrained(vision_name)
                self.vision_dim = self.vision.config.projection_dim
                self.token_dim = self.vision.config.hidden_size
            else:
                self.vision = SiglipVisionModel.from_pretrained(vision_name)
                self.vision_dim = self.vision.config.hidden_size
                self.token_dim = self.vision.config.hidden_size
            self.is_clip = is_clip
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision_mode = vision_mode
            self.seq_len = seq_len

            # ---- text: BanglaBERT, exactly as in models.py -----------------
            self.bert = AutoModel.from_pretrained(TEXT_MODEL)
            self.text_dim = self.bert.config.hidden_size

            # MAF's 512 -> 768; with SigLIP-base this is 768 -> 768
            in_dim = self.token_dim if vision_mode == "tokens" else self.vision_dim
            self.visual_linear = nn.Linear(in_dim, self.text_dim)
            self.attention = nn.MultiheadAttention(self.text_dim, num_heads, dropout=0.1)
            self.fc = nn.Sequential(
                nn.Linear(self.text_dim * 3, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

        def visual_features(self, pixel_values):
            out = self.vision(pixel_values=pixel_values)
            if self.vision_mode == "tokens":
                feats = out.last_hidden_state              # (B, P, D) real patch tokens
            elif self.is_clip:
                feats = out.image_embeds.unsqueeze(1)      # (B, 1, 512) exactly as in MAF
            else:
                feats = out.pooler_output.unsqueeze(1)     # (B, 1, D) SigLIP pooled
            feats = self.visual_linear(feats)
            # MAF's resampling to the text sequence length; for 'pooled' this broadcasts
            # one vector seq_len times, exactly as the original does.
            return F.adaptive_avg_pool1d(feats.permute(0, 2, 1), self.seq_len).permute(0, 2, 1)

        def forward(self, pixel_values, input_ids, attention_mask):
            image_features = self.visual_features(pixel_values)
            text_features = self.bert(input_ids=input_ids,
                                      attention_mask=attention_mask).last_hidden_state

            attended = self.attention(
                image_features.permute(1, 0, 2),
                text_features.permute(1, 0, 2),
                image_features.permute(1, 0, 2),
                need_weights=False,
            )[0].permute(1, 0, 2)

            fusion = torch.cat([attended, image_features, text_features], dim=2)
            return self.fc(fusion.mean(1))

    return MAFSigLIP().to(device)


# ===========================================================================
# Data
# ===========================================================================
def build_loaders(dataset_dir, vision_name, max_len, batch_size, num_workers,
                  imagenet_norm, max_batches, logger):
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoTokenizer

    processor = AutoImageProcessor.from_pretrained(vision_name)
    size = processor.size
    if isinstance(size, dict):
        side = size.get("height") or size.get("shortest_edge")
    else:
        side = size
    if imagenet_norm:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # dataset.py's values
    else:
        mean, std = list(processor.image_mean), list(processor.image_std)
    logger.log(f"Image pipeline: resize {side}x{side}, mean={mean}, std={std}"
               + ("  (ImageNet constants -- reproducing dataset.py)" if imagenet_norm
                  else "  (the encoder's own constants)"))

    transform = transforms.Compose([
        transforms.Resize((side, side)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    label_map = {name: i for i, name in enumerate(CLASS_FULL)}
    images_dir = os.path.join(dataset_dir, "Img")

    class MIMOSA(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.loc[i]
            with Image.open(os.path.join(images_dir, row["image_name"])) as im:
                pixel_values = transform(im.convert("RGB"))
            enc = tokenizer(str(row["Captions"]), return_tensors="pt", padding="max_length",
                            truncation=True, max_length=max_len)
            return {
                "pixel_values": pixel_values,
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": label_map[row["Label"]],
            }

    loaders, frames = {}, {}
    for split, filename in SPLIT_FILES.items():
        df = pd.read_csv(os.path.join(dataset_dir, filename))
        if max_batches:
            df = df.iloc[: max_batches * batch_size].reset_index(drop=True)
        frames[split] = df
        loaders[split] = DataLoader(
            MIMOSA(df), batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders, frames


# ===========================================================================
# Train / evaluate
# ===========================================================================
def evaluate(model, loader, device):
    import numpy as np
    import torch

    model.eval()
    labels, preds, logits = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["pixel_values"].to(device),
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device))
            logits.append(out.float().cpu())
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(batch["label"].numpy())
    return (np.concatenate(labels), np.concatenate(preds),
            torch.cat(logits).softmax(dim=1).numpy())


def train(logger, model, loaders, cfg, device, run_dir):
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from transformers import get_linear_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr_rate"], weight_decay=0.01)
    steps_per_epoch = max(len(loaders["train"]), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=cfg["epochs"] * steps_per_epoch)
    criterion = nn.CrossEntropyLoss()

    best = {"score": -1.0, "epoch": None}
    ckpt = os.path.join(run_dir, "maf_siglip.pth")
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_t0 = time.time()
        loss_sum = acc_sum = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with tqdm(loaders["train"], desc=f"Epoch {epoch}/{cfg['epochs']}", unit="batch") as bar:
            for step, batch in enumerate(bar, start=1):
                images = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                t_batch = time.time()

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images, input_ids, mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if cfg["scheduler"] == "per_batch":
                    scheduler.step()
                global_step += 1

                with torch.no_grad():
                    acc = (outputs.argmax(dim=1) == targets).float().mean().item()
                loss_sum += loss.item()
                acc_sum += acc
                lr_now = optimizer.param_groups[0]["lr"]
                alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None
                logger._row(logger.batch_csv, [
                    round(time.time() - logger.t0, 2), round(time.time() - logger.t0, 2),
                    epoch, global_step, step, round(loss.item(), 6), round(acc, 6),
                    round(loss_sum / step, 6), round(acc_sum / step, 6), lr_now,
                    round(targets.shape[0] / max(time.time() - t_batch, 1e-9), 2),
                    None if alloc is None else round(alloc, 1),
                ])
                bar.set_postfix(loss=loss_sum / step, acc=acc_sum / step)

        train_time = time.time() - epoch_t0
        y_true, y_pred, _ = evaluate(model, loaders["validation"], device)
        report = el.compute_report(y_true, y_pred)
        with open(os.path.join(logger.val_dir, f"epoch_{epoch:03d}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({**report, "epoch": epoch, "split": "validation"}, fh, indent=2)

        score = report["macro_f1"] if cfg["select_on"] == "macro_f1" else report["accuracy"]
        is_best = score > best["score"]
        if is_best:
            best = {"score": score, "epoch": epoch, "val_accuracy": report["accuracy"],
                    "val_macro_f1": report["macro_f1"]}
            torch.save(model.state_dict(), ckpt)

        if cfg["scheduler"] == "per_epoch":
            scheduler.step()   # models.py does this; see the guide's note on the schedule

        peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
        epoch_time = time.time() - epoch_t0
        logger._row(logger.epoch_csv, [
            epoch, round(loss_sum / steps_per_epoch, 6), round(acc_sum / steps_per_epoch, 6),
            round(report["accuracy"], 6), round(report["macro_f1"], 6),
            round(report["weighted_f1"], 6), round(report["macro_precision"], 6),
            round(report["macro_recall"], 6),
            None if report["mmae"] is None else round(report["mmae"], 6),
            optimizer.param_groups[0]["lr"], round(train_time, 2),
            round(epoch_time - train_time, 2), round(epoch_time, 2),
            None if peak is None else round(peak, 1), is_best,
        ])
        row = {"epoch": epoch, "train_loss": round(loss_sum / steps_per_epoch, 6),
               "train_acc": round(acc_sum / steps_per_epoch, 6),
               "val_acc": round(report["accuracy"], 6),
               "val_macro_f1": round(report["macro_f1"], 6),
               "epoch_time_s": round(epoch_time, 2), "is_best": is_best}
        logger.epoch_rows.append(row)
        logger.event("epoch", **row)
        logger.log(f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
                   f"train_acc={row['train_acc'] * 100:.2f}% val_acc={row['val_acc'] * 100:.2f}% "
                   f"val_macroF1={row['val_macro_f1']:.4f} ({epoch_time:.1f}s)"
                   f"{' *best*' if is_best else ''}")

    logger.log(f"Best epoch {best['epoch']} by {cfg['select_on']} "
               f"(val acc {best.get('val_accuracy', float('nan')) * 100:.2f}%, "
               f"val macro F1 {best.get('val_macro_f1', float('nan')):.4f})")
    if os.path.exists(ckpt):
        size_mb = os.path.getsize(ckpt) / 1e6
        logger.log(f"Checkpoint: {ckpt} ({size_mb:.1f} MB)")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.log("Reloaded best checkpoint for testing")
    return best


# ===========================================================================
# Entry point
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="MAF with SigLIP as the vision encoder instead of CLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- the paper's arguments, same names and defaults -------------------
    p.add_argument("--dataset", dest="dataset_path", type=str, default="Dataset")
    p.add_argument("--max_len", dest="maximum_length", type=int, default=70)
    p.add_argument("--batch_size", dest="batch", type=int, default=16)
    p.add_argument("--heads", dest="n_heads", type=int, default=16)
    p.add_argument("--n_iter", dest="epochs", type=int, default=5)
    p.add_argument("--lrate", dest="lr_rate", type=float, default=5e-5)
    # --- SigLIP / ablation switches ---------------------------------------
    p.add_argument("--vision", type=str, default="pooled", choices=["pooled", "tokens"],
                   help="'pooled' mirrors CLIP's single vector; 'tokens' uses the 196 patches")
    p.add_argument("--vision_model", type=str, default="google/siglip-base-patch16-224",
                   help="a SigLIP checkpoint, or an 'openai/clip-*' one to run the "
                        "CLIP control through this exact same pipeline")
    p.add_argument("--imagenet_norm", action="store_true",
                   help="use dataset.py's ImageNet mean/std instead of SigLIP's own")
    p.add_argument("--scheduler", type=str, default="per_epoch",
                   choices=["per_epoch", "per_batch"],
                   help="'per_epoch' reproduces models.py; 'per_batch' is the intended decay")
    p.add_argument("--select_on", type=str, default="accuracy",
                   choices=["accuracy", "macro_f1"],
                   help="validation metric for the best checkpoint ('accuracy' = models.py)")
    # --- logging ----------------------------------------------------------
    p.add_argument("--run_name", type=str, default="maf_siglip")
    p.add_argument("--log_dir", type=str, default="Runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resource_interval", type=float, default=15.0)
    p.add_argument("--max_batches", type=int, default=0, help="smoke test: N batches per split")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(ROOT_DIR, args.log_dir)
    run_dir = os.path.join(log_root, f"{stamp}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    logger = el.RunLogger(run_dir, resource_interval=args.resource_interval)
    exit_code = 0
    try:
        import numpy as np
        import pandas as pd
        import torch

        logger.log(f"Run directory: {run_dir}")
        el.seed_everything(args.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_dir = os.path.join(ROOT_DIR, args.dataset_path)

        cfg = {
            "approach": "MAF with SigLIP vision encoder (CLIP replaced)",
            "vision_model": args.vision_model, "vision_mode": args.vision,
            "text_model": TEXT_MODEL, "imagenet_norm": args.imagenet_norm,
            "max_len": args.maximum_length, "batch_size": args.batch,
            "n_heads": args.n_heads, "epochs": args.epochs, "lr_rate": args.lr_rate,
            "scheduler": args.scheduler, "select_on": args.select_on,
            "seed": args.seed, "max_batches": args.max_batches,
            "dataset_dir": dataset_dir, "run_dir": run_dir,
            "command": " ".join([sys.executable] + sys.argv),
        }
        logger.dump("config.json", cfg)
        logger.event("config", **cfg)
        logger.log("Collecting environment ...")
        logger.dump("environment.json", el.collect_environment())

        logger.log(f"Building MAF with {args.vision_model} ({args.vision} features) "
                   f"+ {TEXT_MODEL} ...")
        model = build_maf_siglip(args.vision_model, len(CLASS_IDS), args.n_heads,
                                 args.maximum_length, args.vision, device)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        groups = {name: sum(p.numel() for p in mod.parameters())
                  for name, mod in model.named_children()}
        summary, _, _ = el.describe_model(model)
        with open(os.path.join(run_dir, "model_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write(summary + "\n")
        logger.dump("model_params.json", {"total": total, "trainable": trainable,
                                          "frozen": total - trainable, "groups": groups,
                                          "vision_dim": model.vision_dim,
                                          "text_dim": model.text_dim})
        logger.log(f"Model: {total:,} params ({trainable:,} trainable, "
                   f"{total - trainable:,} frozen); vision_dim={model.vision_dim}, "
                   f"text_dim={model.text_dim}")

        loaders, frames = build_loaders(dataset_dir, args.vision_model, args.maximum_length,
                                        args.batch, args.num_workers, args.imagenet_norm,
                                        args.max_batches, logger)
        logger.log(f"Loaders: {len(loaders['train'])}/{len(loaders['validation'])}/"
                   f"{len(loaders['test'])} train/val/test batches")

        t0 = time.time()
        best = train(logger, model, loaders, cfg, device, run_dir)
        logger.timings["training_s"] = round(time.time() - t0, 2)

        logger.log("=== TEST EVALUATION ===")
        t0 = time.time()
        y_true, y_pred, probs = evaluate(model, loaders["test"], device)
        logger.timings["test_eval_s"] = round(time.time() - t0, 2)

        metrics = el.compute_report(y_true, y_pred)
        metrics.update({"split": "test", "n_samples": int(len(y_true))})
        logger.dump("classification_report.json", metrics)

        from sklearn.metrics import classification_report as _cr
        text = _cr(y_true, y_pred, labels=CLASS_IDS, target_names=CLASS_SHORT,
                   digits=3, zero_division=0)
        lines = [
            "MAF + SigLIP on MIMOSA -- test set",
            f"run: {os.path.basename(run_dir)}",
            f"vision: {args.vision_model} ({args.vision})   text: {TEXT_MODEL}",
            f"normalisation: {'ImageNet (dataset.py)' if args.imagenet_norm else 'SigLIP own'}",
            f"trainable: {trainable:,} of {total:,}",
            f"samples: {len(y_true)}", "", text, "",
            f"Accuracy           : {metrics['accuracy']:.4f}",
            f"Weighted Precision : {metrics['weighted_precision']:.4f}",
            f"Weighted Recall    : {metrics['weighted_recall']:.4f}",
            f"Weighted F1        : {metrics['weighted_f1']:.4f}",
            f"Macro F1           : {metrics['macro_f1']:.4f}",
            f"MMAE               : {metrics['mmae']}",
            "",
            "Confusion matrix (rows = true, cols = predicted, order "
            + ", ".join(CLASS_SHORT) + "):",
        ]
        for name, row in zip(CLASS_SHORT, metrics["confusion_matrix"]):
            lines.append(f"  {name:<5}" + "".join(f"{v:>7}" for v in row))
        with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        el.save_confusion_matrix(
            metrics["confusion_matrix"],
            os.path.join(run_dir, "confusion_matrix.png"),
            os.path.join(run_dir, "confusion_matrix.csv"),
            f"MAF + SigLIP ({args.vision}) -- test",
        )

        test_df = frames["test"]
        out = pd.DataFrame({
            "index": np.arange(len(y_true)),
            "image_name": test_df["image_name"].values,
            "caption": test_df["Captions"].values,
            "true_label_id": y_true,
            "true_label": [CLASS_FULL[i] for i in y_true],
            "pred_label_id": y_pred,
            "pred_label": [CLASS_FULL[i] for i in y_pred],
            "correct": y_true == y_pred,
        })
        for i, name in enumerate(CLASS_SHORT):
            out[f"prob_{name}"] = probs[:, i]
        out["confidence"] = probs.max(axis=1)
        out.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

        logger.dump("summary.json", {
            "run_dir": run_dir, "config": cfg,
            "params": {"total": total, "trainable": trainable, "groups": groups},
            "best_validation": best, "epochs": logger.epoch_rows,
            "timings_s": logger.timings,
            "test": {k: v for k, v in metrics.items() if k != "per_class"},
            "test_per_class": metrics["per_class"],
        })
        logger.log(f"TEST acc={metrics['accuracy'] * 100:.2f}%  "
                   f"weightedF1={metrics['weighted_f1']:.4f}  "
                   f"macroF1={metrics['macro_f1']:.4f}  MMAE={metrics['mmae']}")

    except KeyboardInterrupt:
        exit_code = 130
        logger.log("Interrupted by user (KeyboardInterrupt)", "ERROR")
    except Exception:
        exit_code = 1
        tb = traceback.format_exc()
        logger.log(f"RUN FAILED\n{tb}", "ERROR")
        logger.event("error", traceback=tb)
        with open(os.path.join(run_dir, "traceback.txt"), "w", encoding="utf-8") as fh:
            fh.write(tb)
    finally:
        logger.timings["wall_clock_s"] = round(time.time() - logger.t0, 2)
        logger.event("run_end", exit_code=exit_code, timings=logger.timings)
        logger.log(f"Run finished with exit code {exit_code} "
                   f"({logger.timings['wall_clock_s']:.1f}s wall clock)")
        logger.close()
        print(f"\nAll logs and artefacts: {run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
