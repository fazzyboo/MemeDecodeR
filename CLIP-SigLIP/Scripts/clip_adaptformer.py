"""
clip_adaptformer.py -- PEFT training of a CLIP vision tower with AdaptFormer on MIMOSA.

The frozen CLIP image encoder from `clip_zeroshot.py`, with an AdaptFormer bottleneck
injected into every residual block (`adaptformer.py`), plus a small classification head.
Only the adapters and the head are trained -- ~1.3% of the vision tower for ViT-B/32.

    frozen CLIP ViT  +  AdaptMLP in every block  +  head  ->  5 aggression classes

Why the default head is `zeroshot`
----------------------------------
The head is initialised from the very class-prompt embeddings `clip_zeroshot.py` uses,
and AdaptFormer's up-projections start at zero, so **at step 0 this model reproduces the
zero-shot run exactly**. The epoch-0 row in `epoch_metrics.csv` is therefore a real
zero-shot baseline measured by the same code path, and every later row shows what PEFT
bought on top of it. Pass `--head linear` for a conventional randomly-initialised probe.

Verified: with `--no_amp` the epoch-0 validation numbers match `clip_zeroshot.py` to four
decimals (36.59% / macro-F1 0.3197 for ViT-B/32). Default bf16 autocast moves ~10 of 727
borderline samples, so epoch 0 reads 36.18% there. Use `--no_amp` when you want the
identity to hold exactly.

Usage
-----
    python clip_adaptformer.py                                  # ViT-B/32, r=64, 10 epochs
    python clip_adaptformer.py --backbone ViT-L/14 --batch_size 32
    python clip_adaptformer.py --bottleneck 8 --run_name r8     # capacity ablation
    python clip_adaptformer.py --blocks 8,9,10,11               # adapt only the last blocks
    python clip_adaptformer.py --max_batches 3 --epochs 2 --run_name smoke

Artefacts match the other runners (`experiment_logger.py` does the logging), so trained,
zero-shot and PEFT runs all compare with the same tooling. The saved checkpoint holds
only the trained tensors -- a few MB, against ~1 GB for the full MAF checkpoint.
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

import adaptformer as af
import clip_zeroshot as zs
import experiment_logger as el
from experiment_logger import CLASS_FULL, CLASS_IDS, CLASS_SHORT, ROOT_DIR


# ===========================================================================
# Model
# ===========================================================================
def build_model(clip_model, num_classes, head, class_features, device):
    import torch
    import torch.nn as nn

    visual = clip_model.visual.float()
    feature_dim = getattr(visual, "output_dim", None)
    if feature_dim is None:
        feature_dim = visual.proj.shape[1]

    class ZeroShotHead(nn.Module):
        """Cosine classifier initialised from the zero-shot prompt embeddings."""

        def __init__(self, weights, logit_scale=100.0):
            super().__init__()
            self.weight = nn.Parameter(weights.clone().float())
            self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))

        def forward(self, feats):
            f = feats / feats.norm(dim=-1, keepdim=True)
            w = self.weight / self.weight.norm(dim=-1, keepdim=True)
            return self.logit_scale.exp().clamp(max=100.0) * (f @ w.T)

    class Classifier(nn.Module):
        def __init__(self, visual, head_module):
            super().__init__()
            self.visual = visual
            self.head = head_module

        def forward(self, images):
            return self.head(self.visual(images))

    if head == "zeroshot":
        if class_features.shape != (num_classes, feature_dim):
            raise ValueError(
                f"class embeddings {tuple(class_features.shape)} do not match "
                f"(num_classes={num_classes}, feature_dim={feature_dim})"
            )
        head_module = ZeroShotHead(class_features)
    elif head == "linear":
        head_module = nn.Linear(feature_dim, num_classes)
        nn.init.zeros_(head_module.bias)
        nn.init.normal_(head_module.weight, std=0.01)
    else:
        raise ValueError(f"unknown head: {head}")

    return Classifier(visual, head_module).to(device), feature_dim


def build_loaders(dataset_dir, preprocess, batch_size, num_workers, augment, max_batches):
    """MIMOSA splits under CLIP's own preprocessing."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from torchvision import transforms

    images_dir = os.path.join(dataset_dir, "Img")
    label_map = {name: i for i, name in enumerate(CLASS_FULL)}

    train_transform = preprocess
    if augment:
        # deliberately no horizontal flip: these memes carry burned-in Bengali text
        n_px = preprocess.transforms[0].size
        n_px = n_px if isinstance(n_px, int) else n_px[0]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(n_px, scale=(0.8, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            preprocess.transforms[-1],  # CLIP's own Normalize
        ])

    class Split(Dataset):
        def __init__(self, df, transform):
            self.df, self.transform = df.reset_index(drop=True), transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.loc[i]
            with Image.open(os.path.join(images_dir, row["image_name"])) as im:
                image = self.transform(im.convert("RGB"))
            return image, label_map[row["Label"]]

    loaders, frames = {}, {}
    for split, filename in zs.SPLIT_FILES.items():
        df = pd.read_csv(os.path.join(dataset_dir, filename))
        if max_batches:
            df = df.iloc[: max_batches * batch_size].reset_index(drop=True)
        frames[split] = df
        is_train = split == "train"
        loaders[split] = DataLoader(
            Split(df, train_transform if is_train else preprocess),
            batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
        )
    return loaders, frames


# ===========================================================================
# Train / evaluate
# ===========================================================================
def run_eval(model, loader, device, amp_dtype):
    import numpy as np
    import torch

    model.eval()
    labels, preds, logits = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = model(images)
            else:
                out = model(images)
            out = out.float()
            logits.append(out.cpu())
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(targets.numpy())
    return (np.concatenate(labels), np.concatenate(preds),
            torch.cat(logits).softmax(dim=1).numpy())


def train(logger, model, loaders, cfg, device, amp_dtype, run_dir):
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    steps_per_epoch = len(loaders["train"])
    total_steps = max(steps_per_epoch * cfg["epochs"], 1)
    warmup = int(cfg["warmup_ratio"] * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    # stepped per batch, unlike the original models.py which steps once per epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    weight = None
    if cfg["class_weights"]:
        counts = np.bincount(
            [loaders["train"].dataset.df["Label"].map(
                {n: i for i, n in enumerate(CLASS_FULL)}).to_numpy()][0], minlength=5)
        weight = torch.tensor((counts.sum() / (5 * np.maximum(counts, 1))),
                              dtype=torch.float32, device=device)
        logger.log(f"class weights: {dict(zip(CLASS_SHORT, weight.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])

    best = {"score": -1.0, "epoch": None}
    select_on = cfg["select_on"]
    ckpt_path = os.path.join(run_dir, "adapter_head.pt")
    global_step = 0

    # ---- epoch 0: the model before any gradient step ----------------------
    y_true, y_pred, _ = run_eval(model, loaders["validation"], device, amp_dtype)
    zero = el.compute_report(y_true, y_pred)
    logger._row(logger.epoch_csv, [
        0, None, None, round(zero["accuracy"], 6), round(zero["macro_f1"], 6),
        round(zero["weighted_f1"], 6), round(zero["macro_precision"], 6),
        round(zero["macro_recall"], 6),
        None if zero["mmae"] is None else round(zero["mmae"], 6),
        optimizer.param_groups[0]["lr"], 0.0, 0.0, 0.0, None, False,
    ])
    logger.epoch_rows.append({"epoch": 0, "val_acc": zero["accuracy"],
                              "val_macro_f1": zero["macro_f1"],
                              "note": "before training (zero-shot equivalent)"})
    logger.event("epoch", epoch=0, val_acc=zero["accuracy"], val_macro_f1=zero["macro_f1"],
                 note="before training")
    logger.log(f"epoch 0 (no training yet): val_acc={zero['accuracy'] * 100:.2f}% "
               f"val_macroF1={zero['macro_f1']:.4f}"
               + ("  <- equals the clip_zeroshot.py baseline exactly under --no_amp; "
                  "bf16 autocast shifts it by a few borderline samples"
                  if cfg["head"] == "zeroshot" else ""))

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_t0 = time.time()
        loss_sum = acc_sum = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with tqdm(loaders["train"], desc=f"Epoch {epoch}/{cfg['epochs']}", unit="batch") as bar:
            for step, (images, targets) in enumerate(bar, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                t_batch = time.time()

                optimizer.zero_grad(set_to_none=True)
                if amp_dtype is not None:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        outputs = model(images)
                        loss = criterion(outputs.float(), targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                loss.backward()
                if cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                optimizer.step()
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
                bar.set_postfix(loss=loss_sum / step, acc=acc_sum / step, lr=f"{lr_now:.2e}")

        train_time = time.time() - epoch_t0
        y_true, y_pred, _ = run_eval(model, loaders["validation"], device, amp_dtype)
        report = el.compute_report(y_true, y_pred)
        with open(os.path.join(logger.val_dir, f"epoch_{epoch:03d}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({**report, "epoch": epoch, "split": "validation"}, fh, indent=2)

        score = report["macro_f1"] if select_on == "macro_f1" else report["accuracy"]
        is_best = score > best["score"]
        if is_best:
            best = {"score": score, "epoch": epoch,
                    "val_accuracy": report["accuracy"], "val_macro_f1": report["macro_f1"]}
            torch.save({"state_dict": af.adapter_state_dict(model), "config": cfg,
                        "epoch": epoch, "val": {k: report[k] for k in
                                                ("accuracy", "macro_f1", "weighted_f1")}},
                       ckpt_path)

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

    logger.log(f"Best epoch {best['epoch']} by {select_on} "
               f"(val macro F1 {best.get('val_macro_f1', float('nan')):.4f}, "
               f"val acc {best.get('val_accuracy', float('nan')) * 100:.2f}%)")
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1e6
        logger.log(f"Trained tensors only: {ckpt_path} ({size_mb:.1f} MB)")
        logger.event("checkpoint", path=ckpt_path, size_mb=round(size_mb, 2))
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"],
                              strict=False)
        logger.log("Reloaded best adapters+head for testing")
    return best


# ===========================================================================
# Entry point
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="AdaptFormer PEFT of a CLIP vision tower on MIMOSA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- AdaptFormer ------------------------------------------------------
    p.add_argument("--bottleneck", type=int, default=64, help="adapter inner dimension r")
    p.add_argument("--scalar", type=str, default="0.1",
                   help="adapter scale s: a float, or 'learnable_scalar'")
    p.add_argument("--adapter_dropout", type=float, default=0.0, help="dropout inside the adapter")
    p.add_argument("--adapter_ln", type=str, default="none", choices=["none", "in", "out"],
                   help="optional LayerNorm in the adapter branch")
    p.add_argument("--adapter_init", type=str, default="lora", choices=["lora", "xavier"],
                   help="'lora' zeroes the up-projection so training starts as a no-op")
    p.add_argument("--adapter_mode", type=str, default="parallel",
                   choices=["parallel", "sequential"], help="AdaptMLP placement")
    p.add_argument("--blocks", type=str, default="all",
                   help="'all' (the paper's setting) or a comma list, e.g. 8,9,10,11")
    p.add_argument("--train_layernorms", action="store_true",
                   help="also train every LayerNorm affine parameter")
    # --- model / data -----------------------------------------------------
    p.add_argument("--backbone", type=str, default="ViT-B/32", help="CLIP ViT backbone")
    p.add_argument("--head", type=str, default="zeroshot", choices=["zeroshot", "linear"],
                   help="'zeroshot' initialises the head from the class prompts")
    p.add_argument("--prompts", type=str, default=None, help="JSON prompt bank for the head")
    p.add_argument("--dataset", dest="dataset_path", type=str, default="Dataset")
    p.add_argument("--augment", action="store_true",
                   help="random-resized-crop + colour jitter on train (no flips: burned-in text)")
    p.add_argument("--num_workers", type=int, default=4)
    # --- optimisation -----------------------------------------------------
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="adapters tolerate a much higher LR")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0, help="0 disables clipping")
    p.add_argument("--class_weights", action="store_true",
                   help="inverse-frequency class weights in the loss")
    p.add_argument("--select_on", type=str, default="macro_f1", choices=["macro_f1", "accuracy"],
                   help="validation metric used to keep the best checkpoint")
    p.add_argument("--no_amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--eval_only", type=str, default=None,
                   help="path to an adapter_head.pt: load it, skip training, just evaluate. "
                        "Use it to score a saved adapter, or to regenerate the test artefacts "
                        "of a run that was interrupted after training.")
    # --- logging ----------------------------------------------------------
    p.add_argument("--run_name", type=str, default="adaptformer")
    p.add_argument("--log_dir", type=str, default="Runs")
    p.add_argument("--seed", type=int, default=42)
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
        import clip as clip_mod

        logger.log(f"Run directory: {run_dir}")
        el.seed_everything(args.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_dir = os.path.join(ROOT_DIR, args.dataset_path)

        amp_dtype = None
        if not args.no_amp and device.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        logger.log(f"Device {device}, autocast={'bf16' if amp_dtype else 'off'}")

        blocks = None if args.blocks.strip().lower() == "all" else \
            [int(b) for b in args.blocks.split(",") if b.strip() != ""]

        cfg = {
            "approach": "AdaptFormer PEFT on the CLIP vision tower",
            "backbone": args.backbone, "head": args.head, "bottleneck": args.bottleneck,
            "scalar": args.scalar, "adapter_dropout": args.adapter_dropout,
            "adapter_ln": args.adapter_ln, "adapter_init": args.adapter_init,
            "adapter_mode": args.adapter_mode, "blocks": args.blocks,
            "train_layernorms": args.train_layernorms, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
            "label_smoothing": args.label_smoothing, "grad_clip": args.grad_clip,
            "class_weights": args.class_weights, "select_on": args.select_on,
            "augment": args.augment, "amp": "bf16" if amp_dtype else "off",
            "seed": args.seed, "max_batches": args.max_batches, "eval_only": args.eval_only,
            "dataset_dir": dataset_dir, "run_dir": run_dir,
            "command": " ".join([sys.executable] + sys.argv),
        }
        logger.dump("config.json", cfg)
        logger.event("config", **cfg)
        logger.log("Collecting environment ...")
        logger.dump("environment.json", el.collect_environment())

        # ---- CLIP + prompts ------------------------------------------------
        logger.log(f"Loading CLIP {args.backbone} ...")
        clip_model, preprocess = clip_mod.load(args.backbone, device=device)
        clip_model.float()
        bank = zs.load_prompts(args.prompts)
        logger.dump("prompts_used.json", bank)
        class_features = zs.encode_class_prompts(clip_mod, clip_model, bank, device).float()

        # ---- AdaptFormer injection -----------------------------------------
        info = af.inject_adaptformer(
            clip_model.visual, bottleneck=args.bottleneck, scalar=args.scalar,
            dropout=args.adapter_dropout, layernorm_option=args.adapter_ln,
            init_option=args.adapter_init, mode=args.adapter_mode, blocks=blocks,
        )
        logger.dump("adaptformer_info.json", info)
        logger.log(f"AdaptFormer injected into {info['blocks_adapted']}/{info['blocks_total']} "
                   f"blocks (width {info['width']}, r={info['bottleneck']}, s={info['scalar']}, "
                   f"{info['mode']}): {info['adapter_params']:,} adapter params "
                   f"({info['params_per_adapter']:,} per block)")

        model, feature_dim = build_model(clip_model, len(CLASS_IDS), args.head,
                                         class_features, device)
        stats = af.freeze_for_peft(model, train_adapters=True,
                                   train_layernorms=args.train_layernorms)
        report_text, groups = af.trainable_report(model)
        with open(os.path.join(run_dir, "model_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write(report_text + "\n\n" + str(model))
        logger.dump("trainable_params.json", {**stats, "groups": groups,
                                              "feature_dim": feature_dim})
        logger.log(f"Trainable: {stats['trainable_params']:,} / {stats['total_params']:,} "
                   f"({100 * stats['trainable_fraction']:.3f}%) -- "
                   + ", ".join(f"{k} {v:,}" for k, v in sorted(groups.items())))
        if stats["trainable_params"] == 0:
            raise RuntimeError("nothing is trainable -- check --blocks / freeze settings")

        # ---- data ------------------------------------------------------------
        loaders, frames = build_loaders(dataset_dir, preprocess, args.batch_size,
                                        args.num_workers, args.augment, args.max_batches)
        logger.log(f"Loaders: {len(loaders['train'])}/{len(loaders['validation'])}/"
                   f"{len(loaders['test'])} train/val/test batches "
                   f"({len(frames['train'])}/{len(frames['validation'])}/{len(frames['test'])} memes)")

        # ---- train (or load a saved adapter) ---------------------------------
        if args.eval_only:
            payload = torch.load(args.eval_only, map_location=device)
            state = payload.get("state_dict", payload)
            af.load_adapter_state_dict(model, state, strict=False)
            best = {"epoch": payload.get("epoch"), "loaded_from": args.eval_only,
                    **(payload.get("val") or {})}
            logger.timings["training_s"] = 0.0
            logger.log(f"--eval_only: loaded {args.eval_only} "
                       f"(trained to epoch {payload.get('epoch')}), skipping training")
        else:
            t0 = time.time()
            best = train(logger, model, loaders, cfg, device, amp_dtype, run_dir)
            logger.timings["training_s"] = round(time.time() - t0, 2)

        # ---- test ------------------------------------------------------------
        logger.log("=== TEST EVALUATION ===")
        t0 = time.time()
        y_true, y_pred, probs = run_eval(model, loaders["test"], device, amp_dtype)
        logger.timings["test_eval_s"] = round(time.time() - t0, 2)

        metrics = el.compute_report(y_true, y_pred)
        metrics.update({"split": "test", "n_samples": int(len(y_true)),
                        "binary_aggression": zs.binary_collapse(y_true, y_pred),
                        "majority_baseline": zs.majority_baseline(y_true)})
        logger.dump("classification_report.json", metrics)

        from sklearn.metrics import classification_report as _cr
        text = _cr(y_true, y_pred, labels=CLASS_IDS, target_names=CLASS_SHORT,
                   digits=3, zero_division=0)
        binm = metrics["binary_aggression"]
        lines = [
            "CLIP + AdaptFormer (PEFT) on MIMOSA -- test set",
            f"run: {os.path.basename(run_dir)}",
            f"backbone: {args.backbone}   head: {args.head}   r={args.bottleneck}   s={args.scalar}",
            f"trainable: {stats['trainable_params']:,} of {stats['total_params']:,} "
            f"({100 * stats['trainable_fraction']:.3f}%)",
            f"samples: {len(y_true)}", "", text, "",
            f"Accuracy           : {metrics['accuracy']:.4f}",
            f"Weighted F1        : {metrics['weighted_f1']:.4f}",
            f"Macro F1           : {metrics['macro_f1']:.4f}",
            f"MMAE               : {metrics['mmae']}",
            "",
            "Aggressive vs non-aggressive (collapsed to binary)",
            f"  accuracy {binm['accuracy']:.4f}  precision {binm['precision']:.4f}  "
            f"recall {binm['recall']:.4f}  F1 {binm['f1']:.4f}",
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
            f"CLIP+AdaptFormer ({args.backbone}, r={args.bottleneck}) -- test",
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
            "run_dir": run_dir, "config": cfg, "adaptformer": info,
            "trainable": stats, "best_validation": best,
            "epochs": logger.epoch_rows, "timings_s": logger.timings,
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
