"""
clip_zeroshot.py -- training-free CLIP zero-shot classification on MIMOSA.

This replaces the trained BERT+CLIP pipeline (MAF) with pure prompt-based
inference: no gradients, no checkpoints, no BanglaBERT.  A meme is classified by
comparing its CLIP embedding against the embeddings of natural-language
descriptions of the five aggression-target classes.

    score(meme, class) = cos( CLIP_image(meme), mean_t CLIP_text(prompt_t(class)) )

Like experiment_logger.py this is an additive module: it imports nothing from
main.py / models.py / dataset.py and edits none of them.  It reuses the run
logger so zero-shot runs produce the same artefact set as trained runs and can
be compared with the same tooling.

Usage
-----
    python clip_zeroshot.py                                  # image-only, ViT-B/32, test split
    python clip_zeroshot.py --backbone ViT-L/14 --split test validation
    python clip_zeroshot.py --mode fused --alpha_sweep       # tune fusion on val, apply to test
    python clip_zeroshot.py --prompts my_prompts.json        # prompt engineering
    python clip_zeroshot.py --max_samples 64 --run_name smoke

Scoring modes
-------------
    image    cosine(image embedding, class prompt embedding)          [default]
    caption  cosine(caption embedding, class prompt embedding)
    fused    alpha * image similarity + (1 - alpha) * caption similarity

A warning about `caption` and `fused`: OpenAI CLIP's text tower is English-only
and its BPE shreds Bengali into ~135 near-per-character tokens, so 77% of MIMOSA
captions overflow the 77-token context window before they are even encoded.  The
caption branch is therefore expected to be weak-to-random.  It is implemented
because it costs nothing to compute and the measurement is worth having, not
because it is expected to work.  `tokenization_stats.json` in every run records
exactly how much of each caption survived.

Image features are cached per (backbone, split) under `.clip_cache/`, so prompt
iteration after the first run costs seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import experiment_logger as el
from experiment_logger import CLASS_IDS, CLASS_SHORT, CLASS_FULL, ROOT_DIR

SPLIT_FILES = {
    "train": "training_set.csv",
    "validation": "validation_set.csv",
    "test": "testing_set.csv",
}

# ---------------------------------------------------------------------------
# Default prompt bank.
#
# CLIP cannot read the Bengali text burned into these memes, so prompts describe
# what the *picture* looks like rather than what it says.  Override the whole
# bank with --prompts <file.json> using this same {class_name: [prompts]} shape.
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = {
    "non-aggressive": [
        "a harmless funny internet meme",
        "a wholesome joke meme with no insult",
        "a neutral photo with caption text",
        "a friendly lighthearted meme",
        "a meme that does not attack anyone",
    ],
    "gendered aggression": [
        "a sexist meme insulting a woman",
        "a misogynistic meme attacking women",
        "a meme mocking a woman's appearance or body",
        "a meme attacking someone for their gender",
        "a meme objectifying a woman",
    ],
    "political aggression": [
        "a meme attacking a politician",
        "a political meme insulting a political party",
        "a meme mocking a government leader",
        "an angry political protest meme",
        "a meme criticising a prime minister or minister",
    ],
    "religious aggression": [
        "a meme attacking a religion",
        "a meme insulting religious people",
        "a meme mocking a religious figure or symbol",
        "a communal hate meme about religion",
        "a meme showing a mosque temple or religious clothing with hostile text",
    ],
    "others": [
        "an aggressive meme insulting a person",
        "an offensive meme with abusive text",
        "a meme attacking someone personally",
        "an angry insulting meme with harsh words",
        "a rude meme bullying an individual",
    ],
}


# ===========================================================================
# Prompt handling
# ===========================================================================
def load_prompts(path=None):
    if path is None:
        return {k: list(v) for k, v in DEFAULT_PROMPTS.items()}
    with open(path, encoding="utf-8") as fh:
        bank = json.load(fh)
    missing = [c for c in CLASS_FULL if c not in bank]
    if missing:
        raise ValueError(
            f"prompt file {path} is missing classes: {missing}. "
            f"Expected exactly these keys: {CLASS_FULL}"
        )
    for name, prompts in bank.items():
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"prompt file {path}: class '{name}' needs a non-empty list")
    return bank


def encode_class_prompts(clip_mod, model, bank, device):
    """One L2-normalised embedding per class, averaged over its prompt ensemble."""
    import torch

    vectors = []
    for name in CLASS_FULL:
        tokens = clip_mod.tokenize(bank[name], truncate=True).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean = feats.mean(dim=0)
        vectors.append(mean / mean.norm())
    return torch.stack(vectors)  # (5, D)


# ===========================================================================
# Feature extraction
# ===========================================================================
class MemeSplit:
    """Images + captions for one split, using CLIP's own preprocessing.

    Note this deliberately uses the `preprocess` transform returned by
    clip.load(), not the ImageNet normalisation in dataset.py -- the frozen CLIP
    tower expects its own mean/std.
    """

    def __init__(self, dataframe, images_dir, preprocess):
        from torch.utils.data import Dataset

        class _DS(Dataset):
            def __init__(self, df, d, pp):
                self.df, self.dir, self.pp = df.reset_index(drop=True), d, pp

            def __len__(self):
                return len(self.df)

            def __getitem__(self, i):
                from PIL import Image
                row = self.df.loc[i]
                with Image.open(os.path.join(self.dir, row["image_name"])) as im:
                    image = self.pp(im.convert("RGB"))
                return image, i

        self.dataset = _DS(dataframe, images_dir, preprocess)


def encode_split(clip_mod, model, preprocess, df, images_dir, device,
                 batch_size, logger, cache_path=None):
    """Return L2-normalised image and caption features for one split."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if cache_path and os.path.exists(cache_path):
        blob = np.load(cache_path)
        if len(blob["image"]) == len(df):
            logger.log(f"Loaded cached features from {os.path.basename(cache_path)}")
            return (torch.from_numpy(blob["image"]).to(device),
                    torch.from_numpy(blob["caption"]).to(device),
                    json.loads(str(blob["token_stats"])))
        logger.log("Cache size mismatch -- re-encoding", "WARN")

    # ---- images --------------------------------------------------------
    loader = DataLoader(MemeSplit(df, images_dir, preprocess).dataset,
                        batch_size=batch_size, shuffle=False, num_workers=4)
    from tqdm import tqdm
    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="encoding images", unit="batch"):
            feats = model.encode_image(images.to(device)).float()
            chunks.append((feats / feats.norm(dim=-1, keepdim=True)).cpu())
    image_features = torch.cat(chunks)
    logger.log(f"Encoded {len(image_features)} images in {time.time() - t0:.1f}s")

    # ---- captions ------------------------------------------------------
    from clip.simple_tokenizer import SimpleTokenizer
    raw_tokenizer = SimpleTokenizer()
    captions = [str(c) for c in df["Captions"]]
    lengths = [len(raw_tokenizer.encode(c)) for c in captions]
    budget = 75  # 77 minus the SOT/EOT sentinels
    token_stats = {
        "n_captions": len(captions),
        "context_length": 77,
        "usable_token_budget": budget,
        "bpe_tokens_mean": round(float(np.mean(lengths)), 2),
        "bpe_tokens_median": float(np.median(lengths)),
        "bpe_tokens_p90": float(np.percentile(lengths, 90)),
        "bpe_tokens_max": int(np.max(lengths)),
        "fraction_truncated": round(float(np.mean([l > budget for l in lengths])), 4),
        "mean_fraction_of_caption_kept": round(
            float(np.mean([min(1.0, budget / max(l, 1)) for l in lengths])), 4),
    }

    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="encoding captions", unit="batch"):
            tokens = clip_mod.tokenize(captions[i:i + batch_size], truncate=True).to(device)
            feats = model.encode_text(tokens).float()
            chunks.append((feats / feats.norm(dim=-1, keepdim=True)).cpu())
    caption_features = torch.cat(chunks)
    logger.log(f"Encoded {len(caption_features)} captions in {time.time() - t0:.1f}s "
               f"({token_stats['fraction_truncated'] * 100:.1f}% truncated)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path,
                 image=image_features.numpy(),
                 caption=caption_features.numpy(),
                 token_stats=json.dumps(token_stats))
        logger.log(f"Cached features -> {cache_path}")

    return image_features.to(device), caption_features.to(device), token_stats


# ===========================================================================
# Scoring
# ===========================================================================
def similarities(image_features, caption_features, class_features, mode, alpha):
    """Cosine similarity of each sample to each class, under the chosen mode.

    All three inputs are L2-normalised, so image- and caption-side similarities
    live on the same [-1, 1] scale and can be blended directly.
    """
    sim_image = image_features @ class_features.T
    sim_caption = caption_features @ class_features.T
    if mode == "image":
        return sim_image
    if mode == "caption":
        return sim_caption
    if mode == "fused":
        return alpha * sim_image + (1.0 - alpha) * sim_caption
    raise ValueError(f"unknown mode: {mode}")


def predict(sims, logit_scale=100.0):
    import torch
    probs = torch.softmax(sims * logit_scale, dim=1)
    return sims.argmax(dim=1).cpu().numpy(), probs.cpu().numpy()


def binary_collapse(y_true, y_pred):
    """Aggressive (any target) vs non-aggressive -- the easier sub-problem.

    Separates 'did the model notice aggression at all' from 'did it identify the
    right target', which the 5-way report conflates.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    bt = (np.asarray(y_true) != 0).astype(int)
    bp = (np.asarray(y_pred) != 0).astype(int)
    return {
        "positive_class": "aggressive (any target)",
        "accuracy": float(accuracy_score(bt, bp)),
        "precision": float(precision_score(bt, bp, zero_division=0)),
        "recall": float(recall_score(bt, bp, zero_division=0)),
        "f1": float(f1_score(bt, bp, zero_division=0)),
        "macro_f1": float(f1_score(bt, bp, average="macro", zero_division=0)),
        "n_aggressive_true": int(bt.sum()),
        "n_aggressive_pred": int(bp.sum()),
    }


def majority_baseline(y_true):
    """What you get by always predicting the most frequent class."""
    import numpy as np
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    major = int(np.bincount(y_true, minlength=5).argmax())
    y_pred = np.full_like(y_true, major)
    return {
        "always_predicts": CLASS_FULL[major],
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_split(logger, out_dir, split, df, image_features, caption_features,
                   class_features, mode, alpha, token_stats):
    """Full artefact set for one split, mirroring a trained run's layout."""
    import numpy as np
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    y_true = df["Label"].map({name: i for i, name in enumerate(CLASS_FULL)}).to_numpy()
    if np.isnan(y_true.astype(float)).any():
        raise ValueError(f"unmapped labels in {split}: {sorted(set(df['Label']) - set(CLASS_FULL))}")
    y_true = y_true.astype(int)

    sims = similarities(image_features, caption_features, class_features, mode, alpha)
    y_pred, probs = predict(sims)

    metrics = el.compute_report(y_true, y_pred)
    metrics.update({
        "split": split,
        "mode": mode,
        "alpha": alpha if mode == "fused" else None,
        "n_samples": int(len(y_true)),
        "binary_aggression": binary_collapse(y_true, y_pred),
        "majority_baseline": majority_baseline(y_true),
        "tokenization": token_stats,
    })
    with open(os.path.join(out_dir, "classification_report.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    from sklearn.metrics import classification_report as _cr
    text = _cr(y_true, y_pred, labels=CLASS_IDS, target_names=CLASS_SHORT,
               digits=3, zero_division=0)
    maj, binm = metrics["majority_baseline"], metrics["binary_aggression"]
    lines = [
        f"CLIP zero-shot on MIMOSA -- {split} split",
        f"mode: {mode}" + (f" (alpha={alpha})" if mode == "fused" else ""),
        f"samples: {len(y_true)}",
        "",
        text,
        "",
        f"Accuracy           : {metrics['accuracy']:.4f}",
        f"Weighted F1        : {metrics['weighted_f1']:.4f}",
        f"Macro F1           : {metrics['macro_f1']:.4f}",
        f"MMAE               : {metrics['mmae']}",
        "",
        "Reference points",
        f"  majority baseline (always '{maj['always_predicts']}'): "
        f"acc {maj['accuracy']:.4f}, macro F1 {maj['macro_f1']:.4f}",
        f"  random 5-way guess                          : acc 0.2000",
        "",
        "Aggressive vs non-aggressive (5-way collapsed to binary)",
        f"  accuracy {binm['accuracy']:.4f}  precision {binm['precision']:.4f}  "
        f"recall {binm['recall']:.4f}  F1 {binm['f1']:.4f}",
        "",
        "Confusion matrix (rows = true, cols = predicted, order " + ", ".join(CLASS_SHORT) + "):",
    ]
    for name, row in zip(CLASS_SHORT, metrics["confusion_matrix"]):
        lines.append(f"  {name:<5}" + "".join(f"{v:>7}" for v in row))
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    el.save_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(out_dir, "confusion_matrix.png"),
        os.path.join(out_dir, "confusion_matrix.csv"),
        f"CLIP zero-shot ({mode}) -- {split}",
    )

    out = pd.DataFrame({
        "index": np.arange(len(y_true)),
        "image_name": df["image_name"].values,
        "caption": df["Captions"].values,
        "true_label_id": y_true,
        "true_label": [CLASS_FULL[i] for i in y_true],
        "pred_label_id": y_pred,
        "pred_label": [CLASS_FULL[i] for i in y_pred],
        "correct": y_true == y_pred,
    })
    sims_np = sims.cpu().numpy()
    for i, name in enumerate(CLASS_SHORT):
        out[f"prob_{name}"] = probs[:, i]
    for i, name in enumerate(CLASS_SHORT):
        out[f"cos_{name}"] = sims_np[:, i]
    out["confidence"] = probs.max(axis=1)
    out["margin"] = np.sort(sims_np, axis=1)[:, -1] - np.sort(sims_np, axis=1)[:, -2]
    out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    logger.event("zeroshot_split", split=split, mode=mode, alpha=alpha,
                 accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                 weighted_f1=metrics["weighted_f1"])
    logger.log(
        f"[{split}/{mode}] acc={metrics['accuracy'] * 100:.2f}% "
        f"weightedF1={metrics['weighted_f1']:.4f} macroF1={metrics['macro_f1']:.4f} "
        f"| majority acc={maj['accuracy'] * 100:.2f}% "
        f"| binary-aggression F1={binm['f1']:.4f}"
    )
    return metrics


# ===========================================================================
# Fusion-weight tuning
# ===========================================================================
def sweep_alpha(logger, run_dir, df, image_features, caption_features, class_features):
    """Pick the fusion weight on validation, never on test.

    Returns the alpha with the best validation macro-F1 and writes the full
    sweep so the choice is auditable.
    """
    import csv as _csv
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    y_true = df["Label"].map({n: i for i, n in enumerate(CLASS_FULL)}).to_numpy().astype(int)
    rows, best = [], (None, -1.0)
    for alpha in [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]:
        sims = similarities(image_features, caption_features, class_features, "fused", alpha)
        y_pred = sims.argmax(dim=1).cpu().numpy()
        acc = float(accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        rows.append({"alpha": alpha, "val_accuracy": round(acc, 4),
                     "val_macro_f1": round(mf1, 4), "val_weighted_f1": round(wf1, 4)})
        if mf1 > best[1]:
            best = (alpha, mf1)

    path = os.path.join(run_dir, "alpha_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.log(f"alpha sweep on validation -> best alpha={best[0]} (macro F1 {best[1]:.4f}); "
               f"full sweep in alpha_sweep.csv")
    logger.event("alpha_sweep", best_alpha=best[0], best_val_macro_f1=best[1], sweep=rows)
    return best[0]


# ===========================================================================
# Entry point
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Training-free CLIP zero-shot classification on MIMOSA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backbone", type=str, default="ViT-B/32",
                   help="CLIP visual backbone (RN50, ViT-B/32, ViT-B/16, ViT-L/14, ...)")
    p.add_argument("--split", type=str, nargs="+", default=["test"],
                   choices=["train", "validation", "test", "all"],
                   help="split(s) to evaluate")
    p.add_argument("--mode", type=str, default="image",
                   choices=["image", "caption", "fused"], help="scoring mode")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="fused mode: weight on image similarity")
    p.add_argument("--alpha_sweep", action="store_true",
                   help="fused mode: choose alpha by validation macro-F1 before scoring")
    p.add_argument("--prompts", type=str, default=None,
                   help="JSON file overriding the prompt bank")
    p.add_argument("--dataset", dest="dataset_path", type=str, default="Dataset",
                   help="dataset folder, relative to repo root")
    p.add_argument("--batch_size", type=int, default=64, help="encoding batch size")
    p.add_argument("--max_samples", type=int, default=0,
                   help="use only the first N rows of each split (smoke test)")
    p.add_argument("--no_cache", action="store_true", help="ignore and do not write feature cache")
    p.add_argument("--run_name", type=str, default="clip_zeroshot", help="tag in the run folder name")
    p.add_argument("--log_dir", type=str, default="Runs", help="root folder for run directories")
    p.add_argument("--seed", type=int, default=42, help="seed for random/numpy/torch")
    p.add_argument("--resource_interval", type=float, default=15.0,
                   help="seconds between GPU/CPU resource samples")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    splits = list(SPLIT_FILES) if "all" in args.split else list(dict.fromkeys(args.split))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(ROOT_DIR, args.log_dir)
    run_dir = os.path.join(log_root, f"{stamp}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    logger = el.RunLogger(run_dir, resource_interval=args.resource_interval, training_csvs=False)
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
        images_dir = os.path.join(dataset_dir, "Img")

        cfg = {
            "approach": "CLIP zero-shot (no training, no BanglaBERT, no checkpoint)",
            "backbone": args.backbone,
            "splits": splits,
            "mode": args.mode,
            "alpha": args.alpha,
            "alpha_sweep": args.alpha_sweep,
            "prompt_file": args.prompts,
            "dataset_dir": dataset_dir,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "device": str(device),
            "run_dir": run_dir,
            "command": " ".join([sys.executable] + sys.argv),
        }
        logger.dump("config.json", cfg)
        logger.event("config", **cfg)
        logger.log("Collecting environment ...")
        logger.dump("environment.json", el.collect_environment())

        logger.log(f"Loading CLIP {args.backbone} ...")
        t0 = time.time()
        model, preprocess = clip_mod.load(args.backbone, device=device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        logger.log(f"CLIP loaded in {time.time() - t0:.1f}s "
                   f"({n_params:,} params, all frozen, 0 trainable)")
        logger.event("model", backbone=args.backbone, params=n_params, trainable=0)

        bank = load_prompts(args.prompts)
        logger.dump("prompts_used.json", bank)
        class_features = encode_class_prompts(clip_mod, model, bank, device)
        logger.log(f"Built class embeddings from {sum(len(v) for v in bank.values())} prompts "
                   f"({len(bank[CLASS_FULL[0]])} per class, ensembled by mean)")

        # sanity check: prompt vectors should not be near-duplicates
        cross = (class_features @ class_features.T).cpu().numpy()
        off = cross[~np.eye(5, dtype=bool)]
        logger.log(f"Inter-class prompt cosine similarity: mean {off.mean():.3f}, max {off.max():.3f} "
                   f"(closer to 1.0 means the prompts do not separate the classes)")
        logger.dump("prompt_similarity.json",
                    {"matrix": cross.tolist(), "classes": CLASS_SHORT,
                     "offdiag_mean": float(off.mean()), "offdiag_max": float(off.max())})

        cache_dir = os.path.join(ROOT_DIR, ".clip_cache")
        tag = args.backbone.replace("/", "-").replace("@", "-")
        features = {}

        def features_for(split):
            if split in features:
                return features[split]
            df = pd.read_csv(os.path.join(dataset_dir, SPLIT_FILES[split]))
            if args.max_samples:
                df = df.iloc[: args.max_samples].reset_index(drop=True)
            cache = None if (args.no_cache or args.max_samples) \
                else os.path.join(cache_dir, f"{tag}_{split}.npz")
            logger.log(f"Encoding split '{split}' ({len(df)} memes) ...")
            img, cap, stats = encode_split(clip_mod, model, preprocess, df, images_dir,
                                           device, args.batch_size, logger, cache)
            features[split] = (df, img, cap, stats)
            return features[split]

        alpha = args.alpha
        if args.mode == "fused" and args.alpha_sweep:
            vdf, vimg, vcap, _ = features_for("validation")
            alpha = sweep_alpha(logger, run_dir, vdf, vimg, vcap, class_features)

        results = {}
        for split in splits:
            df, img, cap, stats = features_for(split)
            logger.dump(f"tokenization_stats_{split}.json", stats)
            out_dir = run_dir if len(splits) == 1 else os.path.join(run_dir, split)
            results[split] = evaluate_split(logger, out_dir, split, df, img, cap,
                                            class_features, args.mode, alpha, stats)

        summary = {
            "run_dir": run_dir,
            "config": {**cfg, "alpha_used": alpha},
            "prompts": bank,
            "results": {
                s: {k: v for k, v in m.items() if k not in ("per_class", "tokenization")}
                for s, m in results.items()
            },
            "per_class": {s: m["per_class"] for s, m in results.items()},
            "timings_s": logger.timings,
        }
        logger.dump("summary.json", summary)

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
