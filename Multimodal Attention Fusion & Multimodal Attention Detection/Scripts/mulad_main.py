"""
MuLAD entry point - trains one cell of the paper's experiment grid and scores it.

Reports the same metric set and writes the same Outputs/ files as the MAF pipeline, so the
two can be compared directly. Per-class probabilities are written alongside the hard
predictions, which is what any later ensembling needs.

Usage:
    python mulad_main.py --modality multimodal --text_model cnn --backbone vgg16 \
                         --embedding keras --run_name mulad_cnn_vgg16
    python mulad_main.py --modality text   --text_model cnn --embedding fasttext
    python mulad_main.py --modality visual --backbone vgg16
"""
import _paths  # noqa: F401

import argparse
import copy
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import metrics as M
import mulad_data as D
import mulad_models as MM
from mulad_features import BACKBONES

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_run_name(args):
    if args.modality == "text":
        name = "mulad_text_{}_{}".format(args.text_model, args.embedding)
    elif args.modality == "visual":
        name = "mulad_visual_{}".format(args.backbone)
    else:
        name = "mulad_{}_{}_{}".format(args.text_model, args.backbone, args.embedding)
    # A smoke run must never land on the same Outputs/ filenames as the real run of the
    # same configuration, or it silently overwrites a finished result.
    return name + ("_subset{}".format(args.subset) if args.subset else "")


@torch.no_grad()
def predict(model, loader, device):
    """Return (true, pred, probabilities) for a whole split."""
    model.eval()
    true, logits_all = [], []
    for tokens, visual, target in loader:
        logits = model(tokens.to(device), visual.to(device))
        logits_all.append(logits.cpu())
        true.extend(int(t) for t in target)
    probs = torch.softmax(torch.cat(logits_all), dim=1).numpy()
    return true, probs.argmax(axis=1).tolist(), probs


def train(model, loaders, device, epochs, lr, patience, monitor, target_names, quiet=False):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for tokens, visual, target in loaders["training_set"]:
            tokens, visual, target = tokens.to(device), visual.to(device), target.to(device)
            optimiser.zero_grad()
            loss = criterion(model(tokens, visual), target)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(target)
            seen += len(target)

        true, pred, _ = predict(model, loaders["validation_set"], device)
        scores = M.summarise(true, pred, target_names)
        score = scores["accuracy"] if monitor == "accuracy" else scores["macro_f1"]

        # The paper keeps "the best intermediate model" via a Keras callback (Sec. 5.1);
        # this is that callback, plus early stopping since the paper gives no epoch count.
        flag = ""
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            flag = "  <- best"
        else:
            stale += 1

        if not quiet:
            print("epoch {:>3}/{}  loss {:.4f}  val_acc {:.4f}  val_macroF1 {:.4f}{}".format(
                epoch, epochs, total / max(seen, 1), scores["accuracy"], scores["macro_f1"], flag),
                flush=True)

        if patience and stale >= patience:
            if not quiet:
                print("early stop: no {} improvement for {} epochs".format(monitor, patience))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score, best_epoch


def run(args, quiet=False):
    """Train and evaluate one configuration; returns the results dict."""
    set_seed(args.seed)
    start = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(path):
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(script_dir, path))

    dataset_dir, features_dir = resolve(args.dataset), resolve(args.features)
    root = os.path.abspath(os.path.join(script_dir, ".."))
    outputs_dir = os.path.join(root, "Outputs")
    models_dir = os.path.join(root, "Saved_Models")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    backbone = args.backbone if args.modality in ("visual", "multimodal") else None
    loaders, meta = D.load(
        dataset_dir, features_dir, backbone,
        max_len=args.max_len, num_words=args.num_words, batch_size=args.batch_size,
        embedding=args.embedding, emb_dim=args.emb_dim, vectors_path=args.vectors,
        subset=args.subset, seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MM.build(
        args.modality, meta["num_classes"],
        vocab_size=meta["vocab_size"], emb_dim=args.emb_dim,
        emb_matrix=meta["emb_matrix"], emb_trainable=meta["emb_trainable"],
        text_model=args.text_model, cnn_filters=args.cnn_filters,
        kernel_size=args.kernel_size, lstm_units=args.lstm_units,
        text_dense=args.text_dense, feature_shape=meta["feature_shape"],
        visual_pool=args.visual_pool, visual_dense=args.visual_dense, dropout=args.dropout,
    ).to(device)

    tag = args.run_name or default_run_name(args)
    if not quiet:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Run         : {}  ({}, {:.2f}M trainable params on {})".format(
            tag, model.modality, trainable / 1e6, device))

    model, best_val, best_epoch = train(
        model, loaders, device, args.epochs, args.lr, args.patience,
        args.monitor, meta["target_names"], quiet=quiet,
    )

    true, pred, probs = predict(model, loaders["testing_set"], device)
    results = M.summarise(true, pred, meta["target_names"])
    results.update({
        "run_name": tag,
        "framework": "MuLAD",
        "modality": args.modality,
        "text_model": args.text_model if args.modality != "visual" else None,
        "backbone": backbone,
        "embedding": args.embedding if args.modality != "visual" else None,
        "best_val_{}".format(args.monitor): best_val,
        "best_epoch": best_epoch,
        "degenerate": M.is_degenerate(pred),
        "runtime_s": round(time.time() - start, 1),
        "hyperparameters": {
            "max_len": args.max_len, "num_words": args.num_words, "emb_dim": args.emb_dim,
            "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
            "patience": args.patience, "monitor": args.monitor, "seed": args.seed,
            "cnn_filters": args.cnn_filters, "kernel_size": args.kernel_size,
            "lstm_units": args.lstm_units, "text_dense": args.text_dense,
            "visual_pool": args.visual_pool, "visual_dense": args.visual_dense,
            "dropout": args.dropout, "subset": args.subset,
        },
    })

    with open(os.path.join(outputs_dir, "results_{}.json".format(tag)), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    frame = meta["frames"]["testing_set"].copy()
    frame["true_id"], frame["pred_id"] = true, pred
    frame["pred_label"] = [meta["target_names"][p] for p in pred]
    for i, name in enumerate(meta["target_names"]):
        frame["prob_{}".format(name)] = probs[:, i]
    frame.to_csv(os.path.join(outputs_dir, "predictions_{}.csv".format(tag)),
                 index=False, encoding="utf-8")

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(models_dir, "mulad_{}.pth".format(tag)))

    if not quiet:
        print("\nClassification Report :")
        M.print_metrices(true, pred, list(range(meta["num_classes"])), meta["target_names"])
        print("\nConfusion matrix (rows = true, cols = predicted)")
        print(pd.DataFrame(results["confusion_matrix"],
                           index=meta["target_names"], columns=meta["target_names"]).to_string())
        if results["degenerate"]:
            print("\n*** DEGENERATE: this model predicted one class for every test meme. "
                  "Its scores are the majority-class baseline, not a learned result. ***")
        print("\nSaved results -> {}".format(
            os.path.join(outputs_dir, "results_{}.json".format(tag))))
        print("Total time : {:.1f}s".format(time.time() - start))

    return results


def build_parser():
    p = argparse.ArgumentParser(description="MuLAD - multimodal aggression detection from memes")
    p.add_argument("--dataset", type=str, default="../Dataset")
    p.add_argument("--features", type=str, default="../Features")
    p.add_argument("--modality", type=str, default="multimodal",
                   choices=["text", "visual", "multimodal"])
    p.add_argument("--text_model", type=str, default="cnn", choices=list(MM.TEXT_MODELS))
    p.add_argument("--backbone", type=str, default="vgg16", choices=list(BACKBONES))
    p.add_argument("--embedding", type=str, default="keras",
                   choices=["keras", "fasttext", "selftrained"],
                   help="keras = 64-d table learned from scratch; selftrained = the "
                        "paper's GloVe row, trained on our captions")
    p.add_argument("--vectors", type=str, default=None,
                   help="path to a pretrained .vec file for --embedding fasttext")
    p.add_argument("--max_len", type=int, default=130, help="paper Sec. 4.2")
    p.add_argument("--num_words", type=int, default=8000, help="paper Sec. 4.2")
    p.add_argument("--emb_dim", type=int, default=64, help="64 for keras, 300 for pretrained")
    p.add_argument("--batch_size", type=int, default=32, help="paper Table 2")
    p.add_argument("--lr", type=float, default=1e-3, help="paper Table 2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8, help="0 disables early stopping")
    p.add_argument("--monitor", type=str, default="accuracy", choices=["accuracy", "macro_f1"])
    p.add_argument("--cnn_filters", type=int, default=128,
                   help="paper Sec. 4.2 (Sec. 5.1 contradicts it with 32)")
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--lstm_units", type=int, default=100)
    p.add_argument("--text_dense", type=int, default=32)
    p.add_argument("--visual_pool", type=str, default="auto", choices=["auto", "flatten", "gap"],
                   help="auto = flatten for fusion, GAP for visual baselines, as the paper has it")
    p.add_argument("--visual_dense", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--subset", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
