"""
Entry point - port of the original MIMOSA Scripts/main.py.

Reports the same metric set as the paper (Table 4): accuracy, weighted F1, macro F1 and
MMAE (macro-averaged mean absolute error), plus the per-class classification report.

Changes from the original:
  1. target_names is four classes, not five - our data has no "others" category.
  2. The hyperparameter defaults follow the paper's Appendix A (batch size 4, 20 epochs,
     lr 5e-5 for MAF) rather than the repository defaults (batch 16, 5 epochs), which do
     not match the published setup.
  3. Results, the confusion matrix and per-meme predictions are written to Outputs/ so a
     run is reproducible after the terminal is closed.
  4. --subset runs the whole pipeline on a handful of memes, for a fast end-to-end check.

Usage:
    python main.py                        # full run, paper hyperparameters
    python main.py --subset 24 --n_iter 1 # smoke test
"""
import SIGLIP.Scripts._paths as _paths  # noqa: F401  - must precede transformers / clip imports

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import SIGLIP.Scripts.dataset as d
import SIGLIP.Scripts.evaluation as e

try:
    from imblearn.metrics import macro_averaged_mean_absolute_error
except ImportError:  # imbalanced-learn is optional; MMAE is then computed directly
    macro_averaged_mean_absolute_error = None

import warnings

warnings.filterwarnings("ignore")

TARGET_NAMES = d.TARGET_NAMES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mmae(true, pred):
    """Macro-averaged mean absolute error (Baccianella et al., 2009).

    Averages the mean absolute ordinal error within each true class, so a class with few
    samples counts as much as a large one.
    """
    if macro_averaged_mean_absolute_error is not None:
        return float(macro_averaged_mean_absolute_error(true, pred))
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    per_class = [np.abs(pred[true == c] - true[true == c]).mean() for c in np.unique(true)]
    return float(np.mean(per_class))


def print_metrices(true, pred, labels):
    print(classification_report(true, pred, labels=labels, target_names=TARGET_NAMES, digits=3))
    print("Accuracy : ", accuracy_score(true, pred))
    print("Precison : ", precision_score(true, pred, average="weighted"))
    print("Recall : ", recall_score(true, pred, average="weighted"))
    print("F1 : ", f1_score(true, pred, average="weighted"))
    print("Macro F1 : ", f1_score(true, pred, average="macro"))
    print("MMAE: ", mmae(true, pred))


def main(args):
    set_seed(args.seed)
    start_time = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_base_path = os.path.join(root_dir, args.dataset_path)
    excel_path = os.path.join(dataset_base_path, "")
    memes_path = os.path.join(dataset_base_path, "Img")

    saved_models_dir = os.path.join(root_dir, args.model_path)
    os.makedirs(saved_models_dir, exist_ok=True)
    outputs_dir = os.path.join(root_dir, "Outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Load the processed data splits
    train_loader, valid_loader, test_loader = d.load_dataset(
        excel_path,
        memes_path,
        args.maximum_length,
        args.batch,
        test_batch_size=args.batch,
        subset=args.subset,
    )

    # One checkpoint file per run. A single shared maf_model.pth (as in the original repo)
    # lets consecutive runs in one notebook silently overwrite each other's models.
    tag = args.run_name or "maf"
    checkpoint_name = "maf_model_{}.pth".format(args.run_name) if args.run_name else "maf_model.pth"
    print("Checkpoint file:", os.path.join(saved_models_dir, checkpoint_name))

    # Train, then evaluate on the held-out test set
    actual, pred = e.pipline(
        train_loader,
        valid_loader,
        test_loader,
        saved_models_dir,
        args.n_heads,
        args.epochs,
        args.lr_rate,
        num_classes=d.NUM_CLASSES,
        seq_len=args.maximum_length,
        attn_variant=args.attn_variant,
        fix_scheduler=args.fix_scheduler,
        checkpoint_name=checkpoint_name,
    )

    actual = [int(x) for x in actual]
    pred = [int(x) for x in pred]
    labels = list(range(d.NUM_CLASSES))

    print("Classification Report :")
    print_metrices(actual, pred, labels)

    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    cm = confusion_matrix(actual, pred, labels=labels)
    print(pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES).to_string())

    # Persist results
    results = {
        "run_name": tag,
        "checkpoint": checkpoint_name,
        "accuracy": accuracy_score(actual, pred),
        "weighted_f1": f1_score(actual, pred, average="weighted"),
        "macro_f1": f1_score(actual, pred, average="macro"),
        "weighted_precision": precision_score(actual, pred, average="weighted"),
        "weighted_recall": recall_score(actual, pred, average="weighted"),
        "mmae": mmae(actual, pred),
        "per_class": classification_report(
            actual, pred, labels=labels, target_names=TARGET_NAMES, digits=3, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "target_names": TARGET_NAMES,
        "hyperparameters": {
            "max_len": args.maximum_length,
            "batch_size": args.batch,
            "heads": args.n_heads,
            "epochs": args.epochs,
            "lr": args.lr_rate,
            "seed": args.seed,
            "attn_variant": args.attn_variant,
            "fix_scheduler": args.fix_scheduler,
            "subset": args.subset,
        },
    }
    with open(os.path.join(outputs_dir, "results_{}.json".format(tag)), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Take the dataframe the loader actually iterated, not a re-read of the CSV - under
    # --subset those differ, and the rows would silently misalign with the predictions.
    preds_frame = test_loader.dataset.data.copy()
    preds_frame["true_id"] = actual
    preds_frame["pred_id"] = pred
    preds_frame["pred_label"] = [TARGET_NAMES[p] for p in pred]
    preds_frame.to_csv(
        os.path.join(outputs_dir, "predictions_{}.csv".format(tag)), index=False, encoding="utf-8"
    )

    print()
    print("Saved results   ->", os.path.join(outputs_dir, "results_{}.json".format(tag)))
    print("Saved preds     ->", os.path.join(outputs_dir, "predictions_{}.csv".format(tag)))
    print("Total time : {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bengali Aggressive Memes Classification (MAF)")

    parser.add_argument("--dataset", dest="dataset_path", type=str, default="Dataset",
                        help="the directory of the dataset folder")
    parser.add_argument("--max_len", dest="maximum_length", type=int, default=70,
                        help="the maximum text length - default 70")
    parser.add_argument("--batch_size", dest="batch", type=int, default=4,
                        help="Batch Size - default 4 (paper Appendix A)")
    parser.add_argument("--model", dest="model_path", type=str, default="Saved_Models",
                        help="the directory of the saved model folder")
    parser.add_argument("--heads", dest="n_heads", type=int, default=16,
                        help="number of attention heads - default 16")
    parser.add_argument("--n_iter", dest="epochs", type=int, default=20,
                        help="Number of Epochs - default 20 (paper Appendix A)")
    parser.add_argument("--lrate", dest="lr_rate", type=float, default=5e-5,
                        help="Learning rate - default 5e-5 (paper Appendix A)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--run_name", type=str, default=None, help="tag for the Outputs/ files")
    parser.add_argument("--attn_variant", type=str, default="code", choices=["code", "paper"],
                        help="attention operand order: released code (default) or paper text")
    parser.add_argument("--fix_scheduler", action="store_true",
                        help="step the LR scheduler per batch instead of per epoch")
    parser.add_argument("--subset", type=int, default=0,
                        help="use only N rows per split - for a fast end-to-end smoke test")

    args = parser.parse_args()
    main(args)
