"""
Entry point - port of the original MIMOSA Scripts/main.py.

Reports the same metric set as the paper (Table 4): accuracy, weighted F1, macro F1 and
MMAE (macro-averaged mean absolute error), plus the per-class classification report.

Changes from the original:
  1. The class count comes from the data (labels.py), not a hardcoded list, and the
     metric code is shared with MuLAD via metrics.py so the two are scored alike.
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
import _paths  # noqa: F401  - must precede transformers / clip imports

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

import dataset as d
import evaluation as e
import metrics as M

import warnings

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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
        suffix=args.captions_suffix,
    )

    # Resolved by load_dataset() from the CSVs: five-way for full MIMOSA, four-way for a
    # subset without "others".
    target_names = d.TARGET_NAMES
    labels = list(range(d.NUM_CLASSES))

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

    print("Classification Report :")
    M.print_metrices(actual, pred, labels, target_names)

    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    results = M.summarise(actual, pred, target_names)
    print(pd.DataFrame(results["confusion_matrix"],
                       index=target_names, columns=target_names).to_string())

    # Persist results
    results.update({
        "run_name": tag,
        "framework": "MAF",
        "checkpoint": checkpoint_name,
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
            "captions_suffix": args.captions_suffix,
        },
    })

    with open(os.path.join(outputs_dir, "results_{}.json".format(tag)), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Take the dataframe the loader actually iterated, not a re-read of the CSV - under
    # --subset those differ, and the rows would silently misalign with the predictions.
    preds_frame = test_loader.dataset.data.copy()
    preds_frame["true_id"] = actual
    preds_frame["pred_id"] = pred
    preds_frame["pred_label"] = [target_names[p] for p in pred]
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
    parser.add_argument("--captions_suffix", type=str, default="",
                        help="caption variant to train on, e.g. _ocr for training_set_ocr.csv")

    args = parser.parse_args()
    main(args)
