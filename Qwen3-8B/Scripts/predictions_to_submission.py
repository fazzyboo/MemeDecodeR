"""
Stage 4 (no checkpoint needed) - turn a predictions_<run>.csv into submission.csv.

main.py's final test evaluation runs over testing_set.csv, which holds exactly the 400
images sample_submission.csv lists. The predictions_<run>.csv it writes therefore already
contains that run's prediction for every submission image. This script keeps only
image_name and pred_label, maps pred_label to the competition vocabulary, and orders the
rows to match sample_submission.csv. The Label and true_id columns in that file are never
read.

This gives the same predictions as running generate_submission.py on that run's best
checkpoint: the same weights, captions and preprocessing produced them, in eval mode. It is
the only route when the checkpoint is gone, and it needs no GPU, model download or PyTorch.

Usage:
    python predictions_to_submission.py --predictions ../Outputs/predictions_maf_full.csv
    python predictions_to_submission.py --predictions ../Outputs/predictions_*.csv   # several runs
"""
import argparse
import glob
import os
import sys

import pandas as pd

from submission_format import SHORT_TO_SUBMISSION, validate_submission

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))


def convert(predictions_path, sample, out_path):
    preds = pd.read_csv(predictions_path)
    missing_cols = {"image_name", "pred_label"} - set(preds.columns)
    if missing_cols:
        raise ValueError("{} lacks columns {} - is it a main.py predictions file?".format(
            os.path.basename(predictions_path), sorted(missing_cols)))
    if preds["image_name"].duplicated().any():
        raise ValueError("duplicate image names in {}".format(os.path.basename(predictions_path)))
    unknown = set(preds["pred_label"]) - set(SHORT_TO_SUBMISSION)
    if unknown:
        raise ValueError("unrecognised pred_label values: {}".format(sorted(unknown)))

    id_col, target_col = sample.columns[0], sample.columns[1]
    by_image = dict(zip(preds["image_name"], preds["pred_label"].map(SHORT_TO_SUBMISSION)))
    absent = [name for name in sample[id_col] if name not in by_image]
    if absent:
        raise ValueError(
            "{} of the {} submission images have no prediction (e.g. {}). "
            "A --subset smoke test only predicts a handful of memes and cannot be submitted.".format(
                len(absent), len(sample), absent[:3]))

    out = pd.DataFrame({
        id_col: sample[id_col].tolist(),
        target_col: [by_image[name] for name in sample[id_col]],
    })
    validate_submission(out, sample)
    out.to_csv(out_path, index=False)
    return out


def main(args):
    sample = pd.read_csv(args.sample_submission)
    target_col = sample.columns[1]
    paths = sorted({p for pattern in args.predictions for p in glob.glob(pattern)})
    if not paths:
        sys.exit("No predictions files matched: {}".format(args.predictions))
    if args.out and len(paths) > 1:
        sys.exit("--out only makes sense with a single predictions file")

    failures = 0
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        run = stem[len("predictions_"):] if stem.startswith("predictions_") else stem
        out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(path)), "submission_{}.csv".format(run))
        try:
            out = convert(path, sample, out_path)
        except ValueError as exc:
            failures += 1
            print("SKIP  {:<24} {}".format(run, exc))
            continue
        counts = out[target_col].value_counts()
        print("OK    {:<24} -> {}".format(run, out_path))
        print("      " + "  ".join("{} {}".format(k, counts.get(k, 0)) for k in SHORT_TO_SUBMISSION.values()))
    if failures == len(paths):
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build submission.csv from main.py predictions files")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="one or more predictions_<run>.csv paths (globs allowed)")
    p.add_argument("--sample_submission", type=str,
                   default=os.path.join(PROJECT_ROOT, "sample_submission.csv"))
    p.add_argument("--out", type=str, default=None,
                   help="output path; default submission_<run>.csv next to the predictions file")
    main(p.parse_args())
