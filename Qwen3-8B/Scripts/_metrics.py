"""
Shared metric reporting for the VLM scripts.

main.py computes these inline for the MAF model. The VLM scripts run in a different conda
environment that has no `clip` package, so importing main.py to reuse its helpers is not
possible - it pulls in dataset.py and evaluation.py and therefore the whole MAF stack.
This module is a standalone copy of the metric set so that a VLM run writes a
results_<run>.json byte-comparable in structure to a MAF run, and the comparison snippets
in RUN_COMMANDS.txt work across both.

The metric set is the paper's Table 4: accuracy, weighted F1, macro F1 and MMAE.
"""
import json
import os

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

try:
    from imblearn.metrics import macro_averaged_mean_absolute_error
except ImportError:
    macro_averaged_mean_absolute_error = None

# Same order and integer encoding as dataset.TARGET_NAMES - the paper's label codes.
TARGET_NAMES = ["NoAg", "GAg", "PAg", "RAg"]
NUM_CLASSES = 4

# The canonical label strings used in the split CSVs, in the paper's integer order.
LABEL_TO_ID = {
    "non-aggressive": 0,
    "gendered aggression": 1,
    "political aggression": 2,
    "religious aggression": 3,
}


def mmae(true, pred):
    """Macro-averaged mean absolute error (Baccianella et al., 2009)."""
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


def report_and_save(actual, pred, outputs_dir, run_name, extra=None, frame=None):
    """Print the paper's metric set, then write results_<run>.json and predictions_<run>.csv.

    `extra` is merged into the JSON (used for the VLM's model name, prompt mode, etc.).
    `frame` is the dataframe the run iterated; prediction columns are appended to it.
    """
    actual = [int(x) for x in actual]
    pred = [int(x) for x in pred]
    labels = list(range(NUM_CLASSES))
    os.makedirs(outputs_dir, exist_ok=True)

    print("Classification Report :")
    print_metrices(actual, pred, labels)

    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    cm = confusion_matrix(actual, pred, labels=labels)
    print(pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES).to_string())

    results = {
        "run_name": run_name,
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
    }
    results.update(extra or {})

    results_path = os.path.join(outputs_dir, "results_{}.json".format(run_name))
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    preds_path = os.path.join(outputs_dir, "predictions_{}.csv".format(run_name))
    if frame is not None:
        frame = frame.copy()
        frame["true_id"] = actual
        frame["pred_id"] = pred
        frame["pred_label"] = [TARGET_NAMES[p] for p in pred]
        frame.to_csv(preds_path, index=False, encoding="utf-8")

    print()
    print("Saved results   ->", results_path)
    if frame is not None:
        print("Saved preds     ->", preds_path)
    return results
