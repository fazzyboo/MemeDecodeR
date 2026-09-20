"""
Shared evaluation metrics, so MAF and MuLAD are scored by identical code.

Reports the MAF paper's metric set (Table 4): accuracy, weighted precision/recall/F1,
macro F1 and MMAE, plus the per-class report and confusion matrix.
"""
import numpy as np
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
except ImportError:  # imbalanced-learn is optional; MMAE is then computed directly
    macro_averaged_mean_absolute_error = None


def mmae(true, pred):
    """Macro-averaged mean absolute error (Baccianella et al., 2009).

    Averages the mean absolute ordinal error within each true class, so a rare class
    counts as much as a common one.
    """
    if macro_averaged_mean_absolute_error is not None:
        return float(macro_averaged_mean_absolute_error(true, pred))
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    per_class = [np.abs(pred[true == c] - true[true == c]).mean() for c in np.unique(true)]
    return float(np.mean(per_class))


def print_metrices(true, pred, labels, target_names):
    print(classification_report(true, pred, labels=labels, target_names=target_names, digits=3))
    print("Accuracy : ", accuracy_score(true, pred))
    print("Precison : ", precision_score(true, pred, average="weighted", zero_division=0))
    print("Recall   : ", recall_score(true, pred, average="weighted", zero_division=0))
    print("F1       : ", f1_score(true, pred, average="weighted", zero_division=0))
    print("Macro F1 : ", f1_score(true, pred, average="macro", zero_division=0))
    print("MMAE     : ", mmae(true, pred))


def summarise(true, pred, target_names):
    """Every headline metric plus the per-class report, as a JSON-serialisable dict."""
    labels = list(range(len(target_names)))
    return {
        "accuracy": accuracy_score(true, pred),
        "weighted_f1": f1_score(true, pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(true, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(true, pred, average="weighted", zero_division=0),
        "mmae": mmae(true, pred),
        "per_class": classification_report(
            true, pred, labels=labels, target_names=target_names,
            digits=3, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
        "target_names": list(target_names),
    }


def is_degenerate(pred):
    """True when a model emitted a single class for every input.

    MuLAD's published Table 5 contains roughly eighteen such runs - the tuple
    (0.812, 0.659, 0.812, 0.728) repeated across configurations, which is exactly the
    majority-class share of their test set. They are reported there as results. Flagging
    the condition explicitly keeps that failure mode visible in our own tables.
    """
    return len(set(int(p) for p in pred)) == 1
