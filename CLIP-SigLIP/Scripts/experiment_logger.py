"""
experiment_logger.py -- full-fidelity logging layer for the MAF / MIMOSA pipeline.

This module is *additive*: it does not require any edit to main.py, dataset.py,
models.py or evaluation.py.  It runs the exact same pipeline as `python main.py`
but wraps it so that every artefact a researcher needs afterwards is written to
disk under `Runs/<timestamp>_<tag>/`.

Usage
-----
    python experiment_logger.py                       # same defaults as main.py
    python experiment_logger.py --n_iter 10 --lrate 2e-5 --run_name maf_10ep
    python experiment_logger.py --max_batches 3 --n_iter 1 --run_name smoke

What gets written (one directory per run)
-----------------------------------------
    console.log            verbatim stdout+stderr, tqdm bars included
    run.log                timestamped structured log
    config.json            every hyper-parameter + resolved paths + seed
    environment.json       python/torch/cuda/gpu/git/pip state
    dataset_stats.json     split sizes, class distribution, loader config
    model_summary.txt      module tree + per-module parameter counts
    batch_metrics.csv      per-training-batch loss / accuracy / lr / throughput
    epoch_metrics.csv      per-epoch train + validation metrics, timing, GPU mem
    val_reports/           per-epoch validation classification reports (json)
    metrics.jsonl          every event above as one JSON object per line
    resources.csv          periodic GPU/CPU/RAM sampling
    test_predictions.csv   per-test-sample true / pred / softmax probabilities
    classification_report.txt|.json
    confusion_matrix.csv|.png
    summary.json           headline numbers for the whole run

Nothing here changes the science: the hooks only observe values that the
original code already computes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Class vocabulary -- mirrors the encoding in dataset.py / main.py
# ---------------------------------------------------------------------------
CLASS_IDS = [0, 1, 2, 3, 4]
CLASS_SHORT = ["NoAg", "GAg", "PAg", "RAg", "Oth"]
CLASS_FULL = [
    "non-aggressive",
    "gendered aggression",
    "political aggression",
    "religious aggression",
    "others",
]

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))


# ===========================================================================
# 1.  Compatibility shims
# ===========================================================================
def apply_compat_shims(verbose=True):
    """Make the original 2024 scripts import cleanly on a modern stack.

    The only incompatibility in this repo is `from transformers import AdamW`:
    that alias was deprecated in transformers 4.x and removed in 5.x.  Nothing
    in the repo actually uses it (models.py builds `torch.optim.AdamW`), so we
    re-attach a harmless alias to the module object instead of editing the
    original files.

    transformers 5 replaces its own entry in `sys.modules` the first time a
    lazy attribute is resolved, which silently drops the alias, so this must be
    called again immediately before each module that needs it is imported.
    It is idempotent and cheap.
    """
    notes = []
    import torch
    import transformers

    transformers = sys.modules["transformers"]
    if not hasattr(transformers, "AdamW"):
        transformers.AdamW = torch.optim.AdamW
        notes.append(
            "transformers.AdamW re-aliased to torch.optim.AdamW "
            f"(transformers {transformers.__version__} removed it)"
        )
    if verbose:
        for n in notes:
            print(f"[compat] {n}")
    return notes


def seed_everything(seed):
    """Seed python / numpy / torch so a run can be repeated."""
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# ===========================================================================
# 2.  stdout/stderr tee
# ===========================================================================
class _Tee:
    """Duplicate a text stream into a file, transparently for tqdm."""

    def __init__(self, stream, handle, lock):
        self._stream = stream
        self._handle = handle
        self._lock = lock

    def write(self, data):
        with self._lock:
            self._stream.write(data)
            try:
                self._handle.write(data)
            except ValueError:  # file closed during shutdown
                pass
        return len(data)

    def flush(self):
        with self._lock:
            self._stream.flush()
            try:
                self._handle.flush()
            except ValueError:
                pass

    def close(self):
        """absl/logging calls close() on whatever sys.stderr was at import time.

        Only flush: the wrapped stream is the real stdout/stderr, which must stay
        open, and RunLogger.close() is what restores the originals.
        """
        try:
            self.flush()
        except Exception:
            pass

    # tqdm inspects these before drawing a bar
    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


# ===========================================================================
# 3.  The run logger
# ===========================================================================
class RunLogger:
    def __init__(self, run_dir, resource_interval=15.0, training_csvs=True):
        """`training_csvs=False` for runs with no training loop (e.g. zero-shot),
        so the run folder is not littered with header-only CSVs."""
        self.run_dir = run_dir
        self.val_dir = os.path.join(run_dir, "val_reports")
        if training_csvs:
            os.makedirs(self.val_dir, exist_ok=True)

        self.t0 = time.time()
        self._lock = threading.Lock()
        self._closed = False

        self.console_fh = open(os.path.join(run_dir, "console.log"), "w", buffering=1, encoding="utf-8")
        self.log_fh = open(os.path.join(run_dir, "run.log"), "w", buffering=1, encoding="utf-8")
        self.jsonl_fh = open(os.path.join(run_dir, "metrics.jsonl"), "w", buffering=1, encoding="utf-8")

        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self.console_fh, self._lock)
        sys.stderr = _Tee(self._orig_stderr, self.console_fh, self._lock)

        if training_csvs:
            self.batch_csv = self._csv(
                "batch_metrics.csv",
                ["wall_time", "elapsed_s", "epoch", "global_step", "step_in_epoch",
                 "loss", "batch_acc", "running_loss", "running_acc", "lr",
                 "samples_per_s", "gpu_alloc_mb"],
            )
            self.epoch_csv = self._csv(
                "epoch_metrics.csv",
                ["epoch", "train_loss", "train_acc", "val_acc", "val_macro_f1",
                 "val_weighted_f1", "val_macro_precision", "val_macro_recall",
                 "val_mmae", "lr_end", "train_time_s", "val_time_s", "epoch_time_s",
                 "gpu_peak_mb", "is_best"],
            )
        else:
            self.batch_csv = self.epoch_csv = None
        self.resource_csv = self._csv(
            "resources.csv",
            ["elapsed_s", "cpu_percent", "rss_mb", "gpu_util_percent",
             "gpu_mem_used_mb", "gpu_temp_c", "torch_alloc_mb", "torch_reserved_mb"],
        )

        # running state filled in by the hooks
        self.global_step = 0
        self.epoch = 1
        self.step_in_epoch = 0
        self.epoch_loss_sum = 0.0
        self.epoch_acc_sum = 0.0
        self.optimizer = None
        self.model = None
        self.best_val_acc = -1.0
        self.best_epoch = None
        self._epoch_t0 = time.time()
        self._last_batch_t = time.time()
        self._batch_size = None
        self.epoch_rows = []
        self.timings = {}

        self._stop_sampler = threading.Event()
        self._sampler = threading.Thread(
            target=self._sample_resources, args=(resource_interval,), daemon=True
        )
        self._sampler.start()

    # -- plumbing ----------------------------------------------------------
    def _csv(self, name, header):
        fh = open(os.path.join(self.run_dir, name), "w", newline="", buffering=1, encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(header)
        return (fh, writer)

    def _row(self, target, values):
        with self._lock:
            if self._closed or target is None:
                return
            try:
                target[1].writerow(values)
            except ValueError:  # file closed underneath us during shutdown
                pass

    def log(self, message, level="INFO"):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} | {level:<7} | +{time.time() - self.t0:8.1f}s | {message}"
        with self._lock:
            if not self._closed:
                self.log_fh.write(line + "\n")
        print(f"[log] {message}")

    def event(self, kind, **payload):
        record = {"t": round(time.time() - self.t0, 3), "kind": kind}
        record.update(payload)
        with self._lock:
            if not self._closed:
                self.jsonl_fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def dump(self, name, obj):
        path = os.path.join(self.run_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        return path

    # -- resource sampling -------------------------------------------------
    def _sample_resources(self, interval):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            psutil, proc = None, None
        import torch

        while not self._stop_sampler.wait(interval):
            cpu = rss = None
            if proc is not None:
                try:
                    cpu = proc.cpu_percent(interval=None)
                    rss = proc.memory_info().rss / 1e6
                except Exception:
                    pass
            util = mem = temp = None
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                if out.returncode == 0 and out.stdout.strip():
                    util, mem, temp = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            except Exception:
                pass
            alloc = reserved = None
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1e6
                reserved = torch.cuda.memory_reserved() / 1e6
            self._row(self.resource_csv, [
                round(time.time() - self.t0, 1), cpu, rss, util, mem, temp,
                None if alloc is None else round(alloc, 1),
                None if reserved is None else round(reserved, 1),
            ])

    # -- shutdown ----------------------------------------------------------
    def close(self):
        self._stop_sampler.set()
        self._sampler.join(timeout=5)
        with self._lock:
            self._closed = True
        sys.stdout, sys.stderr = self._orig_stdout, self._orig_stderr
        for handle in (self.console_fh, self.log_fh, self.jsonl_fh):
            try:
                handle.close()
            except Exception:
                pass
        for target in (self.batch_csv, self.epoch_csv, self.resource_csv):
            if target is None:
                continue
            try:
                target[0].close()
            except Exception:
                pass


# ===========================================================================
# 4.  Environment capture
# ===========================================================================
def _shell(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT_DIR)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_environment():
    import numpy as np
    import torch

    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cwd": os.getcwd(),
        "repo_root": ROOT_DIR,
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        },
        "git": {
            "commit": _shell(["git", "rev-parse", "HEAD"]),
            "branch": _shell(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty_files": (_shell(["git", "status", "--porcelain"]) or "").splitlines(),
        },
        "env_vars": {
            k: v for k, v in os.environ.items()
            if k.startswith(("CUDA", "PYTORCH", "HF_", "TRANSFORMERS", "OMP_", "TOKENIZERS"))
        },
    }

    for mod in ("pandas", "transformers", "sklearn", "PIL", "torchvision", "imblearn", "clip"):
        try:
            env["packages"][mod] = __import__(mod).__version__
        except Exception:
            env["packages"][mod] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["cuda"]["device_name"] = props.name
        env["cuda"]["total_memory_mb"] = round(props.total_memory / 1e6, 1)
        env["cuda"]["capability"] = f"{props.major}.{props.minor}"
    env["nvidia_smi"] = _shell(["nvidia-smi"])
    env["pip_freeze"] = (_shell([sys.executable, "-m", "pip", "freeze"]) or "").splitlines()
    return env


def describe_model(model):
    """Module tree with parameter counts (trainable / frozen)."""
    lines, total, trainable = [], 0, 0
    lines.append(f"{'module':<52}{'params':>14}{'trainable':>12}")
    lines.append("-" * 78)
    for name, module in model.named_children():
        p = sum(x.numel() for x in module.parameters())
        t = sum(x.numel() for x in module.parameters() if x.requires_grad)
        lines.append(f"{name:<52}{p:>14,}{t:>12,}")
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<52}{total:>14,}{trainable:>12,}")
    lines.append(f"{'frozen':<52}{total - trainable:>14,}")
    lines.append("")
    lines.append("Full module tree")
    lines.append("=" * 78)
    lines.append(str(model))
    return "\n".join(lines), total, trainable


# ===========================================================================
# 5.  Metric helpers
# ===========================================================================
def _as_int_array(values):
    import numpy as np
    return np.asarray(values).astype(int).ravel()


def compute_report(y_true, y_pred):
    """Full classification metrics; identical definitions to main.print_metrices."""
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score, precision_score, recall_score, f1_score)

    y_true, y_pred = _as_int_array(y_true), _as_int_array(y_pred)
    report = classification_report(
        y_true, y_pred, labels=CLASS_IDS, target_names=CLASS_SHORT,
        digits=3, output_dict=True, zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    try:
        from imblearn.metrics import macro_averaged_mean_absolute_error
        metrics["mmae"] = float(macro_averaged_mean_absolute_error(y_true, y_pred))
    except Exception as exc:  # metric is ordinal-specific; never fail the run for it
        metrics["mmae"] = None
        metrics["mmae_error"] = str(exc)
    metrics["per_class"] = {
        CLASS_SHORT[i]: {
            "full_name": CLASS_FULL[i],
            **{k: float(v) for k, v in report[CLASS_SHORT[i]].items()},
        }
        for i in CLASS_IDS
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=CLASS_IDS).tolist()
    metrics["support_true"] = {CLASS_SHORT[i]: int((y_true == i).sum()) for i in CLASS_IDS}
    metrics["support_pred"] = {CLASS_SHORT[i]: int((y_pred == i).sum()) for i in CLASS_IDS}
    return metrics


def save_confusion_matrix(cm, path_png, path_csv, title):
    import numpy as np
    with open(path_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred"] + CLASS_SHORT)
        for name, row in zip(CLASS_SHORT, cm):
            writer.writerow([name] + list(row))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(5), CLASS_SHORT)
        ax.set_yticks(range(5), CLASS_SHORT)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thresh = cm.max() / 2 if cm.max() else 0.5
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(path_png, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[log] confusion-matrix plot skipped: {exc}")


# ===========================================================================
# 6.  Loader wrapper used by --max_batches (smoke tests)
# ===========================================================================
class LimitedLoader:
    """Yields at most `limit` batches; keeps len() consistent for schedulers."""

    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = limit

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.limit:
                break
            yield batch

    def __len__(self):
        return min(self.limit, len(self.loader))

    def __getattr__(self, item):
        return getattr(self.loader, item)


# ===========================================================================
# 7.  Hooks -- everything below only observes values the original code
#     already computes.  No training behaviour is altered.
# ===========================================================================
def install_hooks(logger, dataset_mod, models_mod, evaluation_mod, cfg):
    import torch
    import torch.nn.functional as F

    logger._batch_size = cfg["batch_size"]

    # -- 7.1 dataset ------------------------------------------------------
    orig_load_dataset = dataset_mod.load_dataset

    def load_dataset_hook(files_path, memes_path, max_len, batch_size):
        logger.log(f"Building data loaders (max_len={max_len}, batch_size={batch_size})")
        t0 = time.time()
        train_loader, val_loader, test_loader = orig_load_dataset(
            files_path, memes_path, max_len, batch_size)
        elapsed = time.time() - t0
        logger.timings["data_loading_s"] = elapsed

        import pandas as pd
        stats = {"files_path": files_path, "images_path": memes_path,
                 "max_len": max_len, "batch_size": batch_size,
                 "build_time_s": round(elapsed, 2), "splits": {}}
        for split, name in (("train", "training_set.csv"),
                            ("validation", "validation_set.csv"),
                            ("test", "testing_set.csv")):
            try:
                df = pd.read_csv(os.path.join(files_path, name))
                counts = df["Label"].value_counts().to_dict()
                stats["splits"][split] = {
                    "csv": name,
                    "n_samples": int(len(df)),
                    "class_counts": {str(k): int(v) for k, v in counts.items()},
                    "caption_chars_mean": float(df["Captions"].astype(str).str.len().mean()),
                    "caption_chars_max": int(df["Captions"].astype(str).str.len().max()),
                }
            except Exception as exc:
                stats["splits"][split] = {"error": str(exc)}
        for split, loader in (("train", train_loader), ("validation", val_loader), ("test", test_loader)):
            stats["splits"].setdefault(split, {})
            stats["splits"][split]["n_batches"] = len(loader)
            stats["splits"][split]["loader_batch_size"] = loader.batch_size

        if cfg["max_batches"]:
            n = cfg["max_batches"]
            logger.log(f"--max_batches={n}: truncating every loader (smoke-test mode)", "WARN")
            train_loader = LimitedLoader(train_loader, n)
            val_loader = LimitedLoader(val_loader, n)
            test_loader = LimitedLoader(test_loader, n)
            stats["max_batches"] = n

        logger.dump("dataset_stats.json", stats)
        logger.event("dataset", **stats)
        logger.log(f"Loaders ready in {elapsed:.1f}s "
                   f"({len(train_loader)}/{len(val_loader)}/{len(test_loader)} train/val/test batches)")
        return train_loader, val_loader, test_loader

    dataset_mod.load_dataset = load_dataset_hook

    # -- 7.2 model construction -------------------------------------------
    # models.py uses the Python-2 style `super(MAF, self)`, so the class object
    # itself must stay bound to the name `MAF`; we wrap __init__ instead.
    orig_maf_init = models_mod.MAF.__init__

    def maf_init_hook(self, *args, **kwargs):
        orig_maf_init(self, *args, **kwargs)
        logger.model = self
        try:
            summary, total, trainable = describe_model(self)
            with open(os.path.join(logger.run_dir, "model_summary.txt"), "w", encoding="utf-8") as fh:
                fh.write(summary + "\n")
            logger.log(f"MAF built: {total:,} params ({trainable:,} trainable, {total - trainable:,} frozen)")
            logger.event("model_built", total_params=total, trainable_params=trainable,
                         num_classes=self.fc[-1].out_features)
        except Exception as exc:
            logger.log(f"model summary failed: {exc}", "WARN")

    models_mod.MAF.__init__ = maf_init_hook

    # -- 7.3 optimizer capture (via the scheduler factory) -----------------
    orig_sched = models_mod.get_linear_schedule_with_warmup

    def sched_hook(optimizer, *args, **kwargs):
        logger.optimizer = optimizer
        groups = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
        logger.event("optimizer", type=type(optimizer).__name__, param_groups=groups,
                     scheduler_kwargs={k: v for k, v in kwargs.items()})
        logger.log(f"Optimizer {type(optimizer).__name__} lr={groups[0].get('lr')} "
                   f"weight_decay={groups[0].get('weight_decay')}; "
                   f"scheduler num_training_steps={kwargs.get('num_training_steps', args[-1] if args else None)}")
        return orig_sched(optimizer, *args, **kwargs)

    models_mod.get_linear_schedule_with_warmup = sched_hook

    # -- 7.4 per-batch training metrics ------------------------------------
    orig_calc_acc = models_mod.calculate_accuracy

    def calc_acc_hook(predictions, targets):
        acc = orig_calc_acc(predictions, targets)
        try:
            with torch.no_grad():
                # identical to the criterion the training loop uses
                loss = F.cross_entropy(predictions.detach().float(), targets).item()
            acc_val = float(acc.item())
            now = time.time()
            dt = max(now - logger._last_batch_t, 1e-9)
            logger._last_batch_t = now
            logger.global_step += 1
            logger.step_in_epoch += 1
            logger.epoch_loss_sum += loss
            logger.epoch_acc_sum += acc_val
            lr = logger.optimizer.param_groups[0]["lr"] if logger.optimizer else None
            alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None
            n = targets.shape[0]
            row = [round(now - logger.t0, 2), round(now - logger.t0, 2), logger.epoch,
                   logger.global_step, logger.step_in_epoch, round(loss, 6), round(acc_val, 6),
                   round(logger.epoch_loss_sum / logger.step_in_epoch, 6),
                   round(logger.epoch_acc_sum / logger.step_in_epoch, 6),
                   lr, round(n / dt, 2), None if alloc is None else round(alloc, 1)]
            logger._row(logger.batch_csv, row)
            if logger.global_step % cfg["log_every"] == 0:
                logger.event("batch", epoch=logger.epoch, step=logger.global_step,
                             loss=round(loss, 6), acc=round(acc_val, 6), lr=lr,
                             samples_per_s=round(n / dt, 2))
        except Exception as exc:
            logger.log(f"batch hook failed: {exc}", "WARN")
        return acc

    models_mod.calculate_accuracy = calc_acc_hook

    # -- 7.5 per-epoch validation metrics ----------------------------------
    orig_acc_score = models_mod.accuracy_score

    def acc_score_hook(y_true, y_pred, *args, **kwargs):
        value = orig_acc_score(y_true, y_pred, *args, **kwargs)
        try:
            epoch = logger.epoch
            now = time.time()
            train_time = max(logger._last_batch_t - logger._epoch_t0, 0.0)
            val_time = max(now - logger._last_batch_t, 0.0)
            steps = max(logger.step_in_epoch, 1)
            report = compute_report(y_true, y_pred)
            report["epoch"] = epoch
            report["split"] = "validation"
            with open(os.path.join(logger.val_dir, f"epoch_{epoch:03d}.json"), "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)

            is_best = report["accuracy"] > logger.best_val_acc
            if is_best:
                logger.best_val_acc = report["accuracy"]
                logger.best_epoch = epoch
            peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
            lr = logger.optimizer.param_groups[0]["lr"] if logger.optimizer else None
            row = {
                "epoch": epoch,
                "train_loss": round(logger.epoch_loss_sum / steps, 6),
                "train_acc": round(logger.epoch_acc_sum / steps, 6),
                "val_acc": round(report["accuracy"], 6),
                "val_macro_f1": round(report["macro_f1"], 6),
                "val_weighted_f1": round(report["weighted_f1"], 6),
                "val_macro_precision": round(report["macro_precision"], 6),
                "val_macro_recall": round(report["macro_recall"], 6),
                "val_mmae": None if report["mmae"] is None else round(report["mmae"], 6),
                "lr_end": lr,
                "train_time_s": round(train_time, 2),
                "val_time_s": round(val_time, 2),
                "epoch_time_s": round(now - logger._epoch_t0, 2),
                "gpu_peak_mb": None if peak is None else round(peak, 1),
                "is_best": is_best,
            }
            logger._row(logger.epoch_csv, list(row.values()))
            logger.epoch_rows.append(row)
            logger.event("epoch", **row)
            logger.log(
                f"epoch {epoch}: train_loss={row['train_loss']:.4f} train_acc={row['train_acc']*100:.2f}% "
                f"val_acc={row['val_acc']*100:.2f}% val_macroF1={row['val_macro_f1']:.4f} "
                f"({row['epoch_time_s']:.1f}s){' *best*' if is_best else ''}"
            )

            # reset for the next epoch
            logger.epoch += 1
            logger.step_in_epoch = 0
            logger.epoch_loss_sum = 0.0
            logger.epoch_acc_sum = 0.0
            logger._epoch_t0 = time.time()
            logger._last_batch_t = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            logger.log(f"epoch hook failed: {exc}\n{traceback.format_exc()}", "WARN")
        return value

    models_mod.accuracy_score = acc_score_hook

    # -- 7.6 training / evaluation wrappers --------------------------------
    orig_train = models_mod.train

    def train_hook(*args, **kwargs):
        logger.log("=== TRAINING START ===")
        logger._epoch_t0 = time.time()
        logger._last_batch_t = time.time()
        t0 = time.time()
        try:
            model = orig_train(*args, **kwargs)
        finally:
            logger.timings["training_s"] = round(time.time() - t0, 2)
            logger.log(f"=== TRAINING END ({logger.timings['training_s']:.1f}s) ===")
        return model

    models_mod.train = train_hook

    orig_eval = models_mod.evaluation

    def eval_hook(path, model, test_loader, *args, **kwargs):
        logger.log("=== TEST EVALUATION START ===")
        logits = []
        handle = model.register_forward_hook(
            lambda _m, _i, out: logits.append(out.detach().float().cpu()))
        t0 = time.time()
        try:
            labels, preds = orig_eval(path, model, test_loader, *args, **kwargs)
        finally:
            handle.remove()
            logger.timings["test_eval_s"] = round(time.time() - t0, 2)
        if logits:
            probs = torch.softmax(torch.cat(logits, dim=0), dim=1).numpy()
            logger._test_probs = probs
        ckpt = os.path.join(path, "maf_model.pth")
        if os.path.exists(ckpt):
            logger.event("checkpoint", path=ckpt, size_mb=round(os.path.getsize(ckpt) / 1e6, 1))
            logger.log(f"Checkpoint used: {ckpt} ({os.path.getsize(ckpt) / 1e6:.1f} MB)")
        logger.log(f"=== TEST EVALUATION END ({logger.timings['test_eval_s']:.1f}s) ===")
        return labels, preds

    models_mod.evaluation = eval_hook

    # -- 7.7 final artefacts ------------------------------------------------
    orig_pipline = evaluation_mod.pipline

    def pipline_hook(*args, **kwargs):
        actual, pred = orig_pipline(*args, **kwargs)
        try:
            write_final_artifacts(logger, actual, pred, cfg)
        except Exception as exc:
            logger.log(f"final artefact writing failed: {exc}\n{traceback.format_exc()}", "ERROR")
        return actual, pred

    evaluation_mod.pipline = pipline_hook
    logger.log("Instrumentation hooks installed "
               "(dataset.load_dataset, models.MAF, models.calculate_accuracy, "
               "models.accuracy_score, models.train, models.evaluation, evaluation.pipline)")


# ===========================================================================
# 8.  Final artefacts
# ===========================================================================
def write_final_artifacts(logger, actual, pred, cfg):
    import numpy as np
    import pandas as pd

    y_true, y_pred = _as_int_array(actual), _as_int_array(pred)
    metrics = compute_report(y_true, y_pred)
    metrics["split"] = "test"
    metrics["n_samples"] = int(len(y_true))
    logger.dump("classification_report.json", metrics)

    from sklearn.metrics import classification_report as _cr
    text = _cr(y_true, y_pred, labels=CLASS_IDS, target_names=CLASS_SHORT, digits=3, zero_division=0)
    lines = [
        "MAF on MIMOSA -- test-set classification report",
        f"run: {os.path.basename(logger.run_dir)}",
        f"samples: {len(y_true)}",
        "",
        text,
        "",
        f"Accuracy           : {metrics['accuracy']:.4f}",
        f"Weighted Precision : {metrics['weighted_precision']:.4f}",
        f"Weighted Recall    : {metrics['weighted_recall']:.4f}",
        f"Weighted F1        : {metrics['weighted_f1']:.4f}",
        f"Macro F1           : {metrics['macro_f1']:.4f}",
        f"MMAE               : {metrics['mmae']}",
        "",
        "Confusion matrix (rows = true, cols = predicted, order " + ", ".join(CLASS_SHORT) + "):",
    ]
    for name, row in zip(CLASS_SHORT, metrics["confusion_matrix"]):
        lines.append(f"  {name:<5}" + "".join(f"{v:>7}" for v in row))
    with open(os.path.join(logger.run_dir, "classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    save_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(logger.run_dir, "confusion_matrix.png"),
        os.path.join(logger.run_dir, "confusion_matrix.csv"),
        "MAF / MIMOSA -- test confusion matrix",
    )

    # per-sample predictions, aligned with testing_set.csv (test loader is unshuffled)
    try:
        test_df = pd.read_csv(os.path.join(cfg["dataset_dir"], "testing_set.csv")).iloc[: len(y_true)]
        out = pd.DataFrame({
            "index": np.arange(len(y_true)),
            "image_name": test_df["image_name"].values,
            "caption": test_df["Captions"].values,
            "true_label_id": y_true,
            "true_label": [CLASS_FULL[i] for i in y_true],
            "pred_label_id": y_pred,
            "pred_label": [CLASS_FULL[i] for i in y_pred],
            "correct": (y_true == y_pred),
        })
        probs = getattr(logger, "_test_probs", None)
        if probs is not None and len(probs) >= len(y_true):
            probs = probs[: len(y_true)]
            for i, name in enumerate(CLASS_SHORT):
                out[f"prob_{name}"] = probs[:, i]
            out["confidence"] = probs.max(axis=1)
        out.to_csv(os.path.join(logger.run_dir, "test_predictions.csv"), index=False)
        logger.log(f"Wrote test_predictions.csv ({len(out)} rows)")
    except Exception as exc:
        logger.log(f"test_predictions.csv failed: {exc}", "WARN")

    summary = {
        "run_dir": logger.run_dir,
        "config": cfg,
        "total_epochs_logged": len(logger.epoch_rows),
        "best_val_accuracy": logger.best_val_acc if logger.best_val_acc >= 0 else None,
        "best_val_epoch": logger.best_epoch,
        "epochs": logger.epoch_rows,
        "timings_s": logger.timings,
        "test": {k: v for k, v in metrics.items() if k != "per_class"},
        "test_per_class": metrics["per_class"],
    }
    logger.dump("summary.json", summary)
    logger.event("test", **{k: v for k, v in metrics.items() if k != "per_class"})
    logger.log(f"TEST  acc={metrics['accuracy']*100:.2f}%  weightedF1={metrics['weighted_f1']:.4f}  "
               f"macroF1={metrics['macro_f1']:.4f}  MMAE={metrics['mmae']}")


# ===========================================================================
# 9.  Entry point
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="Run the MAF/MIMOSA pipeline with full experiment logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- identical to main.py -------------------------------------------
    p.add_argument('--dataset', dest='dataset_path', type=str, default='Dataset',
                   help='the directory of the dataset folder')
    p.add_argument('--max_len', dest='maximum_length', type=int, default=70,
                   help='the maximum text length')
    p.add_argument('--batch_size', dest='batch', type=int, default=16, help='batch size')
    p.add_argument('--model', dest='model_path', type=str, default=None,
                   help='checkpoint folder; default = <run dir>/checkpoints')
    p.add_argument('--heads', dest='n_heads', type=int, default=16, help='number of attention heads')
    p.add_argument('--n_iter', dest='epochs', type=int, default=5, help='number of epochs')
    p.add_argument('--lrate', dest='lr_rate', type=float, default=5e-5, help='learning rate')
    # --- logging-only additions ------------------------------------------
    p.add_argument('--run_name', type=str, default='maf', help='tag used in the run directory name')
    p.add_argument('--log_dir', type=str, default='Runs', help='root folder for run directories')
    p.add_argument('--seed', type=int, default=42, help='seed for random/numpy/torch')
    p.add_argument('--log_every', type=int, default=10, help='batches between metrics.jsonl batch events')
    p.add_argument('--resource_interval', type=float, default=15.0,
                   help='seconds between GPU/CPU resource samples')
    p.add_argument('--max_batches', type=int, default=0,
                   help='truncate every loader to N batches (0 = full run); use for smoke tests')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(ROOT_DIR, args.log_dir)
    run_dir = os.path.join(log_root, f"{stamp}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    if args.model_path is None:
        ckpt_dir = os.path.join(run_dir, "checkpoints")
    else:
        ckpt_dir = args.model_path if os.path.isabs(args.model_path) \
            else os.path.join(ROOT_DIR, args.model_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = RunLogger(run_dir, resource_interval=args.resource_interval)
    exit_code = 0
    try:
        logger.log(f"Run directory: {run_dir}")
        notes = apply_compat_shims()
        seed_everything(args.seed)
        logger.log(f"Seeded python/numpy/torch with {args.seed}")

        cfg = {
            "dataset_path": args.dataset_path,
            "dataset_dir": os.path.join(ROOT_DIR, args.dataset_path),
            "max_len": args.maximum_length,
            "batch_size": args.batch,
            "checkpoint_dir": ckpt_dir,
            "num_heads": args.n_heads,
            "epochs": args.epochs,
            "learning_rate": args.lr_rate,
            "seed": args.seed,
            "max_batches": args.max_batches,
            "log_every": args.log_every,
            "run_name": args.run_name,
            "run_dir": run_dir,
            "command": " ".join([sys.executable] + sys.argv),
            "compat_shims": notes,
        }
        logger.dump("config.json", cfg)
        logger.event("config", **cfg)

        logger.log("Collecting environment ...")
        logger.dump("environment.json", collect_environment())

        logger.log("Importing pipeline modules (loads CLIP ViT-B/32 onto the device) ...")
        sys.path.insert(0, SCRIPTS_DIR)
        t0 = time.time()
        # re-apply the shim before every import: see apply_compat_shims()
        apply_compat_shims(verbose=False)
        import dataset as dataset_mod
        apply_compat_shims(verbose=False)
        import models as models_mod
        apply_compat_shims(verbose=False)
        import evaluation as evaluation_mod
        import main as main_mod
        logger.timings["module_import_s"] = round(time.time() - t0, 2)
        logger.log(f"Modules imported in {logger.timings['module_import_s']:.1f}s")

        install_hooks(logger, dataset_mod, models_mod, evaluation_mod, cfg)

        pipeline_args = argparse.Namespace(
            dataset_path=args.dataset_path,
            maximum_length=args.maximum_length,
            batch=args.batch,
            model_path=ckpt_dir,          # absolute -> os.path.join keeps it
            n_heads=args.n_heads,
            epochs=args.epochs,
            lr_rate=args.lr_rate,
        )
        logger.log(f"Calling main.main with {vars(pipeline_args)}")
        run_t0 = time.time()
        main_mod.main(pipeline_args)
        logger.timings["total_pipeline_s"] = round(time.time() - run_t0, 2)
        logger.log(f"Pipeline finished in {logger.timings['total_pipeline_s']:.1f}s")

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
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                data["timings_s"] = logger.timings
                data["exit_code"] = exit_code
                with open(summary_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, default=str)
            except Exception:
                pass
        logger.close()
        print(f"\nAll logs and artefacts: {run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
