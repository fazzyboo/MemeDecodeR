"""
Runs MuLAD's full experiment grid and assembles the paper's result tables.

The paper reports three sweeps (Sec. 6):

    Table 3   9 textual baselines    3 text models  x  3 embeddings
    Table 4   3 visual baselines     3 backbones
    Table 5  27 multimodal models    3 text models  x  3 embeddings  x  3 backbones

39 runs total. Each is cheap because the convolutional backbones are frozen and their
feature maps are cached by mulad_features.py, so only the text branch and the fusion head
are ever trained.

Model selection is on the VALIDATION split; the test split is scored once per run and
never used to choose anything. That matters here: with 39 configurations, picking the
winner on test would overfit the test set by construction.

Usage:
    python mulad_grid.py                       # everything
    python mulad_grid.py --part multimodal     # one sweep
    python mulad_grid.py --embeddings keras    # skip the gensim-trained embeddings
"""
import _paths  # noqa: F401

import argparse
import json
import os
import time

import pandas as pd

import mulad_main
import mulad_models as MM
from mulad_features import BACKBONES

EMBEDDINGS = ("keras", "selftrained", "fasttext")
# The paper's Keras embedding is 64-d; its FastText and GloVe rows are both 300-d.
EMB_DIM = {"keras": 64, "selftrained": 300, "fasttext": 300}


def configs(part, text_models, backbones, embeddings):
    if part in ("all", "text"):
        for emb in embeddings:
            for tm in text_models:
                yield {"modality": "text", "text_model": tm, "embedding": emb}
    if part in ("all", "visual"):
        for bb in backbones:
            yield {"modality": "visual", "backbone": bb}
    if part in ("all", "multimodal"):
        for emb in embeddings:
            for bb in backbones:
                for tm in text_models:
                    yield {"modality": "multimodal", "text_model": tm,
                           "backbone": bb, "embedding": emb}


def main(args):
    base = mulad_main.build_parser().parse_args([])
    for key, value in vars(args).items():
        if key in vars(base) and value is not None:
            setattr(base, key, value)

    plan = list(configs(args.part, args.text_models, args.backbones, args.embeddings))
    print("{} configuration(s) to run\n".format(len(plan)))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(os.path.abspath(os.path.join(script_dir, "..")), "Outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    rows, started = [], time.time()
    for i, cfg in enumerate(plan, 1):
        run_args = argparse.Namespace(**vars(base))
        for key, value in cfg.items():
            setattr(run_args, key, value)
        run_args.emb_dim = EMB_DIM.get(cfg.get("embedding", "keras"), base.emb_dim)
        run_args.run_name = None  # let mulad_main derive a stable descriptive name

        name = mulad_main.default_run_name(run_args)
        print("[{:>2}/{}] {}".format(i, len(plan), name), flush=True)

        if args.skip_existing and os.path.isfile(
                os.path.join(outputs_dir, "results_{}.json".format(name))):
            with open(os.path.join(outputs_dir, "results_{}.json".format(name)),
                      encoding="utf-8") as fh:
                rows.append(json.load(fh))
            print("        cached, skipping\n")
            continue

        try:
            result = mulad_main.run(run_args, quiet=True)
        except Exception as exc:  # one bad cell must not lose the other 38
            print("        FAILED: {}: {}\n".format(type(exc).__name__, exc))
            continue

        rows.append(result)
        print("        acc {:.3f}  WF1 {:.3f}  macroF1 {:.3f}  MMAE {:.3f}{}  ({:.0f}s)\n".format(
            result["accuracy"], result["weighted_f1"], result["macro_f1"], result["mmae"],
            "  [DEGENERATE]" if result["degenerate"] else "", result["runtime_s"]), flush=True)

    if not rows:
        raise SystemExit("No runs completed.")

    table = pd.DataFrame([{
        "run": r["run_name"],
        "modality": r["modality"],
        "text": r.get("text_model") or "-",
        "visual": r.get("backbone") or "-",
        "embedding": r.get("embedding") or "-",
        "A": round(r["accuracy"], 3),
        "P": round(r["weighted_precision"], 3),
        "R": round(r["weighted_recall"], 3),
        "WF": round(r["weighted_f1"], 3),
        "macroF1": round(r["macro_f1"], 3),
        "MMAE": round(r["mmae"], 3),
        "degenerate": r["degenerate"],
    } for r in rows]).sort_values("WF", ascending=False)

    csv_path = os.path.join(outputs_dir, "mulad_grid.csv")
    table.to_csv(csv_path, index=False)

    print("=" * 100)
    print("MuLAD grid - {} runs in {:.1f} min".format(len(rows), (time.time() - started) / 60))
    print("=" * 100)
    for modality in ("text", "visual", "multimodal"):
        part = table[table["modality"] == modality]
        if part.empty:
            continue
        print("\n--- {} ---".format(modality.upper()))
        print(part.drop(columns=["modality", "run"]).to_string(index=False))

    degenerate = int(table["degenerate"].sum())
    if degenerate:
        print("\n{} of {} runs predicted a single class for every test meme. In the paper's "
              "Table 5 that failure accounts for roughly 18 of 27 multimodal rows, reported "
              "there as results.".format(degenerate, len(table)))

    print("\nSaved table ->", csv_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the full MuLAD experiment grid")
    p.add_argument("--part", type=str, default="all",
                   choices=["all", "text", "visual", "multimodal"])
    p.add_argument("--text_models", nargs="+", default=list(MM.TEXT_MODELS))
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--embeddings", nargs="+", default=list(EMBEDDINGS))
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--features", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--monitor", type=str, default=None, choices=["accuracy", "macro_f1"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--vectors", type=str, default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true",
                   help="reuse Outputs/results_*.json from an earlier grid run")
    main(p.parse_args())
