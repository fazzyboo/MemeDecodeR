"""
Stage 2 - Build MIMOSA-format splits from our dataset.

Joins the OCR captions from Stage 1 with our label files, maps our label vocabulary onto
the paper's, and writes the three CSVs the MAF codebase expects:

    Dataset/training_set.csv
    Dataset/validation_set.csv
    Dataset/testing_set.csv

each with the exact columns the original dataset.py reads: image_name, Captions, Label.

SPLIT POLICY
------------
The paper splits MIMOSA 70 / 15 / 15 into train / validation / test (Sec. 3.4). Our
dataset arrives pre-split into a Train folder (2,903 memes) and a held-out Test folder
(399 labelled memes), and that held-out test set is kept intact so the evaluation stays
honest. The paper's 70:15 train:validation ratio is then applied *within* our Train
folder - a stratified 82.35 / 17.65 split, which is 70:15 renormalised - so the
train-to-validation proportion matches the paper exactly.

LABEL POLICY
------------
Our data has four classes; MIMOSA has five. The paper's "others" (Oth) category has no
counterpart in our data, so the task is 4-way here. The integer encoding of the four
shared classes is left exactly as the paper assigns it (NoAg=0, GAg=1, PAg=2, RAg=3) so
that MMAE - which treats the labels as an ordinal scale - stays comparable.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --stats     # also reproduce the paper's Tables 1-3
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))

# Canonical label strings, identical to the MIMOSA CSVs. dataset.py maps these to ints.
NOAG = "non-aggressive"
GAG = "gendered aggression"
PAG = "political aggression"
RAG = "religious aggression"

# Our two label files use different vocabularies for the same four classes.
LABEL_MAP = {
    # Train/Train.csv  ("Target" column)
    "neutral": NOAG,
    "genders": GAG,
    "politics": PAG,
    "religion": RAG,
    # Test/test.csv  ("label" column)
    "nonaggressive": NOAG,
    "gendered": GAG,
    "political": PAG,
    "religious": RAG,
}

# Paper ordering, minus the absent "others" class.
LABEL_ORDER = [NOAG, GAG, PAG, RAG]
SHORT = {NOAG: "NoAg", GAG: "GAg", PAG: "PAg", RAG: "RAg"}


def normalise(series):
    return series.astype(str).str.strip().str.lower().map(LABEL_MAP)


def load_source_labels(project_root):
    """Read our two label files and return one frame of (image_name, Label, origin)."""
    train_csv = os.path.join(project_root, "Train", "Train.csv")
    test_csv = os.path.join(project_root, "Test", "test.csv")

    tr = pd.read_csv(train_csv)
    tr = tr.rename(columns={"Image_name": "image_name", "Target": "raw_label"})
    tr = tr[["image_name", "raw_label"]].copy()
    tr["origin"] = "train_pool"

    te = pd.read_csv(test_csv)
    te = te.rename(columns={"filename": "image_name", "label": "raw_label"})
    te = te[["image_name", "raw_label"]].copy()
    te["origin"] = "test"

    for name, frame in (("Train.csv", tr), ("test.csv", te)):
        frame["Label"] = normalise(frame["raw_label"])
        unknown = frame.loc[frame["Label"].isna(), "raw_label"].unique()
        if len(unknown):
            raise ValueError("Unmapped label(s) in {}: {}".format(name, list(unknown)))

    return pd.concat([tr, te], ignore_index=True)[["image_name", "Label", "origin"]]


def main(args):
    project_root = args.project_root or PROJECT_ROOT
    dataset_dir = os.path.join(ROOT_DIR, "Dataset")
    img_dir = os.path.join(dataset_dir, "Img")
    # Prefer the denoised captions when Stage 1b has been run. The paper trained on
    # hand-corrected captions, so the denoised file is the closer analogue; captions_raw
    # stays available via --captions for a strict raw-OCR comparison.
    if args.captions:
        captions_csv = args.captions
    else:
        clean_path = os.path.join(dataset_dir, "captions_clean.csv")
        raw_path = os.path.join(dataset_dir, "captions_raw.csv")
        captions_csv = clean_path if os.path.isfile(clean_path) else raw_path

    print("=" * 74)
    print("Stage 2 - building MIMOSA-format splits")
    print("=" * 74)

    labels = load_source_labels(project_root)
    print("Label rows read          :", len(labels))

    dup = labels["image_name"].duplicated().sum()
    if dup:
        print("Dropping duplicate rows  :", dup)
        labels = labels.drop_duplicates(subset="image_name", keep="first")

    # Keep only rows whose image actually exists in the unified Img folder.
    on_disk = set(os.listdir(img_dir))
    missing = sorted(set(labels["image_name"]) - on_disk)
    if missing:
        print("Labelled but no image    :", len(missing), missing[:5])
        labels = labels[labels["image_name"].isin(on_disk)]
    unlabelled = sorted(on_disk - set(labels["image_name"]))
    if unlabelled:
        print("Image but no label (drop):", len(unlabelled), unlabelled[:5])

    captions = pd.read_csv(captions_csv, keep_default_na=False)
    captions["Captions"] = captions["Captions"].astype(str).str.strip()
    print("Captions source          :", os.path.basename(captions_csv))
    print("Captions read            :", len(captions))

    data = labels.merge(captions, on="image_name", how="left")
    data["Captions"] = data["Captions"].fillna("")
    no_caption = int((data["Captions"].str.len() == 0).sum())
    print("Empty captions (OCR read nothing):", no_caption)
    if args.drop_empty_captions:
        data = data[data["Captions"].str.len() > 0]
        print("  -> dropped (--drop_empty_captions)")

    # --------------------------------------------------------------------------------
    # Splits
    # --------------------------------------------------------------------------------
    test_df = data[data["origin"] == "test"].copy()
    pool_df = data[data["origin"] == "train_pool"].copy()

    # 70:15 from the paper, renormalised over the train pool -> validation is 15/85.
    val_fraction = 15.0 / 85.0
    train_df, valid_df = train_test_split(
        pool_df,
        test_size=val_fraction,
        random_state=args.seed,
        stratify=pool_df["Label"],
        shuffle=True,
    )

    cols = ["image_name", "Captions", "Label"]
    out = {
        "training_set.csv": train_df,
        "validation_set.csv": valid_df,
        "testing_set.csv": test_df,
    }
    for fname, frame in out.items():
        frame = frame[cols].reset_index(drop=True)
        path = os.path.join(dataset_dir, fname)
        frame.to_csv(path, index=False, encoding="utf-8")
        print("Wrote {:<20} {:>5} rows -> {}".format(fname, len(frame), path))

    # --------------------------------------------------------------------------------
    # Class distribution (our analogue of the paper's Table 1)
    # --------------------------------------------------------------------------------
    print()
    print("Class distribution (cf. paper Table 1)")
    print("-" * 74)
    dist = pd.DataFrame(
        {
            "Train": train_df["Label"].value_counts(),
            "Validation": valid_df["Label"].value_counts(),
            "Test": test_df["Label"].value_counts(),
        }
    ).reindex(LABEL_ORDER).fillna(0).astype(int)
    dist.index = [SHORT[i] for i in dist.index]
    dist.loc["Total"] = dist.sum()
    print(dist.to_string())

    total = int(dist.loc["Total"].sum())
    print()
    print(
        "Overall split: train {:.1f}% / validation {:.1f}% / test {:.1f}%  (paper: 70 / 15 / 15)".format(
            100 * len(train_df) / total, 100 * len(valid_df) / total, 100 * len(test_df) / total
        )
    )
    print(
        "Train:validation ratio {:.1f}:{:.1f}  (paper: 70:15)".format(
            100 * len(train_df) / (len(train_df) + len(valid_df)) * 0.85,
            100 * len(valid_df) / (len(train_df) + len(valid_df)) * 0.85,
        )
    )

    if args.stats:
        caption_stats(train_df)
        jaccard_table(train_df)


def caption_stats(train_df):
    """Reproduce the paper's Table 2: total / unique / max / average words per caption."""
    print()
    print("Training-set caption summary (cf. paper Table 2)")
    print("-" * 74)
    rows = []
    for label in LABEL_ORDER:
        caps = train_df.loc[train_df["Label"] == label, "Captions"]
        tokens = [w for c in caps for w in str(c).split()]
        per_caption = [len(str(c).split()) for c in caps]
        rows.append(
            {
                "Class": SHORT[label],
                "Ttw": len(tokens),
                "Tuw": len(set(tokens)),
                "Tmw": max(per_caption) if per_caption else 0,
                "Taw": int(round(np.mean(per_caption))) if per_caption else 0,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("Ttw=total words, Tuw=unique words, Tmw=max words/caption, Taw=avg words/caption")


def jaccard_table(train_df):
    """Reproduce the paper's Table 3: pairwise Jaccard similarity between class vocabularies."""
    print()
    print("Jaccard similarity between class captions (cf. paper Table 3)")
    print("-" * 74)
    vocab = {
        label: set(w for c in train_df.loc[train_df["Label"] == label, "Captions"] for w in str(c).split())
        for label in LABEL_ORDER
    }
    names = [SHORT[l] for l in LABEL_ORDER]
    table = pd.DataFrame("-", index=names, columns=names)
    for i, a in enumerate(LABEL_ORDER):
        for j, b in enumerate(LABEL_ORDER):
            if j <= i:
                continue
            union = vocab[a] | vocab[b]
            score = len(vocab[a] & vocab[b]) / len(union) if union else 0.0
            table.iloc[i, j] = "{:.2f}".format(score)
    print(table.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build MIMOSA-format splits from our dataset")
    p.add_argument("--project_root", type=str, default=None, help="folder holding Train/ and Test/")
    p.add_argument("--captions", type=str, default=None, help="captions CSV from Stage 1")
    p.add_argument("--seed", type=int, default=42, help="split seed (paper-style fixed seed)")
    p.add_argument("--stats", action="store_true", help="also print the paper Tables 2 and 3")
    p.add_argument(
        "--drop_empty_captions",
        action="store_true",
        help="discard memes whose OCR returned nothing instead of keeping an empty caption",
    )
    main(p.parse_args())
