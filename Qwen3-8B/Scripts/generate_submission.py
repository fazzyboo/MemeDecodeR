"""
Stage 4 - Generate a competition submission.csv from a trained MAF checkpoint.

Produces predictions ONLY - Image_name, Target - in exactly the format Kaggle's
sample_submission.csv uses. This script has no access to ground-truth labels at any
point: it reads only image_name + Captions for the images sample_submission.csv lists,
runs the model's forward pass, and writes out what the model predicted. There is nothing
to compare against here - scoring happens on Kaggle's side, against labels this script
never sees.

LABEL MAPPING
-------------
Training uses the paper's canonical label strings and order (dataset.TARGET_NAMES:
NoAg=0, GAg=1, PAg=2, RAg=3). The competition's sample_submission.csv instead expects
the ORIGINAL vocabulary from Train/Train.csv's Target column: Neutral, Genders, Politics,
Religion. This script maps model output index -> paper short code -> submission string,
so the mapping stays tied to dataset.py's canonical order instead of being duplicated.
Getting this wrong (e.g. submitting "non-aggressive" instead of "Neutral") means Kaggle
scores every row as wrong even if the model's predictions are good.

Usage (after training on Colab/Kaggle and downloading that run's maf_model_<run_name>.pth
back here). If the checkpoint is gone, predictions_to_submission.py builds the same file
from the run's predictions CSV:
    python generate_submission.py \
        --checkpoint ../Saved_Models/maf_model_maf_full.pth \
        --sample_submission ../../sample_submission.csv \
        --out ../Outputs/submission.csv
"""
import _paths  # noqa: F401  - must precede transformers / clip imports

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

import dataset as d
import models as m
from submission_format import SHORT_TO_SUBMISSION, validate_submission

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))

# dataset.TARGET_NAMES = ["NoAg", "GAg", "PAg", "RAg"] is index-aligned with the model's
# output classes; submission_format.SHORT_TO_SUBMISSION translates each short code to the
# string sample_submission.csv expects (Train/Train.csv's original vocabulary).
SUBMISSION_LABELS = [SHORT_TO_SUBMISSION[name] for name in d.TARGET_NAMES]


class InferenceDataset(Dataset):
    """Like dataset.MIMOSA, but carries no label field - structurally cannot see truth."""

    def __init__(self, frame, tokenizer, img_dir, max_len, transform):
        self.data = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.img_dir = img_dir
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        image = Image.open(os.path.join(self.img_dir, row["image_name"])).convert("RGB")
        image = self.transform(image)
        inputs = self.tokenizer(
            str(row["Captions"]),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        return {
            "image": image,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "image_name": row["image_name"],
        }


def resolve_captions_path(dataset_dir, explicit):
    if explicit:
        return explicit
    clean = os.path.join(dataset_dir, "captions_clean.csv")
    raw = os.path.join(dataset_dir, "captions_raw.csv")
    return clean if os.path.isfile(clean) else raw


def main(args):
    device = m.device
    print("Device:", device)

    sample = pd.read_csv(args.sample_submission)
    id_col, target_col = sample.columns[0], sample.columns[1]
    print("sample_submission :", args.sample_submission)
    print("  columns          :", list(sample.columns))
    print("  rows             :", len(sample))

    captions_path = resolve_captions_path(args.dataset, args.captions)
    captions = pd.read_csv(captions_path, keep_default_na=False)
    captions["Captions"] = captions["Captions"].astype(str)
    captions = captions.rename(columns={captions.columns[0]: "image_name"})
    print("captions source    :", os.path.basename(captions_path))

    frame = sample[[id_col]].rename(columns={id_col: "image_name"})
    frame = frame.merge(captions[["image_name", "Captions"]], on="image_name", how="left")
    missing = int(frame["Captions"].isna().sum())
    if missing:
        print("WARNING: {} images had no caption match - using an empty caption".format(missing))
    frame["Captions"] = frame["Captions"].fillna("")

    img_dir = os.path.join(args.dataset, "Img")
    on_disk = set(os.listdir(img_dir))
    absent = sorted(set(frame["image_name"]) - on_disk)
    if absent:
        raise FileNotFoundError(
            "{} images listed in sample_submission are missing from {}: {}".format(
                len(absent), img_dir, absent[:5]
            )
        )

    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    infer_ds = InferenceDataset(frame, tokenizer, img_dir, args.max_len, transform)
    infer_loader = DataLoader(infer_ds, batch_size=args.batch_size, shuffle=False)

    print("Loading CLIP + Bangla-BERT and building MAF ...")
    clip_visual = m.load_clip_visual()
    model = m.MAF(
        clip_visual,
        num_classes=len(SUBMISSION_LABELS),
        num_heads=args.heads,
        seq_len=args.max_len,
        attn_variant=args.attn_variant,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint  :", args.checkpoint)

    names = []
    preds = []
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Predicting", unit="batch"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(images, input_ids, attention_mask)  # no label passed in - ever
            pred_ids = torch.argmax(outputs, dim=1).cpu().tolist()
            names.extend(batch["image_name"])
            preds.extend(pred_ids)

    pred_map = dict(zip(names, preds))
    out_rows = []
    for _, row in sample.iterrows():
        img_name = row[id_col]
        pred_id = pred_map.get(img_name)
        if pred_id is None:
            raise RuntimeError("No prediction produced for {}".format(img_name))
        out_rows.append({id_col: img_name, target_col: SUBMISSION_LABELS[pred_id]})

    out_df = pd.DataFrame(out_rows, columns=[id_col, target_col])

    # Validate against sample_submission.csv before writing anything.
    validate_submission(out_df, sample)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print()
    print("Wrote {} predictions -> {}".format(len(out_df), args.out))
    print("Predicted class distribution:")
    print(out_df[target_col].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate a Kaggle submission.csv from a trained MAF checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="path to a trained maf_model_<run_name>.pth")
    p.add_argument(
        "--sample_submission",
        type=str,
        default=os.path.join(PROJECT_ROOT, "sample_submission.csv"),
        help="path to sample_submission.csv (defines image order/columns to match)",
    )
    p.add_argument("--dataset", type=str, default=os.path.join(ROOT_DIR, "Dataset"), help="folder with Img/ and captions")
    p.add_argument("--captions", type=str, default=None, help="override: captions CSV to use (default: clean, else raw)")
    p.add_argument("--out", type=str, default=os.path.join(ROOT_DIR, "Outputs", "submission.csv"))
    p.add_argument("--max_len", type=int, default=70)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--attn_variant", type=str, default="code", choices=["code", "paper"],
                    help="must match whatever the checkpoint was TRAINED with")
    main(p.parse_args())
