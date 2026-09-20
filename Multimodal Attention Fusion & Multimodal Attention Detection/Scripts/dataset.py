"""
Data loading - port of the original MIMOSA Scripts/dataset.py.

Changes from the original, all of them forced by our dataset or by running without a GPU:

  1. The class count is resolved from the split CSVs instead of being hardcoded, so the
     same code runs the five-way MIMOSA corpus and any four-way subset of it. Integer
     codes keep the paper's ordering (NoAg=0..Oth=4) so MMAE stays ordinal. See labels.py.
  2. Captions read from CSV can be empty (the OCR found no text). Those arrive as NaN
     from pandas and would crash the tokenizer, so they are coerced to "".
  3. Image.open(...).convert("RGB") - the original omits the conversion, which breaks
     Normalize() on any grayscale or palettised meme.
  4. The original loads a CLIP model here, half()s it, and never uses it. That dead load
     is removed; models.py owns the real CLIP encoder. No effect on results.
  5. The device string falls back to CPU instead of hardcoding "cuda:0".
"""
import os
import time

import _paths  # noqa: F401  - must precede the transformers import

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

import labels as L

# Set the device to GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

start_time = time.time()

# Label handling lives in labels.py so MAF and MuLAD encode classes identically.
# NUM_CLASSES and TARGET_NAMES below are defaults for the full five-way MIMOSA corpus;
# load_dataset() re-resolves both from the CSVs actually loaded and rewrites them, so a
# four-way subset (no "others" class) still works without edits. Read them AFTER calling
# load_dataset(), never at import time.
LABEL_MAP = L.LABEL_MAP
NUM_CLASSES = len(L.ALL_TARGET_NAMES)
TARGET_NAMES = list(L.ALL_TARGET_NAMES)

_encode = L.encode


def load_dataset(files_path, memes_path, max_len, batch_size, test_batch_size=4, subset=0,
                 suffix=""):

    print("Fetching Dataset... ")
    print("-----------------------------")
    print("Maximum Text Length: ", max_len)
    print("Batch Size: ", batch_size)

    # join the paths
    # `suffix` selects a caption variant that shares the images, labels and split
    # membership - e.g. "_ocr" reads training_set_ocr.csv. That makes the OCR-versus-gold
    # comparison a matched pair: only the Captions column differs.
    train_file = os.path.join(files_path, "training_set{}.csv".format(suffix))
    valid_file = os.path.join(files_path, "validation_set{}.csv".format(suffix))
    test_file = os.path.join(files_path, "testing_set{}.csv".format(suffix))
    for path in (train_file, valid_file, test_file):
        if not os.path.isfile(path):
            raise SystemExit("Missing {} - check --captions_suffix".format(path))
    if suffix:
        print("Caption variant:", suffix)

    # dataset
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)

    # encode labels
    train_data = _encode(train_data, "training_set.csv")
    valid_data = _encode(valid_data, "validation_set.csv")
    test_data = _encode(test_data, "testing_set.csv")

    # Resolve the class count from the data rather than assuming it. The full MIMOSA
    # corpus is five-way; a subset without "others" is four-way. Both are handled.
    global NUM_CLASSES, TARGET_NAMES
    NUM_CLASSES, TARGET_NAMES = L.resolve_classes([train_data, valid_data, test_data])
    print("Classes:", NUM_CLASSES, TARGET_NAMES)

    if subset:
        # Smoke-test mode: take a stratified-ish head of each split so every class is
        # still represented, then reset the index that MIMOSA.__getitem__ relies on.
        def _take(frame):
            per_class = max(1, subset // NUM_CLASSES)
            picked = frame.groupby("Label", group_keys=False).head(per_class)
            return picked.reset_index(drop=True)

        train_data, valid_data, test_data = _take(train_data), _take(valid_data), _take(test_data)
        print("SUBSET MODE: {} rows per split (smoke test, not a real result)".format(len(train_data)))

    print("Training Data:", len(train_data))
    print("Valid Data:", len(valid_data))
    print("Test Data:", len(test_data))

    class MIMOSA(Dataset):
        def __init__(self, dataframe, tokenizer, data_dir, max_seq_length, transform=None):
            self.data = dataframe
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, "image_name"])
            image = Image.open(img_name).convert("RGB")
            caption = self.data.loc[idx, "Captions"]
            label = int(self.data.loc[idx, "Label"])

            if self.transform:
                image = self.transform(image)

            # Tokenize the caption using BERT tokenizer
            inputs = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
            )

            return {
                "image": image,
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "label": label,
            }

    # Data preprocessing and augmentation
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Initialize BERT
    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

    # Create data loaders
    train_dataset = MIMOSA(
        dataframe=train_data,
        tokenizer=tokenizer,
        data_dir=memes_path,
        max_seq_length=max_len,
        transform=data_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MIMOSA(
        dataframe=valid_data,
        tokenizer=tokenizer,
        data_dir=memes_path,
        max_seq_length=max_len,
        transform=data_transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = MIMOSA(
        dataframe=test_data,
        tokenizer=tokenizer,
        data_dir=memes_path,
        max_seq_length=max_len,
        transform=data_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    print("Fetched.")
    end_time = time.time()

    print("Time required for preparing the Data loaders: {:.2f}s".format(end_time - start_time))
    print("--------------------------------")

    return train_loader, val_loader, test_loader
