"""
Data assembly for MuLAD: padded token sequences + cached frozen visual features.

MuLAD needs a different input pipeline from MAF - word indices instead of BERT subword
ids, and a precomputed conv feature map instead of a raw image tensor - but it reads the
same three split CSVs, so both frameworks see exactly the same memes in the same splits.

The vocabulary is built from the TRAINING split only. Fitting a Keras tokenizer on the
full corpus, as is easy to do by accident, leaks test vocabulary into training.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import labels as L
import mulad_text as T

SPLITS = ("training_set", "validation_set", "testing_set")

# Training word vectors with gensim takes ~30s and the experiment grid reuses the same
# embedding across many configurations, so the matrix is memoised on everything that
# changes it. Cleared by restarting the process.
_EMB_CACHE = {}


class MuLADSet(Dataset):
    def __init__(self, sequences, feature_rows, targets, features):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.feature_rows = feature_rows
        self.features = features  # np.memmap or None for text-only runs

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.features is None:
            visual = torch.zeros(1)
        else:
            row = self.features[self.feature_rows[idx]]
            visual = torch.from_numpy(np.asarray(row, dtype=np.float32))
        return self.sequences[idx], visual, self.targets[idx]


def load(dataset_dir, features_dir=None, backbone=None, max_len=130, num_words=8000,
         batch_size=32, embedding="keras", emb_dim=64, vectors_path=None, subset=0,
         seed=42):
    """Return (loaders, meta) where loaders is train/valid/test and meta carries the
    vocabulary, embedding matrix, class names and cached-feature shape."""
    frames = {}
    for split in SPLITS:
        path = os.path.join(dataset_dir, "{}.csv".format(split))
        if not os.path.isfile(path):
            raise SystemExit("Missing {}. Point --dataset at the folder holding the three "
                             "split CSVs and Img/.".format(path))
        frames[split] = L.encode(pd.read_csv(path), split)

    num_classes, target_names = L.resolve_classes(list(frames.values()))

    if subset:
        per_class = max(1, subset // num_classes)
        frames = {k: v.groupby("Label", group_keys=False).head(per_class).reset_index(drop=True)
                  for k, v in frames.items()}
        print("SUBSET MODE: {} rows per split (smoke test, not a real result)"
              .format(len(frames["training_set"])))

    train = frames["training_set"]
    vocab = T.build_vocab(train["Captions"], num_words=num_words)

    key = (embedding, emb_dim, num_words, len(train), vectors_path)
    if key in _EMB_CACHE:
        matrix, trainable = _EMB_CACHE[key]
    else:
        matrix, trainable = T.build_embedding_matrix(
            embedding, vocab, emb_dim, corpus=list(train["Captions"]), vectors_path=vectors_path
        )
        _EMB_CACHE[key] = (matrix, trainable)

    features, feature_shape = None, None
    if backbone:
        index_path = os.path.join(features_dir, "feature_index.json")
        feat_path = os.path.join(features_dir, "features_{}.npy".format(backbone))
        for path in (index_path, feat_path):
            if not os.path.isfile(path):
                raise SystemExit("Missing {}. Run: python mulad_features.py --models {}"
                                 .format(path, backbone))
        with open(index_path, encoding="utf-8") as fh:
            order = json.load(fh)["image_names"]
        row_of = {name: i for i, name in enumerate(order)}
        # mmap keeps the ~1 GB ResNet50 cache off the heap; batches page in on demand.
        features = np.load(feat_path, mmap_mode="r")
        feature_shape = tuple(features.shape[1:])

    loaders = {}
    for split, frame in frames.items():
        sequences = T.texts_to_sequences(frame["Captions"], vocab, max_len)
        rows = None
        if features is not None:
            unknown = [n for n in frame["image_name"] if n not in row_of]
            if unknown:
                raise SystemExit("{} meme(s) in {} have no cached feature, e.g. {}. "
                                 "Rebuild with --overwrite.".format(len(unknown), split, unknown[:3]))
            rows = np.array([row_of[n] for n in frame["image_name"]], dtype=np.int64)
        dataset = MuLADSet(sequences, rows, frame["Label"].to_numpy(), features)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "training_set"),
            num_workers=0, drop_last=False,
        )

    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "emb_matrix": matrix,
        "emb_trainable": trainable,
        "num_classes": num_classes,
        "target_names": target_names,
        "feature_shape": feature_shape,
        "frames": frames,
    }
    print("Classes     : {}  {}".format(num_classes, target_names))
    print("Vocabulary  : {} (cap {}, built on the training split only)".format(len(vocab), num_words))
    print("Split sizes : {}".format({k: len(v) for k, v in frames.items()}))
    if feature_shape:
        print("Visual cache: {} {}".format(backbone, feature_shape))
    return loaders, meta
