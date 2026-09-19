"""Pre-flight check for Stage 2: does our label vocabulary map cleanly onto the paper's?

Read-only - exercises prepare_dataset.load_source_labels without writing any split files.
"""
import os

import pandas as pd

from prepare_dataset import LABEL_ORDER, PROJECT_ROOT, SHORT, load_source_labels

labels = load_source_labels(PROJECT_ROOT)
print("total labelled rows:", len(labels))
print()

pivot = (
    labels.assign(short=labels["Label"].map(SHORT))
    .pivot_table(index="short", columns="origin", aggfunc="size", fill_value=0)
    .reindex([SHORT[l] for l in LABEL_ORDER])
)
print(pivot.to_string())
print()

# Every label must have mapped; load_source_labels raises otherwise, so reaching here is
# already the pass condition for mapping. Check image coverage too.
img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dataset", "Img")
on_disk = set(os.listdir(img_dir))
named = set(labels["image_name"])
print("images on disk        :", len(on_disk))
print("images named in labels:", len(named))
print("labelled, missing file:", len(named - on_disk), sorted(named - on_disk)[:5])
print("on disk, unlabelled   :", len(on_disk - named), sorted(on_disk - named)[:5])
print("duplicate names       :", int(labels["image_name"].duplicated().sum()))
print()
print("LABEL CHECK PASSED - all label strings mapped to the paper's four canonical classes")
