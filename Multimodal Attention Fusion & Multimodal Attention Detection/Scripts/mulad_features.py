"""
Stage V - precompute MuLAD's frozen visual features.

MuLAD uses VGG16 / VGG19 / ResNet50 purely as fixed feature extractors: "the top layers of
these models were discarded" and only the small head on top is trained (Sec. 4.2). Nothing
in the backbone receives a gradient, so every meme's feature map is a constant. Computing
them once and caching turns the paper's 3 x 9 grid from days of repeated convolution into
minutes of training a few dense layers.

Cached tensors are the last conv feature map, before pooling:

    VGG16 / VGG19 -> (512, 7, 7)    ResNet50 -> (2048, 7, 7)

Both of the paper's two poolings are then derivable from that one cache without a second
GPU pass: `flatten` (used for multimodal fusion, Sec. 4.2) and `gap` (global average
pooling, used for the visual baselines, Sec. 5.1).

Usage:
    python mulad_features.py --dataset ../Dataset --models vgg16 vgg19 resnet50
"""
import _paths  # noqa: F401  - redirects TORCH_HOME before torchvision downloads weights

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BACKBONES = {
    "vgg16": (512, 7, 7),
    "vgg19": (512, 7, 7),
    "resnet50": (2048, 7, 7),
}


def build_backbone(name):
    """Return a frozen, eval-mode conv trunk with the classifier head removed."""
    from torchvision import models

    if name == "vgg16":
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    elif name == "vgg19":
        net = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    elif name == "resnet50":
        full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        net = torch.nn.Sequential(*list(full.children())[:-2])  # drop avgpool + fc
    else:
        raise ValueError("Unknown backbone: {}".format(name))

    for param in net.parameters():
        param.requires_grad = False
    return net.eval()


class MemeImages(Dataset):
    """Resize to 224x224 and apply ImageNet normalisation (the torchvision equivalent of
    Keras' per-model `preprocess_input`, which the paper calls out in Sec. 4.1)."""

    def __init__(self, names, img_dir):
        self.names = names
        self.img_dir = img_dir
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.names[idx])
        return self.tf(Image.open(path).convert("RGB")), idx


def collect_image_names(dataset_dir):
    """Every meme referenced by the three split CSVs, de-duplicated, order stable."""
    names, seen = [], set()
    for split in ("training_set", "validation_set", "testing_set"):
        frame = pd.read_csv(os.path.join(dataset_dir, "{}.csv".format(split)))
        for name in frame["image_name"]:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def extract(name, names, img_dir, out_dir, batch_size, workers, device):
    shape = BACKBONES[name]
    out_path = os.path.join(out_dir, "features_{}.npy".format(name))

    net = build_backbone(name).to(device)
    loader = DataLoader(
        MemeImages(names, img_dir), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
    )

    # float16 halves a cache that reaches ~1 GB for ResNet50; these are frozen activations
    # fed to a dense layer, so the precision loss is immaterial.
    store = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(len(names),) + shape
    )

    start, done = time.time(), 0
    with torch.no_grad():
        for images, idx in loader:
            feats = net(images.to(device, non_blocking=True))
            store[idx.numpy()] = feats.cpu().numpy().astype(np.float16)
            done += len(idx)
            if done % (batch_size * 20) < batch_size:
                rate = done / (time.time() - start)
                print("  {:>5}/{}  {:.0f} img/s".format(done, len(names), rate), flush=True)

    store.flush()
    del store, net
    size_mb = os.path.getsize(out_path) / 1e6
    print("{:<9} -> {}  {}  {:.0f} MB  ({:.0f}s)".format(
        name, os.path.basename(out_path), (len(names),) + shape, size_mb, time.time() - start))


def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = args.dataset if os.path.isabs(args.dataset) else \
        os.path.abspath(os.path.join(script_dir, args.dataset))
    img_dir = os.path.join(dataset_dir, "Img")
    out_dir = args.out if os.path.isabs(args.out) else \
        os.path.abspath(os.path.join(script_dir, args.out))
    os.makedirs(out_dir, exist_ok=True)

    names = collect_image_names(dataset_dir)
    missing = [n for n in names if not os.path.isfile(os.path.join(img_dir, n))]
    if missing:
        raise SystemExit("{} image(s) referenced by the CSVs are missing from {}, e.g. {}"
                         .format(len(missing), img_dir, missing[:3]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Memes :", len(names))
    if device.type == "cpu":
        print("WARNING: no GPU. Feature extraction on CPU takes ~1h per backbone; "
              "run this on Kaggle instead.")

    # The row order of every features_*.npy file is this index - the training code joins
    # on it, so it must be written once and shared by all backbones.
    with open(os.path.join(out_dir, "feature_index.json"), "w", encoding="utf-8") as fh:
        json.dump({"image_names": names}, fh, ensure_ascii=False)

    for name in args.models:
        out_path = os.path.join(out_dir, "features_{}.npy".format(name))
        if os.path.isfile(out_path) and not args.overwrite:
            print("{:<9} -> cached, skipping (use --overwrite to rebuild)".format(name))
            continue
        extract(name, names, img_dir, out_dir, args.batch_size, args.workers, device)

    print("\nFeature cache ready at", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute MuLAD frozen visual features")
    parser.add_argument("--dataset", type=str, default="../Dataset")
    parser.add_argument("--out", type=str, default="../Features")
    parser.add_argument("--models", nargs="+", default=list(BACKBONES),
                        choices=list(BACKBONES))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
