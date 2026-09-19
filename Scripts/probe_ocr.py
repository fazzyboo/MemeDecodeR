"""
One-off diagnostic: compare Tesseract configurations on a sample of our memes.

Bengali OCR quality drives the entire text modality, so the language set and page
segmentation mode are worth choosing on evidence rather than by default. This script
reports, per configuration, the share of memes that yield no text at all and how much
Bengali script is actually recovered.

Usage:
    python probe_ocr.py --sample 60
"""
import argparse
import os
import random
import re
import time

from ocr_captions import ROOT_DIR, clean_caption, configure_tesseract, preprocess

BENGALI = re.compile(r"[ঀ-৿]")

CONFIGS = [
    ("ben", 3),
    ("ben", 6),
    ("ben", 11),
    ("ben+eng", 3),
    ("ben+eng", 6),
    ("ben+eng", 11),
]


def main(args):
    pytesseract = configure_tesseract()
    img_dir = os.path.join(ROOT_DIR, "Dataset", "Img")
    images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.seed(args.seed)
    sample = random.sample(images, min(args.sample, len(images)))
    print("Sampling {} memes from {}".format(len(sample), img_dir))
    print("=" * 92)
    print(
        "{:<10} {:>4} {:>9} {:>11} {:>11} {:>11} {:>9}".format(
            "lang", "psm", "empty %", "avg words", "avg Bn ch", "Bn share", "sec/img"
        )
    )
    print("-" * 92)

    cache = {name: preprocess(os.path.join(img_dir, name), args.min_width) for name in sample}

    results = []
    for lang, psm in CONFIGS:
        config = "--psm {} --oem 1".format(psm)
        empty = 0
        words = []
        bn_chars = []
        bn_share = []
        start = time.time()
        for name in sample:
            try:
                raw = pytesseract.image_to_string(cache[name], lang=lang, config=config)
                caption = clean_caption(raw)
            except Exception:
                caption = ""
            if not caption:
                empty += 1
                continue
            words.append(len(caption.split()))
            n_bn = len(BENGALI.findall(caption))
            bn_chars.append(n_bn)
            visible = len(caption.replace(" ", ""))
            bn_share.append(n_bn / visible if visible else 0.0)
        elapsed = time.time() - start
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        row = (lang, psm, 100.0 * empty / len(sample), avg(words), avg(bn_chars), avg(bn_share), elapsed / len(sample))
        results.append(row)
        print(
            "{:<10} {:>4} {:>9.1f} {:>11.1f} {:>11.1f} {:>11.2f} {:>9.2f}".format(*row)
        )

    print("=" * 92)
    print("empty %   - memes where OCR recovered nothing (these become empty captions)")
    print("Bn share  - fraction of recovered characters that are Bengali script")
    print("Pick the config with low empty %, high avg words, and a high Bengali share.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare Tesseract configurations on our memes")
    p.add_argument("--sample", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_width", type=int, default=1000)
    main(p.parse_args())
