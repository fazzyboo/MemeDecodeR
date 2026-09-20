"""
Build the matched-pair OCR splits and measure OCR error against the gold captions.

This is the controlled experiment the caption argument needs. The three split CSVs are
rewritten with the SAME image_name, the SAME Label and the SAME split membership - only
the Captions column changes, from the dataset authors' hand-corrected text to machine OCR.
Caption quality is then the only variable between this run and the existing one.

Because both texts now exist for every meme, OCR error becomes directly measurable:

    CER = edit distance between OCR and gold characters, over gold length
    WER = the same over whitespace tokens

Those per-meme rates are what turn "OCR is noisy" into an accuracy-versus-error curve.

Outputs, all in Dataset/:
    training_set_ocr.csv, validation_set_ocr.csv, testing_set_ocr.csv
    caption_error_rates.csv   - image_name, split, CER, WER, and the raw measures

Usage:
    python make_ocr_splits.py
"""
import argparse
import os

import numpy as np
import pandas as pd

SPLITS = ("training_set", "validation_set", "testing_set")


def edit_distance(a, b):
    """Levenshtein distance with a rolling row - captions are short, so this is plenty."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (ca != cb)  # substitution
            ))
        previous = current
    return previous[-1]


def error_rates(gold, hypothesis):
    """(CER, WER) of `hypothesis` against reference `gold`.

    Normalised by the GOLD length, which is the standard convention: a hypothesis that
    invents text is penalised through insertions, and the rate can exceed 1.0 when OCR
    emits more garbage than there was real text. That is a meaningful signal here, not a
    bug, so the value is left uncapped.
    """
    gold, hypothesis = str(gold or ""), str(hypothesis or "")
    g_chars = gold.replace(" ", "")
    h_chars = hypothesis.replace(" ", "")
    cer = edit_distance(g_chars, h_chars) / max(len(g_chars), 1)

    g_words, h_words = gold.split(), hypothesis.split()
    wer = edit_distance(g_words, h_words) / max(len(g_words), 1)
    return cer, wer


def main(args):
    here = os.path.dirname(os.path.abspath(__file__))
    rs = lambda p: p if os.path.isabs(p) else os.path.abspath(os.path.join(here, p))
    dataset_dir, ocr_csv = rs(args.dataset), rs(args.captions)

    if not os.path.isfile(ocr_csv):
        raise SystemExit("Missing {}. Run ocr_captions.py first.".format(ocr_csv))

    ocr = pd.read_csv(ocr_csv)
    ocr["Captions"] = ocr["Captions"].fillna("").astype(str)
    ocr_map = dict(zip(ocr["image_name"], ocr["Captions"]))
    print("OCR captions available:", len(ocr_map))

    rows, missing_total = [], 0
    for split in SPLITS:
        path = os.path.join(dataset_dir, "{}.csv".format(split))
        frame = pd.read_csv(path)
        frame["Captions"] = frame["Captions"].fillna("").astype(str)

        missing = [n for n in frame["image_name"] if n not in ocr_map]
        missing_total += len(missing)
        if missing:
            print("  {}: {} meme(s) have no OCR caption yet, e.g. {}"
                  .format(split, len(missing), missing[:2]))

        out = frame.copy()
        out["Captions"] = [ocr_map.get(n, "") for n in frame["image_name"]]
        out_path = os.path.join(dataset_dir, "{}_ocr.csv".format(split))
        out.to_csv(out_path, index=False, encoding="utf-8")

        for name, gold, hyp in zip(frame["image_name"], frame["Captions"], out["Captions"]):
            cer, wer = error_rates(gold, hyp)
            rows.append({
                "image_name": name, "split": split, "cer": cer, "wer": wer,
                "gold_words": len(str(gold).split()), "ocr_words": len(str(hyp).split()),
                "gold_chars": len(str(gold).replace(" ", "")),
                "ocr_chars": len(str(hyp).replace(" ", "")),
                "ocr_empty": len(str(hyp).strip()) == 0,
            })
        print("  {:<16} {:>5} rows -> {}".format(split, len(out), os.path.basename(out_path)))

    if missing_total:
        print("\nWARNING: {} meme(s) still lack an OCR caption. Re-run ocr_captions.py to "
              "finish the pass, then run this script again.".format(missing_total))

    errors = pd.DataFrame(rows)
    err_path = os.path.join(dataset_dir, "caption_error_rates.csv")
    errors.to_csv(err_path, index=False, encoding="utf-8")

    print()
    print("OCR ERROR AGAINST GOLD CAPTIONS")
    print("  memes            : {}".format(len(errors)))
    print("  mean CER         : {:.3f}".format(errors["cer"].mean()))
    print("  median CER       : {:.3f}".format(errors["cer"].median()))
    print("  mean WER         : {:.3f}".format(errors["wer"].mean()))
    print("  CER > 0.5        : {:.1%} of memes".format((errors["cer"] > 0.5).mean()))
    print("  CER > 1.0        : {:.1%} of memes  (OCR emitted more error than gold text)"
          .format((errors["cer"] > 1.0).mean()))
    print("  empty OCR output : {}".format(int(errors["ocr_empty"].sum())))
    print("  length ratio     : {:.2f}x gold".format(
        errors["ocr_words"].sum() / max(errors["gold_words"].sum(), 1)))

    test = errors[errors["split"] == "testing_set"]
    print()
    print("  test split only  : mean CER {:.3f}, median {:.3f}"
          .format(test["cer"].mean(), test["cer"].median()))
    print("\nWrote", err_path)
    print("\nNext: upload the *_ocr.csv splits as a Kaggle dataset version and run "
          "MAF_Kaggle.ipynb with the maf_fixed_sched configuration only.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build matched-pair OCR splits and measure CER/WER")
    p.add_argument("--dataset", default="../Dataset")
    p.add_argument("--captions", default="../Dataset/captions_ocr_mimosa.csv")
    main(p.parse_args())
