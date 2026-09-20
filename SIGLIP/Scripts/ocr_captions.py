"""
Stage 1 - Caption extraction (OCR).

The MIMOSA paper (Ahsan et al., EACL 2024, Sec. 3.2) builds the text modality by
running an OCR over every meme:

    "Afterward, we extract the meme caption using an OCR [pytesseract]. However, we
     manually checked the extracted captions to correct any missing words and
     spelling as OCR in Bengali is not well-established."

The released MIMOSA CSVs already ship those hand-corrected captions, so the public
codebase has no OCR stage. Our dataset ships images + labels only, so this script
reproduces that missing stage with the same engine (pytesseract / Tesseract 5 LSTM).

NOTE ON FIDELITY: the manual correction pass described in the paper is not
reproducible here - these captions are raw OCR output. That is the single largest
expected source of divergence from the published scores. See README, section
"Deviations from the paper".

Usage:
    python ocr_captions.py                                   # OCR every image in Dataset/Img
    python ocr_captions.py --lang ben+eng                    # Bengali + Latin script
    python ocr_captions.py --limit 20 --workers 1 --verbose  # quick probe
"""
import argparse
import csv
import os
import re
import sys
import time
from multiprocessing import Pool

from PIL import Image

# --------------------------------------------------------------------------------------
# Tesseract discovery. The binary is a system install (not a pip package); the Bengali
# traineddata lives in a project-local tessdata dir so that adding languages needs no
# administrator rights.
# --------------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))

CANDIDATE_BINARIES = [
    os.environ.get("TESSERACT_CMD", ""),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]
CANDIDATE_TESSDATA = [
    os.environ.get("TESSDATA_PREFIX", ""),
    os.path.join(PROJECT_ROOT, "tessdata"),
    os.path.join(ROOT_DIR, "tessdata"),
]


def configure_tesseract():
    """Point pytesseract at the binary and at a tessdata dir that contains ben."""
    import pytesseract

    for cand in CANDIDATE_BINARIES:
        if cand and os.path.isfile(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            break
    for cand in CANDIDATE_TESSDATA:
        if cand and os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "ben.traineddata")):
            os.environ["TESSDATA_PREFIX"] = cand
            break
    return pytesseract


# --------------------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------------------
# Bengali block U+0980-U+09FF, the danda / double-danda that Bengali borrows from the
# Devanagari block, the ZWNJ/ZWJ joiners that Bengali conjuncts rely on, and printable
# ASCII (memes routinely mix in Latin words and digits).
_ALLOWED = re.compile(r"[^\u0980-\u09FF\u0964\u0965\u200c\u200d\x20-\x7E]")
_WS = re.compile(r"\s+")


def clean_caption(text):
    """Normalise raw OCR output into a single-line caption."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = _ALLOWED.sub(" ", text)
    # Tesseract emits long runs of stray punctuation on noisy meme backgrounds.
    text = re.sub(r"([^\w\u0980-\u09FF\s])\1{2,}", r"\1", text)
    text = _WS.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------------------
def preprocess(path, min_width):
    """Grayscale + upscale.

    The Tesseract LSTM engine expects roughly 300-DPI-sized glyphs. Meme screenshots are
    often far smaller, and upscaling is the single biggest accuracy win for Bengali
    conjunct characters.
    """
    image = Image.open(path)
    image = image.convert("RGB").convert("L")
    if image.width < min_width:
        scale = min_width / float(image.width)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image


_CFG = {}


def _init_worker(lang, psm, oem, min_width, verbose):
    _CFG["pytesseract"] = configure_tesseract()
    _CFG["lang"] = lang
    _CFG["config"] = "--psm {} --oem {}".format(psm, oem)
    _CFG["min_width"] = min_width
    _CFG["verbose"] = verbose


def _ocr_one(task):
    image_name, img_dir = task
    pytesseract = _CFG["pytesseract"]
    path = os.path.join(img_dir, image_name)
    try:
        image = preprocess(path, _CFG["min_width"])
        raw = pytesseract.image_to_string(image, lang=_CFG["lang"], config=_CFG["config"])
        caption = clean_caption(raw)
    except Exception as exc:  # one unreadable meme must not kill a multi-hour run
        sys.stderr.write("[WARN] {}: {}\n".format(image_name, exc))
        caption = ""
    if _CFG["verbose"]:
        print("{}\n  -> {}\n".format(image_name, caption[:160]), flush=True)
    return image_name, caption


def load_done(out_csv):
    """Resume support: a long OCR pass should survive an interruption."""
    done = {}
    if os.path.isfile(out_csv):
        with open(out_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("image_name"):
                    done[row["image_name"]] = row.get("Captions", "")
    return done


def main(args):
    img_dir = args.img_dir or os.path.join(ROOT_DIR, "Dataset", "Img")
    out_csv = args.out or os.path.join(ROOT_DIR, "Dataset", "captions_raw.csv")

    pytesseract = configure_tesseract()
    print("Tesseract binary :", pytesseract.pytesseract.tesseract_cmd)
    print("TESSDATA_PREFIX  :", os.environ.get("TESSDATA_PREFIX", "<default>"))
    print("Version          :", str(pytesseract.get_tesseract_version()).split("\n")[0])
    langs = pytesseract.get_languages()
    print("Languages        :", langs)
    for need in args.lang.split("+"):
        if need not in langs:
            sys.exit("ERROR: language '{}' is not available to Tesseract.".format(need))

    images = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    done = {} if args.overwrite else load_done(out_csv)
    todo = [f for f in images if f not in done]
    if args.limit:
        todo = todo[: args.limit]

    print("Images on disk   :", len(images))
    print("Already captioned:", len(done))
    print("To process       :", len(todo))
    print("lang={}  psm={}  oem={}  workers={}".format(args.lang, args.psm, args.oem, args.workers))
    print("-" * 60)
    if not todo:
        print("Nothing to do.")
        return

    mode = "w" if (args.overwrite or not os.path.isfile(out_csv)) else "a"
    start = time.time()
    written = 0
    pool = None
    with open(out_csv, mode, encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if mode == "w":
            writer.writerow(["image_name", "Captions"])
        payload = [(f, img_dir) for f in todo]
        init_args = (args.lang, args.psm, args.oem, args.min_width, args.verbose)
        if args.workers <= 1:
            _init_worker(*init_args)
            results = map(_ocr_one, payload)
        else:
            pool = Pool(args.workers, initializer=_init_worker, initargs=init_args)
            results = pool.imap(_ocr_one, payload, chunksize=8)
        for image_name, caption in results:
            writer.writerow([image_name, caption])
            written += 1
            if written % 50 == 0:
                fh.flush()
                rate = written / (time.time() - start)
                eta = (len(todo) - written) / rate if rate else 0
                print(
                    "  {}/{}  ({:.2f} img/s, ETA {:.1f} min)".format(
                        written, len(todo), rate, eta / 60
                    ),
                    flush=True,
                )
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - start
    print("-" * 60)
    print("Done. {} captions in {:.1f} min -> {}".format(written, elapsed / 60, out_csv))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Bengali meme captions with pytesseract")
    p.add_argument("--img_dir", type=str, default=None, help="folder of meme images")
    p.add_argument("--out", type=str, default=None, help="output CSV path")
    p.add_argument("--lang", type=str, default="ben", help="tesseract language(s), e.g. ben or ben+eng")
    # PSM 11 ("sparse text") is the default on evidence, not by convention. Measured on a
    # 60-meme sample with probe_ocr.py: PSM 3 (Tesseract's own default) returns NO text at
    # all for 33% of our memes, because meme text is scattered over the image rather than
    # laid out as a page. PSM 11 drops that to 0% and recovers the most Bengali script.
    p.add_argument("--psm", type=int, default=11, help="page segmentation mode (default 11, sparse text)")
    p.add_argument("--oem", type=int, default=1, help="OCR engine mode (1 = LSTM)")
    p.add_argument("--min_width", type=int, default=1000, help="upscale images narrower than this")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=0, help="only process N images (debugging)")
    p.add_argument("--overwrite", action="store_true", help="ignore existing output and restart")
    p.add_argument("--verbose", action="store_true", help="print each caption as it is produced")
    main(p.parse_args())
