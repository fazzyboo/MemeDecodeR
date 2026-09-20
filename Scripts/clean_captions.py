"""
Stage 1b (optional) - denoise the raw OCR captions.

WHY THIS EXISTS
---------------
The paper hand-corrected every OCR caption (Sec. 3.2). That pass is not reproducible
here, and its absence is the largest source of divergence in this replication.

PSM 11 ("sparse text") was chosen because Tesseract's default PSM 3 returns nothing at
all for a third of our memes. The cost of sparse mode is that it also reports watermarks,
page furniture, logos and plain image texture as text - so a caption like

    "ie, wy ~ pt La pw a a ONODEORMEMEST A or well =x Se সাক. he a কালা % ig = TN"

contains two real Bengali words buried in Latin noise. Bangla-BERT sees only this string,
so that noise is fed straight into the text modality.

This script applies a conservative, deterministic filter: Bengali script is always kept,
and Latin/numeric tokens are kept only if they look like real words rather than OCR
debris. It is an automated stand-in for the paper's manual pass, not an equivalent of it.

It rewrites nothing - it reads captions_raw.csv and writes captions_clean.csv, so both
versions remain available and the choice stays visible.

Usage:
    python clean_captions.py
    python clean_captions.py --report 15     # also show before/after examples
"""
import argparse
import csv
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

BENGALI_CHAR = re.compile(r"[ঀ-৿]")
LATIN_CHAR = re.compile(r"[A-Za-z]")
VOWEL = re.compile(r"[aeiouAEIOU]")
WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")

# Latin tokens this short are almost always OCR debris unless they are real words.
SHORT_WHITELIST = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in",
    "is", "it", "me", "my", "no", "of", "oh", "ok", "on", "or", "so", "to", "up", "us",
    "we", "yo", "the", "and", "you", "not", "but", "for", "her", "his", "our", "out",
    "who", "why", "how", "all", "can", "did", "get", "got", "had", "has", "her", "him",
    "new", "now", "old", "one", "see", "she", "too", "two", "was", "way", "yes", "yet",
}


def keep_token(token):
    """Decide whether a single whitespace-delimited token survives the filter."""
    # Any Bengali content is always kept - that is the signal we are protecting.
    if BENGALI_CHAR.search(token):
        return True

    stripped = token.strip(".,!?;:'\"()[]{}<>|/\\-_=~*+#@%^&`")
    if not stripped:
        return False

    # Pure numbers: keep short ones (years, counts), drop long OCR digit runs.
    if stripped.isdigit():
        return len(stripped) <= 4

    if not LATIN_CHAR.search(stripped):
        return False

    lowered = stripped.lower()
    if lowered in SHORT_WHITELIST:
        return True

    # Must look like a word: letters only, no digit/letter salad.
    if not WORDLIKE.match(stripped):
        return False
    # Debris is typically 1-2 characters, or a consonant run with no vowel.
    if len(stripped) < 3:
        return False
    if not VOWEL.search(stripped):
        return False
    # Mixed case inside a word ("ONODEORMEMEST" is fine, "aBcDe" is not).
    if not (stripped.islower() or stripped.isupper() or stripped.istitle()):
        return False
    return True


def clean(caption):
    if not caption:
        return ""
    kept = [t for t in caption.split() if keep_token(t)]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def bengali_share(text):
    visible = text.replace(" ", "")
    if not visible:
        return 0.0
    return len(BENGALI_CHAR.findall(text)) / len(visible)


def main(args):
    src = args.src or os.path.join(ROOT_DIR, "Dataset", "captions_raw.csv")
    dst = args.dst or os.path.join(ROOT_DIR, "Dataset", "captions_clean.csv")

    rows = []
    with open(src, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["image_name"], row.get("Captions", "")))

    examples = []
    n_before_words = n_after_words = 0
    share_before = share_after = 0.0
    empty_before = empty_after = 0

    with open(dst, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_name", "Captions"])
        for name, raw in rows:
            cleaned = clean(raw)
            writer.writerow([name, cleaned])

            n_before_words += len(raw.split())
            n_after_words += len(cleaned.split())
            share_before += bengali_share(raw)
            share_after += bengali_share(cleaned)
            empty_before += 1 if not raw.strip() else 0
            empty_after += 1 if not cleaned.strip() else 0
            if len(examples) < args.report and raw.strip() and cleaned != raw:
                examples.append((name, raw, cleaned))

    n = max(1, len(rows))
    print("Captions processed : {}".format(len(rows)))
    print("-" * 78)
    print("{:<26}{:>12}{:>12}".format("", "raw", "clean"))
    print("{:<26}{:>12.1f}{:>12.1f}".format("avg words / caption", n_before_words / n, n_after_words / n))
    print("{:<26}{:>12.2f}{:>12.2f}".format("avg Bengali share", share_before / n, share_after / n))
    print("{:<26}{:>12}{:>12}".format("empty captions", empty_before, empty_after))
    print("-" * 78)
    print("Tokens removed     : {:,} ({:.1f}% of all tokens)".format(
        n_before_words - n_after_words,
        100.0 * (n_before_words - n_after_words) / max(1, n_before_words),
    ))
    print("Wrote ->", dst)

    if examples:
        print()
        print("Examples (raw -> clean)")
        print("=" * 78)
        for name, raw, cleaned in examples:
            print(name)
            print("  raw  :", raw[:150])
            print("  clean:", cleaned[:150] if cleaned else "<emptied>")
            print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Denoise raw OCR captions")
    p.add_argument("--src", type=str, default=None)
    p.add_argument("--dst", type=str, default=None)
    p.add_argument("--report", type=int, default=8, help="show N before/after examples")
    main(p.parse_args())
