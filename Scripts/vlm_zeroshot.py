"""
Zero-shot VLM baseline - classify each meme with an instruction-tuned vision-language
model, with no task-specific training at all.

WHY THIS EXISTS

README.md section 5.1 names caption noise as the largest single source of divergence
between this replication and the published result. MAF cannot see the meme's text: it
sees whatever pytesseract managed to extract, and Bengali OCR is weak enough that the
paper's own authors hand-corrected every caption. A modern VLM reads the image directly,
so the OCR stage - and its errors - disappear from the pipeline entirely.

That makes this the natural counterpart to the MAF numbers rather than just a different
model: it isolates how much of MAF's ceiling was set by the OCR stage rather than by the
fusion architecture.

HOW A PREDICTION IS MADE

The four classes are presented as options A-D and the model is asked for one letter. The
answer is then read off the first generated position by comparing the logits of exactly
the four letter tokens, and taking the largest.

This is deliberate, and worth understanding when reading the numbers:

  - It always yields a valid class. Free-text generation needs parsing, and a model that
    answers "This meme appears to be..." either has to be re-prompted or scored as a
    failure - which quietly becomes a wrong answer and depresses the metric for a reason
    that has nothing to do with the model's understanding.
  - It is one forward pass per meme rather than an autoregressive decode, so a 400-meme
    run takes minutes instead of an hour.
  - It is fully deterministic, so a rerun reproduces the number exactly.
  - The cost is that the model cannot reason step by step before answering. --decode text
    generates free text and parses it instead, if you want to measure that difference.

Usage:
    python vlm_zeroshot.py --subset 20 --run_name vlm_zs_smoke   # quick check
    python vlm_zeroshot.py --run_name vlm_zeroshot               # the real run
"""
import _paths  # noqa: F401  - keeps model downloads inside Replication/.cache

import argparse
import os
import re
import time

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import _metrics as m

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# The class definitions are the paper's (Sec. 3), phrased for a model that has never seen
# this dataset. "Aggression" here is about the target of the attack, not its intensity -
# the distinction the paper draws between its four categories.
SYSTEM_PROMPT = (
    "You are an expert content moderator annotating Bengali (Bangla) memes for a research "
    "dataset on targeted aggression. You are precise and you always answer with a single "
    "letter."
)

TASK_PROMPT = """This is a Bengali meme. Read the Bengali text written in the image, look at the imagery, and decide who the meme attacks, if anyone.

Choose exactly one category:

A. Non-aggressive - the meme is a joke, an observation, an advertisement or a harmless statement. It does not attack or demean any person or group. Most ordinary memes are this.

B. Gendered aggression - the meme attacks, demeans, stereotypes or sexualises someone on the basis of gender. Typically misogynistic content about women, wives, girlfriends, or about gender roles generally.

C. Political aggression - the meme attacks a political party, a political leader, a government, or a person because of their political affiliation.

D. Religious aggression - the meme attacks a religion, a religious group, a religious figure or a person because of their faith.

Answer with one letter only: A, B, C or D."""

CAPTION_SUFFIX = """

For reference, an automatic OCR of the meme's text produced the following, which may contain recognition errors:
{caption}"""

LETTERS = ["A", "B", "C", "D"]  # index == the paper's class id (NoAg, GAg, PAg, RAg)


def build_messages(image, caption):
    text = TASK_PROMPT
    if caption is not None:
        text += CAPTION_SUFFIX.format(caption=caption.strip() or "(no text recognised)")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
        },
    ]


def letter_token_ids(processor):
    """Token ids for 'A'..'D' as they appear at the start of an assistant turn.

    A tokenizer may encode "A" and " A" differently, so both spellings are collected and
    the logits of all of them are considered for each class.
    """
    tok = processor.tokenizer
    ids = []
    for letter in LETTERS:
        variants = set()
        for form in (letter, " " + letter):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                variants.add(enc[0])
        ids.append(sorted(variants))
    return ids


def main(args):
    torch.manual_seed(args.seed)
    start = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_dir = os.path.join(root, "Dataset")
    outputs_dir = os.path.join(root, "Outputs")
    images_dir = os.path.join(dataset_dir, "Img")

    frame = pd.read_csv(os.path.join(dataset_dir, args.split + ".csv"))
    frame["Captions"] = frame["Captions"].fillna("")
    if args.subset:
        frame = frame.groupby("Label", group_keys=False).head(
            max(1, args.subset // m.NUM_CLASSES)
        ).reset_index(drop=True)
        print("SUBSET MODE: {} rows (smoke test, not a real result)".format(len(frame)))

    print("Split           :", args.split, "|", len(frame), "memes")
    print("Model           :", args.model)
    print("Load in 4-bit   :", args.load_4bit)
    print("Captions given  :", args.use_caption, "(image-only is the point of this baseline)")
    print("Decode          :", args.decode)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(args.model)
    # Cap the visual token budget. Memes are small but text-heavy, so resolution matters
    # for legibility; left uncapped, a large meme can blow up the sequence length.
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.min_pixels
        processor.image_processor.max_pixels = args.max_pixels

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant,
    )
    model.eval()
    print("Model loaded. VRAM allocated: {:.1f} GB".format(torch.cuda.memory_allocated() / 1e9))

    ids_per_class = letter_token_ids(processor)
    print("Letter token ids:", ids_per_class)

    preds, actual, raw_answers = [], [], []
    unparsed = 0

    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Classifying"):
        image = Image.open(os.path.join(images_dir, row["image_name"])).convert("RGB")
        caption = row["Captions"] if args.use_caption else None
        messages = build_messages(image, caption)

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            if args.decode == "letter":
                logits = model(**inputs).logits[0, -1]
                scores = [max(float(logits[i]) for i in ids) for ids in ids_per_class]
                choice = int(max(range(m.NUM_CLASSES), key=lambda k: scores[k]))
                raw_answers.append(LETTERS[choice])
            else:
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                text = processor.decode(
                    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                raw_answers.append(text)
                found = re.search(r"\b([ABCD])\b", text.upper())
                if found:
                    choice = LETTERS.index(found.group(1))
                else:
                    choice = 0  # counted and reported; see below
                    unparsed += 1

        preds.append(choice)
        actual.append(m.LABEL_TO_ID[row["Label"]])

    if unparsed:
        print("\nWARNING: {} of {} answers could not be parsed and were scored as "
              "'{}'. Treat the metrics as a lower bound.".format(
                  unparsed, len(frame), m.TARGET_NAMES[0]))

    frame["vlm_answer"] = raw_answers
    m.report_and_save(
        actual,
        preds,
        outputs_dir,
        args.run_name,
        extra={
            "approach": "vlm_zeroshot",
            "model": args.model,
            "load_4bit": args.load_4bit,
            "decode": args.decode,
            "use_caption": args.use_caption,
            "split": args.split,
            "subset": args.subset,
            "seed": args.seed,
            "unparsed_answers": unparsed,
            "trained_on_this_dataset": False,
        },
        frame=frame,
    )
    print("Total time : {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zero-shot VLM baseline for Bengali aggression memes")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--split", type=str, default="testing_set",
                   choices=["training_set", "validation_set", "testing_set"])
    p.add_argument("--run_name", type=str, default="vlm_zeroshot")
    p.add_argument("--decode", type=str, default="letter", choices=["letter", "text"],
                   help="letter = compare A-D logits in one pass; text = generate and parse")
    p.add_argument("--use_caption", action="store_true",
                   help="also show the model the OCR caption (default: image only)")
    p.add_argument("--load_4bit", action="store_true", help="4-bit quantisation via bitsandbytes")
    p.add_argument("--max_new_tokens", type=int, default=16, help="only used by --decode text")
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--subset", type=int, default=0, help="N rows total, class-balanced")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_log", action="store_true")

    args = p.parse_args()
    if not args.no_log:
        import _logging

        _logging.start(args.run_name)
    main(args)
