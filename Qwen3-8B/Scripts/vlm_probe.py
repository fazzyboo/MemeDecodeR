"""
Diagnostic - can the VLM actually read the Bengali text in our memes?

The zero-shot and fine-tuned VLM runs both rest on one assumption: that the model can read
Bengali script off a meme. If it cannot, its scores will be poor for a reason that has
nothing to do with aggression classification, and the whole comparison against MAF would be
measuring the wrong thing. This script checks that assumption on a handful of memes before
either of the long runs is worth starting.

For each sampled meme it prints, side by side:
  - what pytesseract extracted (what MAF is given, from captions_raw.csv)
  - what the VLM transcribes from the same image
  - the VLM's one-line description of what the meme is doing

Read the pairs yourself - this prints evidence, it does not score anything. What you are
looking for is whether the VLM's transcription is recognisably Bengali prose where the OCR
line is garbled. That gap, if present, is the entire argument for the VLM approach.

This mirrors probe_ocr.py, which compares Tesseract configurations the same way.

Usage:
    python vlm_probe.py --n 5
"""
import _paths  # noqa: F401

import argparse
import os

import pandas as pd
import torch
from PIL import Image

PROMPT = (
    "This is a Bengali (Bangla) meme.\n"
    "1. Transcribe every piece of text you can read in the image, exactly as written.\n"
    "2. Then, on a new line starting with 'MEANING:', explain in one English sentence what "
    "the meme is saying and who - if anyone - it is attacking."
)


def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_dir = os.path.join(root, "Dataset")
    images_dir = os.path.join(dataset_dir, "Img")

    test = pd.read_csv(os.path.join(dataset_dir, "testing_set.csv"))
    raw = pd.read_csv(os.path.join(dataset_dir, "captions_raw.csv"))
    raw_map = dict(zip(raw["image_name"], raw["Captions"].fillna("")))

    # One meme per class, so the sample is not all of a kind.
    sample = test.groupby("Label", group_keys=False).head(
        max(1, args.n // test["Label"].nunique())).reset_index(drop=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 1280 * 28 * 28
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", quantization_config=quant)
    model.eval()
    print("Model:", args.model, "| VRAM {:.1f} GB".format(torch.cuda.memory_allocated() / 1e9))

    for _, row in sample.iterrows():
        image = Image.open(os.path.join(images_dir, row["image_name"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        print("\n" + "=" * 78)
        print("MEME      :", row["image_name"], "| gold label:", row["Label"])
        print("-" * 78)
        print("TESSERACT :", (raw_map.get(row["image_name"], "") or "(nothing)")[:400])
        print("-" * 78)
        print("VLM       :", answer[:900])
    print("\n" + "=" * 78)
    print("Judge by eye: is the VLM line readable Bengali where the Tesseract line is not?")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Check the VLM can read Bengali meme text")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--no_log", action="store_true")
    args = p.parse_args()
    if not args.no_log:
        import _logging
        _logging.start("vlm_probe")
    main(args)
