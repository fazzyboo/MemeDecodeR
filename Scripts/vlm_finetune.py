"""
QLoRA fine-tune of a vision-language model on the meme training split.

This is the trained counterpart to vlm_zeroshot.py, and the run that is directly
comparable to MAF: both see the same 2,390 training memes, the same 513 validation memes
and the same held-out 400 test memes, and both are scored with the paper's metric set.

HOW IT IS TRAINED

The model is shown the same prompt the zero-shot script uses and trained to emit the one
letter naming the correct class. The loss is computed on that single answer token only -
every prompt token is masked out with -100.

Training the exact thing inference reads is the point. vlm_zeroshot.py decides a class by
comparing the A-D logits at the first answer position, so putting the loss on that same
position means there is no train/test mismatch: no parsing, no format drift, and no way
for the model to score well on a proxy objective that inference never consults.

The base weights stay frozen in 4-bit and only LoRA adapters train, which is what keeps an
8B model inside 16 GB of VRAM alongside the vision tower.

Usage:
    python vlm_finetune.py --subset 32 --epochs 1 --run_name vlm_ft_smoke   # pipeline check
    python vlm_finetune.py --run_name vlm_finetuned                         # the real run
"""
import _paths  # noqa: F401

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import _metrics as m

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Prompt and label scheme are imported so the two scripts can never drift apart.
from vlm_zeroshot import LETTERS, build_messages, letter_token_ids  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(dataset_dir, name, subset):
    frame = pd.read_csv(os.path.join(dataset_dir, name + ".csv"))
    frame["Captions"] = frame["Captions"].fillna("")
    if subset:
        frame = frame.groupby("Label", group_keys=False).head(
            max(1, subset // m.NUM_CLASSES)
        ).reset_index(drop=True)
    return frame


def encode(processor, images_dir, row, use_caption, device, answer=None):
    """Tokenise one meme. With `answer`, the assistant turn is included for training.

    The answer is appended as a chat message and the processor builds the whole sequence,
    rather than tokenising the prompt and concatenating the answer token by hand. Qwen3-VL
    returns several parallel per-token tensors - input_ids, attention_mask and
    mm_token_type_ids - and its 3D rope index is computed from them together, so extending
    some but not all of them by one token raises a shape mismatch deep inside the model.
    Letting the processor produce the full sequence keeps them consistent by construction.
    """
    image = Image.open(os.path.join(images_dir, row["image_name"])).convert("RGB")
    caption = row["Captions"] if use_caption else None
    messages = build_messages(image, caption)
    if answer is not None:
        messages = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
    enc = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=answer is None,
        return_dict=True, return_tensors="pt",
    )
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}


def answer_position(input_ids, answer_id):
    """Index of the assistant's answer token.

    The letters A-D also occur in the prompt itself, so the first match is not the answer.
    The assistant turn is last and is followed only by the end-of-turn marker, so the final
    occurrence is the one being trained on.
    """
    hits = (input_ids[0] == answer_id).nonzero(as_tuple=True)[0]
    if len(hits) == 0:
        raise RuntimeError("answer token not found in the tokenised sequence")
    return int(hits[-1])


@torch.inference_mode()
def evaluate(model, processor, frame, images_dir, use_caption, ids_per_class, device, desc):
    model.eval()
    preds, actual = [], []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=desc):
        enc = encode(processor, images_dir, row, use_caption, device)
        logits = model(**enc).logits[0, -1]
        scores = [max(float(logits[i]) for i in ids) for ids in ids_per_class]
        preds.append(int(max(range(m.NUM_CLASSES), key=lambda k: scores[k])))
        actual.append(m.LABEL_TO_ID[row["Label"]])
    return actual, preds


def main(args):
    set_seed(args.seed)
    start = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    dataset_dir = os.path.join(root, "Dataset")
    images_dir = os.path.join(dataset_dir, "Img")
    outputs_dir = os.path.join(root, "Outputs")
    adapter_dir = os.path.join(root, "Saved_Models", "lora_" + args.run_name)
    os.makedirs(adapter_dir, exist_ok=True)

    train = load_split(dataset_dir, "training_set", args.subset)
    valid = load_split(dataset_dir, "validation_set", args.subset)
    test = load_split(dataset_dir, "testing_set", args.subset)
    if args.subset:
        print("SUBSET MODE: {}/{}/{} rows (smoke test, not a real result)".format(
            len(train), len(valid), len(test)))

    print("Model           :", args.model)
    print("Train/Valid/Test:", len(train), len(valid), len(test))
    print("Epochs          :", args.epochs, "| LR:", args.lr, "| grad accum:", args.grad_accum)
    print("LoRA r/alpha    :", args.lora_r, "/", args.lora_alpha)
    print("Captions given  :", args.use_caption)

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.min_pixels
        processor.image_processor.max_pixels = args.max_pixels

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", quantization_config=quant,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    device = model.device
    ids_per_class = letter_token_ids(processor)
    answer_ids = [ids[0] for ids in ids_per_class]  # the bare "A".."D" spelling

    import bitsandbytes as bnb

    optim = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    steps_per_epoch = (len(train) + args.grad_accum - 1) // args.grad_accum
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(
        optim, int(0.03 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs
    )

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    best_acc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train))
        running, seen, correct = 0.0, 0, 0
        bar = tqdm(enumerate(order), total=len(order), desc="Epoch {}/{}".format(epoch, args.epochs))
        for step, idx in bar:
            row = train.iloc[int(idx)]
            gold = m.LABEL_TO_ID[row["Label"]]
            enc = encode(processor, images_dir, row, args.use_caption, device,
                         answer=LETTERS[gold])
            target = answer_ids[gold]
            pos = answer_position(enc["input_ids"], target)

            # Loss on the answer token alone - every prompt token is masked out.
            labels = torch.full_like(enc["input_ids"], -100)
            labels[0, pos] = target

            out = model(**enc, labels=labels)
            (out.loss / args.grad_accum).backward()

            with torch.no_grad():
                logits = out.logits[0, pos - 1]  # position that predicts the answer token
                scores = [max(float(logits[i]) for i in ids) for ids in ids_per_class]
                correct += int(max(range(m.NUM_CLASSES), key=lambda k: scores[k]) == gold)
            running += float(out.loss)
            seen += 1

            if (step + 1) % args.grad_accum == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            bar.set_postfix(loss=running / seen, acc=correct / seen)

        v_true, v_pred = evaluate(model, processor, valid, images_dir, args.use_caption,
                                  ids_per_class, device, "Validation")
        v_acc = float(np.mean(np.array(v_true) == np.array(v_pred)))
        print("Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Acc: {:.2f}%".format(
            epoch, args.epochs, running / seen, 100 * correct / seen, 100 * v_acc))

        if v_acc > best_acc:
            best_acc, best_epoch = v_acc, epoch
            # Kept in CPU memory for the test pass, and on disk so the run is reusable.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in get_peft_model_state_dict(model).items()}
            model.save_pretrained(adapter_dir)
            print("Adapter Saved. ->", adapter_dir)

    print("Best Validation Accuracy: {:.2f}% (epoch {})".format(100 * best_acc, best_epoch))

    print("-" * 32)
    print("Restoring best adapter (epoch {}) for the held-out test set..".format(best_epoch))
    if best_state is not None:
        set_peft_model_state_dict(model, best_state)
    model.eval()

    actual, preds = evaluate(model, processor, test, images_dir, args.use_caption,
                             ids_per_class, device, "Testing")

    m.report_and_save(
        actual, preds, outputs_dir, args.run_name,
        extra={
            "approach": "vlm_qlora_finetune",
            "model": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "use_caption": args.use_caption,
            "best_val_accuracy": best_acc,
            "best_epoch": best_epoch,
            "adapter": adapter_dir,
            "subset": args.subset,
            "seed": args.seed,
            "trained_on_this_dataset": True,
        },
        frame=test,
    )
    print("Total time : {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QLoRA fine-tune a VLM on Bengali aggression memes")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--run_name", type=str, default="vlm_finetuned")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--use_caption", action="store_true")
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_log", action="store_true")

    args = p.parse_args()
    if not args.no_log:
        import _logging

        _logging.start(args.run_name)
    main(args)
