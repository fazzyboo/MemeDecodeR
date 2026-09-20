# Running MAF on MIMOSA — logged-experiment guide

Two additive modules sit on top of the **original** pipeline (`main.py` → `dataset.py` →
`models.py` → `evaluation.py`):

* **`Scripts/experiment_logger.py`** — runs the original trained MAF pipeline and writes
  every metric, prediction and environment detail to a timestamped folder under `Runs/`
  (§1–§5).
* **`Scripts/clip_zeroshot.py`** — a training-free CLIP zero-shot alternative that drops
  BanglaBERT and the training loop entirely, writing the same artefact set (§6b).
* **`Scripts/adaptformer.py` + `Scripts/clip_adaptformer.py`** — AdaptFormer PEFT: an
  adapter in every block of the CLIP vision tower, training ~1% of the weights. On
  ViT-L/14 this beats the paper's MAF using images alone (§6c).
* **`Scripts/block_ablation.py`** — trains one adapter per block in isolation to find which
  block earns it, with head-only and all-blocks reference points (§6d). Read §6d before
  quoting §6c: the head-only control changes how the §6c win should be attributed.
* **`Scripts/maf_siglip.py`** — MAF with SigLIP swapped in for CLIP, plus a CLIP control
  through the same pipeline (§6e). The two encoders tie; preprocessing and seed variance
  matter more than the swap.

No original script was edited. Both modules are purely additive: delete them and the repo
behaves exactly as it did before.

---

## 1. TL;DR

```bash
conda activate memedecoder
cd ~/PR/Bengali-Aggression-Memes/Scripts

# 2-minute sanity check (3 batches per split, 2 epochs)
python experiment_logger.py --max_batches 3 --n_iter 2 --run_name smoke

# the real thing (paper defaults: 5 epochs, bs 16, lr 5e-5, 16 heads)
python experiment_logger.py --n_iter 5 --run_name maf_full_5ep
```

Results land in `Runs/<timestamp>_<run_name>/`. Start with `classification_report.txt`
and `summary.json`.

Long runs — launch detached so an SSH drop or closed terminal does not kill training:

```bash
nohup python experiment_logger.py --n_iter 15 --run_name maf_15ep > /dev/null 2>&1 &
tail -f ../Runs/*_maf_15ep/run.log      # follow progress
```

`run.log` and `console.log` are written live, so you can always reattach to a run with
`tail -f`.

---

## 1b. Baseline already on disk

A full 5-epoch run with the paper's default arguments is already complete:
`Runs/20260920-104922_maf_full_5ep/`. Use it as your reference point.

```
python experiment_logger.py --n_iter 5 --run_name maf_full_5ep
```

5 min 38 s end to end on the RTX 5060 Ti (318 s training, 9.5 s test, 3.7 s data loading).

**Test set (728 memes)** — the logger's numbers and the original `print_metrices`
output agree to every digit:

| | precision | recall | F1 | support |
|---|---|---|---|---|
| NoAg (non-aggressive) | 0.495 | 0.852 | 0.626 | 182 |
| GAg (gendered) | 0.723 | 0.562 | 0.633 | 144 |
| PAg (political) | 0.852 | 0.766 | 0.807 | 128 |
| RAg (religious) | 0.956 | 0.826 | 0.886 | 132 |
| Oth (others) | 0.622 | 0.324 | 0.426 | 142 |
| **accuracy** | | | **0.672** | 728 |
| macro avg | 0.730 | 0.666 | 0.676 | 728 |
| weighted avg | 0.711 | 0.672 | 0.667 | 728 |

Weighted F1 0.6673 · Macro F1 0.6756 · MMAE 0.8477.

**Learning curve** (`epoch_metrics.csv`):

| epoch | train loss | train acc | val acc | val macro-F1 |
|---|---|---|---|---|
| 1 | 1.085 | 56.5% | 65.6% | 0.639 |
| 2 | 0.586 | 78.8% | **68.2%** | 0.686 |
| 3 | 0.331 | 89.0% | 67.1% | 0.668 |
| 4 | 0.185 | 93.8% | 68.2% | **0.687** |
| 5 | 0.122 | 96.3% | 64.5% | 0.654 |

Two things to take from this before you run anything longer:

* **It overfits from epoch 3 on.** Training accuracy reaches 96% while validation
  accuracy peaks at 68% in epoch 2 and then falls. More epochs alone will not help;
  regularization, a working LR decay (see §7.1), or early stopping on macro-F1 will.
* **`Oth` is the weak class** (recall 0.324) and 72 of its 142 memes are predicted
  `NoAg`. That single confusion accounts for most of the gap between macro-F1 and
  the per-class scores of the political/religious classes, which are already strong
  (0.81 / 0.89). Start error analysis there — `test_predictions.csv` has the captions,
  image names and softmax probabilities for exactly those rows.

Note the checkpoint that was evaluated is epoch 2's: `models.py` keeps the first epoch
that achieves the best validation *accuracy*, and epochs 2 and 4 tie at 68.2256%.

---

## 2. Environment

Use the existing conda env — it already has every dependency:

```bash
conda activate memedecoder
```

| | |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.17.0 |
| pandas / numpy / scikit-learn | 3.0.5 / 2.5.2 / 1.9.1 |
| GPU | RTX 5060 Ti, 16 GB (peak use at batch 16: **≈ 3.8 GB**) |

`requirements.txt` in this repo pins the *2024* stack (torch 2.4, transformers 4.44).
Do **not** `pip install -r requirements.txt` into `memedecoder` — those pins predate the
Blackwell GPU and would replace working CUDA 12.8 wheels with ones that cannot run on
this card.

### The one incompatibility, and how it is handled

`dataset.py` and `models.py` both do:

```python
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
```

`AdamW` was deprecated in transformers 4.x and **removed in 5.x**, so on this env
`python main.py` fails at import with:

```
ImportError: cannot import name 'AdamW' from 'transformers'
```

Nothing in the repo actually uses that import — `models.py` builds `torch.optim.AdamW`
itself. Rather than edit the original files, `experiment_logger.py` re-attaches the alias
(`transformers.AdamW = torch.optim.AdamW`) right before each import.

One subtlety worth knowing: transformers 5 **replaces its own entry in `sys.modules`**
the first time a lazy attribute is resolved, which silently drops the alias. That is why
the shim is re-applied before *every* pipeline import rather than once at startup.

If you ever want plain `python main.py` to work, delete the two words `AdamW,` from line
13 of `dataset.py` and line 8 of `models.py`. That is the entire fix.

### Pretrained weights

Already downloaded and cached; no further network access is needed:

* CLIP ViT-B/32 → `~/.cache/clip/ViT-B-32.pt` (338 MB)
* `sagorsarker/bangla-bert-base` → `~/.cache/huggingface/` (164 M params)

To run fully offline: `export HF_HUB_OFFLINE=1`.

---

## 3. Command-line arguments

Original arguments, unchanged in name, dest and default:

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `Dataset` | dataset folder, relative to repo root |
| `--max_len` | `70` | max caption length in BERT tokens |
| `--batch_size` | `16` | training/validation batch size |
| `--heads` | `16` | attention heads in the fusion block |
| `--n_iter` | `5` | epochs |
| `--lrate` | `5e-5` | learning rate |
| `--model` | *run folder* | checkpoint directory |

Added by the logger:

| Flag | Default | Meaning |
|---|---|---|
| `--run_name` | `maf` | tag in the run folder name |
| `--log_dir` | `Runs` | where run folders are created |
| `--seed` | `42` | seeds `random`, `numpy` **and** `torch` |
| `--log_every` | `10` | batches between `metrics.jsonl` batch events |
| `--resource_interval` | `15` | seconds between GPU/CPU samples |
| `--max_batches` | `0` | truncate every loader to N batches (smoke test) |

Note on `--model`: the original default was the shared folder `Saved_Models`, so every
run overwrote the previous `maf_model.pth`. The logger defaults to
`Runs/<run>/checkpoints/` instead, so each run keeps its own weights. Pass
`--model Saved_Models` for the original behaviour. **Each checkpoint is ~1 GB** (BERT and
the fusion head are trainable), so prune old runs if disk gets tight.

---

## 4. What a run directory contains

```
Runs/20260920-104922_maf_full_5ep/
├── console.log             verbatim stdout+stderr, tqdm bars included
├── run.log                 timestamped structured log — read this first
├── config.json             every hyper-parameter, resolved paths, seed, exact command
├── environment.json        python/torch/CUDA/GPU/git commit/full pip freeze
├── dataset_stats.json      split sizes, class counts, caption lengths, batch counts
├── model_summary.txt       per-submodule parameter counts + full module tree
├── batch_metrics.csv       per training batch: loss, acc, lr, samples/s, GPU MB
├── epoch_metrics.csv       per epoch: train loss/acc, val acc/macro-F1/MMAE, timing
├── val_reports/epoch_NNN.json   full per-class validation report for every epoch
├── metrics.jsonl           every event above, one JSON object per line
├── resources.csv           GPU util/mem/temp + process CPU/RAM every 15 s
├── test_predictions.csv    per test meme: image, caption, true, pred, 5 softmax probs
├── classification_report.txt / .json
├── confusion_matrix.csv / .png
├── summary.json            headline numbers for the whole run
└── checkpoints/maf_model.pth
```

`batch_metrics.csv` loss values are recomputed with `F.cross_entropy` on the exact same
logits and labels the training loop used, so they match the loop's own numbers to the
digit — verified against the original per-epoch prints in `console.log`.

---

## 5. Reading the results

```python
import json, pandas as pd

run = "Runs/20260920-104922_maf_full_5ep"

print(open(f"{run}/classification_report.txt").read())

ep = pd.read_csv(f"{run}/epoch_metrics.csv")        # learning curve
print(ep[["epoch", "train_loss", "train_acc", "val_acc", "val_macro_f1"]])

pred = pd.read_csv(f"{run}/test_predictions.csv")   # error analysis
print(pred[~pred.correct].sort_values("confidence", ascending=False).head(20)
          [["image_name", "true_label", "pred_label", "confidence"]])
```

Compare every run you have ever done:

```python
import glob, json, pandas as pd
rows = []
for f in sorted(glob.glob("Runs/*/summary.json")):
    s = json.load(open(f))
    rows.append({"run": s["run_dir"].split("/")[-1],
                 "epochs": s["config"]["epochs"],
                 "lr": s["config"]["learning_rate"],
                 "heads": s["config"]["num_heads"],
                 "best_val": s["best_val_accuracy"],
                 "test_acc": s["test"]["accuracy"],
                 "test_wF1": s["test"]["weighted_f1"],
                 "test_mF1": s["test"]["macro_f1"]})
print(pd.DataFrame(rows).sort_values("test_wF1", ascending=False))
```

Label encoding used everywhere (from `dataset.py`):

| id | short | full |
|---|---|---|
| 0 | NoAg | non-aggressive |
| 1 | GAg | gendered aggression |
| 2 | PAg | political aggression |
| 3 | RAg | religious aggression |
| 4 | Oth | others |

---

## 6. Dataset

4,848 memes, all RGB JPEG/PNG, every `image_name` in the three CSVs resolves to a file in
`Dataset/Img/` — verified, nothing missing, no null captions.

| split | memes | NoAg | GAg | PAg | RAg | Oth |
|---|---|---|---|---|---|---|
| train | 3,393 | 846 | 672 | 597 | 618 | 660 |
| validation | 727 | 181 | 144 | 128 | 133 | 141 |
| test | 728 | 182 | 144 | 128 | 132 | 142 |

---

## 6b. CLIP zero-shot (no training at all)

`Scripts/clip_zeroshot.py` replaces the whole trained pipeline — no BanglaBERT, no
gradients, no checkpoint — with prompt-based inference:

```
score(meme, class) = cos( CLIP_image(meme), mean_t CLIP_text(prompt_t(class)) )
```

It is additive in the same way as the logger: it imports nothing from `main.py`,
`models.py` or `dataset.py`, and reuses `experiment_logger.py` for run folders and
metrics, so zero-shot runs drop into the same comparison tooling as trained runs.

```bash
conda activate memedecoder && cd Scripts

python clip_zeroshot.py                                    # image-only, ViT-B/32, test
python clip_zeroshot.py --mode caption                     # Bengali caption vs class prompts
python clip_zeroshot.py --mode fused --alpha_sweep         # tune alpha on val, apply to test
python clip_zeroshot.py --backbone ViT-L/14 --split test validation
python clip_zeroshot.py --prompts my_prompts.json          # prompt engineering
```

A full run takes **~13 seconds** (vs 5.5 minutes to train MAF). Image features are cached
per (backbone, split) in `.clip_cache/`, so re-running with new prompts takes seconds.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--backbone` | `ViT-B/32` | any `clip.available_models()` entry |
| `--split` | `test` | one or more of `train`, `validation`, `test`, `all` |
| `--mode` | `image` | `image`, `caption`, or `fused` |
| `--alpha` | `0.5` | fused: weight on the image similarity |
| `--alpha_sweep` | off | fused: pick alpha by **validation** macro-F1, then apply |
| `--prompts` | built-in bank | JSON `{class_name: [prompt, ...]}` for all 5 classes |
| `--max_samples` | `0` | first N rows per split (smoke test) |
| `--no_cache` | off | ignore/skip the feature cache |

Each run writes `classification_report.txt/.json`, `predictions.csv` (with per-class
softmax probabilities **and** raw cosine similarities, plus a `margin` column for
confidence analysis), `confusion_matrix.csv/.png`, `prompts_used.json`,
`prompt_similarity.json`, `tokenization_stats_<split>.json` and `summary.json`.

### Results (test split, 728 memes)

| approach | accuracy | weighted F1 | macro F1 | cost |
|---|---|---|---|---|
| random 5-way | 0.200 | — | — | — |
| majority class | 0.250 | 0.080 | 0.080 | — |
| zero-shot, caption (ViT-B/32) | 0.251 | 0.110 | 0.090 | 13 s |
| zero-shot, image (ViT-B/32) | 0.349 | 0.278 | 0.296 | 13 s |
| zero-shot, fused (ViT-B/32, α=0.8 tuned on val) | 0.348 | 0.310 | 0.323 | 13 s |
| **zero-shot, image (ViT-L/14)** | **0.422** | **0.357** | **0.378** | 25 s |
| trained MAF (5 epochs) | **0.672** | **0.667** | **0.676** | 5 min 38 s |

Per-class F1:

| | NoAg | GAg | PAg | RAg | Oth |
|---|---|---|---|---|---|
| zero-shot ViT-B/32 | 0.059 | 0.428 | 0.437 | 0.467 | 0.089 |
| zero-shot ViT-L/14 | 0.061 | 0.608 | 0.529 | 0.468 | 0.226 |
| trained MAF | 0.626 | 0.633 | 0.807 | 0.886 | 0.426 |

**Backbone size matters a lot here** — ViT-L/14 adds +7.3 accuracy points and +8.2 macro-F1
over ViT-B/32 for free (25 s, no training), with the biggest jump on gendered aggression
(F1 0.428 → 0.608). It also separates the class prompts better: mean inter-class cosine
0.898 vs 0.949. Try `--backbone ViT-L/14@336px` next; the first run downloads ~900 MB, after
which features are cached.

**Zero-shot is well above chance but roughly half of the trained model.** Read that as a
measurement of how much of this task is visual and English-describable, not as a failure
of the implementation. Four things the artifacts show clearly:

Note that the `NoAg` failure (F1 ~0.06) and the everything-is-aggressive skew persist at
both backbone sizes, so they are a property of the prompting setup, not of model capacity.

1. **The caption branch is dead, and predictably so.** OpenAI CLIP's text tower is
   English-only and its BPE shatters Bengali into ~135 near-per-character tokens, so
   **77% of MIMOSA captions overflow the 77-token context window** before encoding
   (`tokenization_stats_test.json`). Caption mode scores 0.251 accuracy against a 0.250
   majority baseline — i.e. exactly nothing. This is the single biggest structural reason
   zero-shot trails MAF: the trained pipeline reads Bengali through BanglaBERT, and this
   one cannot read it at all.
2. **It calls almost everything aggressive.** Collapsed to binary aggressive
   vs non-aggressive, recall is 0.974 but precision only 0.751 (F1 0.849). It predicts
   `NoAg` just 20 times out of 728. So CLIP *is* picking up hostility-correlated imagery;
   it just has no calibrated notion of "ordinary meme".
3. **Political and religious targets are visually legible; gender and 'others' are not.**
   PAg recall 0.805 and RAg recall 0.712 — politicians' faces and religious dress/buildings
   are exactly what CLIP was pretrained on. `Oth` gets F1 0.089, which is unsurprising:
   it is a catch-all defined by what it is *not*, and no prompt can describe that.
4. **The prompts barely separate.** Mean inter-class cosine similarity between the five
   class embeddings is **0.949** for ViT-B/32 and 0.898 for ViT-L/14
   (`prompt_similarity.json`). CLIP's text embeddings for
   five sentences that all start "a meme..." are nearly parallel, so the decision is made
   in a very narrow angular margin. Lowering that number is the most direct lever you have
   — check it after every prompt edit.

### Where to take it next

* **Prompt engineering** is the cheapest lever: edit a JSON, re-run in seconds against
  cached features. Watch `offdiag_mean` in `prompt_similarity.json`.
* **Prior calibration.** The skew in (2) is a known zero-shot failure mode with a known
  fix: z-score each class's similarity column across the evaluated set, or subtract a
  per-class bias fitted on validation. This typically buys a lot when predictions collapse
  onto two classes. Note the transductive version uses test-set statistics, so fit the
  bias on validation if you plan to report it.
* **A linear probe on cached CLIP features** (`sklearn.linear_model.LogisticRegression`
  over `.clip_cache/*.npz`) is ~20 lines, trains in seconds, and is the honest middle
  point between zero-shot and full MAF fine-tuning.
* **A multilingual text tower** (M-CLIP, or `sentence-transformers` LaBSE aligned to CLIP)
  is the principled fix for the dead caption branch, since it would actually encode
  Bengali. That needs a new pip install into `memedecoder`.

---

## 6c. AdaptFormer PEFT on the CLIP vision tower

`Scripts/adaptformer.py` implements AdaptFormer (Chen et al., NeurIPS 2022) and injects it
into **every** residual block of a CLIP ViT; `Scripts/clip_adaptformer.py` is the training
runner. A stock CLIP block is pre-norm:

```
x = x + Attention(ln_1(x))
x = x + MLP(ln_2(x))
```

AdaptFormer turns the MLP sub-layer into *AdaptMLP* — the frozen MLP plus a **parallel**
trainable bottleneck fed the same input:

```
x = x + Attention(ln_1(x))
x = x + MLP(ln_2(x)) + s * W_up( ReLU( W_down(x) ) )        W_down: d->r, W_up: r->d
```

Defaults follow the official implementation: r=64, s=0.1, parallel placement, no adapter
LayerNorm, LoRA-style init. Everything else — attention, MLP, norms, embeddings — stays
frozen, so **1.34% of ViT-B/32 and 1.03% of ViT-L/14 is trainable**.

```bash
conda activate memedecoder && cd Scripts

python clip_adaptformer.py --epochs 10 --batch_size 64                  # ViT-B/32
python clip_adaptformer.py --backbone ViT-L/14 --epochs 10 --batch_size 32
python clip_adaptformer.py --bottleneck 8 --run_name r8                 # capacity ablation
python clip_adaptformer.py --blocks 8,9,10,11 --run_name last4          # placement ablation
python clip_adaptformer.py --eval_only ../Runs/<run>/adapter_head.pt    # score a saved adapter
```

### Two properties worth knowing

**It starts as an exact no-op.** `W_up` is zero-initialised, so the adapter branch
contributes literally nothing at step 0 — verified as a bitwise-identical forward pass
(max abs diff 0.0 over a random batch). Combined with the default `--head zeroshot`, which
initialises the classifier from the same prompt embeddings `clip_zeroshot.py` uses, the
epoch-0 row of `epoch_metrics.csv` *is* the zero-shot baseline, measured by this code path.
Checked end to end: with `--no_amp`, epoch 0 reports 36.59% / macro-F1 0.3197 on
validation, matching the zero-shot run to four decimals. The default bf16 autocast shifts
~10 of 727 borderline samples, so epoch 0 reads 36.18% there — use `--no_amp` if you want
the identity to hold exactly.

**The checkpoint is tiny.** Only trained tensors are saved: 4.6 MB (ViT-B/32) and 12.7 MB
(ViT-L/14), against 1021 MB for the MAF checkpoint. `--eval_only` reloads one and skips
straight to evaluation.

### Results (test split, 728 memes)

| approach | trained params | acc | weighted F1 | macro F1 | MMAE ↓ | train time | ckpt |
|---|---|---|---|---|---|---|---|
| zero-shot ViT-B/32 | 0 | 0.349 | 0.278 | 0.296 | 1.184 | — | — |
| zero-shot ViT-L/14 | 0 | 0.422 | 0.357 | 0.378 | 1.079 | — | — |
| **AdaptFormer ViT-B/32** (r=64) | 1.19 M (1.34%) | 0.644 | 0.645 | 0.653 | 0.835 | 2 min 26 s | 4.6 MB |
| MAF — BERT+CLIP, image **+ text** | 167.4 M | 0.672 | 0.667 | 0.676 | 0.848 | 5 min 18 s | 1021 MB |
| **AdaptFormer ViT-L/14** (r=64) | 3.18 M (1.03%) | **0.727** | **0.729** | **0.738** | **0.659** | ~10 min | 12.7 MB |

Per-class F1:

| | NoAg | GAg | PAg | RAg | Oth |
|---|---|---|---|---|---|
| zero-shot ViT-L/14 | 0.061 | 0.608 | 0.529 | 0.468 | 0.226 |
| AdaptFormer ViT-B/32 | 0.586 | 0.614 | 0.776 | 0.779 | 0.511 |
| MAF (image + text) | 0.626 | 0.633 | 0.807 | 0.886 | 0.426 |
| AdaptFormer ViT-L/14 | 0.654 | 0.712 | 0.854 | 0.857 | **0.612** |

**The headline: AdaptFormer on ViT-L/14 beats the paper's MAF on every metric** — +5.5
accuracy points, +6.2 weighted F1, +6.2 macro F1, and MMAE down from 0.848 to 0.659 —
while training **1.03% of the parameters** and, notably, **using only the image**. It never
sees the Bengali caption at all, whereas MAF reads it through BanglaBERT.

Take that as a statement about where the signal is, not as proof that text is useless. Two
readings are consistent with it: MIMOSA's aggression targets are more visually determined
than the multimodal framing suggests, and/or MAF's 5-epoch training at an effectively
constant learning rate (§7.1) leaves performance on the table. Both are testable with the
tooling here.

Other things the runs show:

* **The gains concentrate exactly where MAF was weakest.** `Oth` goes 0.426 → 0.612 F1 and
  `NoAg` 0.626 → 0.654. These are the two classes zero-shot could not touch at all
  (`NoAg` F1 0.061), so the adapters are learning precisely the distinctions that generic
  CLIP prompting cannot express.
* **It overfits fast, and worse than MAF.** Training accuracy hits 99.9% by epoch 5–6 in
  both runs. ViT-L/14's best validation macro-F1 is at **epoch 2** — the remaining eight
  epochs are wasted. Start from `--epochs 3` for L/14 and spend the saved time on seeds.
* **Backbone beats budget.** ViT-L/14 with r=64 gains far more than any r you could give
  ViT-B/32. If you have one run to spend, spend it on the bigger tower.

### Ablations worth running

```bash
for r in 4 8 16 32 64 128; do
  python clip_adaptformer.py --backbone ViT-L/14 --epochs 3 --batch_size 32 \
         --bottleneck $r --run_name l14_r$r
done

python clip_adaptformer.py --backbone ViT-L/14 --epochs 3 --blocks 18,19,20,21,22,23 \
       --run_name l14_last6                       # does every block actually earn its adapter?
python clip_adaptformer.py --backbone ViT-L/14 --epochs 3 --head linear --run_name l14_linear
python clip_adaptformer.py --backbone ViT-L/14 --epochs 3 --scalar learnable_scalar \
       --run_name l14_learned_s
python clip_adaptformer.py --backbone ViT-L/14 --epochs 3 --class_weights --run_name l14_cw
```

The obvious next model: put AdaptFormer's visual branch back together with a Bengali text
encoder, i.e. MAF's fusion over an adapted CLIP tower instead of a frozen one.
`adaptformer.inject_adaptformer(clip_model, ...)` works on any CLIP ViT, so it drops into
`models.py`'s `clip_model` with one line if you want to try it there.

### A note on running these concurrently

Don't. This box has 15 GB of RAM and 16 GB of VRAM; a ViT-L/14 PEFT run alone uses ~10 GB
VRAM and ~2.9 GB RSS. I lost the tail of one L/14 run to the host OOM killer by running
probe scripts alongside it — the process died silently after writing its report but before
`summary.json` (a CUDA OOM would have raised and been logged; a host OOM kill is silent).
Everything was recoverable from `adapter_head.pt` via `--eval_only`, which is one reason
that flag exists, but the cheaper fix is to run one job at a time.

---

## 6d. Per-block ablation: which block earns its adapter?

`Scripts/block_ablation.py` trains one model per block with AdaptFormer attached to **that
block only**, then aggregates. Two reference points come for free:

* **head-only** — no adapters at all, only the classifier head trains (2,561 params). The
  floor: what frozen CLIP features alone are worth.
* **all-blocks** — every block adapted. The ceiling from §6c.

```bash
python block_ablation.py                                     # ViT-B/32, 12 blocks, 10 epochs
python block_ablation.py --blocks 0,8 --seeds 0,1,2          # error bars on chosen blocks
python block_ablation.py --backbone ViT-L/14 --epochs 3 --batch_size 32
python block_ablation.py --aggregate_only Runs/<sweep-dir>   # re-aggregate, train nothing
```

Child runs are separate processes (one crash can't kill the sweep, GPU memory is released
between runs), each a normal `clip_adaptformer.py` run with the full artefact set. The
sweep folder holds `block_ablation.csv/.txt/.png` and `records.json`.

### Results — ViT-B/32, r=64, 10 epochs

| block adapted | trainable | macro F1 | ± | seeds |
|---|---|---|---|---|
| 0 | 101,697 | 0.6325 | 0.0077 | 3 |
| 1 | 101,697 | 0.6444 | — | 1 |
| 2 | 101,697 | 0.6451 | — | 1 |
| 3 | 101,697 | 0.6552 | — | 1 |
| 4 | 101,697 | 0.6464 | — | 1 |
| 5 | 101,697 | 0.6579 | — | 1 |
| 6 | 101,697 | 0.6510 | — | 1 |
| **7** | 101,697 | **0.6655** | — | 1 |
| 8 | 101,697 | 0.6596 | 0.0104 | 3 |
| 9 | 101,697 | 0.6614 | — | 1 |
| 10 | 101,697 | 0.6445 | — | 1 |
| 11 | 101,697 | 0.6326 | — | 1 |
| *head only (floor)* | 2,561 | 0.6389 | 0.0057 | 3 |
| *all 12 blocks (ceiling)* | 1,192,193 | 0.6559 | 0.0032 | 3 |

Seed-to-seed std over the four repeated configs is **0.0067**, so treat gaps below about
**0.013 macro F1 as noise**. Ten blocks have a single seed and carry that same uncertainty.

**1. Adapter depth follows a clear inverted U.** Blocks 7–9 are best (0.660–0.666), the
first and last blocks are worst (0.633), and the 0.033 spread across depth is ~2.5× the
noise band — so the *shape* is real even though adjacent blocks are not separable from
each other. Early blocks encode generic edges and textures that need no task adaptation;
the last block is too close to the output to reshape anything downstream. The useful
capacity sits two-thirds of the way up.

**2. One well-chosen block matches all twelve.** Block 7 at 0.6655 vs all-blocks at
0.6559 ± 0.0032 — a +0.0096 gap that sits *inside* the noise band, so the honest reading
is that they tie, at **1/12 of the adapter budget** (101 K vs 1.19 M params). Adapting
every block, the paper's setting, buys nothing here. This is the practical takeaway: if
you adopt AdaptFormer on this dataset, adapt block 7–9 and skip the rest.

*(A single-seed run initially showed block 8 apparently beating all-blocks by +0.017. It
did not survive three seeds — block 8's mean is 0.6596 ± 0.0104. This is exactly the gap
the noise band exists to catch.)*

**3. Adapting the earliest block is worse than no adapters at all.** Block 0 scores
0.6325 ± 0.0077 against a 0.6389 ± 0.0057 head-only floor. Within ~1σ, so call it "gives
nothing" rather than "hurts", but it certainly does not help.

### The control that reframes §6c

Head-only runs — a cosine probe on frozen CLIP features, nothing else trained:

| model | trained params | acc | weighted F1 | macro F1 |
|---|---|---|---|---|
| ViT-B/32 head-only | 2,561 | 0.628 | 0.630 | 0.639 |
| ViT-B/32 + AdaptFormer (all blocks) | 1.19 M | 0.647 | 0.648 | 0.656 |
| **ViT-L/14 head-only** | **3,841** | **0.695** | **0.695** | **0.702** |
| ViT-L/14 + AdaptFormer (all blocks) | 3.18 M | 0.727 | 0.729 | 0.738 |
| MAF — BERT+CLIP, image **+ text** | 167.4 M | 0.672 | 0.667 | 0.676 |

Two things follow, and the second qualifies what §6c claims.

* **AdaptFormer's gain is real but modest, and scale-dependent.** On ViT-B/32 it adds
  +0.017 macro F1 over head-only — barely outside the 0.013 noise band. On ViT-L/14 it
  adds **+0.036**, comfortably outside it. The adapters earn their place on the bigger
  tower, much less so on the smaller one.
* **A 3,841-parameter head on frozen ViT-L/14 features already beats MAF** (0.695 vs
  0.667 weighted F1). So the headline in §6c — "AdaptFormer beats the paper's model" — is
  true but attributes the win to the wrong component. **Most of it comes from the CLIP
  ViT-L/14 features themselves**, not from AdaptFormer and not from PEFT. AdaptFormer
  contributes the last third of the margin over MAF; the backbone contributes the first
  two thirds. If you report these numbers, report the head-only control alongside them.

### Worth running next

```bash
# the curve at ViT-L/14 scale -- 24 blocks, the peak may sit elsewhere
python block_ablation.py --backbone ViT-L/14 --epochs 3 --batch_size 32 --seeds 0,1,2

# fill in seeds for the 10 single-seed ViT-B/32 blocks
python block_ablation.py --seeds 0,1 --sweep_name block_ablation_fill

# does a contiguous mid-late band beat the single best block?
python clip_adaptformer.py --blocks 7,8,9 --epochs 10 --run_name b789
```

---

## 6e. SigLIP in place of CLIP inside MAF

`Scripts/maf_siglip.py` keeps MAF's architecture exactly — frozen vision tower,
BanglaBERT, `MultiheadAttention(query=image, key=text, value=image)`, concat of
[attention, image, text], mean over the sequence, the same 2304→128→5 head, AdamW
lr 5e-5 / wd 0.01, linear schedule stepped once per epoch, best checkpoint by validation
accuracy, 5 epochs / batch 16 / 16 heads / max_len 70 — and swaps **only** the vision
encoder for SigLIP (`google/siglip-base-patch16-224`, 92.9 M params, hidden 768).

Two conveniences of the swap: SigLIP's width is 768, exactly BanglaBERT's, so MAF's
512→768 widening becomes a same-width map; and SigLIP exposes a real 196-token patch
sequence where CLIP gives one pooled vector that MAF broadcasts 70 times.

```bash
python maf_siglip.py                                            # SigLIP, paper defaults
python maf_siglip.py --vision tokens                            # use the 196 patch tokens
python maf_siglip.py --imagenet_norm                            # normalisation control
python maf_siglip.py --vision_model openai/clip-vit-base-patch32   # the CLIP control
```

The `--vision_model openai/clip-*` path runs CLIP through this **same** pipeline, which is
what makes the comparison meaningful. It uses `CLIPVisionModelWithProjection`, so `pooled`
returns the identical 512-d projected embedding the original MAF consumes — and it
reproduces the original's parameter counts exactly (255,297,797 total / 167,448,581
trainable / 87,849,216 frozen), confirming the architecture is unchanged.

### Results — test set, 728 memes

| vision encoder | normalisation | acc | weighted F1 | macro F1 | seeds |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | CLIP's own | 0.7102 ± 0.0143 | 0.7117 ± 0.0145 | 0.7197 ± 0.0156 | 3 |
| **SigLIP base/16** | **SigLIP's own** | **0.7161 ± 0.0163** | **0.7151 ± 0.0173** | **0.7226 ± 0.0174** | 3 |
| CLIP ViT-B/32 | ImageNet | 0.7225 | 0.7238 | 0.7311 | 1 |
| SigLIP base/16 | ImageNet | 0.6896 | 0.6845 | 0.6914 | 1 |
| SigLIP base/16, 196 tokens | SigLIP's own | 0.7005 | 0.7070 | 0.7166 | 1 |
| *original `models.py` run* | ImageNet | 0.6717 | 0.6673 | 0.6756 | 1 |

**1. SigLIP and CLIP are a tie.** +0.0034 weighted F1 for SigLIP, against a pooled
seed-to-seed sd of **0.0159** — a 0.2σ difference. Per-seed weighted F1 was
CLIP {0.7271, 0.6982, 0.7098} vs SigLIP {0.7347, 0.7085, 0.7021}; the distributions
overlap almost completely. On this dataset, at this scale, swapping the encoder is not
what moves the number. (SigLIP is ~35% faster per epoch, 41 s vs 64 s, which is a real if
undramatic reason to prefer it.)

**2. Normalisation matters for SigLIP but not for CLIP — and now we know why.** Feeding
SigLIP the ImageNet constants costs ~0.030 weighted F1 (0.6845 vs 0.7151); doing the same
to CLIP costs nothing measurable. The reason is arithmetic: CLIP's mean
[0.481, 0.458, 0.408] is almost exactly ImageNet's [0.485, 0.456, 0.406], whereas SigLIP's
is [0.5, 0.5, 0.5] with std [0.5, 0.5, 0.5]. So §7.3's complaint about `dataset.py` using
ImageNet normalisation for CLIP was right in principle but nearly harmless in practice —
it would have been a genuine bug had the original used SigLIP. **If you swap encoders,
swap the preprocessing with it.**

**3. Real patch tokens are not better than the broadcast vector.** `--vision tokens` scores
0.7070 against 0.7151 pooled (single seed, so within noise, but certainly no gain). MAF's
attention uses the image as both query and value with text only as key, so richer visual
tokens are averaged back down by `fusion.mean(1)` regardless. Exploiting a patch sequence
would need the fusion redesigned, not just fed more tokens.

**4. This recipe is noisy — sd ≈ 0.016 weighted F1.** That is more than twice the 0.007
seen in the PEFT setup (§6d), because 5 epochs plus best-checkpoint-by-validation-accuracy
on 727 samples is a high-variance selection rule. **Treat any MAF-recipe difference below
~0.03 weighted F1 as unmeasured**, including single-run numbers reported from this recipe.

### An open discrepancy, stated plainly

The CLIP control in this pipeline scores 0.7117 ± 0.0145 weighted F1, while the original
`models.py` run scores 0.6673 — below all three of my CLIP seeds. Same architecture
(verified by parameter count), same hyperparameters, same normalisation in the
`--imagenet_norm` cell, and near-identical training-loss curves
(1.085/0.586/0.331/0.185/0.122 original vs 1.125/0.633/0.369/0.230/0.135 here). The
selection is what differs: the original's validation accuracy peaked at epoch 2, mine at
epoch 4.

Seed variance plausibly covers part of a ~0.045 gap but probably not all of it. Two
candidates remain untested:

* the OpenAI `clip` package loads **fp16** weights and casts them to fp32, so the original's
  frozen features come from fp16-rounded weights, while `transformers` loads fp32 directly;
* `main.py` seeds only numpy, leaving torch initialisation and shuffling unseeded, whereas
  these runs seed python, numpy and torch.

Until that is resolved, **compare within one pipeline**. The CLIP-vs-SigLIP conclusion
above is safe because both arms ran through identical code; comparing either against the
0.6673 figure is not.

### Next

```bash
for s in 2 3 4; do python maf_siglip.py --seed $s --run_name siglip_s$s; done   # tighten the CI
python maf_siglip.py --scheduler per_batch --n_iter 15 --select_on macro_f1     # fix §7.1 too
python maf_siglip.py --vision_model google/siglip-so400m-patch14-384 --batch_size 8
```

`google/siglip2-*` would need transformers ≥ 4.49; this environment has 4.44.2.

---

## 7. Things I noticed in the original code

Observations only — **nothing below has been changed**. They are here because they affect
how you read the numbers, and several are cheap experiments.

1. **The LR scheduler is stepped once per epoch, not per batch.** `models.py` builds
   `get_linear_schedule_with_warmup(..., num_training_steps=epochs * len(train_loader))`
   — 1,065 steps for a 5-epoch run — but calls `lr_scheduler.step()` at the end of each
   epoch. After 5 epochs the LR has decayed by 5/1065, i.e. it is still ~99.5% of its
   initial value. Effectively this trains at a constant LR. Moving `.step()` inside the
   batch loop is a one-line change and a real experiment.
2. **Two CLIP models are loaded.** `dataset.py` loads CLIP and calls `.half()` on it, then
   never uses it; `models.py` loads its own copy and uses the `.float()` visual tower.
   Costs startup time and VRAM, changes no results.
3. **Image normalization is ImageNet's, not CLIP's.** `dataset.py` normalizes with
   mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`, while the frozen CLIP
   tower was trained with mean `[0.4815, 0.4578, 0.4082]` / std `[0.2686, 0.2613, 0.2758]`.
   The `preprocess` transform returned by `clip.load` is assigned but never used.
   **Measured in §6e: this costs essentially nothing**, because CLIP's mean is almost
   exactly ImageNet's. I originally called this the highest-value fix; it is not. It
   becomes a real bug the moment you swap in an encoder whose constants differ — SigLIP
   loses ~0.030 weighted F1 to it.
4. **The visual branch carries one vector, broadcast 70 times.** CLIP gives a single
   512-d embedding per image; `adaptive_avg_pool1d(..., 70)` repeats it across the text
   sequence length. So the attention block attends over 70 identical visual tokens.
5. **`test_loader` batch size is hard-coded to 4** in `dataset.py`, ignoring
   `--batch_size`. Harmless, just slower at test time.
6. **Class imbalance is never used.** `compute_class_weight` is imported in `dataset.py`
   but never called; the loss is unweighted `CrossEntropyLoss`. The splits are only mildly
   imbalanced, so this matters more for macro-F1 than accuracy.
7. **`models.evaluation` casts test labels to `float`** before returning them. sklearn and
   imblearn both cope, so the original report is correct; the logger casts to `int` for its
   own artifacts regardless.
8. **The tqdm postfix is slightly wrong**: it divides the running totals by `t.n + 1`,
   which lags the true batch count by one. The end-of-epoch print is correct, and
   `batch_metrics.csv` is exact.
9. **Only validation *accuracy* selects the best checkpoint**, not macro-F1 — relevant
   because the paper headlines weighted/macro F1. Per-epoch macro-F1 is in
   `epoch_metrics.csv` if you want to pick differently.

---

## 8. Suggested next experiments

```bash
# longer schedule
python experiment_logger.py --n_iter 15 --run_name lr5e5_15ep

# learning-rate sweep
for lr in 1e-5 2e-5 5e-5; do
  python experiment_logger.py --n_iter 10 --lrate $lr --run_name lr$lr
done

# attention-head ablation
for h in 4 8 12 16; do
  python experiment_logger.py --n_iter 5 --heads $h --run_name heads$h
done

# seed variance (report mean ± std over these)
for s in 0 1 2; do
  python experiment_logger.py --n_iter 5 --seed $s --run_name seed$s
done
```

Every run is self-describing, so the comparison snippet in §5 will pick them all up.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'AdamW'` | you ran `main.py` directly; use `experiment_logger.py` (see §2) |
| CUDA out of memory | `--batch_size 8` (peak at bs 16 is ~3.8 GB of 16 GB, so unlikely) |
| Run died, want the partial logs | they are already on disk — `run.log`, `batch_metrics.csv` and `epoch_metrics.csv` are flushed line by line; a crash also writes `traceback.txt` |
| Hangs at startup | it is downloading CLIP/BERT weights; check network or set `HF_HUB_OFFLINE=1` once cached |
| Disk filling up | each run keeps a ~1 GB checkpoint: `rm -rf Runs/<old-run>/checkpoints` |
| `MMAE: null` in a smoke run | `macro_averaged_mean_absolute_error` needs all 5 classes present; a truncated loader will not have them. Full runs are fine. |

---

## 10. Reproducing a past run

`config.json` in every run folder stores the exact command line:

```bash
python -c "import json;print(json.load(open('Runs/<run>/config.json'))['command'])"
```

`--seed` seeds `random`, `numpy` and `torch`, but cuDNN kernels are not forced into
deterministic mode, so expect small run-to-run differences on identical seeds. Treat
single-run differences under ~1 point of F1 as noise, and use the seed sweep in §8 before
claiming an improvement.
