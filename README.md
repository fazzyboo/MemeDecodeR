# MAF Replication on the MemeDecode Dataset

A replication of **"A Multimodal Framework to Detect Target Aware Aggression in Memes"**
(Ahsan et al., EACL 2024) — the **MAF** (Multimodal Attentive Fusion) model — trained and
evaluated on *our* meme dataset instead of the authors' MIMOSA corpus.

- Paper: `../Published Work/Actual_Paper_Regarding_meme decode_2024.eacl-long.153.pdf`
- Original code: `../Published Work/Bengali-Aggression-Memes`
- Our data: `../Train` (2,903 memes) and `../Test` (400 labelled memes)

---

## 1. What had to be rebuilt, and why

The original repository ships **no OCR stage**. This is easy to miss: MAF is a multimodal
model that consumes an image *and its caption*, and the released MIMOSA CSVs already
contain finished captions, so the code simply reads a `Captions` column and never asks
where it came from.

Our dataset has images and labels but **no caption text at all**. The paper explains where
MIMOSA's captions came from (Sec. 3.2):

> "Afterward, we extract the meme caption using an OCR [pytesseract]. However, we manually
> checked the extracted captions to correct any missing words and spelling as OCR in
> Bengali is not well-established."

So replicating MAF on our data required rebuilding that missing stage first. The pipeline
here is therefore three stages, where the original had one:

| Stage | Script | Produces |
|-------|--------|----------|
| 1. OCR | `Scripts/ocr_captions.py` | `Dataset/captions_raw.csv` — a Bengali caption per meme |
| 1b. Denoise *(optional)* | `Scripts/clean_captions.py` | `Dataset/captions_clean.csv` — automated stand-in for the paper's manual correction |
| 2. Splits | `Scripts/prepare_dataset.py` | `training_set.csv`, `validation_set.csv`, `testing_set.csv` |
| 3. Train + evaluate | `Scripts/main.py` | `Outputs/results_*.json`, `Outputs/predictions_*.csv` |
| 4. Submission | `Scripts/generate_submission.py` (from a checkpoint) or `Scripts/predictions_to_submission.py` (from a predictions CSV) | `Outputs/submission*.csv` — for the Kaggle competition, see §8 |

Stages 2 and 3 produce exactly the file format and run exactly the model the original
codebase does.

---

## 2. Mapping our dataset onto MIMOSA

### Classes: 5 → 4

MIMOSA has five categories. Our data has four — there is no counterpart to the paper's
**"others"** (`Oth`) class, which collected aggression targeting race, occupation,
disability, nationality and so on. The task here is therefore **4-way, not 5-way**.

The integer encoding of the four shared classes is left exactly as the paper assigns it,
so that MMAE — which treats the labels as an ordinal scale — stays comparable:

| Paper | Code | Our `Train.csv` | Our `test.csv` | Canonical label |
|-------|------|-----------------|----------------|-----------------|
| NoAg | 0 | `Neutral` | `NonAggressive` | `non-aggressive` |
| GAg | 1 | `Genders` | `Gendered` | `gendered aggression` |
| PAg | 2 | `Politics` | `Political` | `political aggression` |
| RAg | 3 | `Religion` | `Religious` | `religious aggression` |
| Oth | 4 | — absent — | — absent — | — |

Our two label files use different vocabularies for the same four classes
(`Politics` vs `Political`); `prepare_dataset.py` normalises both onto the canonical
strings above and fails loudly on anything it cannot map.

### Splits: 70 / 15 / 15, applied honestly

The paper splits MIMOSA 70 / 15 / 15 into train / validation / test. Our data instead
arrives pre-split into a `Train` folder and a held-out `Test` folder.

**The held-out test set is kept intact** — it is never mixed back in, so the final numbers
stay honest. The paper's **70:15 train-to-validation ratio is then applied within our
`Train` folder**: a stratified 82.35 / 17.65 split, which is 70:15 renormalised. The result
is a train:validation proportion identical to the paper's, with a test set that is slightly
smaller in relative terms than the paper's 15%:

| | Train | Validation | Test |
|---|---|---|---|
| Paper (MIMOSA) | 70% | 15% | 15% |
| Here | ~72% | ~15% | ~12% |

The split is stratified by class and seeded (`--seed 42`), so it is reproducible.

### Images

`Train/Image` (2,903) and `Test/Image` (400) are copied into one `Dataset/Img/` folder, as
the original layout expects. Filenames do not collide (`Gendered1.jpg` vs
`testimage001.jpg`). All 3,303 images are valid RGB JPEGs; none are corrupt.

---

## 3. Setup

### Python packages

```bash
python -m venv .venv
.venv\Scripts\activate                                        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Tesseract (Stage 1 only)

`pytesseract` is only a wrapper — it needs the Tesseract **binary** plus the **Bengali**
language data, neither of which pip installs.

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

The installer ships only `eng` and `osd`. Bengali is added by dropping
`ben.traineddata` into a project-local folder, which needs no administrator rights:

```powershell
mkdir ..\tessdata
curl -L -o ..\tessdata\ben.traineddata `
  https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata
```

`ocr_captions.py` finds the binary and that `tessdata` folder automatically. Override with
the `TESSERACT_CMD` and `TESSDATA_PREFIX` environment variables if your paths differ.

### Model caches

Bangla-BERT (~650 MB) and CLIP ViT-B/32 (~350 MB) download on first run. `Scripts/_paths.py`
redirects them to `Replication/.cache/` instead of the `C:` user profile, which on this
machine has little headroom. An existing `HF_HOME` wins, so Colab and Kaggle are unaffected.

---

## 4. Running it

```bash
cd Scripts

# Stage 1 - OCR every meme (~30 min on 7 CPU workers). Resumable: rerun to continue,
# it skips memes already in captions_raw.csv. Do NOT pass --overwrite to resume.
python ocr_captions.py

# Stage 1b - optional caption denoiser (seconds; reads the CSV, no re-OCR)
python clean_captions.py --report 10

# Stage 2 - build the three split CSVs, and print the paper's Tables 1-3
python prepare_dataset.py --stats
# ...or build them from the denoised captions instead:
python prepare_dataset.py --stats --captions ../Dataset/captions_clean.csv

# Stage 3a - fast end-to-end smoke test (a few minutes, NOT a real result)
python main.py --subset 24 --n_iter 1 --run_name smoketest

# Stage 3b - the real run, paper hyperparameters
python main.py --run_name maf_full
```

### Arguments

`main.py` keeps the original's argument names. **The defaults have been changed to the
paper's published setup** (Appendix A), which the repository's own defaults did not match:

| Argument | Here | Original repo | Paper (Appendix A) |
|----------|------|---------------|--------------------|
| `--batch_size` | **4** | 16 | 4 |
| `--n_iter` (epochs) | **20** | 5 | 20 |
| `--lrate` | 5e-5 | 5e-5 | 5e-5 (MAF) |
| `--max_len` | 70 | 70 | — |
| `--heads` | 16 | 16 | — |

Added here: `--seed`, `--run_name`, `--subset`, `--attn_variant`, `--fix_scheduler`.

---

## 5. Deviations from the paper

Read this section before comparing any number produced here against the published Table 4.

### 5.1 Captions are machine-OCR'd, never hand-corrected *(largest source of divergence)*

The paper hand-corrected every OCR caption. That pass is not reproducible here, so our
captions carry the OCR's errors. Bengali OCR is genuinely weak — the paper itself notes it
"is not well-established" — and since Bangla-BERT sees only this text, caption noise
propagates straight into the text modality. **Expect our scores to fall below the
published WF1 = 0.742 substantially for this reason alone**, independent of anything else.

`clean_captions.py` (Stage 1b) is an automated stand-in, not an equivalent: it is a
deterministic token filter that keeps all Bengali script and discards Latin/numeric tokens
that do not look like words. It lifts the Bengali share of characters from 0.59 to 0.70
and removes 29% of tokens. It cannot fix a *misrecognised* Bengali word, which is what the
paper's annotators were actually correcting — it only removes obvious debris.

`prepare_dataset.py` uses `captions_clean.csv` when it exists, since the paper trained on
corrected captions. Pass `--captions ../Dataset/captions_raw.csv` to train on strict raw
OCR instead; running both is a cheap ablation on how much caption noise costs.

### 5.2 Four classes instead of five

Our data has no `Others` class, so this is a 4-way problem. A 4-way task is easier than a
5-way one at equal difficulty, which pushes in the *opposite* direction to 5.1. The two
effects do not cancel in any principled way, so **our headline number is not directly
comparable to the paper's** — it is a replication of the *method*, not a reproduction of
the *score*.

### 5.3 Page segmentation mode chosen on evidence

Tesseract's default PSM 3 assumes page-like layout. Measured on a 60-meme sample with
`Scripts/probe_ocr.py`, PSM 3 **returns no text at all for 33% of our memes**, because meme
text is scattered rather than laid out as a page. PSM 11 ("sparse text") drops that to 0%
and recovers the most Bengali script, so PSM 11 is the default. Rerun `probe_ocr.py` to
see the comparison. `ben` and `ben+eng` were measured as byte-identical on our memes, so
the simpler `ben` is used.

### 5.4 The paper and the released code disagree on the attention operands

The paper (Sec. 4.2) states:

> "we generate Q from textual features and K and V from visual features"

The released `models.py` does something different — `query=image`, `key=text`,
`value=image`. Q comes from vision, and V is vision rather than text.

This replication **defaults to the released code**, on the grounds that the released code
is what produced the published numbers. `--attn_variant paper` runs the configuration the
paper's text describes, if you want to measure the gap.

### 5.5 A scheduler bug is reproduced deliberately

`get_linear_schedule_with_warmup` is designed to be stepped once per optimizer step, and
the original computes `num_training_steps = epochs * len(train_loader)` accordingly — but
then calls `lr_scheduler.step()` only **once per epoch**. Over 20 epochs the learning rate
therefore decays by well under 1% instead of annealing to zero.

This is kept as-is for fidelity. `--fix_scheduler` opts into correct per-step stepping.

### 5.6 Small correctness fixes that the original needed

| Fix | Why it was necessary |
|-----|----------------------|
| `Image.open(...).convert("RGB")` | The original omits it; `Normalize()` crashes on any non-RGB meme. |
| `Captions` NaN → `""` | pandas reads an empty caption as NaN, which crashes the tokenizer. |
| Visual `seq_len` fed from `--max_len` | The original hardcoded `70` in the pooling call while `--max_len` was separately configurable; any other value made the concatenation fail. |
| Test labels kept as `int` | The original cast them to `float`, which distorts MMAE's ordinal arithmetic. |
| Dropped an unused CLIP load in `dataset.py` | It loaded and `.half()`ed a model that was never used; on CPU it is pure cost. No effect on results. |
| `transformers.AdamW` import dropped | Removed in modern transformers; the original imported it but used `torch.optim.AdamW`. |

### 5.7 Environment

The paper ran on Colab GPUs. This machine has **no CUDA GPU**, so the code falls back to
CPU. The architecture is unchanged, but full training on CPU is impractically slow — see
below.

---

## 6. What the data actually looks like after Stages 1–2

All numbers below were measured on this dataset, not copied from the paper.

### OCR yield (3,303 memes)

| | raw (`captions_raw.csv`) | denoised (`captions_clean.csv`) |
|---|---|---|
| Empty captions | 2 (0.1%) | 3 (0.1%) |
| Contain Bengali script | 98.1% | — |
| Avg words / caption | 29.9 | 21.2 |
| Avg Bengali share of characters | 0.59 | 0.70 |
| Tokens removed by denoiser | — | 28,597 (29.0%) |

For reference, PSM 3 would have produced **~33% empty captions**. The paper's MIMOSA
captions average 12–18 words; our denoised 21.2 is in the right neighbourhood, with the
excess being residual OCR noise.

### Splits

| | Train | Validation | Test |
|---|---|---|---|
| NoAg | 790 | 169 | 75 |
| GAg | 583 | 125 | 97 |
| PAg | 496 | 107 | 128 |
| RAg | 521 | 112 | 100 |
| **Total** | **2,390** | **513** | **400** |

Overall 72.4 / 15.5 / 12.1; train:validation is exactly **70.0 : 15.0**, as the paper has it.

### ⚠️ The train and test sets have different class priors

This is a property of the dataset as supplied, and it matters for interpreting results:

| Class | Train share | Test share |
|-------|-------------|------------|
| NoAg | 33.1% | **18.8%** |
| GAg | 24.4% | 24.3% |
| PAg | 20.8% | **32.0%** |
| RAg | 21.8% | 25.0% |

The most common class in training (NoAg) is the *rarest* in test, and PAg is over-weighted
in test by half again. A model that learns the training prior is mis-calibrated for this
test set, which will depress accuracy and weighted F1 independently of model quality.
Worth stating explicitly when reporting results; macro F1 is the more robust metric here.

### Caption statistics (cf. paper Table 2)

| Class | Ttw | Tuw | Tmw | Taw |
|-------|-----|-----|-----|-----|
| NoAg | 14,272 | 6,526 | 110 | 18 |
| GAg | 10,051 | 4,891 | 115 | 17 |
| PAg | 11,350 | 5,852 | 183 | 23 |
| RAg | 13,058 | 6,271 | 150 | 25 |

Pairwise Jaccard similarity between class vocabularies comes out at **0.12–0.15**, below
the paper's 0.16–0.24. Residual OCR noise contributes many spurious one-off tokens, which
inflates the union and pushes the ratio down.

### Pipeline verification

`main.py --subset 24 --n_iter 1` completes the full path — load → train → validate →
checkpoint → reload → test → metrics → `Outputs/`. Its **20.8% accuracy is not a result**:
24 training memes for one epoch, against a 25% chance baseline for four classes. It
demonstrates that the pipeline runs, nothing more.

---

## 7. A note on compute

Training MAF fine-tunes all 110M parameters of Bangla-BERT (CLIP stays frozen). On this
CPU-only machine, the paper's setup — batch 4, 20 epochs, ~2,400 training memes — is a
multi-day run and is not recommended locally.

`Scripts/MAF_Replication_Colab.ipynb` and `Scripts/MAF_Replication_Kaggle.ipynb` run the
identical code on a free Colab or Kaggle GPU, where the same configuration finishes in
roughly 1-2 hours. Use `--subset` locally to verify the pipeline works, then run the real
training on whichever platform you prefer.

Both notebooks read from the same `Replication.zip`, but the platforms differ in how data
gets in and results come out:

|  | Colab | Kaggle |
|---|---|---|
| Upload as | a file in Google Drive | a **Dataset** (kaggle.com/datasets -> New Dataset) |
| Mount | Drive is mounted, then the zip is unzipped into `/content` | the dataset is auto-extracted and mounted **read-only** at `/kaggle/input/<slug>` |
| GPU quota | varies by plan | ~30 hrs/week free, ~9-12 hr session limit |
| Getting results out | explicit copy back to Drive (last cell) | automatic - anything under `/kaggle/working` is kept as the notebook's Output on commit |

Kaggle's `/kaggle/input` being read-only is why that notebook copies only `Scripts/` (a
few hundred KB) into the writable `/kaggle/working` and reads `Dataset/` (365 MB of
images) straight from the mount by absolute path, rather than duplicating everything the
way the Colab notebook does.

### Packaging for Colab / Kaggle

Upload a **zip**, not the folder — Drive uploads 3,303 individual files very slowly, and
Colab reading them back through the Drive FUSE mount is slow again. One archive uploads
once and unpacks onto Colab's local disk in seconds.

```powershell
powershell -File Scripts\make_colab_zip.ps1
```

Then put the resulting `D:\MemeDecode\Replication.zip` in Drive at
`MyDrive/MemeDecode/Replication.zip`, which is where the notebook looks.

> **Do not build this archive with `Compress-Archive`.** Windows PowerShell 5.1 writes ZIP
> entry names with backslash separators (`Dataset\Img\x.jpg`), which violates the ZIP spec.
> Linux `unzip` — which is what Colab runs — then reads each name as one flat filename
> containing literal backslashes, so the directory structure is never recreated and the
> notebook cannot find `Dataset/Img`. `make_colab_zip.ps1` builds the archive through .NET
> and writes the entry names itself, guaranteeing forward slashes. It also excludes
> `.cache/` (~1 GB of model downloads that Colab re-fetches anyway).

---

## 8. Generating a Kaggle competition submission

`D:\MemeDecode\sample_submission.csv` fixes the exact format the competition scores
against: columns `Image_name,Target`, one row per test image, using the **original
`Train/Train.csv` vocabulary** — `Neutral`, `Genders`, `Politics`, `Religion` — which is a
*different* string set from the paper's canonical labels used internally
(`non-aggressive`, etc.) and different again from `Test/test.csv`'s own vocabulary
(`NonAggressive`, etc.). `generate_submission.py` handles that translation; see the
mapping table below.

**A submission file contains predictions only.** It is built by a script
(`generate_submission.py`) that never reads a label column at all — only `image_name` and
`Captions` go in, matching the separation described in §5: the model's `forward()` has no
`label` parameter, so nothing here could leak ground truth even by accident.

### Route A — from a predictions CSV (no checkpoint, no GPU, seconds)

`main.py`'s final evaluation runs on `testing_set.csv`, which contains exactly the 400
images `sample_submission.csv` lists. Every full run's `Outputs/predictions_<run>.csv`
therefore already holds that run's prediction for each submission image:

```bash
cd Scripts
python predictions_to_submission.py --predictions ../Outputs/predictions_maf_*.csv
```

This writes `Outputs/submission_<run>.csv` for each run. It reads only `image_name` and
`pred_label`, ignores the `Label`/`true_id` columns, and refuses `--subset` smoke-test files
because they don't cover all 400 images. The predictions are the same ones
`generate_submission.py` would produce from that run's best checkpoint.

### Route B — from a trained checkpoint

Each run saves `Saved_Models/maf_model_<run_name>.pth`. Early versions of the notebooks
wrote every run to one shared `maf_model.pth`, so running `maf_full`, `maf_paper_attn` and
`maf_fixed_sched` in sequence left only the last run's model on disk. Route A still works
for those runs.

```bash
cd Scripts
python generate_submission.py \
    --checkpoint ../Saved_Models/maf_model_maf_full.pth \
    --sample_submission ../../sample_submission.csv \
    --out ../Outputs/submission.csv
```

`--sample_submission` defaults to `../../sample_submission.csv` (i.e.
`D:\MemeDecode\sample_submission.csv`), so on this machine the flag can usually be
dropped. `--attn_variant` and `--heads` must match whatever the checkpoint was **trained**
with — the defaults match `main.py`'s defaults, so if you didn't override them for
training, don't override them here either.

The script validates its own output before writing: same columns as
`sample_submission.csv`, same 400 rows in the same order, every value inside
`{Neutral, Genders, Politics, Religion}`. `Scripts/_check_submission.py` proves the whole
path end-to-end with a freshly-initialized (untrained) model, if you want to re-verify the
mechanics before a long training run — its predictions are meaningless, but the file
shape is real.

### Label mapping (internal -> submission)

| Model output index | `dataset.TARGET_NAMES` | Submitted as |
|---|---|---|
| 0 | NoAg | **Neutral** |
| 1 | GAg | **Genders** |
| 2 | PAg | **Politics** |
| 3 | RAg | **Religion** |

Then upload `Outputs/submission.csv` on the competition's Submit Predictions page.

---

## 9. File map

```
Replication/
├── Dataset/
│   ├── Img/                    3,303 memes (Train + Test unified)
│   ├── captions_raw.csv        Stage 1 output: image_name, Captions
│   ├── training_set.csv        Stage 2 output: image_name, Captions, Label
│   ├── validation_set.csv
│   └── testing_set.csv
├── Scripts/
│   ├── ocr_captions.py         Stage 1 - pytesseract caption extraction
│   ├── clean_captions.py       Stage 1b - optional caption denoiser
│   ├── probe_ocr.py            Diagnostic - compares Tesseract configurations
│   ├── prepare_dataset.py      Stage 2 - label mapping, splits, paper Tables 1-3
│   ├── _paths.py               Redirects HF / CLIP caches off the C: drive
│   ├── _check_env.py           Diagnostic - verifies the installed stack imports
│   ├── _check_model.py         Diagnostic - builds MAF, runs a synthetic fwd/bwd pass
│   ├── dataset.py              Port of the original dataset.py
│   ├── models.py               Port of the original models.py (MAF)
│   ├── evaluation.py           Port of the original evaluation.py
│   ├── main.py                 Stage 3 - entry point, metrics, result files
│   ├── generate_submission.py  Stage 4 - submission.csv from a checkpoint
│   ├── predictions_to_submission.py  Stage 4 - submission_<run>.csv from a predictions CSV
│   ├── submission_format.py    Shared label mapping + submission validation
│   ├── _check_submission.py    Diagnostic - proves the submission path end-to-end
│   ├── make_colab_zip.ps1      Packs Replication.zip for the Colab/Kaggle upload
│   ├── MAF_Replication_Colab.ipynb
│   └── MAF_Replication_Kaggle.ipynb
├── Saved_Models/               maf_model_<run_name>.pth, one per run (best validation accuracy)
├── Outputs/                    results_*.json, predictions_*.csv
└── requirements.txt
```

---

## 10. Citation

```bibtex
@inproceedings{ahsan2024multimodal,
  title={A Multimodal Framework to Detect Target Aware Aggression in Memes},
  author={Ahsan, Shawly and Hossain, Eftekhar and Sharif, Omar and Das, Avishek and Hoque, Mohammed Moshiul and Dewan, M},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2487--2500},
  year={2024}
}
```
