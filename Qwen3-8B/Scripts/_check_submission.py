"""Pre-flight check for generate_submission.py: does the whole path actually work?

Builds a MAF with random (untrained) weights, saves it as a throwaway checkpoint, runs
generate_submission.py against it, and validates the output file - then deletes the
throwaway checkpoint and output so nothing fake is left lying around. Predictions from an
untrained model are meaningless; this only proves the mechanics, not accuracy.
"""
import os
import subprocess
import sys

import torch

import models as m

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

fake_ckpt = os.path.join(ROOT_DIR, "Saved_Models", "_fake_untrained.pth")
fake_out = os.path.join(ROOT_DIR, "Outputs", "_fake_submission.csv")

print("Building an untrained MAF to get a checkpoint-shaped file ...")
clip_visual = m.load_clip_visual()
model = m.MAF(clip_visual, num_classes=4, num_heads=16, seq_len=70, attn_variant="code")
os.makedirs(os.path.dirname(fake_ckpt), exist_ok=True)
torch.save(model.state_dict(), fake_ckpt)
print("Saved throwaway checkpoint:", fake_ckpt)

cmd = [
    sys.executable, os.path.join(SCRIPT_DIR, "generate_submission.py"),
    "--checkpoint", fake_ckpt,
    "--out", fake_out,
    "--batch_size", "8",
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, cwd=SCRIPT_DIR)
if result.returncode != 0:
    os.remove(fake_ckpt)
    sys.exit("generate_submission.py FAILED (exit {})".format(result.returncode))

# Validate the output against the real sample_submission.csv.
import pandas as pd

sample = pd.read_csv(os.path.join(ROOT_DIR, "..", "sample_submission.csv"))
out = pd.read_csv(fake_out)

assert list(out.columns) == list(sample.columns), (out.columns, sample.columns)
assert len(out) == len(sample) == 400, len(out)
assert list(out["Image_name"]) == list(sample["Image_name"]), "row order mismatch"
assert set(out["Target"]) <= {"Neutral", "Genders", "Politics", "Religion"}, set(out["Target"])
assert out["Target"].notna().all()

print()
print("Output columns match sample_submission.csv:", list(out.columns))
print("Row count matches (400):", len(out) == 400)
print("Row order matches sample_submission.csv exactly: OK")
print("All Target values are in the expected vocabulary: OK")
print("SUBMISSION PIPELINE CHECK PASSED")

os.remove(fake_ckpt)
os.remove(fake_out)
print("Cleaned up throwaway checkpoint and output.")
