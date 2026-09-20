"""
Cache redirection - import this BEFORE transformers / clip in every entry point.

Bangla-BERT (~650 MB) and CLIP ViT-B/32 (~350 MB) otherwise land in the user profile on
C:, which on this machine has very little headroom. Keeping the caches next to the
project also makes the whole replication self-contained and easy to relocate.

Any pre-existing HF_HOME / TORCH_HOME in the environment wins, so this is a default, not
an override - Colab and Kaggle keep their own caches untouched.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")

CLIP_DOWNLOAD_ROOT = os.path.join(CACHE_DIR, "clip")

_DEFAULTS = {
    "HF_HOME": os.path.join(CACHE_DIR, "huggingface"),
    "TORCH_HOME": os.path.join(CACHE_DIR, "torch"),
}

for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)
    os.makedirs(os.environ[_key], exist_ok=True)

os.makedirs(CLIP_DOWNLOAD_ROOT, exist_ok=True)
