import SIGLIP.Scripts._paths as _paths  # noqa: F401

import torch
import transformers

print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)

try:
    from transformers import get_linear_schedule_with_warmup  # noqa: F401

    print("get_linear_schedule_with_warmup: OK")
except Exception as exc:
    print("get_linear_schedule_with_warmup: FAIL ->", exc)

try:
    import clip

    print("clip import : OK", clip.available_models()[:3])
except Exception as exc:
    print("clip import : FAIL ->", exc)

try:
    from imblearn.metrics import macro_averaged_mean_absolute_error  # noqa: F401

    print("imblearn MMAE: OK")
except Exception as exc:
    print("imblearn MMAE: FAIL ->", exc)

import pandas as pd
import sklearn

print("pandas      ", pd.__version__)
print("sklearn     ", sklearn.__version__)
print("HF_HOME     ", __import__("os").environ.get("HF_HOME"))
