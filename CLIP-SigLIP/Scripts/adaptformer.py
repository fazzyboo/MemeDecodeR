"""
adaptformer.py -- AdaptFormer parameter-efficient fine-tuning for CLIP vision towers.

Implements AdaptFormer (Chen et al., "AdaptFormer: Adapting Vision Transformers for
Scalable Visual Recognition", NeurIPS 2022) and injects it into **every** residual
block of an OpenAI-CLIP `VisionTransformer`.

What AdaptFormer does
---------------------
A stock CLIP vision block is pre-norm:

    x = x + Attention(ln_1(x))
    x = x + MLP(ln_2(x))

AdaptFormer replaces the MLP sub-layer with *AdaptMLP*: the original frozen MLP plus a
**parallel** trainable bottleneck branch, both fed the same input:

    x = x + Attention(ln_1(x))
    x = x + MLP(ln_2(x)) + s * W_up( ReLU( W_down(x) ) )

with W_down: d -> r, W_up: r -> d and r << d (default r = 64). Everything else --
attention, MLP, layer norms, embeddings -- stays frozen. Only the adapters and the
classification head receive gradients, so a ViT-B/32 tower trains ~1.2% of its weights.

Because `W_up` is zero-initialised (LoRA-style), the parallel branch outputs exactly zero
at step 0: an AdaptFormer-injected model is **numerically identical to the frozen CLIP it
started from** until training moves the weights. That makes the zero-shot run a genuine
step-0 baseline for the fine-tuned one.

Parallel vs sequential placement, the LoRA-style init, the scaling factor `s` and the
optional adapter LayerNorm all follow the official implementation
(github.com/ShoufaChen/AdaptFormer); its ViT defaults are r=64, s=0.1, no adapter
LayerNorm, parallel placement, which are the defaults here too.

This module is standalone: it touches no file in the repo and knows nothing about MIMOSA.
`clip_adaptformer.py` is the training runner that uses it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# The adapter
# ===========================================================================
class Adapter(nn.Module):
    """AdaptFormer's bottleneck branch: down-project, ReLU, up-project, scale.

    Args:
        d_model: width of the residual stream (768 for ViT-B, 1024 for ViT-L).
        bottleneck: inner dimension r. The whole parameter budget is ~2*d_model*r.
        dropout: dropout on the bottleneck activations.
        scalar: fixed float scale `s`, or "learnable_scalar" for a trained scalar.
        layernorm_option: "none" (official ViT default), "in" (LayerNorm the input),
            or "out" (LayerNorm the branch output).
        init_option: "lora" zeroes the up-projection so the branch starts as a no-op.
            "xavier" initialises both projections normally (branch is active at step 0).
    """

    def __init__(self, d_model, bottleneck=64, dropout=0.0, scalar="0.1",
                 layernorm_option="none", init_option="lora"):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.layernorm_option = layernorm_option

        if layernorm_option not in ("none", "in", "out"):
            raise ValueError(f"layernorm_option must be none/in/out, got {layernorm_option!r}")
        self.adapter_layer_norm = (
            nn.LayerNorm(d_model) if layernorm_option in ("in", "out") else None
        )

        if scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(scalar)

        self.down_proj = nn.Linear(d_model, bottleneck)
        self.non_linear = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = dropout

        with torch.no_grad():
            if init_option == "lora":
                # zero up-projection => the branch contributes nothing at step 0
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
            elif init_option == "xavier":
                nn.init.xavier_uniform_(self.down_proj.weight)
                nn.init.xavier_uniform_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
            else:
                raise ValueError(f"init_option must be lora/xavier, got {init_option!r}")

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.layernorm_option == "in":
            x = self.adapter_layer_norm(x)

        down = self.non_linear(self.down_proj(x))
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale

        if self.layernorm_option == "out":
            up = self.adapter_layer_norm(up)
        return up + residual if add_residual else up

    def extra_repr(self):
        scale = "learnable" if isinstance(self.scale, nn.Parameter) else self.scale
        return (f"d_model={self.d_model}, bottleneck={self.bottleneck}, "
                f"scale={scale}, layernorm={self.layernorm_option}")


# ===========================================================================
# Block wrapper
# ===========================================================================
class AdaptFormerBlock(nn.Module):
    """Wraps one CLIP `ResidualAttentionBlock` so its MLP sub-layer becomes AdaptMLP.

    The wrapped block is kept whole and untouched, so this works on any CLIP checkpoint
    and survives upstream changes to the block internals.
    """

    def __init__(self, block, adapter, mode="parallel"):
        super().__init__()
        if mode not in ("parallel", "sequential"):
            raise ValueError(f"mode must be parallel/sequential, got {mode!r}")
        self.block = block
        self.adapter = adapter
        self.mode = mode

    def forward(self, x):
        b = self.block
        # ---- attention sub-layer: untouched --------------------------------
        x = x + b.attention(b.ln_1(x))

        # ---- AdaptMLP sub-layer --------------------------------------------
        if self.mode == "parallel":
            # adapter sees the same input as the MLP, and is summed into the residual
            adapt = self.adapter(x, add_residual=False)
            x = x + b.mlp(b.ln_2(x)) + adapt
        else:  # sequential: adapter post-processes the MLP output
            h = b.mlp(b.ln_2(x))
            x = x + self.adapter(h, add_residual=True)
        return x

    def extra_repr(self):
        return f"mode={self.mode}"


# ===========================================================================
# Injection
# ===========================================================================
def _resblocks(visual):
    """Locate the block list of a CLIP visual tower, with a clear error for ResNets."""
    transformer = getattr(visual, "transformer", None)
    if transformer is None or not hasattr(transformer, "resblocks"):
        raise TypeError(
            f"{type(visual).__name__} has no transformer.resblocks. AdaptFormer needs a "
            "ViT vision tower -- use a ViT-* backbone (ViT-B/32, ViT-B/16, ViT-L/14, ...), "
            "not an RN* one."
        )
    return transformer.resblocks


def inject_adaptformer(visual, bottleneck=64, scalar="0.1", dropout=0.0,
                       layernorm_option="none", init_option="lora", mode="parallel",
                       blocks=None):
    """Wrap every residual block of a CLIP visual tower in AdaptFormer.

    Args:
        visual: a CLIP `VisionTransformer` (i.e. `clip_model.visual`). Modified in place.
        blocks: optional iterable of block indices to adapt. `None` (default) means
            **every** block, which is what the paper does.

    Returns:
        dict describing what was injected.
    """
    resblocks = _resblocks(visual)
    width = visual.transformer.width
    total = len(resblocks)
    targets = list(range(total)) if blocks is None else sorted(set(blocks))
    for i in targets:
        if not 0 <= i < total:
            raise IndexError(f"block index {i} out of range for {total} blocks")

    reference = next(resblocks[0].parameters())
    device, dtype = reference.device, reference.dtype

    for i in targets:
        block = resblocks[i]
        if isinstance(block, AdaptFormerBlock):
            raise RuntimeError(f"block {i} already has AdaptFormer injected")
        adapter = Adapter(
            d_model=width, bottleneck=bottleneck, dropout=dropout, scalar=scalar,
            layernorm_option=layernorm_option, init_option=init_option,
        ).to(device=device, dtype=dtype)
        resblocks[i] = AdaptFormerBlock(block, adapter, mode=mode)

    n_adapter_params = sum(
        p.numel() for i in targets for p in resblocks[i].adapter.parameters()
    )
    return {
        "blocks_total": total,
        "blocks_adapted": len(targets),
        "block_indices": targets,
        "width": width,
        "bottleneck": bottleneck,
        "scalar": scalar,
        "dropout": dropout,
        "layernorm_option": layernorm_option,
        "init_option": init_option,
        "mode": mode,
        "adapter_params": n_adapter_params,
        "params_per_adapter": n_adapter_params // max(len(targets), 1),
    }


# ===========================================================================
# Freezing / inspection / checkpointing
# ===========================================================================
def is_adapter_param(name):
    return ".adapter." in name or name.startswith("adapter.")


def freeze_for_peft(model, train_adapters=True, train_layernorms=False,
                    train_head_prefixes=("head", "classifier", "logit")):
    """Freeze everything, then re-enable adapters (and optionally LNs / the head).

    Returns a dict of parameter counts. `train_layernorms=True` additionally tunes every
    LayerNorm affine parameter, a common and very cheap PEFT add-on -- but note it means
    the model is no longer a no-op at step 0.
    """
    for p in model.parameters():
        p.requires_grad = False

    enabled = []
    for name, p in model.named_parameters():
        want = False
        if train_adapters and is_adapter_param(name):
            want = True
        elif train_layernorms and (".ln_" in name or "ln_post" in name
                                   or "ln_pre" in name or "layer_norm" in name):
            # never unfreeze an adapter-internal LN through this path twice
            want = True
        elif any(name.startswith(prefix) for prefix in train_head_prefixes):
            want = True
        if want:
            p.requires_grad = True
            enabled.append(name)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_fraction": round(trainable / total, 6) if total else 0.0,
        "trainable_tensors": len(enabled),
        "trainable_names_sample": enabled[:12],
    }


def trainable_report(model):
    """Human-readable breakdown of what will actually be trained."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_adapter_param(name):
            key = "adapters"
        elif name.startswith(("head", "classifier", "logit")):
            key = "head"
        elif ".ln_" in name or "ln_post" in name or "ln_pre" in name:
            key = "layernorms"
        else:
            key = "other"
        groups[key] = groups.get(key, 0) + p.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [f"{'group':<16}{'params':>14}{'% of model':>13}", "-" * 43]
    for key in sorted(groups):
        lines.append(f"{key:<16}{groups[key]:>14,}{100 * groups[key] / total:>12.3f}%")
    lines.append("-" * 43)
    lines.append(f"{'TRAINABLE':<16}{trainable:>14,}{100 * trainable / total:>12.3f}%")
    lines.append(f"{'FROZEN':<16}{total - trainable:>14,}{100 * (total - trainable) / total:>12.3f}%")
    lines.append(f"{'TOTAL':<16}{total:>14,}")
    return "\n".join(lines), groups


def adapter_state_dict(model):
    """Only the tensors that were trained -- a few MB instead of ~1 GB."""
    return {
        name: p.detach().cpu()
        for name, p in model.state_dict().items()
        if is_adapter_param(name) or name.startswith(("head", "classifier", "logit"))
    }


def load_adapter_state_dict(model, state, strict=True):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict:
        stale = [k for k in unexpected]
        if stale:
            raise RuntimeError(f"unexpected keys in adapter checkpoint: {stale[:8]}")
    return {"loaded": len(state), "missing_in_checkpoint": len(missing)}
