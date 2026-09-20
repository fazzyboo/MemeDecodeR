"""Architecture smoke test: build MAF and push a synthetic batch through it.

Runs without any dataset, so it can validate shapes and API compatibility while OCR is
still going. Also forces the Bangla-BERT and CLIP downloads to happen up front.
"""
import _paths  # noqa: F401

import torch

import models as m

BATCH, SEQ, CLASSES, HEADS = 2, 70, 4, 16

print("device:", m.device)
print("loading CLIP visual tower ...")
clip_visual = m.load_clip_visual()
print("loading Bangla-BERT and building MAF ...")

for variant in ("code", "paper"):
    model = m.MAF(clip_visual, CLASSES, HEADS, seq_len=SEQ, attn_variant=variant).to(m.device)
    model.eval()

    images = torch.randn(BATCH, 3, 224, 224, device=m.device)
    input_ids = torch.randint(0, 1000, (BATCH, SEQ), device=m.device)
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device=m.device)

    with torch.no_grad():
        out = model(images, input_ids, attention_mask)

    assert out.shape == (BATCH, CLASSES), "bad output shape {}".format(out.shape)
    print("  attn_variant={:<6} -> logits {}  OK".format(variant, tuple(out.shape)))

# Confirm CLIP really is frozen and BERT really is trainable.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("trainable params: {:,}".format(trainable))
print("frozen params   : {:,}  (CLIP visual tower)".format(frozen))
assert frozen > 0, "CLIP should be frozen"
assert any(p.requires_grad for p in model.bert.parameters()), "BERT should be trainable"

# One backward pass, to be sure gradients actually flow through the fusion.
model.train()
logits = model(
    torch.randn(BATCH, 3, 224, 224, device=m.device),
    torch.randint(0, 1000, (BATCH, SEQ), device=m.device),
    torch.ones(BATCH, SEQ, dtype=torch.long, device=m.device),
)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3], device=m.device))
loss.backward()
grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
print("backward OK - {} trainable tensors received gradients, loss={:.4f}".format(grads, loss.item()))
print("MODEL CHECK PASSED")
