"""
MAF (Multimodal Attentive Fusion) - port of the original MIMOSA Scripts/models.py.

The architecture is unchanged: a frozen CLIP ViT-B/32 image encoder projected 512 -> 768,
a fine-tuned Bangla-BERT text encoder, a multi-head attention block that fuses them, and a
dense head over the concatenation of [attention output | visual | textual].

Changes from the original, all forced by our dataset or by running without a GPU:

  1. num_classes is supplied by the caller and is 4 here, not 5 (no "others" class).
  2. The visual sequence length passed to adaptive_avg_pool1d was hardcoded to 70 while
     max_len was independently configurable; the two must be equal or the concatenation
     fails. It is now a parameter fed from max_len, so --max_len works as documented.
  3. Device falls back to CPU; CLIP is loaded via a project-local download_root.
  4. transformers no longer exports AdamW (the original imported it but used
     torch.optim.AdamW), and madgrad was imported but never used. Both are now optional.

KNOWN DISCREPANCY - attention operand order
-------------------------------------------
The paper (Sec. 4.2) states: "we generate Q from textual features and K and V from
visual features". The released code does the opposite: query=image, key=text, value=image.
We default to the released code, since that is the implementation that produced the
published numbers, and expose --attn_variant paper to run the configuration the text
describes. See README, section "Deviations from the paper".

KNOWN BUG KEPT FOR FIDELITY - scheduler cadence
-----------------------------------------------
get_linear_schedule_with_warmup is built for per-optimizer-step stepping, and
num_training_steps is computed as epochs * len(train_loader), but the original calls
lr_scheduler.step() once per epoch. The learning rate therefore decays by only a tiny
fraction over training. This is reproduced as-is; --fix_scheduler opts into per-step
stepping if you want to measure the difference.
"""
import os

import SIGLIP.Scripts._paths as _paths  # noqa: F401  - must precede transformers / clip imports

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModel, get_linear_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        output, _ = self.attention(query, key, value, attn_mask=mask)
        return output


def load_clip_visual():
    """Load the CLIP ViT-B/32 image tower, frozen, in float32."""
    clip_model, _preprocess = clip.load(
        "ViT-B/32", device=device, download_root=_paths.CLIP_DOWNLOAD_ROOT
    )
    clip_model = clip_model.visual

    # Convert model weights to single-precision (clip.load returns fp16 on CUDA).
    clip_model = clip_model.float()
    clip_model = clip_model.to(device)

    # Freeze the parameters of the CLIP model
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model


# Define the model in PyTorch
class MAF(nn.Module):
    def __init__(self, clip_model, num_classes, num_heads, seq_len=70, attn_variant="code"):
        super(MAF, self).__init__()

        # Visual feature extractor (CLIP)
        self.clip = clip_model
        self.visual_linear = nn.Linear(512, 768)
        self.seq_len = seq_len
        self.attn_variant = attn_variant

        # Textual feature extractor (BERT)
        self.bert = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")

        # Multihead attention
        self.attention = MultiheadAttention(d_model=768, nhead=num_heads)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(768 + 768 + 768, 128),  # visual + BERT + attention output
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_input, input_ids, attention_mask):

        # Extract visual features using CLIP
        image_features = self.clip(image_input)
        image_features = self.visual_linear(image_features)
        image_features = image_features.unsqueeze(1)
        # Broadcast the single visual vector across the text sequence length
        image_features = F.adaptive_avg_pool1d(
            image_features.permute(0, 2, 1), self.seq_len
        ).permute(0, 2, 1)

        # Extract BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.last_hidden_state

        # Multihead attention between visual features and BERT embeddings.
        # nn.MultiheadAttention here is seq-first, so swap batch and sequence dims.
        vis = image_features.permute(1, 0, 2)
        txt = bert_output.permute(1, 0, 2)
        if self.attn_variant == "paper":
            # Paper Sec. 4.2: Q from text, K and V from vision.
            attention_output = self.attention(query=txt, key=vis, value=vis, mask=None)
        else:
            # Released implementation: Q from vision, K from text, V from vision.
            attention_output = self.attention(query=vis, key=txt, value=vis, mask=None)

        # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output = attention_output.permute(1, 0, 2)

        # Concatenate the attention output, visual features, and BERT embeddings
        fusion_input = torch.cat([attention_output, image_features, bert_output], dim=2)

        output = self.fc(fusion_input.mean(1))  # Pool over the sequence dimension
        return output


# Define a function to calculate accuracy
def calculate_accuracy(predictions, targets):
    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def train(
    train_loader,
    val_loader,
    path,
    heads,
    epochs,
    lr_rate,
    num_classes=4,
    seq_len=70,
    attn_variant="code",
    fix_scheduler=False,
    checkpoint_name="maf_model.pth",
):

    # Create an instance of the model
    num_heads = heads  # Number of attention heads for multihead attention
    clip_model = load_clip_visual()
    model = MAF(clip_model, num_classes, num_heads, seq_len=seq_len, attn_variant=attn_variant)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.01)

    # Define learning rate scheduler
    num_epochs = epochs
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    best_val_accuracy = -1.0

    print("Start Training MAF")
    print("--------------------------------")
    print("Device:", device)
    print("Classes#:", num_classes)
    print("Attention Heads#:", heads)
    print("Attention variant:", attn_variant)
    print("Epochs#:", epochs)
    print("Learning Rate:", lr_rate)
    print("--------------------------------")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0

        with tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, num_epochs), unit="batch") as t:
            for batch in t:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if fix_scheduler:
                    lr_scheduler.step()
                total_loss += loss.item()
                total_accuracy += calculate_accuracy(outputs, labels).item()

                t.set_postfix(loss=total_loss / (t.n + 1), acc=total_accuracy / (t.n + 1))

        # Calculate training accuracy and loss
        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)

        # Validation loop
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", unit="batch"):
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images, input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(
            "Epoch {}/{}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Acc: {:.2f}%".format(
                epoch + 1, num_epochs, avg_train_loss, avg_train_accuracy * 100, val_accuracy * 100
            )
        )

        # Save the model with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(path, checkpoint_name))
            print("Model Saved.")

        if not fix_scheduler:
            lr_scheduler.step()  # Update learning rate (original cadence: once per epoch)

    print("Best Validation Accuracy: {:.2f}%".format(best_val_accuracy * 100))
    print("--------------------------------")
    return model


def evaluation(path, model, test_loader, checkpoint_name="maf_model.pth"):
    # Load the saved model
    print("Model is Loading..", checkpoint_name)
    model.load_state_dict(torch.load(os.path.join(path, checkpoint_name), map_location=device))
    model.eval()
    print("Loaded.")

    test_labels = []
    test_preds = []

    print("--------------------------------")
    print("Start Evaluating..")
    with torch.no_grad(), tqdm(test_loader, desc="Testing", unit="batch") as t:
        for batch in t:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds)

    print("Evaluation Done.")
    print("--------------------------------")
    return test_labels, test_preds
