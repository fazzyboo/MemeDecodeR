import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from madgrad import MADGRAD
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import clip
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device:",device)


# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        output, _ = self.attention(query, key, value, attn_mask=mask, key_padding_mask=key_padding_mask)
        return output


clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.visual

# Convert model's weights to single-precision
clip_model = clip_model.float()

# Create an instance of the CLIP model
clip_model = clip_model.to(device)

# Freeze the parameters of the CLIP model
for param in clip_model.parameters():
    param.requires_grad = False   


# AdaptFormer-style parallel adapter: down-project -> ReLU -> up-project, scaled
class ParallelAdapter(nn.Module):
    def __init__(self, dim, bottleneck, scale=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = scale
        # zero-init the up projection so training starts from the pretrained CLIP behaviour
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.relu(self.down(x))) * self.scale


def add_clip_adapters(visual, bottleneck):
    """Insert a trainable parallel adapter beside the MLP of every (frozen) CLIP ViT block."""
    for blk in visual.transformer.resblocks:
        if hasattr(blk, "adapter"):
            continue
        blk.adapter = ParallelAdapter(visual.transformer.width, bottleneck).to(device).float()

        def forward(x, blk=blk):
            x = x + blk.attention(blk.ln_1(x))
            return x + blk.mlp(blk.ln_2(x)) + blk.adapter(x)
        blk.forward = forward


def add_bert_adapters(bert, bottleneck):
    """Parallel adapter beside the FFN of every BERT layer (BertOutput): LN(ffn + x + adapter(x))."""
    for layer in bert.encoder.layer:
        out = layer.output
        if hasattr(out, "adapter"):
            continue
        out.adapter = ParallelAdapter(bert.config.hidden_size, bottleneck).to(device).float()

        def forward(hidden_states, input_tensor, out=out):
            h = out.dropout(out.dense(hidden_states))
            return out.LayerNorm(h + input_tensor + out.adapter(input_tensor))
        out.forward = forward


# Define the model in PyTorch
class MAF(nn.Module):
    def __init__(self, clip_model, num_classes, num_heads, adapter_dim=0, freeze_bert=False, fix_attn=False):
        super(MAF, self).__init__()
        self.fix_attn = fix_attn

        # Visual feature extractor (CLIP)
        self.clip = clip_model # Load the CLIP model
        self.visual_linear = nn.Linear(512, 768)

        # Textual feature extractor (BERT)
        self.bert = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            if adapter_dim > 0:
                add_bert_adapters(self.bert, adapter_dim)

        # Multihead attention
        self.attention = MultiheadAttention(d_model=768, nhead=num_heads)

        # Fully connected layers (updated for binary classification)
        self.fc = nn.Sequential(
            nn.Linear(768+768+768, 128),  # Input size includes visual features, BERT embeddings, and attention output
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),  # Updated for binary classification
        )

    def forward(self, image_input, input_ids, attention_mask):

        # Extract visual features using CLIP
        image_features = self.clip(image_input)
        image_features = self.visual_linear(image_features)
        image_features = image_features.unsqueeze(1)
        # Apply average pooling to reduce the sequence length to 50
        image_features = F.adaptive_avg_pool1d(image_features.permute(0, 2, 1), 70).permute(0, 2, 1)
        # print(image_features.shape)


        # Extract BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.last_hidden_state
        # print(bert_output.shape)

        # # Apply multihead attention between visual_features and BERT embeddings
        # # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
        if self.fix_attn:
            # image attends over the caption tokens: Value is the text (padding masked out),
            # so the attention weights actually change the output
            attention_output = self.attention(
                query=image_features.permute(1, 0, 2),
                key=bert_output.permute(1, 0, 2),
                value=bert_output.permute(1, 0, 2),
                key_padding_mask=(attention_mask == 0),
            )
        else:
            attention_output = self.attention(
                query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                value=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
                mask=None  # You can add a mask if needed
            )

        # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output = attention_output.permute(1, 0, 2)
        # print(attention_output.shape)

        # Concatenate the context vector, visual features, BERT embeddings, and attention output
        fusion_input = torch.cat([attention_output, image_features, bert_output], dim=2)
        # print(fusion_input.shape)

        output = self.fc(fusion_input.mean(1))  # Pool over the sequence dimension
        return output


# Define a function to calculate accuracy
def calculate_accuracy(predictions, targets):
    # For multi-class classification, you can use torch.argmax to get the predicted class
    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy



# num_epochs = 1

def train(train_loader, val_loader, path, heads, epochs, lr_rate, adapter_dim=0, freeze_bert=False, fix_attn=False, head_lr=None):

  if adapter_dim > 0:
    add_clip_adapters(clip_model, adapter_dim)

  # Create an instance of the model
  num_classes = 5  # Number of output classes
  num_heads = heads  # Number of attention heads for multihead attention
  model = MAF(clip_model, num_classes, num_heads, adapter_dim, freeze_bert, fix_attn)
  model = model.to(device)  

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  pretrained = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bert.") and ".adapter." not in n]
  new = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("bert.") and ".adapter." not in n)]
  groups = [{"params": new, "lr": head_lr or lr_rate}]
  if pretrained:
      groups.append({"params": pretrained, "lr": lr_rate})
  optimizer = torch.optim.AdamW(groups, lr=lr_rate,  weight_decay = 0.01)

  # Define learning rate scheduler
  num_epochs = epochs
  num_training_steps = num_epochs * len(train_loader)
  lr_scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps)


  # Training loop
  best_val_accuracy = 0.0

  print(f"Start Training MAF on MIMOSA")
  print("--------------------------------")
  print("Attention Heads#:",heads)
  print("Epochs#:",epochs)
  print("Learning Rate:",lr_rate )
  n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
  n_all = sum(p.numel() for p in model.parameters())
  n_adapt = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)
  print(f"freeze_bert={freeze_bert} fix_attn={fix_attn} head_lr={head_lr}")
  print(f"Adapter dim: {adapter_dim} | adapter params: {n_adapt:,}")
  print(f"Trainable params: {n_train:,} / {n_all:,}")
  print("--------------------------------")
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    # Wrap the train_loader with tqdm for the progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
        for batch in t:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels).item()

            # Update the tqdm progress bar
            t.set_postfix(loss=total_loss / (t.n + 1), acc=total_accuracy / (t.n + 1))

    # Calculate training accuracy and loss
    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)

    # Validation loop
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation", unit="batch"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds)

    # Calculate validation accuracy and loss
    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%")

    # Save the model with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(),os.path.join(path, 'maf_model.pth'))
        print("Model Saved.")

    lr_scheduler.step()  # Update learning rate

  print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
  print("--------------------------------")
  return model


def evaluation(path, model, test_loader):
  # Load the saved model
  # model = MultimodalModel(bert_model, resnet_model).to(device)
  print("Model is Loading..")
  model.load_state_dict(torch.load(os.path.join(path, 'maf_model.pth')))
  model.eval()
  print("Loaded.")

  test_labels = []
  test_preds = []

  print("--------------------------------")
  print("Start Evaluating..")
 # testing
  with torch.no_grad(), tqdm(test_loader, desc="Testing", unit="batch") as t:
      for batch in t:
          images = batch['image'].to(device)
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['label'].float().to(device)

          outputs = model(images, input_ids, attention_mask)
          preds = torch.argmax(outputs, dim=1).cpu().numpy()

          test_labels.extend(labels.cpu().numpy())
          test_preds.extend(preds)

  print('Evaluation Done.')
  print("--------------------------------")
  return test_labels, test_preds