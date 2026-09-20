"""
MuLAD architectures (Hasan et al.), reimplemented in PyTorch.

The paper builds these in Keras. Porting to PyTorch keeps MuLAD inside the same data
splits, metric code and Outputs/ format as the MAF replication, so the two frameworks are
compared on identical footing rather than across two toolchains. The layer shapes,
activations and hyperparameters below are the paper's.

Text branches (Sec. 4.2, Sec. 5.1)
  cnn         Conv1d(128 filters, kernel 5) -> ReLU -> global max pool -> Dense(32) -> ReLU
  bilstm      BiLSTM(100 units) -> final hidden states concatenated -> Dense(32) -> ReLU
  bilstm_cnn  Conv1d -> ReLU -> MaxPool1d(2) -> BiLSTM(100) -> Dense(32) -> ReLU

Visual branch (Sec. 4.2, Sec. 5.1)
  frozen conv map -> flatten | global average pool -> Dropout -> Dense(32) -> ReLU

Fusion (Sec. 4.3)
  concatenate(text, visual) -> Dropout -> Dense(num_classes)

Three deviations, all forced and all deliberate:

  1. The paper's output layer is a single sigmoid unit with `binary_crossentropy`, because
     AMemD is a binary corpus. MIMOSA is five-way, so the head is a softmax over
     `num_classes` trained with cross-entropy. This is the minimum change that makes the
     architecture applicable at all.
  2. The paper contradicts itself on the CNN's width: Sec. 4.2 says "a filter size of 128",
     Sec. 5.1 says "32 units". Sec. 4.2 describes the proposed model, so 128 is the
     default; `--cnn_filters 32` runs the other reading.
  3. Only the CNN branch is given an explicit "dense layer with 32 neurons" in the paper.
     The same projection is applied to the BiLSTM branches so that all three contribute
     comparably sized vectors to the concatenation instead of the BiLSTM's 200 dominating
     the visual branch. `--text_dense 0` removes it.
"""
import torch
import torch.nn as nn

TEXT_MODELS = ("cnn", "bilstm", "bilstm_cnn")


def _embedding(vocab_size, emb_dim, matrix=None, trainable=True):
    layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    if matrix is not None:
        layer.weight.data.copy_(torch.as_tensor(matrix, dtype=torch.float32))
        layer.weight.requires_grad = bool(trainable)
    return layer


class TextBranch(nn.Module):
    """One of the paper's three textual feature extractors, minus the classifier head."""

    def __init__(self, kind, vocab_size, emb_dim=64, matrix=None, trainable=True,
                 cnn_filters=128, kernel_size=5, lstm_units=100, dense=32, dropout=0.3):
        super().__init__()
        if kind not in TEXT_MODELS:
            raise ValueError("Unknown text model: {}".format(kind))
        self.kind = kind
        self.embedding = _embedding(vocab_size, emb_dim, matrix, trainable)
        self.dropout = nn.Dropout(dropout)

        if kind == "cnn":
            self.conv = nn.Conv1d(emb_dim, cnn_filters, kernel_size, padding=kernel_size // 2)
            feat_dim = cnn_filters
        elif kind == "bilstm":
            self.lstm = nn.LSTM(emb_dim, lstm_units, batch_first=True, bidirectional=True)
            feat_dim = lstm_units * 2
        else:  # bilstm_cnn - the BiLSTM sits "atop the CNN network" (Sec. 5.1)
            self.conv = nn.Conv1d(emb_dim, cnn_filters, kernel_size, padding=kernel_size // 2)
            self.pool = nn.MaxPool1d(2)
            self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True, bidirectional=True)
            feat_dim = lstm_units * 2

        self.project = nn.Sequential(nn.Linear(feat_dim, dense), nn.ReLU()) if dense else nn.Identity()
        self.out_dim = dense if dense else feat_dim

    @staticmethod
    def _last_hidden(hidden):
        """Concatenate the forward and backward final hidden states."""
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

    def forward(self, tokens):
        emb = self.dropout(self.embedding(tokens))

        if self.kind == "cnn":
            feats = torch.relu(self.conv(emb.transpose(1, 2))).amax(dim=2)
        elif self.kind == "bilstm":
            _, (hidden, _) = self.lstm(emb)
            feats = self._last_hidden(hidden)
        else:
            conv = self.pool(torch.relu(self.conv(emb.transpose(1, 2))))
            _, (hidden, _) = self.lstm(conv.transpose(1, 2))
            feats = self._last_hidden(hidden)

        return self.project(feats)


class VisualBranch(nn.Module):
    """Head over a cached frozen conv map: flatten or GAP, dropout, dense, ReLU."""

    def __init__(self, feature_shape, pool="flatten", dense=32, dropout=0.5):
        super().__init__()
        channels = feature_shape[0]
        self.pool = pool
        in_dim = int(channels * feature_shape[1] * feature_shape[2]) if pool == "flatten" else channels
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, dense), nn.ReLU())
        self.out_dim = dense

    def forward(self, features):
        flat = features.flatten(1) if self.pool == "flatten" else features.mean(dim=(2, 3))
        return self.net(flat)


class MuLAD(nn.Module):
    """Text-only, visual-only, or the concatenation fusion the paper proposes."""

    def __init__(self, num_classes, text=None, visual=None, dropout=0.3):
        super().__init__()
        if text is None and visual is None:
            raise ValueError("MuLAD needs at least one modality")
        self.text = text
        self.visual = visual
        fused = (text.out_dim if text else 0) + (visual.out_dim if visual else 0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused, num_classes))

    @property
    def modality(self):
        if self.text and self.visual:
            return "multimodal"
        return "text" if self.text else "visual"

    def forward(self, tokens=None, features=None):
        parts = []
        if self.text is not None:
            parts.append(self.text(tokens))
        if self.visual is not None:
            parts.append(self.visual(features))
        return self.classifier(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])


def build(modality, num_classes, vocab_size=None, emb_dim=64, emb_matrix=None,
          emb_trainable=True, text_model="cnn", cnn_filters=128, kernel_size=5,
          lstm_units=100, text_dense=32, feature_shape=None, visual_pool="flatten",
          visual_dense=32, dropout=0.3):
    """Assemble the model for one cell of the paper's experiment grid."""
    text = None
    if modality in ("text", "multimodal"):
        text = TextBranch(text_model, vocab_size, emb_dim, emb_matrix, emb_trainable,
                          cnn_filters, kernel_size, lstm_units, text_dense, dropout)

    visual = None
    if modality in ("visual", "multimodal"):
        # The visual baselines pool globally (Sec. 5.1); fusion flattens (Sec. 4.2).
        pool = "gap" if (modality == "visual" and visual_pool == "auto") else \
               ("flatten" if visual_pool == "auto" else visual_pool)
        visual = VisualBranch(feature_shape, pool, visual_dense, dropout=0.5)

    return MuLAD(num_classes, text, visual, dropout)
