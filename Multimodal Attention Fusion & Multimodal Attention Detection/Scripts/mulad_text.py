"""
Text pipeline for MuLAD (Hasan et al.) - vocabulary, padded sequences, embeddings.

MuLAD's text branch is not a transformer: it embeds tokens with a plain lookup table and
feeds the sequence to a small CNN / BiLSTM. That means we need the Keras-tokenizer stage
the paper describes (Sec. 4.1) rather than the Bangla-BERT tokenizer MAF uses.

Paper settings reproduced here:
  * Keras embedding  - vocab 8000, 64 dimensions, max sequence length 130 (Sec. 4.2)
  * FastText / GloVe - 300 dimensions, window 5, min_count 5 (Sec. 4.2)

A note on "GloVe": the paper reports it as "trained with a dimension of 300, a window size
of 5, and a min_count of 5". `window` and `min_count` are gensim training parameters, not
properties of a downloaded GloVe release, and no official Bengali GloVe exists. We read
that row as embeddings the authors trained themselves on their own corpus, so `--embedding
selftrained` trains word vectors on our captions with exactly those hyperparameters.
"""
import os
import re
from collections import Counter

import numpy as np

PAD, OOV = 0, 1
PAD_TOKEN, OOV_TOKEN = "<pad>", "<oov>"

# Keras' text_to_word_sequence filter set, minus characters that carry meaning in Bengali.
# Bengali danda/double-danda are treated as punctuation, as the paper's preprocessing does
# (Sec. 4.1: "remove punctuation, hyperlinks, emojis, and special characters").
_URL = re.compile(r"https?://\S+|www\.\S+")
_KEEP = re.compile(r"[^\u0980-\u09FFa-zA-Z0-9\s]")


def preprocess(text):
    """Lowercase, strip URLs, drop punctuation/emoji/symbols, collapse whitespace."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _KEEP.sub(" ", text)
    return " ".join(text.split())


def tokenize(text):
    return preprocess(text).split()


def build_vocab(texts, num_words=8000, min_count=1):
    """Keras-Tokenizer-equivalent vocabulary: most frequent `num_words` tokens.

    Index 0 is padding and index 1 is out-of-vocabulary, so the embedding layer needs
    `len(vocab)` rows and real tokens start at 2.
    """
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    kept = [(w, c) for w, c in counts.most_common() if c >= min_count]
    if num_words:
        kept = kept[: max(0, num_words - 2)]

    vocab = {PAD_TOKEN: PAD, OOV_TOKEN: OOV}
    for word, _ in kept:
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts, vocab, max_len):
    """Encode and post-pad/truncate to `max_len`, matching keras pad_sequences defaults."""
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for row, text in enumerate(texts):
        ids = [vocab.get(tok, OOV) for tok in tokenize(text)][:max_len]
        out[row, : len(ids)] = ids
    return out


def _load_vec_file(path, vocab, dim):
    """Read a word2vec/fastText text-format .vec file, keeping only words we need."""
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    rng = np.random.RandomState(42)
    matrix[OOV] = rng.normal(0, 0.1, dim)
    found = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first = fh.readline().split()
        if len(first) > 2:  # no header line - rewind and treat it as data
            fh.seek(0)
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            idx = vocab.get(word)
            if idx is None or len(parts) != dim + 1:
                continue
            matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
            found += 1
    return matrix, found


def _train_gensim(corpus, vocab, dim, kind, window=5, min_count=5, seed=42):
    """Train word vectors on our own captions with the paper's hyperparameters."""
    try:
        from gensim.models import FastText, Word2Vec
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "gensim is required for --embedding {}. Install it with `pip install gensim`, "
            "or use --embedding keras (no external vectors needed).".format(kind)
        ) from exc

    sentences = [tokenize(t) for t in corpus]
    cls = FastText if kind == "fasttext" else Word2Vec
    model = cls(
        sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=0,            # CBOW, as the paper states for FastText
        workers=4,
        seed=seed,
        epochs=20,
    )

    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    rng = np.random.RandomState(seed)
    matrix[OOV] = rng.normal(0, 0.1, dim)
    found = 0
    for word, idx in vocab.items():
        if idx in (PAD, OOV):
            continue
        try:
            matrix[idx] = model.wv[word]
            found += 1
        except KeyError:
            matrix[idx] = rng.normal(0, 0.1, dim)
    return matrix, found


def build_embedding_matrix(kind, vocab, dim, corpus=None, vectors_path=None):
    """Return (matrix, trainable) for the requested embedding source.

    kind='keras'       -> None; the model learns a 64-d table from scratch (paper Sec. 4.2)
    kind='fasttext'    -> pretrained cc.bn.300 vectors if `vectors_path` exists,
                          else FastText trained on our captions
    kind='selftrained' -> Word2Vec trained on our captions (the paper's "GloVe" row)
    """
    if kind == "keras":
        return None, True

    if kind == "fasttext" and vectors_path and os.path.isfile(vectors_path):
        matrix, found = _load_vec_file(vectors_path, vocab, dim)
        print("Loaded pretrained FastText: {}/{} vocabulary words found".format(found, len(vocab)))
        return matrix, False

    if kind == "fasttext":
        print("No pretrained vectors at {} - training FastText on our captions instead."
              .format(vectors_path))

    if corpus is None:
        raise ValueError("corpus is required to train embeddings for kind={}".format(kind))

    matrix, found = _train_gensim(corpus, vocab, dim, "fasttext" if kind == "fasttext" else "w2v")
    print("Trained {} on our captions: {}/{} vocabulary words covered"
          .format(kind, found, len(vocab)))
    return matrix, False
