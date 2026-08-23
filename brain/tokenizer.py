"""
brain.tokenizer — Custom tokenizer for spoken transcriptions.

Handles speech artifacts, lowercases everything, builds vocabulary
from training data. Outputs integer sequences padded to MAX_SEQ_LEN.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from brain.config import (
    VOCAB_SIZE, MAX_SEQ_LEN, SPECIAL_TOKENS,
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN,
)


class Tokenizer:
    """Word-level tokenizer built from training data."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._fitted = False

        # Initialize with special tokens
        for i, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = i
            self.idx2word[i] = token

    @property
    def pad_id(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.word2idx[CLS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.word2idx[SEP_TOKEN]

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from a list of texts."""
        counter = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            counter.update(tokens)

        # Reserve slots for special tokens
        max_words = self.vocab_size - len(SPECIAL_TOKENS)

        # Add most common words
        for word, _count in counter.most_common(max_words):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self._fitted = True

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Tokenize text to list of token IDs, padded to MAX_SEQ_LEN.

        Args:
            text: Input text string.
            add_special: If True, prepend [CLS] and append [SEP].

        Returns:
            List of integer token IDs, length == MAX_SEQ_LEN.
        """
        tokens = self._tokenize_text(text)

        ids = []
        if add_special:
            ids.append(self.cls_id)

        for token in tokens:
            ids.append(self.word2idx.get(token, self.unk_id))

        if add_special:
            ids.append(self.sep_id)

        # Truncate
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN - 1] + [self.sep_id]

        # Pad
        pad_len = MAX_SEQ_LEN - len(ids)
        ids.extend([self.pad_id] * pad_len)

        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text."""
        words = []
        for tid in token_ids:
            word = self.idx2word.get(tid, UNK_TOKEN)
            if skip_special and word in SPECIAL_TOKENS:
                continue
            words.append(word)
        return " ".join(words)

    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """Tokenize a batch of texts."""
        return [self.encode(t, add_special=add_special) for t in texts]

    def vocab_size_actual(self) -> int:
        """Actual number of tokens in vocabulary."""
        return len(self.word2idx)

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        tok._fitted = True
        return tok

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize a raw text string.

        Rules:
            1. Lowercase
            2. Split on whitespace and punctuation
            3. Keep apostrophes within words (don't → ["don't"])
        """
        text = text.lower().strip()

        # Split on whitespace
        raw_tokens = text.split()

        # Further split on punctuation, but keep apostrophes in contractions
        tokens = []
        for raw in raw_tokens:
            # Split trailing/leading punctuation
            sub_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[^\s]", raw)
            tokens.extend(sub_tokens)

        return tokens
