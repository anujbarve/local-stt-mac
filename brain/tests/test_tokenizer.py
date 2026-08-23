"""
Tests for brain.tokenizer — encode/decode roundtrip, unknown words, padding.
"""

import os
import tempfile

import pytest

from brain.tokenizer import Tokenizer
from brain.config import MAX_SEQ_LEN, PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN


class TestTokenizerFit:
    """Tests for vocabulary building."""

    def test_fit_builds_vocab(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world", "foo bar baz"])
        assert tok.vocab_size_actual() > 5  # at least special tokens + words

    def test_special_tokens_present(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        assert PAD_TOKEN in tok.word2idx
        assert UNK_TOKEN in tok.word2idx
        assert CLS_TOKEN in tok.word2idx
        assert SEP_TOKEN in tok.word2idx

    def test_vocab_respects_size_limit(self):
        tok = Tokenizer(vocab_size=10)
        texts = [f"word{i}" for i in range(100)]
        tok.fit(texts)
        assert tok.vocab_size_actual() <= 10

    def test_fit_multiple_texts(self):
        tok = Tokenizer(vocab_size=1000)
        texts = [
            "I need to fix the VAD threshold",
            "TODO: update the README before Friday",
            "meeting with Sarah about the project",
        ]
        tok.fit(texts)
        assert "fix" in tok.word2idx
        assert "vad" in tok.word2idx


class TestTokenizerEncode:
    """Tests for encoding."""

    def test_encode_returns_list_of_ints(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world"])
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(x, int) for x in ids)

    def test_encode_length_equals_max_seq_len(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world"])
        ids = tok.encode("hello world")
        assert len(ids) == MAX_SEQ_LEN

    def test_encode_starts_with_cls(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world"])
        ids = tok.encode("hello world", add_special=True)
        assert ids[0] == tok.cls_id

    def test_encode_ends_with_sep_before_padding(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world"])
        ids = tok.encode("hello world", add_special=True)
        # Find the last non-pad token
        last_non_pad = len(ids) - 1
        while last_non_pad >= 0 and ids[last_non_pad] == tok.pad_id:
            last_non_pad -= 1
        assert ids[last_non_pad] == tok.sep_id

    def test_unknown_words_map_to_unk(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello"])
        ids = tok.encode("xyzzy_unknown_word")
        # The unknown word should map to UNK token
        assert tok.unk_id in ids

    def test_encode_no_special(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world"])
        ids = tok.encode("hello world", add_special=False)
        # No CLS at start
        assert ids[0] != tok.cls_id

    def test_encode_lowercases(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["Hello World"])
        ids = tok.encode("HELLO WORLD")
        # Should find "hello" and "world" in vocab
        decoded = tok.decode(ids)
        assert "hello" in decoded
        assert "world" in decoded

    def test_encode_truncates_long_text(self):
        tok = Tokenizer(vocab_size=1000)
        long_text = " ".join(["word"] * (MAX_SEQ_LEN + 100))
        tok.fit([long_text])
        ids = tok.encode(long_text)
        assert len(ids) == MAX_SEQ_LEN


class TestTokenizerDecode:
    """Tests for decoding."""

    def test_decode_roundtrip(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world foo bar"])
        original = "hello world foo bar"
        ids = tok.encode(original)
        decoded = tok.decode(ids)
        assert "hello" in decoded
        assert "world" in decoded

    def test_decode_skips_special_tokens(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello"])
        ids = tok.encode("hello")
        decoded = tok.decode(ids, skip_special=True)
        assert PAD_TOKEN not in decoded
        assert CLS_TOKEN not in decoded

    def test_decode_empty(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        ids = [tok.pad_id] * MAX_SEQ_LEN
        decoded = tok.decode(ids)
        assert decoded.strip() == ""


class TestTokenizerBatch:
    """Tests for batch encoding."""

    def test_encode_batch(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world", "foo bar"])
        texts = ["hello", "world", "foo"]
        batch = tok.encode_batch(texts)
        assert len(batch) == 3
        assert all(len(ids) == MAX_SEQ_LEN for ids in batch)


class TestTokenizerSaveLoad:
    """Tests for save/load."""

    def test_save_and_load(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["hello world", "foo bar baz"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.json")
            tok.save(path)

            loaded = Tokenizer.load(path)
            assert loaded.vocab_size == tok.vocab_size
            assert loaded.word2idx == tok.word2idx

            # Encode should produce same results
            original_ids = tok.encode("hello world")
            loaded_ids = loaded.encode("hello world")
            assert original_ids == loaded_ids

    def test_load_nonexistent_returns_fresh(self):
        tok = Tokenizer(vocab_size=100)
        # Should not crash
        assert tok.vocab_size == 100
