"""
Tests for brain.data — synthetic data generator.
"""

import os
import tempfile

import pytest

from brain.data import SyntheticDataGenerator, generate_training_data


class TestDataGenerator:
    """Tests for SyntheticDataGenerator."""

    def test_generates_correct_count(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=100)
        assert len(data) == 100

    def test_all_required_keys(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=50)
        required_keys = {"text", "type", "priority", "urgency", "sentiment"}
        for sample in data:
            assert required_keys.issubset(sample.keys()), \
                f"Missing keys: {required_keys - sample.keys()}"

    def test_text_is_string(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=20)
        for sample in data:
            assert isinstance(sample["text"], str)
            assert len(sample["text"]) > 0

    def test_priority_in_range(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=100)
        for sample in data:
            assert 0.0 <= sample["priority"] <= 1.0

    def test_valid_urgency_values(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=100)
        valid_urgencies = {"immediate", "soon", "later", "someday"}
        for sample in data:
            assert sample["urgency"] in valid_urgencies

    def test_valid_sentiment_values(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=100)
        valid_sentiments = {"positive", "negative", "neutral"}
        for sample in data:
            assert sample["sentiment"] in valid_sentiments

    def test_all_note_types_covered(self):
        gen = SyntheticDataGenerator(seed=42)
        # Generate enough samples to cover all types
        data = gen.generate(num_samples=1000)
        types_found = {s["type"] for s in data}
        expected_types = {
            "task", "idea", "observation", "learning", "decision",
            "meeting", "reminder", "person_info", "preference", "journal",
        }
        assert expected_types.issubset(types_found)

    def test_entities_are_dicts(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=50)
        for sample in data:
            assert isinstance(sample["entities"], list)
            for ent in sample["entities"]:
                assert isinstance(ent, dict)
                assert "name" in ent
                assert "type" in ent

    def test_reproducible_with_seed(self):
        gen1 = SyntheticDataGenerator(seed=42)
        data1 = gen1.generate(num_samples=20)

        gen2 = SyntheticDataGenerator(seed=42)
        data2 = gen2.generate(num_samples=20)

        # Same seed should produce same texts
        for s1, s2 in zip(data1, data2):
            assert s1["text"] == s2["text"]

    def test_speech_artifacts_injected(self):
        gen = SyntheticDataGenerator(seed=42)
        # Generate many samples — at least some should have artifacts
        data = gen.generate(num_samples=200)
        has_filler = any(
            any(f in s["text"].lower() for f in ["um", "uh", "so basically", "you know"])
            for s in data
        )
        assert has_filler, "No speech artifacts found in any sample"

    def test_different_seeds_produce_different_data(self):
        gen1 = SyntheticDataGenerator(seed=1)
        data1 = gen1.generate(num_samples=20)
        gen2 = SyntheticDataGenerator(seed=2)
        data2 = gen2.generate(num_samples=20)
        texts1 = [s["text"] for s in data1]
        texts2 = [s["text"] for s in data2]
        assert texts1 != texts2


class TestDataSaveLoad:
    """Tests for save/load."""

    def test_save_and_load(self):
        gen = SyntheticDataGenerator(seed=42)
        data = gen.generate(num_samples=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            gen.save(data, path)
            assert os.path.exists(path)

            loaded = SyntheticDataGenerator.load(path)
            assert len(loaded) == len(data)
            for s1, s2 in zip(data, loaded):
                assert s1["text"] == s2["text"]
                assert s1["type"] == s2["type"]


class TestConvenienceFunction:
    """Tests for generate_training_data."""

    def test_generate_training_data(self):
        data = generate_training_data(num_samples=50)
        assert len(data) == 50

    def test_generate_with_custom_seed(self):
        data = generate_training_data(num_samples=20, seed=99)
        assert len(data) == 20
