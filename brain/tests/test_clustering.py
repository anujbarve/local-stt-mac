"""Tests for brain/clustering.py — TopicClusterer."""

import pytest

from brain.clustering import TopicClusterer, Cluster


@pytest.fixture
def clusterer():
    return TopicClusterer()


def _make_note(idx, topics=None, text=None):
    return {
        "id": str(idx),
        "text": text or f"note {idx}",
        "topics": topics or [],
    }


class TestTopicClusterer:
    def test_cluster_empty(self, clusterer):
        result = clusterer.cluster([])
        assert result == []

    def test_cluster_by_topic(self, clusterer):
        notes = [
            _make_note(0, topics=["technology", "python"]),
            _make_note(1, topics=["technology", "rust"]),
            _make_note(2, topics=["health", "fitness"]),
            _make_note(3, topics=["health", "nutrition"]),
        ]
        clusters = clusterer.cluster(notes)
        assert len(clusters) >= 2
        # Each cluster should have a label
        for c in clusters:
            assert c.label != ""

    def test_cluster_single_note(self, clusterer):
        notes = [_make_note(0, topics=["tech"])]
        clusters = clusterer.cluster(notes)
        assert len(clusters) >= 1

    def test_cluster_untagged(self, clusterer):
        notes = [
            _make_note(0, topics=[]),
            _make_note(1, topics=[]),
        ]
        clusters = clusterer.cluster(notes)
        assert any(c.label == "untagged" for c in clusters)

    def test_cluster_returns_clusters(self, clusterer):
        notes = [_make_note(i, topics=["t1"]) for i in range(10)]
        clusters = clusterer.cluster(notes)
        for c in clusters:
            assert isinstance(c, Cluster)
            assert c.size > 0

    def test_find_emerging_topics(self, clusterer):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        notes = [
            {"id": "1", "topics": ["ai", "python"], "timestamp": now.isoformat()},
            {"id": "2", "topics": ["ai", "ml"], "timestamp": now.isoformat()},
            {"id": "3", "topics": ["ai"], "timestamp": (now - timedelta(days=30)).isoformat()},
        ]
        emerging = clusterer.find_emerging_topics(notes, time_window_days=7)
        assert isinstance(emerging, list)
        # ai should be emerging (more recent mentions)
        topics = [e["topic"] for e in emerging]
        assert "ai" in topics

    def test_find_emerging_topics_empty(self, clusterer):
        result = clusterer.find_emerging_topics([])
        assert result == []

    def test_suggest_topic_hierarchy(self, clusterer):
        notes = [
            _make_note(0, topics=["python", "coding"]),
            _make_note(1, topics=["python", "data"]),
            _make_note(2, topics=["coding", "debugging"]),
        ]
        hierarchy = clusterer.suggest_topic_hierarchy(notes)
        assert isinstance(hierarchy, dict)

    def test_find_outliers(self, clusterer):
        notes = [
            _make_note(0, topics=["tech"]),
            _make_note(1, topics=[]),
            _make_note(2, topics=["health"]),
        ]
        outliers = clusterer.find_outliers(notes)
        assert len(outliers) == 1
        assert outliers[0]["reason"] == "no_topics"

    def test_cluster_label(self, clusterer):
        notes = [
            _make_note(0, topics=["python", "python", "python"]),
            _make_note(1, topics=["rust"]),
        ]
        clusters = clusterer.cluster(notes)
        labels = [c.label for c in clusters]
        assert "python" in labels

    def test_euclidean_distance(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        dist = TopicClusterer._euclidean(a, b)
        assert abs(dist - 1.414) < 0.01
