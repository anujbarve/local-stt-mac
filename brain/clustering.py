"""
brain.clustering — Topic clustering for notes.

Auto-clusters notes by semantic similarity, detects emerging topics,
suggests topic hierarchies, and finds outliers.

Usage:
    from brain.clustering import TopicClusterer

    clusterer = TopicClusterer()
    clusters = clusterer.cluster(notes)
    emerging = clusterer.find_emerging_topics(notes)
"""

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Cluster:
    """A cluster of semantically similar notes."""
    id: int
    label: str
    notes: List[str]  # note IDs or text snippets
    size: int = 0
    centroid: Optional[List[float]] = None


class TopicClusterer:
    """Cluster notes by semantic similarity and topic."""

    def __init__(self, embedder=None):
        self.embedder = embedder

    def cluster(
        self,
        notes: List[Dict],
        n_clusters: Optional[int] = None,
    ) -> List[Cluster]:
        """Cluster notes into groups.

        Uses simple K-means on embeddings if available, otherwise
        falls back to topic-based grouping.

        Args:
            notes: List of note dicts with id, text, topics.
            n_clusters: Number of clusters (auto-detect if None).

        Returns:
            List of Cluster objects.
        """
        if not notes:
            return []

        # Try embedding-based clustering if embeddings available
        has_embeddings = any(n.get("embedding") for n in notes)

        if has_embeddings and self.embedder is not None:
            return self._cluster_by_embedding(notes, n_clusters)

        # Fallback: cluster by topic tags
        return self._cluster_by_topic(notes)

    def _cluster_by_embedding(
        self,
        notes: List[Dict],
        n_clusters: Optional[int] = None,
    ) -> List[Cluster]:
        """K-means clustering on note embeddings."""
        # Collect embeddings
        embedded = [(i, n) for i, n in enumerate(notes) if n.get("embedding")]
        if len(embedded) < 2:
            return [Cluster(id=0, label="all", notes=[str(i) for i, _ in embedded], size=len(embedded))]

        embeddings = [n["embedding"] for _, n in embedded]
        indices = [i for i, _ in embedded]

        # Auto-determine k
        if n_clusters is None:
            n_clusters = min(max(2, len(embedded) // 5), 10)

        # Simple K-means
        centroids, assignments = self._kmeans(embeddings, n_clusters)

        # Build clusters
        clusters = []
        for k in range(n_clusters):
            member_indices = [indices[i] for i, a in enumerate(assignments) if a == k]
            if not member_indices:
                continue

            # Generate label from most common topic
            member_notes = [notes[i] for i in member_indices]
            label = self._generate_cluster_label(member_notes)

            clusters.append(Cluster(
                id=k,
                label=label,
                notes=[str(i) for i in member_indices],
                size=len(member_indices),
                centroid=centroids[k] if k < len(centroids) else None,
            ))

        return clusters

    def _cluster_by_topic(self, notes: List[Dict]) -> List[Cluster]:
        """Cluster notes by their topic tags."""
        topic_groups: Dict[str, List[str]] = defaultdict(list)

        for idx, note in enumerate(notes):
            topics = note.get("topics", [])
            note_id = str(note.get("id", note.get("note_id", idx)))

            if topics:
                for topic in topics:
                    topic_groups[topic].append(note_id)
            else:
                topic_groups["untagged"].append(note_id)

        clusters = []
        for idx, (topic, note_ids) in enumerate(sorted(topic_groups.items())):
            clusters.append(Cluster(
                id=idx,
                label=topic,
                notes=note_ids,
                size=len(note_ids),
            ))

        return clusters

    def find_emerging_topics(
        self,
        notes: List[Dict],
        time_window_days: int = 7,
    ) -> List[Dict]:
        """Detect topics that are gaining traction.

        Compares recent notes against older notes to find
        topics with increasing frequency.

        Args:
            notes: List of note dicts with topics and timestamps.
            time_window_days: Days to consider "recent".

        Returns:
            List of emerging topics with growth rates.
        """
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=time_window_days)

        recent_topics: Counter = Counter()
        older_topics: Counter = Counter()

        for note in notes:
            ts_str = note.get("timestamp", note.get("created", ""))
            topics = note.get("topics", [])

            try:
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    is_recent = ts >= cutoff
                else:
                    is_recent = True  # no timestamp, treat as recent
            except (ValueError, TypeError):
                is_recent = True

            for topic in topics:
                if is_recent:
                    recent_topics[topic] += 1
                else:
                    older_topics[topic] += 1

        emerging = []
        for topic, recent_count in recent_topics.items():
            older_count = older_topics.get(topic, 0)
            if older_count == 0:
                growth = float("inf") if recent_count > 1 else 1.0
            else:
                growth = recent_count / older_count

            if growth > 1.0 and recent_count >= 2:
                emerging.append({
                    "topic": topic,
                    "count_recent": recent_count,
                    "count_previous": older_count,
                    "growth_rate": round(growth, 2),
                })

        emerging.sort(key=lambda x: x["growth_rate"], reverse=True)
        return emerging

    def suggest_topic_hierarchy(
        self,
        notes: List[Dict],
    ) -> Dict[str, List[str]]:
        """Suggest parent-child topic relationships based on co-occurrence.

        Returns:
            Dict mapping parent topics to child topics.
        """
        # Count topic co-occurrence
        cooccurrence: Dict[str, Counter] = defaultdict(Counter)

        for note in notes:
            topics = note.get("topics", [])
            for i, t1 in enumerate(topics):
                for t2 in topics[i + 1:]:
                    cooccurrence[t1][t2] += 1
                    cooccurrence[t2][t1] += 1

        # Build hierarchy: most connected topic is parent
        all_topics = set(cooccurrence.keys())
        parent_scores: Dict[str, int] = {}
        for topic, connections in cooccurrence.items():
            parent_scores[topic] = sum(connections.values())

        # Sort by connectivity
        sorted_topics = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

        hierarchy: Dict[str, List[str]] = {}
        assigned: set = set()

        for parent_topic, _ in sorted_topics:
            if parent_topic in assigned:
                continue
            children = [
                t for t, count in cooccurrence[parent_topic].most_common(5)
                if t not in assigned and t != parent_topic
            ]
            if children:
                hierarchy[parent_topic] = children
                assigned.add(parent_topic)
                assigned.update(children)

        return hierarchy

    def find_outliers(
        self,
        notes: List[Dict],
        threshold: float = 0.3,
    ) -> List[Dict]:
        """Find notes that don't fit any cluster well.

        Returns notes with no topics or very unusual content.
        """
        outliers = []
        for idx, note in enumerate(notes):
            topics = note.get("topics", [])
            note_id = note.get("id", note.get("note_id", idx))

            # Notes with no topics are outliers
            if not topics:
                outliers.append({
                    "note_id": str(note_id),
                    "text": note.get("text", "")[:100],
                    "reason": "no_topics",
                })

        return outliers

    # ------------------------------------------------------------------
    # K-means implementation
    # ------------------------------------------------------------------

    def _kmeans(
        self,
        embeddings: List[List[float]],
        k: int,
        max_iterations: int = 50,
    ) -> Tuple[List[List[float]], List[int]]:
        """Simple K-means clustering.

        Returns (centroids, assignments).
        """
        n = len(embeddings)
        dim = len(embeddings[0]) if embeddings else 0

        # Initialize centroids randomly
        indices = list(range(n))
        random.shuffle(indices)
        centroids = [list(embeddings[indices[i]]) for i in range(min(k, n))]

        assignments = [0] * n

        for _ in range(max_iterations):
            # Assign points to nearest centroid
            new_assignments = []
            for emb in embeddings:
                min_dist = float("inf")
                best = 0
                for c_idx, cent in enumerate(centroids):
                    dist = self._euclidean(emb, cent)
                    if dist < min_dist:
                        min_dist = dist
                        best = c_idx
                new_assignments.append(best)

            # Check convergence
            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Update centroids
            for c_idx in range(len(centroids)):
                members = [embeddings[i] for i in range(n) if assignments[i] == c_idx]
                if members:
                    centroids[c_idx] = [
                        sum(m[d] for m in members) / len(members)
                        for d in range(dim)
                    ]

        return centroids, assignments

    @staticmethod
    def _euclidean(a: List[float], b: List[float]) -> float:
        """Euclidean distance between two vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _generate_cluster_label(self, notes: List[Dict]) -> str:
        """Generate a label for a cluster from its notes' topics."""
        topic_counts: Counter = Counter()
        for note in notes:
            for topic in note.get("topics", []):
                topic_counts[topic] += 1

        if topic_counts:
            return topic_counts.most_common(1)[0][0]
        return "mixed"
