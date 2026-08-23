"""
brain — The Intelligence Layer
===============================

Custom neural network for understanding transcribed text.
Classifies, extracts entities, determines priority, and generates
embeddings — all in a single forward pass.

Usage:
    from brain.understand import BrainEngine
    engine = BrainEngine.load("checkpoints/brain/latest")
    result = engine.understand("I need to fix the VAD threshold by Friday")
"""
