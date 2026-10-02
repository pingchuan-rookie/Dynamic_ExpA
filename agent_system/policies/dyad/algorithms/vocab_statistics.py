"""Compatibility import for shared bounded-memory vocabulary statistics."""

from verl_extensions.vocab_statistics import VOCAB_STATISTICS_CHUNK_SIZE, compute_vocab_statistics

__all__ = ["VOCAB_STATISTICS_CHUNK_SIZE", "compute_vocab_statistics"]
