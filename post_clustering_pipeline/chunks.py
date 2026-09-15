"""Generic list chunking helper used across the pipeline.

Kept standalone so claim and assignment phases share one, well-defined
iteration contract rather than re-implementing slice math.
"""
from __future__ import annotations


def iter_chunks(seq: list, size: int):
    """Yield ``seq`` in fixed-size slices as ``(chunk, start, end)`` tuples.

    ``end`` is exclusive and ``end - start == len(chunk)`` always holds, so
    parallel lists (e.g. items and their embeddings) stay index-aligned.
    """
    if not seq:
        return
    if size <= 0:
        yield seq, 0, len(seq)
        return
    for start in range(0, len(seq), size):
        yield seq[start:start + size], start, min(start + size, len(seq))