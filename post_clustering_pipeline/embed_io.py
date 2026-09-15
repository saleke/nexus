"""Single seam for pgvector <-> numpy <-> SQL text/binary conversion.

Centralizing the codec here means every embedding/centroid that crosses the
PostgreSQL boundary uses the exact same, locale-independent representation.
Text form is the psycopg2 bound-parameter format (pgvector-python's psycopg2
adapter emits text, not true wire-binary); the binary codec mirrors pgvector
0.8+ server binary I/O (``Vector.to_binary``) for bulk/edge paths.
"""
from __future__ import annotations

import struct

import numpy as np

VECTOR_SEPARATOR = ","

_BINARY_HEADER = ">HH"  # uint16 dimensions, uint16 unused (must be 0)
_FLOAT32 = np.dtype(">f4")


def vector_to_array_literal(vector) -> str:
    """Format an embedding/centroid as a PostgreSQL vector literal.

    Uses ``repr(float(v))`` which is locale-independent (always '.', never ',')
    and round-trips exactly, unlike ``str(np.float64)`` which is both
    locale-dependent and truncates precision.
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def parse_vector_literal(text: str) -> np.ndarray:
    """Parse a PostgreSQL ``[1.0,2.0,...]`` vector text into a float64 array."""
    if text is None:
        return np.array([], dtype=np.float64)
    body = text.strip().strip("[]").strip()
    if not body:
        return np.array([], dtype=np.float64)
    return np.array(body.split(VECTOR_SEPARATOR), dtype=np.float64)


def vector_to_binary(vector) -> bytes:
    """Encode a vector in pgvector's fixed-width binary format.

    Layout: uint16 dims, uint16 unused (0), then dim big-endian float32
    values. This is byte-for-byte ``pgvector.Vector.to_binary()``.
    """
    array = np.asarray(vector, dtype=np.float32).reshape(-1).astype(_FLOAT32, copy=True)
    return struct.pack(_BINARY_HEADER, array.size, 0) + array.tobytes()


def binary_to_vector(data: bytes) -> np.ndarray:
    """Decode pgvector fixed-width binary bytes into a native float32 array."""
    if len(data) < 4:
        raise ValueError("pgvector binary payload too short")
    dim, unused = struct.unpack_from(_BINARY_HEADER, data, 0)
    if unused != 0:
        raise ValueError(f"pgvector binary payload has non-zero unused field: {unused}")
    body = data[4:]
    if len(body) != dim * _FLOAT32.itemsize:
        raise ValueError(f"pgvector binary payload length mismatch: expected {dim * _FLOAT32.itemsize} bytes, got {len(body)}")
    return np.frombuffer(body, dtype=_FLOAT32).astype(np.float32)