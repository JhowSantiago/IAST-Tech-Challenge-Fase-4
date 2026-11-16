"""Módulo de geração de embeddings e indexação vetorial com FAISS."""

from src.indexing.embeddings import get_embeddings_model, mask_secret
from src.indexing.vector_store import build_faiss_index, load_faiss_index

__all__ = [
    "get_embeddings_model",
    "mask_secret",
    "build_faiss_index",
    "load_faiss_index",
]
