"""Construção, persistência e carregamento do banco vetorial FAISS (compatível com caminhos Unicode no Windows)."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import DEFAULT_FAISS_INDEX_DIR


def _save_faiss_unicode_safe(vector_store: FAISS, save_dir: Path) -> None:
    """Salva index.faiss e index.pkl via I/O nativo do Python, evitando falha do C++ fopen em caminhos com acento."""
    save_dir.mkdir(parents=True, exist_ok=True)
    index_bytes = faiss.serialize_index(vector_store.index).tobytes()
    (save_dir / "index.faiss").write_bytes(index_bytes)
    with open(save_dir / "index.pkl", "wb") as f:
        pickle.dump((vector_store.docstore, vector_store.index_to_docstore_id), f)


def _load_faiss_unicode_safe(index_dir: Path, embeddings: Embeddings) -> FAISS:
    """Carrega index.faiss e index.pkl via I/O nativo do Python, suportando caminhos com acento no Windows."""
    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    if not index_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"Índice FAISS não encontrado em {index_dir}. Execute a etapa de indexação primeiro."
        )

    raw_bytes = index_file.read_bytes()
    index = faiss.deserialize_index(np.frombuffer(raw_bytes, dtype=np.uint8))
    with open(pkl_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        normalize_L2=True,
    )


def build_faiss_index(
    documents: Sequence[Document],
    embeddings: Embeddings,
    save_dir: Path | None = DEFAULT_FAISS_INDEX_DIR,
) -> FAISS:
    """Cria o índice vetorial FAISS a partir dos documentos com normalização L2 e persiste em disco."""
    if not documents:
        raise ValueError("A lista de documentos para indexação não pode estar vazia.")

    vector_store = FAISS.from_documents(
        list(documents),
        embeddings,
        normalize_L2=True,
    )

    if save_dir is not None:
        _save_faiss_unicode_safe(vector_store, save_dir)

    return vector_store


def load_faiss_index(
    embeddings: Embeddings,
    index_dir: Path = DEFAULT_FAISS_INDEX_DIR,
) -> FAISS:
    """Carrega um índice vetorial FAISS previamente persistido em disco."""
    return _load_faiss_unicode_safe(index_dir=index_dir, embeddings=embeddings)
