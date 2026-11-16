"""Construção, persistência e carregamento do banco vetorial FAISS."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import DEFAULT_FAISS_INDEX_DIR


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
        save_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(save_dir))

    return vector_store


def load_faiss_index(
    embeddings: Embeddings,
    index_dir: Path = DEFAULT_FAISS_INDEX_DIR,
) -> FAISS:
    """Carrega um índice vetorial FAISS previamente persistido em disco."""
    index_file = index_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Índice FAISS não encontrado em {index_dir}. Execute a etapa de indexação primeiro."
        )

    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )
