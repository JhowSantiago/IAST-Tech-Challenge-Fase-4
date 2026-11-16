"""Inicialização e validação do provedor de embeddings OpenAI."""

from __future__ import annotations

import os
import re
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import RAGConfig


def mask_secret(text: object) -> str:
    """Remove chaves de API que possam aparecer em logs ou mensagens de erro."""
    return re.sub(r"(sk-[A-Za-z0-9_\-]{6,})", "sk-<REDACTED>", str(text))


def get_embeddings_model(config: RAGConfig | None = None) -> Embeddings:
    """Instancia o modelo de embeddings da OpenAI configurado no projeto."""
    cfg = config or RAGConfig()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sua-chave-openai-aqui":
        raise EnvironmentError(
            "Variável OPENAI_API_KEY não configurada. Defina sua chave no arquivo .env."
        )
    return OpenAIEmbeddings(model=cfg.embedding_model)
