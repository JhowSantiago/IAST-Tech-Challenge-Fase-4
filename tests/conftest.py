"""Configuração global do Pytest: garante inicialização do FAISS antes das bibliotecas nativas do Pandas no Windows."""

from __future__ import annotations

import faiss  # noqa: F401
