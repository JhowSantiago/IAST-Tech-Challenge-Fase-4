"""Funções de limpeza, normalização e tratamento de valores ausentes em avaliações."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
import pandas as pd

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.]){4,}")
_ALPHANUM_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]")

MIN_TEXT_CHARS = 8
MIN_ALPHA_TOKENS = 2


def _is_missing(value: Any) -> bool:
    """Verifica se um valor é nulo, NaN ou string vazia."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def clean_review_text(text: Any) -> str:
    """Limpa e normaliza um texto de avaliação preservando acentuação e semântica em PT-BR."""
    if _is_missing(text):
        return ""

    raw = str(text)
    normalized = unicodedata.normalize("NFC", raw)
    without_ctrl = _CONTROL_CHARS_RE.sub(" ", normalized)
    collapsed_ws = _WHITESPACE_RE.sub(" ", without_ctrl).strip()
    cleaned = _REPEATED_PUNCT_RE.sub(r"\1\1\1", collapsed_ws)
    return cleaned


def combine_title_and_message(title: Any, message: Any) -> str:
    """Consolida título e mensagem tratando valores ausentes em qualquer um dos campos."""
    clean_title = clean_review_text(title)
    clean_msg = clean_review_text(message)

    if clean_title and clean_msg:
        if clean_msg.lower().startswith(clean_title.lower()):
            return clean_msg
        if clean_title.endswith((".", "!", "?", ":")):
            return f"{clean_title} {clean_msg}"
        return f"{clean_title}. {clean_msg}"
    return clean_msg or clean_title


def is_informative_text(text: str, min_chars: int = MIN_TEXT_CHARS, min_tokens: int = MIN_ALPHA_TOKENS) -> bool:
    """Valida se o texto possui conteúdo semântico mínimo para indexação no RAG."""
    if not text or len(text.strip()) < min_chars:
        return False
    tokens = [tok for tok in text.split() if _ALPHANUM_RE.search(tok)]
    return len(tokens) >= min_tokens
