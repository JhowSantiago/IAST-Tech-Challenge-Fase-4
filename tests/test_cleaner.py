"""Testes unitários para higienização textual e tratamento de valores ausentes."""

from __future__ import annotations

import math
from src.preprocessing.cleaner import (
    clean_review_text,
    combine_title_and_message,
    is_informative_text,
)


def test_clean_review_text_handles_missing_and_whitespace() -> None:
    assert clean_review_text(None) == ""
    assert clean_review_text(math.nan) == ""
    assert clean_review_text("   \n\t  ") == ""
    assert clean_review_text("Produto   muito\n\nbom!!!!") == "Produto muito bom!!!"


def test_combine_title_and_message_merges_fields_correctly() -> None:
    assert combine_title_and_message("Recomendo", "Entrega super rápida e produto ótimo.") == (
        "Recomendo. Entrega super rápida e produto ótimo."
    )
    assert combine_title_and_message(None, "Veio com defeito na embalagem.") == (
        "Veio com defeito na embalagem."
    )
    assert combine_title_and_message("Péssimo atendimento!", None) == "Péssimo atendimento!"
    assert combine_title_and_message("Bom", "Bom demais, chegou antes do prazo") == (
        "Bom demais, chegou antes do prazo"
    )


def test_is_informative_text_filters_short_or_noise_entries() -> None:
    assert not is_informative_text("")
    assert not is_informative_text("ok")
    assert not is_informative_text("bom")
    assert not is_informative_text("...")
    assert is_informative_text("O produto chegou antes do prazo e bem embalado.")
