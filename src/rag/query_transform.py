"""Transformação da pergunta do usuário via expansão Multi-Query com LCEL."""

from __future__ import annotations

import re
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um especialista em Voice of Customer (VoC) para e-commerce brasileiro. "
            "Sua tarefa é gerar {num_variations} reformulações de busca semântica para a pergunta do usuário, "
            "utilizando termos e expressões reais que clientes usam ao escrever avaliações de compras online "
            "(como elogios, reclamações de prazo, entrega, defeito, qualidade ou atendimento). "
            "Se a pergunta original estiver totalmente fora do contexto de e-commerce ou avaliações de clientes, "
            "apenas repita a pergunta original. "
            "Responda APENAS com as frases de busca, uma por linha, sem numeração e sem comentários adicionais.",
        ),
        ("human", "Pergunta original: {question}"),
    ]
)


def extract_clean_variations(raw_output: str, limit: int) -> List[str]:
    """Limpa a saída textual da LLM extraindo uma variação de pergunta por linha."""
    variations: List[str] = []
    for line in raw_output.splitlines():
        cleaned = re.sub(r"^\s*(\d+[\.\)\-]|[-*•])\s*", "", line).strip()
        cleaned = cleaned.strip("\"'")
        if cleaned and cleaned not in variations:
            variations.append(cleaned)
    return variations[:limit]


def expand_query_variations(
    question: str,
    llm: BaseChatModel,
    num_variations: int = 3,
) -> List[str]:
    """Gera a lista de consultas combinando a pergunta original com as variações da LLM."""
    base_question = question.strip()
    if not base_question:
        raise ValueError("A pergunta do usuário não pode ser vazia.")

    if num_variations <= 0:
        return [base_question]

    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    raw_response = chain.invoke(
        {"question": base_question, "num_variations": num_variations}
    )
    variations = extract_clean_variations(raw_response, limit=num_variations)

    queries = [base_question]
    for var in variations:
        if var.lower() != base_question.lower() and var not in queries:
            queries.append(var)
    return queries
