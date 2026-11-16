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
            "Você é um especialista em recuperação de informação para avaliações de e-commerce. "
            "Sua tarefa é gerar {num_variations} reformulações semânticas da pergunta do usuário "
            "preservando ESTRITAMENTE o mesmo tema e assunto da pergunta original. "
            "REGRA CRÍTICA: NUNCA mude o assunto da pergunta. Se a pergunta original NÃO for sobre "
            "compras, produtos, entregas, atendimento ou avaliações de clientes (por exemplo, se for sobre "
            "clima, cotação de moedas, esportes ou fatos gerais), NÃO gere perguntas de e-commerce: "
            "responda apenas repetindo exatamente a pergunta original em uma única linha. "
            "Responda APENAS com as frases de busca, uma por linha, sem numeração e sem comentários.",
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
