"""Pipeline completo de RAG com fundamentação em evidências e tratamento de recusa."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import INSUFFICIENT_EVIDENCE_MESSAGE, RAGConfig
from src.rag.query_transform import expand_query_variations
from src.rag.retriever import RetrievedEvidence, retrieve_relevant_reviews

NO_EVIDENCE_TOKEN = "SEM_EVIDENCIAS_SUFICIENTES"

GROUNDED_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um analista especialista em Voice of Customer (VoC) de e-commerce. "
            "Sua missão é responder à pergunta do usuário fundamentando-se EXCLUSIVAMENTE "
            "nas avaliações reais de clientes fornecidas no bloco de EVIDÊNCIAS abaixo.\n\n"
            "Regras obrigatórias:\n"
            "1. Utilize APENAS as informações presentes nas avaliações recuperadas. Nunca invente dados, percentuais ou fatos externos.\n"
            "2. Sempre cite explicitamente os identificadores das avaliações que sustentam cada ponto no formato [review_id: <id>, Nota: <nota>/5] e mencione trechos reais como evidência.\n"
            "3. Caso as avaliações fornecidas não contenham informações suficientes para responder à pergunta, ou caso a pergunta esteja fora do escopo das experiências de compra relatadas, inicie sua resposta EXATAMENTE com a palavra-chave "
            f"'{NO_EVIDENCE_TOKEN}' seguida de uma breve explicação de que a base não possui suporte para responder.",
        ),
        (
            "human",
            "EVIDÊNCIAS RECUPERADAS DA BASE DE CONHECIMENTO:\n"
            "{context}\n\n"
            "PERGUNTA DO USUÁRIO:\n"
            "{question}\n\n"
            "Elabore uma resposta estruturada, objetiva e fundamentada nas evidências acima:",
        ),
    ]
)


def format_context_block(evidences: Sequence[RetrievedEvidence]) -> str:
    """Formata as avaliações selecionadas em um bloco estruturado para o modelo de linguagem."""
    if not evidences:
        return "Nenhuma evidência relevante encontrada."

    lines: List[str] = []
    for idx, ev in enumerate(evidences, start=1):
        meta = ev.document.metadata or {}
        review_id = meta.get("review_id", "N/A")
        score = meta.get("review_score", "N/A")
        date = meta.get("review_creation_date", "N/A")
        lines.append(
            f"[Evidência #{idx} | review_id: {review_id} | Nota: {score}/5 | "
            f"Data: {date} | Similaridade: {ev.relevance_score:.3f}]\n"
            f"Comentário: \"{ev.document.page_content}\""
        )
    return "\n\n".join(lines)


def get_chat_model(config: RAGConfig | None = None) -> BaseChatModel:
    """Instancia o modelo de linguagem da OpenAI configurado no projeto."""
    cfg = config or RAGConfig()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sua-chave-openai-aqui":
        raise EnvironmentError(
            "Variável OPENAI_API_KEY não configurada. Defina sua chave no arquivo .env."
        )
    return ChatOpenAI(model=cfg.chat_model, temperature=cfg.temperature)


@dataclass
class RAGResponse:
    """Resultado estruturado de uma execução do pipeline RAG."""

    question: str
    expanded_queries: List[str]
    evidences: List[Dict[str, Any]]
    context_used: str
    answer: str
    has_sufficient_evidence: bool = True
    refusal_reason: str | None = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "refusal_reason": self.refusal_reason,
            "expanded_queries": self.expanded_queries,
            "retrieved_count": len(self.evidences),
            "answer": self.answer,
            "evidences": self.evidences,
        }


class VoiceOfCustomerRAG:
    """Orquestrador do pipeline RAG para exploração da voz do cliente."""

    def __init__(
        self,
        vector_store: FAISS,
        llm: BaseChatModel | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.vector_store = vector_store
        self.llm = llm or get_chat_model(self.config)

    def ask(self, question: str) -> RAGResponse:
        """Executa o fluxo completo: transformação, recuperação, contexto e geração fundamentada."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("A pergunta não pode ser vazia.")

        # 1. Transformação da pergunta (Multi-Query)
        expanded_queries = expand_query_variations(
            question=clean_question,
            llm=self.llm,
            num_variations=self.config.multi_query_count,
        )

        # 2. Recuperação semântica e seleção das avaliações mais relevantes
        retrieved = retrieve_relevant_reviews(
            vector_store=self.vector_store,
            queries=expanded_queries,
            top_k=self.config.top_k,
            relevance_threshold=self.config.relevance_threshold,
        )

        # 3. Verificação pré-LLM de evidências suficientes (Retrieval Gate)
        if len(retrieved) < self.config.min_evidence_docs:
            return RAGResponse(
                question=clean_question,
                expanded_queries=expanded_queries,
                evidences=[],
                context_used="",
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                has_sufficient_evidence=False,
                refusal_reason="below_relevance_threshold",
            )

        # 4. Construção do contexto estruturado
        context_block = format_context_block(retrieved)
        evidence_dicts = [ev.to_dict() for ev in retrieved]

        # 5. Geração da resposta final fundamentada nas evidências
        generation_chain = GROUNDED_RAG_PROMPT | self.llm | StrOutputParser()
        raw_answer = generation_chain.invoke(
            {"context": context_block, "question": clean_question}
        ).strip()

        # 6. Verificação pós-LLM de recusa por falta de suporte nos dados
        if raw_answer.startswith(NO_EVIDENCE_TOKEN) or NO_EVIDENCE_TOKEN in raw_answer[:60]:
            cleaned_explanation = raw_answer.replace(NO_EVIDENCE_TOKEN, "").lstrip(": -").strip()
            final_refusal = (
                f"{INSUFFICIENT_EVIDENCE_MESSAGE}\n\nDetalhamento: {cleaned_explanation}"
                if cleaned_explanation
                else INSUFFICIENT_EVIDENCE_MESSAGE
            )
            return RAGResponse(
                question=clean_question,
                expanded_queries=expanded_queries,
                evidences=evidence_dicts,
                context_used=context_block,
                answer=final_refusal,
                has_sufficient_evidence=False,
                refusal_reason="llm_insufficient_context_guardrail",
            )

        return RAGResponse(
            question=clean_question,
            expanded_queries=expanded_queries,
            evidences=evidence_dicts,
            context_used=context_block,
            answer=raw_answer,
            has_sufficient_evidence=True,
            refusal_reason=None,
        )
