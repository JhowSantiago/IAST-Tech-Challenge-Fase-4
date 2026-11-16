"""Testes unitários para o pipeline RAG: geração fundamentada em evidências e recusa sem suporte."""

from __future__ import annotations

from typing import Any, List
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

from src.config import RAGConfig
from src.indexing.vector_store import build_faiss_index
from src.rag.pipeline import NO_EVIDENCE_TOKEN, VoiceOfCustomerRAG
from tests.test_retriever import KeywordToyEmbeddings


class StubVoCChatModel(SimpleChatModel):
    """LLM determinística de teste para validar o fluxo Multi-Query, Grounding e Guardrail."""

    @property
    def _llm_type(self) -> str:
        return "stub-voc-chat"

    def _call(self, messages: List[BaseMessage], stop: List[str] | None = None, **kwargs: Any) -> str:
        prompt_text = "\n".join(str(m.content) for m in messages)
        if "reformulações de busca semântica" in prompt_text:
            return "reclamações de atraso na entrega\nproblemas no prazo de entrega"
        if "EVIDÊNCIAS RECUPERADAS" in prompt_text:
            if "r_entrega_1" in prompt_text and "entrega" in prompt_text.lower():
                return (
                    "Os clientes relatam atraso no prazo de entrega como principal problema "
                    "[review_id: r_entrega_1, Nota: 1/5]."
                )
            return f"{NO_EVIDENCE_TOKEN}: As avaliações recuperadas não tratam do tema perguntado."
        return "Resposta padrão"


def test_rag_pipeline_returns_grounded_answer_with_citations() -> None:
    docs = [
        Document(
            page_content="Atraso grande na entrega, passou muito do prazo prometido.",
            metadata={"review_id": "r_entrega_1", "order_id": "o1", "review_score": 1, "review_creation_date": "2018-01-10"},
        ),
        Document(
            page_content="Produto de excelente qualidade.",
            metadata={"review_id": "r_elogio_2", "order_id": "o2", "review_score": 5, "review_creation_date": "2018-02-15"},
        ),
    ]
    store = build_faiss_index(docs, KeywordToyEmbeddings(), save_dir=None)
    config = RAGConfig(top_k=2, multi_query_count=2, relevance_threshold=0.40)
    rag = VoiceOfCustomerRAG(vector_store=store, llm=StubVoCChatModel(), config=config)

    response = rag.ask("Quais problemas existem na entrega?")

    assert response.has_sufficient_evidence is True
    assert response.refusal_reason is None
    assert len(response.expanded_queries) >= 2
    assert len(response.evidences) == 1
    assert response.evidences[0]["review_id"] == "r_entrega_1"
    assert "[review_id: r_entrega_1, Nota: 1/5]" in response.answer


def test_rag_pipeline_refuses_when_no_evidence_meets_threshold() -> None:
    docs = [
        Document(
            page_content="Atraso grande na entrega, passou muito do prazo prometido.",
            metadata={"review_id": "r_entrega_1", "order_id": "o1", "review_score": 1, "review_creation_date": "2018-01-10"},
        )
    ]
    store = build_faiss_index(docs, KeywordToyEmbeddings(), save_dir=None)
    # Multi-query desativado para testar pergunta pura fora de escopo no Retrieval Gate
    config = RAGConfig(top_k=2, multi_query_count=0, relevance_threshold=0.40)
    rag = VoiceOfCustomerRAG(vector_store=store, llm=StubVoCChatModel(), config=config)

    response = rag.ask("Qual é a capital da Austrália?")

    assert response.has_sufficient_evidence is False
    assert response.refusal_reason == "below_relevance_threshold"
    assert response.evidences == []
    assert "não foram encontradas evidências suficientes" in response.answer
