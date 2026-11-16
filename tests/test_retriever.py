"""Testes unitários para indexação FAISS, busca semântica Multi-Query e limiar de relevância."""

from __future__ import annotations

from pathlib import Path
from typing import List
import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.indexing.vector_store import build_faiss_index, load_faiss_index
from src.rag.retriever import retrieve_relevant_reviews


class KeywordToyEmbeddings(Embeddings):
    """Embeddings determinísticos de teste para validar busca vetorial no FAISS sem chamada de rede."""

    KEYWORDS = ["entrega", "atraso", "prazo", "qualidade", "excelente", "defeito", "quebrado"]

    def _embed(self, text: str) -> List[float]:
        lower = text.lower()
        vec = [float(lower.count(kw)) for kw in self.KEYWORDS]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Vetor ortogonal para consultas fora de escopo
            return [0.0] * len(self.KEYWORDS) + [1.0]
        return [v / norm for v in vec] + [0.0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def test_faiss_persistence_and_semantic_retrieval(tmp_path: Path) -> None:
    docs = [
        Document(
            page_content="Atraso grande na entrega, passou muito do prazo prometido.",
            metadata={"review_id": "r_entrega_1", "order_id": "o1", "review_score": 1, "review_creation_date": "2018-01-10"},
        ),
        Document(
            page_content="Produto de excelente qualidade, superou as expectativas.",
            metadata={"review_id": "r_elogio_2", "order_id": "o2", "review_score": 5, "review_creation_date": "2018-02-15"},
        ),
        Document(
            page_content="Veio quebrado e com defeito na peça principal.",
            metadata={"review_id": "r_defeito_3", "order_id": "o3", "review_score": 1, "review_creation_date": "2018-03-20"},
        ),
    ]

    emb = KeywordToyEmbeddings()
    index_dir = tmp_path / "faiss_test"
    build_faiss_index(docs, emb, save_dir=index_dir)
    loaded_store = load_faiss_index(emb, index_dir=index_dir)

    results = retrieve_relevant_reviews(
        vector_store=loaded_store,
        queries=["problemas de atraso na entrega", "demora no prazo de entrega"],
        top_k=2,
        relevance_threshold=0.40,
    )

    assert len(results) == 1
    assert results[0].document.metadata["review_id"] == "r_entrega_1"
    assert results[0].relevance_score >= 0.40

    # Pergunta fora de escopo não deve retornar falsos positivos acima do limiar
    out_of_scope = retrieve_relevant_reviews(
        vector_store=loaded_store,
        queries=["cotação do bitcoin em marte"],
        top_k=2,
        relevance_threshold=0.40,
    )
    assert out_of_scope == []
