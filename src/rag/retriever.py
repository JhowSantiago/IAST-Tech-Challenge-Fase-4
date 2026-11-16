"""Recuperação semântica de avaliações com deduplicação e corte por limiar de relevância."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document


def l2_distance_to_cosine_similarity(squared_l2_distance: float) -> float:
    """Converte distância L2 quadrática de vetores normalizados em similaridade do cosseno [0, 1]."""
    sim = 1.0 - (float(squared_l2_distance) / 2.0)
    return max(0.0, min(1.0, sim))


@dataclass(frozen=True)
class RetrievedEvidence:
    """Representa uma avaliação recuperada com seu score de similaridade e rastreabilidade."""

    document: Document
    relevance_score: float
    matched_query: str

    def to_dict(self) -> dict:
        meta = self.document.metadata or {}
        return {
            "review_id": str(meta.get("review_id", "N/A")),
            "order_id": str(meta.get("order_id", "N/A")),
            "review_score": int(meta.get("review_score", 0)),
            "review_creation_date": str(meta.get("review_creation_date", "")),
            "relevance_score": round(float(self.relevance_score), 4),
            "matched_query": self.matched_query,
            "text": self.document.page_content,
        }


def retrieve_relevant_reviews(
    vector_store: FAISS,
    queries: Sequence[str],
    top_k: int = 5,
    relevance_threshold: float = 0.30,
) -> List[RetrievedEvidence]:
    """Executa busca semântica para múltiplas consultas, remove duplicatas e filtra por limiar."""
    if not queries:
        return []

    best_by_id: Dict[str, RetrievedEvidence] = {}

    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        for doc, l2_dist in results:
            numeric_score = l2_distance_to_cosine_similarity(float(l2_dist))
            if numeric_score < relevance_threshold:
                continue

            doc_key = str(doc.metadata.get("review_id") or doc.page_content)
            existing = best_by_id.get(doc_key)
            if existing is None or numeric_score > existing.relevance_score:
                best_by_id[doc_key] = RetrievedEvidence(
                    document=doc,
                    relevance_score=numeric_score,
                    matched_query=query,
                )

    ranked = sorted(
        best_by_id.values(),
        key=lambda item: item.relevance_score,
        reverse=True,
    )
    return ranked[:top_k]
