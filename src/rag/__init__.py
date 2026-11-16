"""Módulo do pipeline RAG: transformação de consulta, recuperação semântica e geração fundamentada."""

from src.rag.query_transform import expand_query_variations, extract_clean_variations
from src.rag.retriever import RetrievedEvidence, retrieve_relevant_reviews
from src.rag.pipeline import RAGResponse, VoiceOfCustomerRAG, format_context_block

__all__ = [
    "expand_query_variations",
    "extract_clean_variations",
    "RetrievedEvidence",
    "retrieve_relevant_reviews",
    "RAGResponse",
    "VoiceOfCustomerRAG",
    "format_context_block",
]
