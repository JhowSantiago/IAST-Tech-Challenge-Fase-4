"""Configurações centralizadas do projeto Voice of Customer Intelligence com RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDICES_DIR = DATA_DIR / "indices"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_RAW_CSV = RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
DEFAULT_PROCESSED_CSV = PROCESSED_DATA_DIR / "olist_reviews_cleaned.csv"
DEFAULT_FAISS_INDEX_DIR = INDICES_DIR / "faiss_olist_reviews"
CLEANING_AUDIT_PATH = REPORTS_DIR / "data-cleaning-audit.json"
EVALUATION_EXAMPLES_PATH = REPORTS_DIR / "rag-evaluation-examples.json"


@dataclass(frozen=True)
class RAGConfig:
    """Parâmetros de modelo e governança do pipeline RAG."""

    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    multi_query_count: int = int(os.getenv("RAG_MULTI_QUERY_COUNT", "3"))
    relevance_threshold: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.30"))
    min_evidence_docs: int = 1
    temperature: float = 0.0


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Com base nas avaliações disponíveis na base de conhecimento da Olist, "
    "não foram encontradas evidências suficientes para responder a esta pergunta "
    "com segurança. A questão pode estar fora do escopo dos relatos de clientes "
    "ou não possuir ocorrências relevantes registradas."
)
