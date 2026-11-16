"""Carregamento da base Olist, filtragem de registros e construção de Documents LangChain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
from langchain_core.documents import Document

from src.config import DEFAULT_RAW_CSV, DEFAULT_PROCESSED_CSV, CLEANING_AUDIT_PATH
from src.preprocessing.cleaner import combine_title_and_message, is_informative_text

REQUIRED_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
]


def load_and_clean_reviews(
    raw_csv_path: Path = DEFAULT_RAW_CSV,
    save_processed_path: Path | None = DEFAULT_PROCESSED_CSV,
    audit_path: Path | None = CLEANING_AUDIT_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carrega o CSV bruto da Olist, aplica tratamento de nulos, limpeza e desduplicação."""
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Arquivo de avaliações não encontrado em: {raw_csv_path}")

    df_raw = pd.read_csv(raw_csv_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {missing_cols}")

    total_rows = int(len(df_raw))
    df = df_raw[REQUIRED_COLUMNS].copy()

    # Contabiliza ausentes antes do tratamento
    null_title_count = int(df["review_comment_title"].isna().sum())
    null_message_count = int(df["review_comment_message"].isna().sum())
    both_null_count = int((df["review_comment_title"].isna() & df["review_comment_message"].isna()).sum())

    # Consolida título e comentário tratando ausentes
    df["cleaned_text"] = [
        combine_title_and_message(t, m)
        for t, m in zip(df["review_comment_title"], df["review_comment_message"])
    ]

    # Filtra comentários vazios após limpeza
    df_non_empty = df[df["cleaned_text"].str.len() > 0].copy()
    non_empty_rows = int(len(df_non_empty))

    # Filtra comentários curtos/não informativos (ex: "ok", "bom", "...")
    informative_mask = df_non_empty["cleaned_text"].apply(is_informative_text)
    df_informative = df_non_empty[informative_mask].copy()
    informative_rows = int(len(df_informative))

    # Normaliza score e data
    df_informative["review_score"] = pd.to_numeric(df_informative["review_score"], errors="coerce").fillna(0).astype(int)
    df_informative = df_informative[df_informative["review_score"].between(1, 5)].copy()
    df_informative["review_creation_date"] = (
        df_informative["review_creation_date"].fillna("").astype(str).str.slice(0, 10)
    )

    # Desduplicação de textos idênticos (case-insensitive) mantendo a primeira ocorrência
    df_informative["_text_key"] = df_informative["cleaned_text"].str.lower()
    df_dedup = df_informative.drop_duplicates(subset=["_text_key"], keep="first").drop(columns=["_text_key"]).copy()
    df_dedup = df_dedup.drop_duplicates(subset=["review_id"], keep="first").reset_index(drop=True)
    final_rows = int(len(df_dedup))

    score_distribution = {
        str(k): int(v)
        for k, v in df_dedup["review_score"].value_counts().sort_index().items()
    }

    audit = {
        "source_file": str(raw_csv_path.name),
        "total_raw_rows": total_rows,
        "null_title_rows": null_title_count,
        "null_message_rows": null_message_count,
        "both_title_and_message_null_rows": both_null_count,
        "non_empty_text_rows": non_empty_rows,
        "informative_text_rows": informative_rows,
        "removed_uninformative_or_short_rows": non_empty_rows - informative_rows,
        "removed_duplicate_texts": informative_rows - final_rows,
        "final_indexed_reviews": final_rows,
        "score_distribution": score_distribution,
    }

    if save_processed_path is not None:
        save_processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_dedup.to_csv(save_processed_path, index=False, encoding="utf-8")

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    return df_dedup, audit


def build_documents_from_df(df: pd.DataFrame) -> List[Document]:
    """Transforma o DataFrame limpo em objetos Document do LangChain com metadados rastreáveis."""
    documents: List[Document] = []
    for row in df.itertuples(index=False):
        metadata = {
            "review_id": str(row.review_id),
            "order_id": str(row.order_id),
            "review_score": int(row.review_score),
            "review_creation_date": str(row.review_creation_date),
        }
        documents.append(Document(page_content=str(row.cleaned_text), metadata=metadata))
    return documents
