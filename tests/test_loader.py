"""Testes unitários para carregamento da base Olist, desduplicação e criação de Documents."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.preprocessing.loader import build_documents_from_df, load_and_clean_reviews


def test_load_and_clean_reviews_filters_nulls_and_duplicates(tmp_path: Path) -> None:
    raw_csv = tmp_path / "olist_sample.csv"
    processed_csv = tmp_path / "processed.csv"
    audit_json = tmp_path / "audit.json"

    df_input = pd.DataFrame(
        [
            {
                "review_id": "rev_1",
                "order_id": "ord_1",
                "review_score": 1,
                "review_comment_title": "Atraso na entrega",
                "review_comment_message": "O pedido atrasou mais de duas semanas e ninguém responde.",
                "review_creation_date": "2018-05-10 00:00:00",
            },
            {
                "review_id": "rev_2",
                "order_id": "ord_2",
                "review_score": 1,
                "review_comment_title": "Atraso na entrega",
                "review_comment_message": "O pedido atrasou mais de duas semanas e ninguém responde.",
                "review_creation_date": "2018-05-11 00:00:00",
            },
            {
                "review_id": "rev_3",
                "order_id": "ord_3",
                "review_score": 5,
                "review_comment_title": None,
                "review_comment_message": None,
                "review_creation_date": "2018-05-12 00:00:00",
            },
            {
                "review_id": "rev_4",
                "order_id": "ord_4",
                "review_score": 5,
                "review_comment_title": "",
                "review_comment_message": "ok",
                "review_creation_date": "2018-05-13 00:00:00",
            },
            {
                "review_id": "rev_5",
                "order_id": "ord_5",
                "review_score": 5,
                "review_comment_title": "Excelente compra",
                "review_comment_message": "Material de alta qualidade, recomendo a loja.",
                "review_creation_date": "2018-05-14 00:00:00",
            },
        ]
    )
    df_input.to_csv(raw_csv, index=False)

    df_clean, audit = load_and_clean_reviews(
        raw_csv_path=raw_csv,
        save_processed_path=processed_csv,
        audit_path=audit_json,
    )

    assert audit["total_raw_rows"] == 5
    assert audit["both_title_and_message_null_rows"] == 1
    assert audit["removed_uninformative_or_short_rows"] == 1
    assert audit["removed_duplicate_texts"] == 1
    assert audit["final_indexed_reviews"] == 2
    assert len(df_clean) == 2
    assert processed_csv.exists()
    assert audit_json.exists()

    docs = build_documents_from_df(df_clean)
    assert len(docs) == 2
    assert docs[0].metadata["review_id"] == "rev_1"
    assert docs[0].metadata["review_score"] == 1
    assert "atrasou mais de duas semanas" in docs[0].page_content
