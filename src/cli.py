"""Interface de linha de comando (CLI) para ingestão, indexação e exploração do RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from src.config import (
    CLEANING_AUDIT_PATH,
    DEFAULT_FAISS_INDEX_DIR,
    DEFAULT_PROCESSED_CSV,
    DEFAULT_RAW_CSV,
    EVALUATION_EXAMPLES_PATH,
    RAGConfig,
)
from src.indexing.embeddings import get_embeddings_model, mask_secret
from src.indexing.vector_store import build_faiss_index, load_faiss_index
from src.preprocessing.loader import build_documents_from_df, load_and_clean_reviews
from src.rag.pipeline import VoiceOfCustomerRAG

CANONICAL_QUESTIONS: List[str] = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "Quais reclamações relacionadas à entrega aparecem com maior frequência?",
    "O que os clientes mais elogiam em suas experiências de compra?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "Qual é a cotação atual do dólar e a previsão do tempo em Tóquio amanhã?",
]


def cmd_ingest(raw_csv: Path, sample_size: int | None = None) -> int:
    """Executa limpeza da base Olist, gera auditoria e constrói o índice vetorial FAISS."""
    print(f"[1/3] Carregando e higienizando base de avaliações: {raw_csv}")
    df_clean, audit = load_and_clean_reviews(
        raw_csv_path=raw_csv,
        save_processed_path=DEFAULT_PROCESSED_CSV,
        audit_path=CLEANING_AUDIT_PATH,
    )
    print(
        f"      Total bruto: {audit['total_raw_rows']} | "
        f"Após limpeza e desduplicação: {audit['final_indexed_reviews']}"
    )

    if sample_size is not None and sample_size > 0 and len(df_clean) > sample_size:
        df_clean = df_clean.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"      Amostra selecionada para indexação: {len(df_clean)} avaliações")

    docs = build_documents_from_df(df_clean)
    print(f"[2/3] Inicializando modelo de embeddings OpenAI ({RAGConfig().embedding_model})...")
    embeddings = get_embeddings_model()

    print(f"[3/3] Gerando embeddings e salvando índice FAISS em: {DEFAULT_FAISS_INDEX_DIR}")
    build_faiss_index(docs, embeddings, save_dir=DEFAULT_FAISS_INDEX_DIR)
    print("Indexação concluída com sucesso!")
    return 0


def _print_rag_result(result_dict: dict) -> None:
    print("\n" + "=" * 80)
    print(f"PERGUNTA: {result_dict['question']}")
    print("=" * 80)
    print("\n[Transformação da Pergunta - Consultas Expandidas]:")
    for q in result_dict["expanded_queries"]:
        print(f"  - {q}")

    print(f"\n[Evidências Recuperadas ({result_dict['retrieved_count']})]:")
    if not result_dict["evidences"]:
        print("  (Nenhuma avaliação atingiu o limiar mínimo de relevância semântica)")
    else:
        for idx, ev in enumerate(result_dict["evidences"], start=1):
            print(
                f"  {idx}. [review_id: {ev['review_id']} | Nota: {ev['review_score']}/5 | "
                f"Data: {ev['review_creation_date']} | Relevância: {ev['relevance_score']:.3f}]"
            )
            print(f"     \"{ev['text']}\"")

    status = "FUNDAMENTADA EM EVIDÊNCIAS" if result_dict["has_sufficient_evidence"] else "RECUSA POR EVIDÊNCIA INSUFICIENTE / FORA DE ESCOPO"
    print(f"\n[Status de Governança]: {status}")
    print("\n[Resposta Final]:")
    print(result_dict["answer"])
    print("=" * 80 + "\n")


def cmd_query(question: str) -> int:
    """Executa uma consulta no pipeline RAG e exibe as evidências e a resposta fundamentada."""
    embeddings = get_embeddings_model()
    vector_store = load_faiss_index(embeddings, index_dir=DEFAULT_FAISS_INDEX_DIR)
    rag = VoiceOfCustomerRAG(vector_store=vector_store)
    response = rag.ask(question)
    _print_rag_result(response.to_dict())
    return 0


def cmd_evaluate(output_path: Path = EVALUATION_EXAMPLES_PATH) -> int:
    """Executa a bateria de perguntas de negócio e de recusa, salvando o relatório JSON."""
    embeddings = get_embeddings_model()
    vector_store = load_faiss_index(embeddings, index_dir=DEFAULT_FAISS_INDEX_DIR)
    rag = VoiceOfCustomerRAG(vector_store=vector_store)

    results = []
    for question in CANONICAL_QUESTIONS:
        resp = rag.ask(question)
        resp_dict = resp.to_dict()
        _print_rag_result(resp_dict)
        results.append(resp_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório de avaliação salvo em: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI do sistema Voice of Customer Intelligence com RAG (Base Olist)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Limpa a base Olist e constrói o índice FAISS.")
    ingest_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help="Caminho para o arquivo olist_order_reviews_dataset.csv.",
    )
    ingest_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Quantidade máxima de avaliações para indexar (opcional).",
    )

    query_parser = subparsers.add_parser("query", help="Realiza uma pergunta ao pipeline RAG.")
    query_parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="Pergunta em linguagem natural sobre a voz do cliente.",
    )

    subparsers.add_parser("evaluate", help="Executa os exemplos de exploração do desafio e salva relatório.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            return cmd_ingest(raw_csv=args.csv, sample_size=args.sample_size)
        if args.command == "query":
            return cmd_query(question=args.question)
        if args.command == "evaluate":
            return cmd_evaluate()
        return 1
    except Exception as exc:
        print(f"Erro na execução: {mask_secret(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
