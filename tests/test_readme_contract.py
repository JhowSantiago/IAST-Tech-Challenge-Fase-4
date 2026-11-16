"""Testes de contrato documental garantindo que todos os entregáveis exigidos no PDF constam no repositório."""

from __future__ import annotations

from src.config import (
    CLEANING_AUDIT_PATH,
    DEFAULT_PROCESSED_CSV,
    DEFAULT_RAW_CSV,
    DOCS_DIR,
    EVALUATION_EXAMPLES_PATH,
    PROJECT_ROOT,
)


def test_readme_and_required_deliverables_exist() -> None:
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")

    required_sections = [
        "## 1. Problema de negócio e contexto",
        "## 2. Objetivo da solução",
        "## 3. Arquitetura desenvolvida",
        "## 4. Preparação da base de conhecimento",
        "## 5. Estratégia utilizada para construção do pipeline RAG",
        "## 6. Tecnologias e bibliotecas utilizadas",
        "## 7. Justificativa das principais decisões técnicas",
        "## 8. Exemplos de perguntas, respostas e evidências",
        "## 9. Tratamento de perguntas sem evidências suficientes",
        "## 10. Estrutura do repositório",
        "## 11. Instruções para instalação, execução e reprodução",
    ]
    for section in required_sections:
        assert section in content, f"Seção obrigatória ausente no README.md: {section}"

    assert DEFAULT_RAW_CSV.exists()
    assert DEFAULT_PROCESSED_CSV.exists()
    assert CLEANING_AUDIT_PATH.exists()
    assert EVALUATION_EXAMPLES_PATH.exists()
    assert (DOCS_DIR / "arquitetura.md").exists()
    assert (DOCS_DIR / "decisoes-tecnicas.md").exists()
    assert (DOCS_DIR / "dicionario-dados.md").exists()
    assert (DOCS_DIR / "exemplos-exploracao.md").exists()
    assert (DOCS_DIR / "git-workflow.md").exists()
