# Arquitetura da Solução — Voice of Customer Intelligence com RAG

A arquitetura foi projetada em dois fluxos desacoplados: **Construção da Base de Conhecimento (Offline/Batch)** e **Pipeline RAG de Consulta e Governança (Online)**.

---

## 1. Visão Geral do Fluxo

```mermaid
flowchart TD
    subgraph Ingestao ["1. Construção da Base de Conhecimento"]
        A["Base Bruta Olist\n(data/raw/olist_order_reviews_dataset.csv)\n100.000 linhas"] --> B["Limpeza e Tratamento de Nulos\n(src/preprocessing/cleaner.py)"]
        B --> C["Filtragem de Ruído e Desduplicação\n(src/preprocessing/loader.py)\n36.576 avaliações únicas"]
        C --> D["Objetos LangChain Document\n(page_content + metadados rastreáveis)"]
        D --> E["Geração de Embeddings\nOpenAI text-embedding-3-small (1536 dim)"]
        E --> F[("Índice Vetorial Local\nFAISS c/ Normalização L2\n(data/indices/faiss_olist_reviews)")]
    end

    subgraph PipelineRAG ["2. Pipeline RAG e Governança de Evidências"]
        Q["Pergunta em Linguagem Natural"] --> T["Transformação da Pergunta\nMulti-Query Expansion (gpt-4o-mini)"]
        T --> R["Recuperação Semântica no FAISS\nBusca vetorial + Conversão Cosseno [0, 1]"]
        F --> R
        R --> S["Seleção e Deduplicação\nTop-K por maior similaridade + Corte < 0.30"]
        S --> G{"Retrieval Gate\nEvidências >= min_docs?"}
        G -- Não --> REF1["Recusa Determinística\nSem chamada desnecessária ao LLM"]
        G -- Sim --> CTX["Construção do Contexto Estruturado\n(review_id, nota, data, similaridade e texto)"]
        CTX --> GEN["Geração Fundamentada (gpt-4o-mini)\nSystem Prompt Estrito + Guardrail de Escopo"]
        GEN --> RESP["Resposta Sintetizada + Citações Rastreáveis"]
    end
```

---

## 2. Camadas e Responsabilidades

| Camada | Módulo | Responsabilidade Técnica |
| :--- | :--- | :--- |
| **Configuração** | `src/config.py` | Centraliza caminhos de arquivos, nomes de modelos (`text-embedding-3-small`, `gpt-4o-mini`), *top-k*, *relevance threshold* (`0.30`) e mensagens padronizadas de recusa. |
| **Pré-processamento** | `src/preprocessing/cleaner.py` | Normalização Unicode (`NFC`), remoção de caracteres de controle, consolidação entre título e comentário e filtro de informatividade mínima. |
| **Carga e Auditoria** | `src/preprocessing/loader.py` | Filtra registros vazios, remove duplicatas textuais, gera auditoria em `reports/data-cleaning-audit.json` e instancia objetos `Document`. |
| **Indexação Vetorial** | `src/indexing/embeddings.py` e `vector_store.py` | Conecta à API de embeddings da OpenAI e persiste o índice vetorial `FAISS` normalizado em disco. |
| **Transformação de Consulta** | `src/rag/query_transform.py` | Expande a pergunta original em múltiplas variações semânticas usando *LangChain Expression Language* (LCEL). |
| **Recuperação Semântica** | `src/rag/retriever.py` | Executa busca vetorial para todas as variações, converte distância L2 quadrática em similaridade de cosseno, deduplica por `review_id` e aplica corte por limiar. |
| **Geração e Governança** | `src/rag/pipeline.py` | Monta o contexto rastreável, aplica o *Retrieval Gate* pré-LLM e o *Guardrail* pós-LLM para garantir respostas 100% ancoradas nas avaliações. |
| **Interface CLI** | `src/cli.py` | Permite reproduzir via linha de comando a ingestão (`ingest`), consultas livres (`query`) e a bateria oficial de avaliação (`evaluate`). |
