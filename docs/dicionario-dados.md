# Dicionário de Dados e Auditoria da Base

## 1. Origem dos Dados

- **Dataset:** *Brazilian E-Commerce Public Dataset by Olist*
- **Arquivo utilizado:** `olist_order_reviews_dataset.csv` (`data/raw/olist_order_reviews_dataset.csv`)
- **Fonte pública:** [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)

---

## 2. Schema da Tabela Bruta (`olist_order_reviews_dataset.csv`)

| Coluna | Tipo Original | Descrição | Uso na Solução RAG |
| :--- | :--- | :--- | :--- |
| `review_id` | `string` (hash) | Identificador único da avaliação | **Metadado obrigatório** para citação e rastreabilidade da evidência |
| `order_id` | `string` (hash) | Identificador único do pedido associado | **Metadado** preservado no `Document` |
| `review_score` | `int` (1 a 5) | Nota atribuída pelo cliente (1 = muito insatisfeito, 5 = muito satisfeito) | **Metadado obrigatório** exibido no contexto e nas citações da resposta |
| `review_comment_title` | `string` (opcional) | Título resumido da avaliação escrita pelo consumidor | **Conteúdo textual** higienizado e concatenado ao corpo da mensagem |
| `review_comment_message` | `string` (opcional) | Relato textual detalhado da experiência de compra | **Conteúdo textual principal** (`page_content` do `Document`) |
| `review_creation_date` | `timestamp` | Data de criação da avaliação | **Metadado** normalizado (`YYYY-MM-DD`) |
| `review_answer_timestamp` | `timestamp` | Data/hora de resposta do lojista | Descartado na seleção (sem informação qualitativa do cliente) |

---

## 3. Contrato do Documento Indexado (`langchain_core.documents.Document`)

Cada registro em `data/processed/olist_reviews_cleaned.csv` é convertido no seguinte formato:

```python
Document(
    page_content="Sempre compro pela Internet e a entrega ocorre antes do prazo combinado...",
    metadata={
        "review_id": "58044bca115705a48fe0e00a21390c54",
        "order_id": "68e55ca79d04a79f20d4bfc0146f4b66",
        "review_score": 1,
        "review_creation_date": "2018-04-08",
    },
)
```

---

## 4. Funil de Higienização da Base (`reports/data-cleaning-audit.json`)

| Etapa do Funil | Quantidade de Registros | Proporção sobre o Total Bruto |
| :--- | ---: | ---: |
| **Total de linhas no CSV bruto** | `100.000` | `100,00%` |
| Linhas sem título (`review_comment_title` nulo) | `88.285` | `88,29%` |
| Linhas sem mensagem (`review_comment_message` nulo) | `58.247` | `58,25%` |
| Linhas com título e mensagem simultaneamente nulos (descartadas) | `56.518` | `56,52%` |
| **Avaliações com texto não vazio após limpeza** | `43.463` | `43,46%` |
| Removidas por ruído / baixa informatividade (< 8 chars ou < 2 tokens) | `-3.446` | `-3,45%` |
| Removidas por texto duplicado exato (*case-insensitive*) | `-3.441` | `-3,44%` |
| **Total final de avaliações únicas indexáveis** | **`36.576`** | **`36,58%`** |

### Distribuição por Nota (`review_score`) na Base Limpa (`36.576` avaliações)

| Nota (`review_score`) | Quantidade | Participação (%) | Perfil Predominante |
| :---: | ---: | ---: | :--- |
| **1** | `8.739` | `23,89%` | Reclamações severas (atraso, não entrega, defeito, produto errado) |
| **2** | `2.118` | `5,79%` | Insatisfação moderada (entrega parcial, demora, qualidade inferior) |
| **3** | `3.336` | `9,12%` | Experiências neutras ou mistas (produto bom, mas frete demorado) |
| **4** | `5.092` | `13,92%` | Avaliações positivas com ressalvas pontuais |
| **5** | `17.291` | `47,27%` | Elogios à entrega antes do prazo, qualidade e confiabilidade |
