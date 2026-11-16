# Decisões Técnicas e Justificativas

Este documento registra e justifica as escolhas arquiteturais e algorítmicas adotadas para atender aos requisitos do desafio.

---

## 1. Seleção e Preparação da Base de Conhecimento

### 1.1. Consolidação entre Título e Mensagem
- **Decisão:** Unificar `review_comment_title` e `review_comment_message` por meio da função `combine_title_and_message`.
- **Justificativa:** Na base da Olist, diversos consumidores escrevem o motivo central da reclamação apenas no título (ex.: *"Atraso na entrega"*) ou detalham no corpo sem repetir o título. A combinação preserva o contexto completo sem descartar registros em que apenas um dos dois campos foi preenchido. Evita-se a duplicação quando a mensagem já inicia com o mesmo texto do título.

### 1.2. Tratamento de Valores Ausentes e Ruído Textual
- **Decisão:** Descartar os `56.518` registros em que título e mensagem são simultaneamente nulos, além de remover `3.446` avaliações curtas/não informativas (menos de 8 caracteres ou menos de 2 tokens alfanuméricos, como *"ok"*, *"bom"*, *"10"* ou *"..."*).
- **Justificativa:** Avaliações sem texto ou monossilábicas não oferecem evidência qualitativa sobre o *porquê* da satisfação ou insatisfação do cliente e apenas poluem o espaço vetorial.

### 1.3. Desduplicação de Textos Idênticos
- **Decisão:** Remover `3.441` avaliações com texto exatamente idêntico (comparação *case-insensitive*), resultando em `36.576` avaliações únicas.
- **Justificativa:** Sem desduplicação, uma busca por *"produto bom"* ou *"não recebi"* retornaria no *top-k* cinco cópias idênticas da mesma frase curta, reduzindo a diversidade de relatos no contexto enviado ao LLM.

---

## 2. Organização dos Documentos (Sem Fragmentação Artificial)

- **Decisão:** Representar cada avaliação única como um único `Document` do LangChain, sem quebrar um mesmo comentário em múltiplos *chunks* menores.
- **Justificativa:** As avaliações da Olist possuem limite de caracteres imposto pela plataforma original (média de ~60 a 180 caracteres por relato). Dividir um comentário curto em pedaços menores separaria a causa do efeito (ex.: separar *"Recebi bem rápido"* de *"porém veio outro produto diferente do que pedi"*), destruindo a coerência semântica da evidência.

---

## 3. Modelo de Embeddings e Banco Vetorial

### 3.1. `OpenAIEmbeddings` (`text-embedding-3-small`)
- **Decisão:** Utilizar o modelo `text-embedding-3-small` da OpenAI (1536 dimensões).
- **Justificativa:** Excelente desempenho multilíngue em português brasileiro, capturando variações coloquiais, abreviações e erros de digitação frequentes em avaliações reais de e-commerce, além de manter o ambiente local leve (sem necessidade de baixar pesos de gigabytes de modelos PyTorch/HuggingFace na máquina local).

### 3.2. `FAISS` com Normalização L2
- **Decisão:** Indexar os vetores no `FAISS` (`faiss-cpu`) com `normalize_L2=True` e converter a distância L2 quadrática em similaridade de cosseno no intervalo $[0, 1]$ via $s = 1 - \frac{d^2}{2}$.
- **Justificativa:** Para vetores unitários normalizados, a distância euclidiana quadrática possui relação monotônica exata com a similaridade de cosseno. Essa abordagem entrega busca exata em milissegundos para os `36.576` documentos, persistência simples em arquivo local (`data/indices/faiss_olist_reviews`) e um score de relevância padronizado entre `0.0` e `1.0` para aplicação de *threshold*.

---

## 4. Estratégia do Pipeline RAG

### 4.1. Transformação de Consulta (*Multi-Query Expansion*)
- **Decisão:** Antes da busca vetorial, utilizar o `gpt-4o-mini` para expandir a pergunta do gestor em 3 variações semânticas voltadas ao vocabulário do consumidor.
- **Justificativa:** Gestores fazem perguntas corporativas abstratas (*"Quais são os padrões de insatisfação?"*), enquanto clientes escrevem relatos concretos (*"veio quebrado"*, *"prazo esgotou"*, *"diferente da foto"*). A expansão *Multi-Query* reduz o *gap* léxico e semântico, elevando o *recall* da recuperação.

### 4.2. Governança em Dupla Camada para Perguntas sem Evidências
- **Decisão:** Implementar duas travas complementares:
  1. **Retrieval Gate (Pré-LLM):** Filtra qualquer avaliação com similaridade de cosseno inferior a `0.30`. Se nenhuma avaliação superar o limiar, o sistema retorna imediatamente a mensagem padronizada de insuficiência de evidências, economizando tokens e impedindo alucinações.
  2. **Prompt Guardrail (Pós-LLM):** O *system prompt* obriga o modelo a citar `[review_id: <id>, Nota: <nota>/5]` para cada afirmação e a emitir o token `SEM_EVIDENCIAS_SUFICIENTES` caso os documentos recuperados não respondam diretamente à pergunta feita.
