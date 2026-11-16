# Exemplos de Exploração — Perguntas, Respostas e Evidências

Este documento apresenta os resultados reais obtidos na execução do pipeline RAG com a API da OpenAI (`text-embedding-3-small` + `gpt-4o-mini`) sobre o índice vetorial `FAISS` (`data/indices/faiss_olist_reviews`). Todas as saídas estruturadas encontram-se em [reports/rag-evaluation-examples.json](../reports/rag-evaluation-examples.json).

---

## 1. Investigação de Reclamações Gerais

**Pergunta:** *Quais são os principais problemas relatados pelos clientes em suas avaliações?*

**Consultas Expandidas (*Multi-Query*):**
- `Quais são os principais problemas relatados pelos clientes em suas avaliações?`
- `Quais são os problemas mais comuns mencionados pelos clientes em suas avaliações?`
- `Quais dificuldades os clientes costumam relatar em suas avaliações?`
- `Quais são as queixas mais frequentes dos clientes nas avaliações?`

**Evidências Recuperadas do FAISS:**
1. `[review_id: facaaf5d1ea43d6ead5d1288fb8eed21 | Nota: 1/5 | Data: 2018-05-04 | Relevância: 0.507]`
   > *"Faltando produto. Veio faltando produto Comprei 2 cortinas e só veio 1 Quando resolver o problema mudo minha avaliação."*
2. `[review_id: 949eca021ed60d00a2d2bda3cc686d4d | Nota: 1/5 | Data: 2017-06-07 | Relevância: 0.505]`
   > *"muitos desencontros nas entregas de produtos"*
3. `[review_id: 2b2a270315cee9d67f55a02c96cce8ee | Nota: 1/5 | Data: 2018-08-29 | Relevância: 0.505]`
   > *"negativa. utilizo o site das amaericanas a muitos anos, sempre tive um bom retorno, as vezes demora,mas os produtos chegam sempre embalados. mas dessa vez não recebi, por isso a avaliação negativa."*
4. `[review_id: 4dcdbd92efbddf89c6ece8d3b0cb6ad6 | Nota: 3/5 | Data: 2017-03-25 | Relevância: 0.501]`
   > *"Comprei vários produtos alguns ótimo mas alguns. Com problemas"*

**Resposta Fundamentada Gerada:**
> Os principais problemas relatados pelos clientes em suas avaliações incluem:
> 1. **Falta de produtos:** Um cliente mencionou que recebeu apenas um dos dois itens comprados, afirmando que *"Veio faltando produto"* `[review_id: facaaf5d1ea43d6ead5d1288fb8eed21, Nota: 1/5]`.
> 2. **Desencontros nas entregas:** Outro cliente relatou *"muitos desencontros nas entregas de produtos"* `[review_id: 949eca021ed60d00a2d2bda3cc686d4d, Nota: 1/5]`, indicando problemas na logística e na pontualidade das entregas.
> 3. **Não recebimento de produtos:** Um cliente expressou insatisfação ao afirmar que não recebeu o produto: *"mas dessa vez não recebi, por isso a avaliação negativa"* `[review_id: 2b2a270315cee9d67f55a02c96cce8ee, Nota: 1/5]`.
> 4. **Problemas com a qualidade de alguns produtos:** Relato de inconsistência em pedidos múltiplos: *"Comprei vários produtos alguns ótimo mas alguns. Com problemas"* `[review_id: 4dcdbd92efbddf89c6ece8d3b0cb6ad6, Nota: 3/5]`.

---

## 2. Investigação de um Tema Específico (Entrega)

**Pergunta:** *Quais reclamações relacionadas à entrega aparecem com maior frequência?*

**Consultas Expandidas (*Multi-Query*):**
- `Quais reclamações relacionadas à entrega aparecem com maior frequência?`
- `Quais são as reclamações mais comuns sobre entregas?`
- `Quais tipos de reclamações sobre entrega são mais frequentes?`
- `Quais reclamações sobre entrega ocorrem com mais regularidade?`

**Evidências Recuperadas do FAISS:**
- `[review_id: 72926e5236768ffc912764ca429d0e00 | Nota: 1/5 | Data: 2018-03-09 | Relevância: 0.699]`: *"Muita demora na entrega"*
- `[review_id: 949eca021ed60d00a2d2bda3cc686d4d | Nota: 1/5 | Data: 2017-06-07 | Relevância: 0.654]`: *"muitos desencontros nas entregas de produtos"*
- `[review_id: 7528a396f7625967f05cb09946286608 | Nota: 1/5 | Data: 2018-03-30 | Relevância: 0.648]`: *"Ultimamente estão atrasando muito as entregas"*
- `[review_id: 214deaabf69eb3d26c7bae540e03d1b2 | Nota: 2/5 | Data: 2018-03-14 | Relevância: 0.638]`: *"Demora demasiada na entrega"*
- `[review_id: f242ce44c6572a94907a446b8fda32c8 | Nota: 1/5 | Data: 2018-03-02 | Relevância: 0.638]`: *"Olá! Minha queixa é a não entrega na residência. Tive que buscar no correio. Tenho um produto que foi devolvido, e ainda nem sei o que fazer. Talvez um telefonema ajudasse."*

**Resposta Fundamentada Gerada:**
> As reclamações relacionadas à entrega que aparecem com maior frequência nas avaliações são:
> 1. **Demora na entrega:** Vários clientes mencionaram lentidão no processo, como *"Muita demora na entrega"* `[review_id: 72926e5236768ffc912764ca429d0e00, Nota: 1/5]` e *"Ultimamente estão atrasando muito as entregas"* `[review_id: 7528a396f7625967f05cb09946286608, Nota: 1/5]`.
> 2. **Desencontros nas entregas:** Problemas de coordenação logística em *"muitos desencontros nas entregas de produtos"* `[review_id: 949eca021ed60d00a2d2bda3cc686d4d, Nota: 1/5]`.
> 3. **Não entrega no endereço residencial:** O consumidor precisou retirar o objeto na agência dos Correios por ausência de entrega domiciliar `[review_id: f242ce44c6572a94907a446b8fda32c8, Nota: 1/5]`.

---

## 3. Busca por Experiências Positivas

**Pergunta:** *O que os clientes mais elogiam em suas experiências de compra?*

**Evidências Recuperadas do FAISS:**
- `[review_id: 382d855ffdd4f5e66594dbfb1566e96f | Nota: 5/5 | Data: 2018-06-03 | Relevância: 0.642]`: *"10. Cada vez que compro mais fico satisfeita parabéns pela honestidade com seus clientes 👏👏👏👏?"*
- `[review_id: d7571a8910cfc6319c8acbba887fd33b | Nota: 5/5 | Data: 2018-06-20 | Relevância: 0.622]`: *"Nota 10. Comprometimento e dedicação com o cliente.por isso indico pra quem está com alguma dúvida referente a loja;e sobre o produto atingiu perfeitamente as minhas perspectivas."*
- `[review_id: f8bc084927f01cb84fce083eb9aa5819 | Nota: 5/5 | Data: 2018-07-12 | Relevância: 0.611]`: *"otimo. Produtos entregues no prazo , corretamente. Certamente comprarei mais vezes."*
- `[review_id: a644c23f94b9da0ca04ae1de4538e42a | Nota: 5/5 | Data: 2018-06-20 | Relevância: 0.603]`: *"Nota 10. Gosto muito de comprar online pela lojas lannister sempre com um atendimento excelente nota 10.saudações."*
- `[review_id: b4f0bb9b7de99c1cdb8572b7914449f9 | Nota: 5/5 | Data: 2018-04-24 | Relevância: 0.601]`: *"Já comprei vários produtos, sempre fiquei satisfeito."*

**Resposta Fundamentada Gerada:**
> Os clientes elogiam principalmente: (1) **Honestidade e satisfação recorrente** `[review_id: 382d855ffdd4f5e66594dbfb1566e96f, Nota: 5/5]`; (2) **Comprometimento com o cliente e aderência do produto às expectativas** `[review_id: d7571a8910cfc6319c8acbba887fd33b, Nota: 5/5]`; (3) **Pontualidade e correção na entrega** `[review_id: f8bc084927f01cb84fce083eb9aa5819, Nota: 5/5]`; e (4) **Excelência no atendimento online** `[review_id: a644c23f94b9da0ca04ae1de4538e42a, Nota: 5/5]`.

---

## 4. Investigação de Insatisfação

**Pergunta:** *Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?*

**Resposta Fundamentada Gerada:**
> Os padrões identificados nas avaliações de clientes insatisfeitos incluem:
> 1. **Falta de respeito e suporte ao consumidor:** Relatos diretos de decepção com a postura da loja `[review_id: 5e7a3ac257327b889989c9ede7d35ff7, Nota: 1/5]` `[review_id: 48dfb09da10ec86135fedeb9941364c2, Nota: 2/5]`.
> 2. **Expectativas frustradas e produto divergente do comprado:** Recebimento de itens diferentes do adquirido ou abaixo do esperado `[review_id: db2ad68a772d476d92bae6d8c9a8eb18, Nota: 2/5]` `[review_id: 4a3ee26dbf183679868310fcb72ade3d, Nota: 1/5]`.
> 3. **Cobrança de avaliação antes da entrega efetiva do pedido:** Clientes que recebem pesquisa de avaliação sem ainda terem recebido o produto `[review_id: 5fa4fb44964b47967a99bef856ba418d, Nota: 1/5]`.

---

## 5. Exploração Baseada em Evidências

**Pergunta:** *Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?*

**Resposta Fundamentada Gerada:**
> As avaliações que sustentam diretamente a existência de problemas recorrentes com o prazo de entrega são:
> - `[review_id: 7528a396f7625967f05cb09946286608, Nota: 1/5]`: *"Ultimamente estão atrasando muito as entregas"*.
> - `[review_id: 72926e5236768ffc912764ca429d0e00, Nota: 1/5]`: *"Muita demora na entrega"*.
> - `[review_id: 949eca021ed60d00a2d2bda3cc686d4d, Nota: 1/5]`: *"muitos desencontros nas entregas de produtos"*.
> - `[review_id: 2a1993cf1867bbfc58d8a8d8db314408, Nota: 2/5]`: *"Deveria melhorar o prazo. Demora muito tempo para entregar a mercadoria. O produto permaneceu alguns dias parado e já podia ter sido encaminhado"*.
> - `[review_id: 18b759902d50cc4b835b7d65e1f293c1, Nota: 2/5]`: *"Prazo de entrega muito longo"*.

---

## 6. Tratamento de Pergunta Fora de Escopo / Sem Evidências Suficientes

**Pergunta:** *Qual é a cotação atual do dólar e a previsão do tempo em Tóquio amanhã?*

- **Consultas Expandidas:** `["Qual é a cotação atual do dólar e a previsão do tempo em Tóquio amanhã?"]` (preservação estrita do escopo original)
- **Status:** `has_sufficient_evidence = false` (`below_relevance_threshold`)
- **Documentos recuperados acima do limiar (`0.45`):** `0`
- **Resposta do Sistema:**
  > *"Com base nas avaliações disponíveis na base de conhecimento da Olist, não foram encontradas evidências suficientes para responder a esta pergunta com segurança. A questão pode estar fora do escopo dos relatos de clientes ou não possuir ocorrências relevantes registradas."*
