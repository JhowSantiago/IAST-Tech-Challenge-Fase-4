# Exemplos de Exploração — Perguntas, Respostas e Evidências

Este documento apresenta os resultados detalhados das cinco investigações de negócio propostas no desafio, além do teste de governança para perguntas sem evidências suficientes. Todos os dados brutos encontram-se serializados em [reports/rag-evaluation-examples.json](../reports/rag-evaluation-examples.json).

---

## 1. Investigação de Reclamações Gerais

**Pergunta:** *Quais são os principais problemas relatados pelos clientes em suas avaliações?*

**Consultas Expandidas (*Multi-Query*):**
- `Quais são os principais problemas relatados pelos clientes em suas avaliações?`
- `principais reclamações de clientes defeito atraso produto errado`
- `motivos de insatisfação problemas na compra entrega e qualidade`
- `relatos negativos recorrentes pedido incompleto ou avariado`

**Evidências Recuperadas do FAISS:**
1. `[review_id: 58044bca115705a48fe0e00a21390c54 | Nota: 1/5 | Data: 2018-04-08 | Relevância: 0.842]`
   > *"Sempre compro pela Internet e a entrega ocorre antes do prazo combinado, que acredito ser o prazo máximo. No stark o prazo máximo já se esgotou e ainda não recebi o produto."*
2. `[review_id: c40a6b6e0181e5ec0d12cbc2e12c49d3 | Nota: 1/5 | Data: 2018-05-23 | Relevância: 0.832]`
   > *"falta de produto e quebra. Faltou 1 produto e os que recebi 1 veio quebrado"*
3. `[review_id: 6f885e2fd69412264d9e74e805175d53 | Nota: 1/5 | Data: 2017-06-22 | Relevância: 0.819]`
   > *"Recebi bem rápido, porém veio outro produto, e não o que eu pedi, o modelo veio diferente, a cor, veio uma capinha que não tem zíper, enfim, não recomendo."*
4. `[review_id: 2d0d80576ee9bbab49cc108d9ec7a0fb | Nota: 1/5 | Data: 2017-02-21 | Relevância: 0.808]`
   > *"meu produto veio com defeito,devolvi conforme informado.postado no correio conforme indicado.e até hoje não recebi nenhuma notificação da empresa quanto a devolução e análise para ter meuressarcimento"*
5. `[review_id: 6d06808638ec0701bccd70bc8d462c28 | Nota: 1/5 | Data: 2016-12-09 | Relevância: 0.797]`
   > *"O produto não chegou no prazo estipulado e causou transtorno, pq programei a viagem de férias do meu filho, baseado no prazo. Moro na Bahia e ele em Cuiabá sozinho. Agora, a casa está vazia."*

**Resposta Fundamentada Gerada:**
> Com base nas avaliações recuperadas na base da Olist, os principais problemas relatados pelos clientes concentram-se em quatro frentes recorrentes:
> 1. **Atrasos e descumprimento do prazo de entrega:** Clientes relatam pedidos que ultrapassam o prazo máximo estipulado sem recebimento, gerando transtornos pessoais `[review_id: 58044bca115705a48fe0e00a21390c54, Nota: 1/5]` `[review_id: 6d06808638ec0701bccd70bc8d462c28, Nota: 1/5]`.
> 2. **Entrega incompleta ou avaria física:** Há registros de pedidos entregues faltando itens adquiridos e com peças quebradas no transporte `[review_id: c40a6b6e0181e5ec0d12cbc2e12c49d3, Nota: 1/5]`.
> 3. **Divergência entre o produto anunciado e o recebido:** Consumidores apontam o envio de modelos, cores ou materiais inferiores e diferentes da foto do anúncio `[review_id: 6f885e2fd69412264d9e74e805175d53, Nota: 1/5]`.
> 4. **Defeito funcional e falha no pós-venda/reembolso:** Relatos de produtos que apresentam defeito logo no primeiro uso ou ausência de retorno e ressarcimento após devolução pelos Correios `[review_id: 2d0d80576ee9bbab49cc108d9ec7a0fb, Nota: 1/5]`.

---

## 2. Investigação de um Tema Específico (Entrega)

**Pergunta:** *Quais reclamações relacionadas à entrega aparecem com maior frequência?*

**Evidências Recuperadas do FAISS:**
- `[review_id: d7ea05bb4f9c77d606fd90b1a27e317d | Nota: 1/5 | Relevância: 0.864]`: *"Comprei o produto dia 25 de fevereiro e hoje dia 29 de marco não fora entregue na minha residência. Não sei se os correios desse Brasil e péssimo ou foi a própria loja que demorou postar."*
- `[review_id: 185cb85ede91009d7a8e685687c037ee | Nota: 1/5 | Relevância: 0.851]`: *"fiz minha compra faz 30 dias e não recebi ainda meu produto. Precisa melhorar nas entregas"*
- `[review_id: 06f47dacd0575e592bf6a17e922ce6cd | Nota: 1/5 | Relevância: 0.839]`: *"não recebi ainda. aqui está descrevendo como entregue só que ate agora não recebi"*
- `[review_id: ae728c1061bf163b4bd256ad9ee0bb83 | Nota: 1/5 | Relevância: 0.812]`: *"Este foi o pedido Balde Com 128 Peças - Blocos De Montar 2 un - R$ 25,00 cada (NÃO FOI ENTREGUE)..."*

**Resposta Fundamentada Gerada:**
> Nas avaliações relacionadas à logística e entrega, destacam-se quatro reclamações principais: (1) Extrapolar o prazo prometido com esperas superiores a 30 dias `[review_id: d7ea05bb4f9c77d606fd90b1a27e317d, Nota: 1/5]` `[review_id: 185cb85ede91009d7a8e685687c037ee, Nota: 1/5]`; (2) Status de rastreamento apontando pedido entregue sem que o cliente tenha recebido a mercadoria `[review_id: 06f47dacd0575e592bf6a17e922ce6cd, Nota: 1/5]`; (3) Entrega parcial de pedidos compostos por múltiplos itens `[review_id: ae728c1061bf163b4bd256ad9ee0bb83, Nota: 1/5]`; e (4) Descumprimento da data limite afetando o planejamento familiar do consumidor `[review_id: 6d06808638ec0701bccd70bc8d462c28, Nota: 1/5]`.

---

## 3. Busca por Experiências Positivas

**Pergunta:** *O que os clientes mais elogiam em suas experiências de compra?*

**Evidências Recuperadas do FAISS:**
- `[review_id: 4940d39efaa50e020312724f6a6e5efa | Nota: 5/5 | Relevância: 0.871]`: *"super recomendo. produto de qualidade e eficiência na entrega. recebi muito antes do combinado. recomendo"*
- `[review_id: 5241fcad10d7b45a1761f1e1e1536faa | Nota: 5/5 | Relevância: 0.859]`: *"super recomendo. chegou dentro do prazo e o produto é de excelente qualidade! Acabou as dores nas costas, estou dormindo muito bem!"*
- `[review_id: fe3db7c069d694bab50cc43463f91608 | Nota: 5/5 | Relevância: 0.846]`: *"A compra foi realizada facilmente. A entrega foi efetuada muito antes do prazo dado. O produto já começou a ser usado e até o presente, sem problemas."*
- `[review_id: 07b48dbc11677f5021dbefcaba550ed2 | Nota: 5/5 | Relevância: 0.834]`: *"O produto chegou bem antes do prazo estipulado, amei o atendimento! Super recomendo"*

**Resposta Fundamentada Gerada:**
> Os aspectos mais valorizados pelos consumidores nas avaliações nota 5/5 são: (1) **Antecipação do prazo de entrega**, recebendo o produto antes da data combinada `[review_id: 4940d39efaa50e020312724f6a6e5efa, Nota: 5/5]` `[review_id: fe3db7c069d694bab50cc43463f91608, Nota: 5/5]`; (2) **Qualidade superior e benefício prático do produto** `[review_id: 5241fcad10d7b45a1761f1e1e1536faa, Nota: 5/5]`; e (3) **Qualidade do atendimento e facilidade de compra** `[review_id: 07b48dbc11677f5021dbefcaba550ed2, Nota: 5/5]`.

---

## 4. Investigação de Insatisfação

**Pergunta:** *Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?*

**Resposta Fundamentada Gerada:**
> Observam-se três padrões principais: (1) **Quebra de expectativa visual e material** (produtos que parecem tecido na foto mas chegam em plástico ou sem o acabamento anunciado) `[review_id: bdd3dd9b37833960a30cbdd2f2e7e888, Nota: 1/5]` `[review_id: 71f7323f884804c6d38bbcd658e07318, Nota: 1/5]`; (2) **Falha operacional agravada por omissão no pós-venda** (clientes que devolvem itens defeituosos e ficam sem resposta ou reembolso) `[review_id: b396ba75350276a6cc1993b6627dceea, Nota: 1/5]` `[review_id: 2d0d80576ee9bbab49cc108d9ec7a0fb, Nota: 1/5]`; e (3) **Abandono declarado da marca** após atrasos combinados com mau atendimento `[review_id: a36150969fe6bd09ed011dd78aeb0189, Nota: 1/5]`.

---

## 5. Exploração Baseada em Evidências

**Pergunta:** *Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?*

**Resposta Fundamentada Gerada:**
> A conclusão é sustentada diretamente pelas avaliações:
> - `[review_id: 58044bca115705a48fe0e00a21390c54, Nota: 1/5]`: *"o prazo máximo já se esgotou e ainda não recebi o produto"*.
> - `[review_id: 6d06808638ec0701bccd70bc8d462c28, Nota: 1/5]`: *"O produto não chegou no prazo estipulado e causou transtorno"*.
> - `[review_id: d7ea05bb4f9c77d606fd90b1a27e317d, Nota: 1/5]`: *"Comprei o produto dia 25 de fevereiro e hoje dia 29 de marco não fora entregue na minha residência"*.
> - `[review_id: 185cb85ede91009d7a8e685687c037ee, Nota: 1/5]`: *"fiz minha compra faz 30 dias e não recebi ainda meu produto"*.

---

## 6. Tratamento de Pergunta Fora de Escopo / Sem Evidências Suficientes

**Pergunta:** *Qual é a cotação atual do dólar e a previsão do tempo em Tóquio amanhã?*

- **Status:** `has_sufficient_evidence = false` (`below_relevance_threshold`)
- **Documentos recuperados acima do limiar (`0.30`):** `0`
- **Resposta do Sistema:**
  > *"Com base nas avaliações disponíveis na base de conhecimento da Olist, não foram encontradas evidências suficientes para responder a esta pergunta com segurança. A questão pode estar fora do escopo dos relatos de clientes ou não possuir ocorrências relevantes registradas."*
