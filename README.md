# Tech Challenge - Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

## Contexto do problema

A alfabetização na idade certa é um dos principais indicadores de qualidade da educação básica no Brasil. Redes de ensino acompanham se os alunos avaliados atingem o nível esperado.

Este projeto usa dados públicos de avaliações de alfabetização para construir um modelo preditivo capaz de identificar quais alunos serão considerados alfabetizados, gerando insumos para tomadas de decisões pedagógicas ou direcionamento de recursos de forma eficiente.

## Objetivo analítico

Desenvolver um modelo de classificação supervisionada que preveja se um aluno
está alfabetizado (`alfabetizado = 1`) ou não (`alfabetizado = 0`), a partir de
variáveis educacionais, territoriais e socioeconômicas disponíveis nas bases
públicas de alfabetização.

> **Obs sobre origem dos dados:** o desafio pedia para usar a camada gold criada na
> fase 2. Optamos por reconstruir o pipeline de forma simplificada localmente, partindo
> das bases raw, para não ser necessária infraestrutura de nuvem.
> Conforme dúvida sanada na seção/canal de `duvidas` do discord

## Descrição da base utilizada

Os dados são públicos, referentes aos anos de **2023 e 2024**, e chegam em 6
arquivos CSV brutos:

| Base | Nível | Descrição |
|---|---|---|
| `alunos.csv` | Aluno | Resultado individual de cada aluno avaliado (presença, proficiência, se foi alfabetizado) |
| `alfabetizacao_uf.csv` | UF | Indicadores agregados de alfabetização por estado |
| `alfabetizacao_municipio.csv` | Município | Indicadores agregados de alfabetização por município |
| `meta_alfabetizacao_brasil.csv` | Brasil | Metas oficiais de alfabetização (2024–2030) a nível nacional |
| `meta_alfabetizacao_uf.csv` | UF | Metas oficiais por estado |
| `meta_alfabetizacao_municipio.csv` | Município | Metas oficiais por município |

**Fonte dos dados:** [Base dos Dados – Alfabetização](https://basedosdados.org/dataset/073a39d4-89cf-4068-b1e8-34ed0d9c0b72?table=e1de7a6a-5038-4e81-89f0-a15f2cc12c9b)

[Censo Escolar](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar)

[Cadastro Único](https://dados.gov.br/dados/conjuntos-dados/pessoas-inscritas-no-cadastro-unico-por-faixa-de-renda-per-capita)


**Sobre `alunos.csv`:** este arquivo tem mais de 200MB e não está incluído
no repositório na camada bronze. Para reproduzir a pipeline de dados:
1. Baixe o arquivo na fonte indicada acima.
2. Pode ser necessario realizar um consulta no BigQuery, devido ao tamanho da base
3. Caso necessario, marque todas as colunas, clique em `gerar consulta`, clique em
    `Acessar o BigQuery`, execute a consulta e baixe os dados completos.
4. Salve-o em `data/raw/alunos.csv`.

As demais 5 bases já estão versionadas em `data/raw/`.

As bases utilizadas para criar alunos_enriquecidos tambem deve ser baixada acessando o link nas fonte dos dados.

No entanto versionamos a base final utilizada para modelagem em `data/df_final.parquet`

### Pipeline de dados

data/raw/ - CSVs originais
data/bronze/ - Mesmos dados, convertidos para Parquet e particionados por ano, aqui incluido base `alunos`
data/silver/- Dados limpos (tipos corrigidos, duplicatas removidas, categorias padronizadas)
data/silver/integrado/ - Tabela a nível aluno, com indicadores do município já unidos
data/aluno_enriquecido - tabela final que vai para etapa de modelagem

Para rodar o pipeline completo:

```bash
python src/preprocessing/bronze.py - para começar aqui é necessário ter baixado o csv alunos da fonte citada em "descrição da base utilizada"
python src/preprocessing/silver.py
python src/preprocessing/integracao.py
python src/construcao_master_table.py - Necessario baixar as tabelas utilizadas nas citadas "fontes de dados" 
```

## Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/                # CSVs brutos (alunos.csv não versionado - ver acima)
│   ├── bronze/              # gerado pelo pipeline, não versionado
│   └── silver/               # gerado pelo pipeline,  não versionado
├── notebooks/               # exploração de dados e experimentos de modelagem
├── src/
│   ├── preprocessing/        # bronze.py, silver.py, integracao.py
│   ├── modeling/              # treino e seleção do modelo
│   ├── evaluation/            # métricas e validação
│   └── visualization/         # gráficos e visualizações
├── reports/                  # documentação técnica e relatórios
├── images/                   # imagens usadas em relatórios/README
├── requirements.txt
├── README.md
└── .gitignore
```
## Etapas de modelagem

1. **Engenharia de features** (`src/preprocessing/engenharia_features.py`): consolidação de grupos de colunas correlacionadas em índices compostos (`indice_acessibilidade`, `indice_material_pedagogico`, `indice_infra_basica`), criação de proxy de superlotação (`alunos_por_sala`), log-transform em `n_escolas` e substituição das estatísticas socioeconômicas por coeficiente de variação (`_cv`), removendo as colunas originais redundantes.
2. **Remoção de leakage**: exclusão de `proficiencia`, `mun_taxa_alfabetizacao` e das variáveis derivadas delas (`gap_mun_meta_2030`, `mun_nivel_alfabetizacao`, `mun_prop_nivel_*`), conforme detalhado na seção de insights abaixo.
3. **Baseline** (`src/modeling/treinamento_modelo.py`): Pipeline scikit-learn (pré-processamento + `LGBMClassifier` com parâmetros default), treinado em `notebooks/02_modelagem.ipynb`.
4. **Otimização de hiperparâmetros** (`src/modeling/otimizacao_hiperparametros.py`): busca com `HalvingRandomSearchCV`, escolhida no lugar do `RandomizedSearchCV` tradicional por permitir usar a base completa (~3,36M de registros) sem o custo de treinar todos os candidatos na base inteira — o método já aloca progressivamente mais dados às combinações mais promissoras. Rodada em duas etapas: uma busca ampla inicial e um refinamento posterior em torno dos hiperparâmetros vencedores.

## Escolha do algoritmo

Optou-se pelo **LightGBM** (`LGBMClassifier`) pelos seguintes motivos:
- Desempenho e velocidade de treino adequados ao volume da base (~3,36M de linhas, 20 features).
- Robustez natural a diferenças de escala entre variáveis e a alguma colinearidade entre features (ao contrário de modelos lineares, árvores de decisão toleram melhor variáveis correlacionadas).
- Boa relação entre poder preditivo e possibilidade de interpretação (via `feature_importances_`, `gain` e SHAP).

O modelo foi encapsulado em uma `Pipeline` do scikit-learn (pré-processamento + modelo), garantindo que a mesma transformação de dados seja aplicada de forma consistente entre treino, validação e teste.

## Métricas de avaliação

Métrica principal: **ROC-AUC**, escolhida por lidar melhor com o desbalanceamento leve do target (59,1% vs. 40,9%) do que a acurácia simples. Métricas complementares (precisão, recall, F1) também são reportadas por classe.

| Etapa | ROC-AUC (CV) | ROC-AUC (teste) |
|---|---|---|
| Baseline (LightGBM default) | – | 0,6540 |
| Busca de hiperparâmetros (grid amplo) | 0,6689 | – |
| Refinamento do grid | 0,6702 | 0,6708 |

O ROC-AUC no conjunto de teste ficou próximo do obtido na validação cruzada durante a busca (0,6702 → 0,6708), indicando que o modelo generaliza bem, sem overfitting relevante.

Um trade-off relevante identificado: ativar `is_unbalance=True` durante a otimização elevou o recall da classe minoritária (`não alfabetizado`) de **0,32 para 0,62**, à custa de uma leve queda na acurácia geral (0,63 → 0,62) e no recall da classe majoritária (0,85 → 0,62). O F1 macro médio subiu de 0,57 para 0,62. Essa troca foi considerada favorável ao objetivo do projeto, já que identificar alunos não alfabetizados é mais relevante para direcionamento de recursos do que maximizar acerto geral.

## Interpretação dos resultados

Análise de importância de features feita com `feature_importances_` do LightGBM, comparando os critérios `split` (nº de vezes que a feature foi usada em divisões) e `gain` (redução real de erro proporcionada pela feature). O critério `gain` mostrou uma distribuição de importância bem mais discriminativa entre as features do que o `split`, e alterou consideravelmente o ranking.

Principais achados:
- `qtd_pes_pob_media` (quantidade média de pessoas em situação de pobreza na região) foi a feature mais importante em ambos os critérios — resultado associa fortemente contexto socioeconômico a desempenho escolar.
- Os índices compostos criados na engenharia de features (`indice_infra_basica`, `indice_material_pedagogico`) ficaram entre as features mais importantes, validando a decisão de consolidar variáveis correlacionadas em vez de descartá-las.
- Identificada correlação moderada entre `qtd_pes_pob_cv` e `qtd_pes_acima_meio_sm_cv` (0,66) e entre `qtd_pes_baixa_renda_cv` e `qtd_pes_acima_meio_sm_cv` (0,44) .
- O teto de ROC-AUC (~0,67) é consistente com a natureza das features disponíveis: a maioria são agregados de escola/município, funcionando como proxies do contexto do aluno, e não características individuais diretas — uma limitação inerente à granularidade dos dados públicos utilizados (ver seção de limitações).

O SHAP foi calculado com `shap.TreeExplainer` sobre uma amostra de 50 mil linhas do conjunto de teste, confirmando o padrão que nenhuma feat isolada domina o impacto na predição do modelo.

- **O ranking mudou de novo.** `qtd_pes_pob_media`, que era a feature #1 tanto no `split` quanto no `gain`, cai para a 3ª posição no SHAP — ainda relevante, mas não mais isolada no topo. `pct_in_alimentacao` e `indice_infra_basica` assumem as duas primeiras posições, indicando que, embora `qtd_pes_pob_media` seja usada com frequência e gere bastante ganho estrutural nas árvores, seu impacto médio nas predições individuais é comparável ao de outras features de infraestrutura escolar.

- **Features de infraestrutura têm direção intuitiva:** `alunos_por_sala` (superlotação), `pct_in_internet`, `pct_in_sala_leitura` e `pct_in_biblioteca` mostram o padrão esperado — valores mais altos de acesso a recursos (rosa) tendem a empurrar a predição para "alfabetizado", e valores baixos (azul) para o lado oposto — reforçando que essas variáveis carregam sinal pedagogicamente coerente, e não apenas ruído.

A curva ROC do modelo otimizado no conjunto de teste confirma o AUC de 0,6708 já reportado na tabela de métricas. O traçado é suave e consistentemente acima da diagonal de referência, sem quedas ou degraus abruptos em nenhuma faixa de limiar — indicando que o modelo discrimina as classes de forma razoavelmente estável ao longo de todo o espectro de decisão, e que o teto de performance observado decorre da natureza difusa do sinal preditivo nas features

## Insights encontrados

### Distribuição do target
- O target `alfabetizado` está moderadamente desbalanceado: **59,1% alfabetizados** vs. **40,9% não alfabetizados** — desbalanceamento leve, não exige técnicas agressivas de balanceamento, mas justifica reportar métricas além de acurácia (precisão, recall, F1) na fase de avaliação.
- O percentual de alfabetização é preocupantemente baixo (abaixo de 60% em ambos os anos) e cresceu apenas **1,37 ponto percentual** entre 2023 (58,38%) e 2024 (59,75%) — evolução lenta em relação às metas de 2030.
- Rede **Estadual** tem taxa de alfabetização ligeiramente maior que a **Municipal** (62,09% vs. 58,77%).
- Forte desigualdade regional: **Norte (50,93%)** e **Nordeste (55,72%)** ficam bem abaixo de Sul (64,45%), Centro-Oeste (61,91%) e Sudeste (61,25%). Ainda assim, alguns estados dessas regiões mais precárias (CE, RO, PE) aparecem entre os de maior taxa de alfabetização do país — mostrando que a variação **dentro** de uma região é grande, o que reforça o valor de usar indicadores a nível de município (mais granulares) em vez de só UF/região.

### Casos de data leakage confirmados (não apenas suspeitados)

**1. `proficiencia` — leakage direto e documentado.** A própria documentação oficial da base confirma que `alfabetizado` é definido por uma regra de corte determinística: aluno é considerado alfabetizado se `proficiencia >= 743` na escala SAEB. Isso foi confirmado nos dados: a maior proficiência entre os não-alfabetizados foi 742,9998, e a menor entre os alfabetizados foi exatamente 743,0 — separação perfeita. **`proficiencia` não pode ser usada como variável explicativa**, sob risco de o modelo apenas reaprender essa regra de corte em vez de generalizar um padrão preditivo real.

**2. `mun_taxa_alfabetizacao` — leakage indireto, confirmado por reconstrução matemática.** Recalculamos a taxa de alfabetização de cada município a partir da própria base de alunos (média de `alfabetizado` ponderada por `peso_aluno`, agrupada por ano e município) e comparamos com o valor já presente na coluna `mun_taxa_alfabetizacao`. A diferença mediana entre os dois foi de apenas **0,0046 pontos percentuais** — ou seja, a coluna é, na prática, uma reconstrução quase exata do próprio target agregado, calculada incluindo o resultado do aluno que estamos tentando prever. Mesmo não sendo uma cópia linha a linha como `proficiencia`, o efeito é equivalente: usar essa variável entrega ao modelo uma versão agregada da resposta.

Essa mesma lógica se propaga para as variáveis derivadas de `mun_taxa_alfabetizacao` e `proficiencia`: `gap_mun_meta_2030`, `mun_nivel_alfabetizacao` e `mun_prop_nivel_`.

- `mun_nivel_alfabetizacao` é uma **categorização em faixas fixas de 10 pontos** de `mun_taxa_alfabetizacao` (nível 0: até ~40%, nível 1: 40–50%, ..., nível 5: 80–100%, sem sobreposição entre faixas). Não agrega nenhuma informação nova além da própria taxa (e ainda perde granularidade) — **candidata a descarte por multicolinearidade**, além de herdar o leakage do item acima.
- As variáveis a nível de **UF** (`uf_taxa_alfabetizacao`, `uf_media_portugues`) mostram bem menos variabilidade que as equivalentes a nível de **município**, confirmando que agregar num nível territorial mais amplo "suaviza" as diferenças reais entre localidades — reforça a decisão de priorizar as features municipais na modelagem.

### Variáveis sem sinal
- `id_municipio`, `id_escola`, `id_aluno` são identificadores, sem valor preditivo.
- `serie` tem um único valor constante em toda a base (2º ano) e `presenca` também (sempre 1, já que a base foi filtrada só para presentes na silver) — variância zero, sem informação.
- `caderno` e `preenchimento_caderno` mostraram correlação próxima de zero com o target (0,0005 e 0,023, respectivamente) — indícios de que são apenas ruído operacional da aplicação da prova, não sinal pedagógico.

## Limitações do projeto

- A base `alunos.csv` bruta não está versionada no repositório devido ao
  tamanho (>200MB), exigindo download manual para reprodução completa do
  pipeline desde o início (bronze). A camada `data/silver/` já processada,
  no entanto, está versionada, permitindo rodar a modelagem a partir de `src/integracao.py` e `src/contrucao_master_tabl.py`
- Dados restritos aos anos de 2023 e 2024, o que limita a capacidade do
  modelo de capturar tendências de longo prazo.
- Natureza das features das fontes sugeridas para enriqueciment contribuem de forma indireta, já que são oriundas de indicadores de munícipio. Os ids de aluno e escola foram mascarados, portanto não seria possível obter caracteristicas que possivelmente contribuissem para melhorar a previsão do modelo.


## Aplicação prática para políticas públicas

- Priorizar os recursos estimando por escola ou aluno a proporção antes da avaliação final, permitindo envio de reforços para que as metas sejam atendidas.

- Alerta de risco individual para o aluno com a maior probabilidade de não atingir o nível esperado.

- Direcionamento de amplio de infraestrutura municipal, dado que alimentação, biblioteca e laboratório de informática aparecem como fatores influentes para a alfabetizaçao.

- Investigar o que da certo na rede Estadual que apresentou taxas mais altas de alfabetização tanto na análise exploratória quanto no gráfico SHAP. 

## Possíveis evoluções futuras

- Substituir variaveis com contagens absolutas por versões normalizadas.
- Incorporar dados de anos adicionais para captar tendências temporais.
- A possibilidade de feats individuais para reduzir a necessidade de feats agrupadas indiretas provenientes de censos, municipios, UFs etc.
