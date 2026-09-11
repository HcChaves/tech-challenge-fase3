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

**Sobre `alunos.csv`:** este arquivo tem mais de 200MB e não está incluído
no repositório. Para reproduzir a pipeline de dados:
1. Baixe o arquivo na fonte indicada acima.
2. Pode ser necessario realizar um consulta no BigQuery, devido ao tamanho da base
3. Caso necessario, marque todas as colunas, clique em `gerar consulta`, clique em
    `Acessar o BigQuery`, execute a consulta e baixe os dados completos.
4. Salve-o em `data/raw/alunos.csv`.

As demais 5 bases já estão versionadas em `data/raw/`.

### Pipeline de dados

data/raw/ - CSVs originais
data/bronze/ - Mesmos dados, convertidos para Parquet e particionados por ano
data/silver/- Dados limpos (tipos corrigidos, duplicatas removidas, categorias padronizadas)
data/silver/integrado/ - Tabela final a nível aluno, com indicadores do município já unidos (esta é a base usada para treinar o modelo)

Para rodar o pipeline completo:

```bash
python src/preprocessing/bronze.py
python src/preprocessing/silver.py
python src/preprocessing/integracao.py
```
**Obs sobre versionamento dos dados:** Já `data/silver/`, incluindo a tabela
`integrado/` usada para treinar o modelo, **é versionada**, permitindo ir direto para a etapa de modelagem.

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

A ser preenchido...

## Escolha do algoritmo

A ser preenchido...

## Métricas de avaliação

A ser preenchido...

## Interpretação dos resultados

A ser preenchido...

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

### Variáveis sem sinal (ruído)
- `id_municipio`, `id_escola`, `id_aluno` são identificadores, sem valor preditivo.
- `serie` tem um único valor constante em toda a base (2º ano) e `presenca` também (sempre 1, já que a base foi filtrada só para presentes na silver) — variância zero, sem informação.
- `caderno` e `preenchimento_caderno` mostraram correlação próxima de zero com o target (0,0005 e 0,023, respectivamente) — indícios de que são apenas ruído operacional da aplicação da prova, não sinal pedagógico.

## Limitações do projeto

- A base `alunos.csv` bruta não está versionada no repositório devido ao
  tamanho (>200MB), exigindo download manual para reprodução completa do
  pipeline desde o início (bronze). A camada `data/silver/` já processada,
  no entanto, está versionada, permitindo rodar a modelagem sem esse download.
- Dados restritos aos anos de 2023 e 2024, o que limita a capacidade do
  modelo de capturar tendências de longo prazo.


## Aplicação prática para políticas públicas

_A ser preenchido: como o modelo pode apoiar decisões de gestores_
_educacionais (ex: priorização de recursos, alunos em risco, municípios_
_com maior necessidade de apoio pedagógico)._

## Possíveis evoluções futuras

_A ser preenchido: extensões possíveis do projeto (novas bases, novos_
_anos, granularidade diferente, etc.)._
