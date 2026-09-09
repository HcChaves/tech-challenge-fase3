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

A ser preenchido conforme a análise exploratória e os resultados do modelo.

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
