# Documentação Técnica — Pipeline de Dados e Modelagem (Tech Challenge Fase 3)

Este documento detalha o funcionamento de cada script do projeto, complementando o `README.md` com uma visão técnica aprofundada: funções, parâmetros, fluxo de dados e decisões de implementação.

Como registrado no `README.md`, esta fase reconstrói o pipeline de forma **simplificada e local**, partindo das bases raw, sem depender da infraestrutura de nuvem (S3/Kafka) utilizada na arquitetura original da Fase 2.

---

## Índice

1. [Camada Bronze](#camada-bronze)
   - [`bronze.py`](#bronzepy)
2. [Camada Silver](#camada-silver)
   - [`silver.py`](#silverpy)
   - [`integracao.py`](#integracaopy)
3. [Construção da Master Table](#construção-da-master-table)
   - [`contrucao_master_table.py`](#contrucao_master_tablepy)
4. [Modelagem](#modelagem)
   - [`engenharia_features.py`](#engenharia_featurespy)
   - [`preprocessamento.py`](#preprocessamentopy)
   - [`treinamento_modelo.py`](#treinamento_modelopy)
   - [`otimizacao_hiperparametros.py`](#otimizacao_hiperparametrospy)
5. [Avaliação e Interpretabilidade](#avaliação-e-interpretabilidade)
   - [`avaliacao_modelo.py`](#avaliacao_modelopy)
   - [`interpretabilidade.py`](#interpretabilidadepy)

---

## Camada Bronze

### `bronze.py`

**Propósito:** Ler os 6 CSVs brutos armazenados em `data/raw/`, filtrar os registros por ano e salvá-los em formato Parquet, particionados por ano, em `data/bronze/`.

**Fluxo de execução:**
```
1. Para cada base do dicionário BASES, lê o CSV correspondente de data/raw/
2. Filtra os registros de cada base pelos anos em ANOS (2023, 2024)
3. Salva cada subconjunto anual como Parquet em data/bronze/{nome_base}/year={ano}/dados.parquet
```

**Função principal:**

```python
processa_base(nome_base: str, arquivo_csv: str, col_ano: str) -> None
```
| Parâmetro | Tipo | Descrição |
|---|---|---|
| `nome_base` | str | Nome lógico da base — usado para montar o caminho de destino |
| `arquivo_csv` | str | Nome do arquivo CSV em `data/raw/` |
| `col_ano` | str | Coluna usada para filtrar os registros por ano |

**Bases processadas:** `alunos`, `meta_alfabetizacao_brasil`, `meta_alfabetizacao_municipio`, `meta_alfabetizacao_uf`, `alfabetizacao_municipio`, `alfabetizacao_uf`

**Decisão de dados:** versão local e simplificada da arquitetura original da Fase 2 (que usava S3 + `boto3`) — leitura e gravação inteiramente locais via `pathlib`, eliminando a necessidade de infraestrutura de nuvem.

**Saída:**
```
data/bronze/{base}/year={ano}/dados.parquet
```

---

## Camada Silver

### `silver.py`

**Propósito:** Ler as bases da camada Bronze, aplicar limpeza e padronização, e salvar o resultado tratado na Silver.

**Fluxo de execução:**
```
1. Lê cada base da Bronze, concatenando os anos 2023 e 2024 (le_bronze)
2. Aplica a função de limpeza específica de cada tipo de base
3. Remove duplicatas
4. Salva cada base tratada na Silver, particionada por ano (salva_silver)
```

**Funções utilitárias:**

```python
le_bronze(nome_base)
```
Concatena os arquivos Parquet de 2023 e 2024 de uma base da Bronze em um único DataFrame.

```python
salva_silver(df, nome_base)
```
Salva o DataFrame particionado por ano em `data/silver/{nome_base}/year={ano}/dados.parquet`.

```python
dropa_duplicadas(df, nome)
```
Remove linhas duplicadas de um DataFrame e reporta quantas foram removidas.

**Funções de limpeza por tipo de base:**

```python
limpa_alunos(alunos)
```
- Mapeia `rede` (inteiro → texto) via `MAPA_REDE_ALUNOS`
- Remove a rede **Privada** (não possui meta de alfabetização no projeto)
- Remove alunos ausentes na avaliação (`presenca != 1`)
- Converte `alfabetizado` para inteiro
- Remove duplicatas

```python
limpa_agregado(df, nome)
```
Usada para `alfabetizacao_uf` e `alfabetizacao_municipio`:
- Remove registros com `rede = 0` (valor inválido conforme dicionário de dados)
- Mapeia `rede` (inteiro → texto) via `MAPA_REDE_AGREGADO`
- Remove duplicatas

```python
limpa_meta(df, nome)
```
Usada nas 3 bases de meta (`meta_alfabetizacao_brasil`, `meta_alfabetizacao_uf`, `meta_alfabetizacao_municipio`):
- Corrige o tipo da coluna `meta_alfabetizacao_2030` (int → float)
- Remove duplicatas

**Mapeamentos de `rede` utilizados:**
```python
# Base alunos
MAPA_REDE_ALUNOS = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}

# Bases agregadas (UF e Município)
MAPA_REDE_AGREGADO = {2: "Estadual", 3: "Municipal", 5: "Publica"}
```

**Saída:**
```
data/silver/{base}/year={ano}/dados.parquet
```

---

### `integracao.py`

**Propósito:** Juntar a base de alunos (Silver) com os indicadores agregados de município e UF e com as metas oficiais, criando a tabela `data/silver/integrado/` a nível de aluno — insumo para o enriquecimento posterior feito em `contrucao_master_table.py`.

**Fluxo de execução:**
```
1. Lê alunos, alfabetizacao_municipio, meta_alfabetizacao_municipio e alfabetizacao_uf da Silver (le_silver)
2. Deriva sigla_uf e regiao a partir dos 2 primeiros dígitos de id_municipio (adiciona_sigla_uf)
3. Junta indicadores de município (junta_municipio)
4. Junta metas de município (junta_meta_municipio)
5. Junta indicadores de UF (junta_uf)
6. Valida (assert) que nenhum dos joins left multiplicou o número de linhas
7. Salva o resultado particionado por ano em data/silver/integrado/
```

**Funções principais:**

```python
adiciona_sigla_uf(alunos)
```
Deriva `sigla_uf` e `regiao` a partir do código de 2 dígitos da UF embutido em `id_municipio`, usando um mapa fixo (`MAPA_UF`) construído a partir da tabela oficial de códigos de município do IBGE.

```python
junta_municipio(alunos, municipio)
```
| Junção | Chave |
|---|---|
| alunos + alfabetizacao_municipio | `ano, id_municipio, serie, rede` |

Traz `mun_taxa_alfabetizacao`, `mun_media_portugues` e a distribuição de desempenho do município (`mun_prop_nivel_0` a `mun_prop_nivel_8`).

```python
junta_meta_municipio(alunos, meta_municipio)
```
| Junção | Chave |
|---|---|
| alunos + meta_alfabetizacao_municipio | `ano, id_municipio, rede` |

Traz `mun_nivel_alfabetizacao`, `mun_percentual_participacao` e calcula:
```
gap_mun_meta_2030 = mun_taxa_alfabetizacao - meta_alfabetizacao_2030
```

```python
junta_uf(alunos, uf)
```
| Junção | Chave |
|---|---|
| alunos + alfabetizacao_uf | `ano, sigla_uf, serie, rede` |

Traz `uf_taxa_alfabetizacao`, `uf_media_portugues` e calcula:
```
gap_mun_vs_uf = mun_taxa_alfabetizacao - uf_taxa_alfabetizacao
```

**Decisão de dados:** `sigla_uf` e `regiao` são derivados matematicamente a partir do próprio `id_municipio` (mapa fixo baseado na tabela do IBGE), em vez de virem de um merge com uma tabela externa de-para — reduz uma dependência de dados e simplifica o pipeline.

**Validação:** um `assert` garante que a contagem de linhas da base de alunos não muda após nenhum dos merges `left` — protege contra duplicação silenciosa de registros por chaves não únicas nas bases de município/UF/meta.

**Saída:**
```
data/silver/integrado/year={ano}/dados.parquet
```

---

## Construção da Master Table

### `contrucao_master_table.py`

**Propósito:** Enriquecer a tabela integrada da Silver (nível aluno) com duas fontes externas de dados a nível de município — **Censo Escolar** (infraestrutura e recursos pedagógicos) e **Cadastro Único** (indicadores socioeconômicos) — construindo a master table final usada na modelagem.

**Fluxo de execução:**
```
1. Lê a base Silver de alunos (2023 + 2024) e adiciona a coluna ANO
2. Remove colunas de leakage e colunas constantes/sem utilidade
3. Carrega o Censo Escolar (2023 e 2024), filtrando apenas as colunas relevantes
4. Agrega o Censo Escolar por município + ano
5. Faz merge (left) da base de alunos com o Censo agregado, por id_municipio + ano
6. Ajusta id_municipio para o padrão de 6 dígitos (sem dígito verificador), usado pelo Cadastro Único
7. Carrega o Cadastro Único (2023 e 2024) e deriva o ano a partir de anomes_s
8. Agrega o Cadastro Único por município + ano
9. Faz merge (left, validate="many_to_one") com o Cadastro Único agregado
10. Salva a master table final em Parquet
```

**Colunas removidas na etapa 2 (leakage / sem utilidade):**

| Coluna | Motivo |
|---|---|
| `proficiencia` | Leakage direto — `alfabetizado` é definido por um corte determinístico sobre esta variável (`proficiencia >= 743`) |
| `preenchimento_caderno`, `caderno` | Ruído operacional da aplicação da prova, correlação próxima de zero com o target |
| `presenca` | Variância zero após o filtro de alunos presentes feito na Silver |
| `serie` | Valor único em toda a base (2º ano) |
| `peso_aluno` | Peso amostral, não é feature preditiva (usado depois, à parte, em análises agregadas por município) |

**Função principal:**

```python
carregar_censo(ano)
```
Lê o CSV de microdados do Censo Escolar de um ano específico (`sep=";"`, `encoding="latin-1"`), seleciona apenas as colunas de infraestrutura/recursos relevantes (`colunas_censo`) e renomeia `CO_ENTIDADE` → `id_escola` e `CO_MUNICIPIO` → `id_municipio_censo`.

**Agregação do Censo Escolar (por `id_municipio` + `ano`):**
- Percentual médio de cada indicador binário (`pct_<indicador>` = média dos `IN_*`)
- `qt_mat_bas_media`, `qt_salas_media`: médias de matrícula e salas
- `n_escolas`: contagem de escolas distintas (`nunique` de `id_escola`)

**Agregação do Cadastro Único (por `codigo_ibge` + `ano`):**
Para cada uma das 3 variáveis socioeconômicas (`qtd_pes_pob`, `qtd_pes_baixa_renda`, `qtd_pes_acima_meio_sm`), calcula média, mínimo, máximo e desvio padrão.

**Decisão de dados:** o join com o Censo Escolar e com o Cadastro Único é feito **a nível de município**, não de escola, porque `id_escola` na base Silver é uma máscara/código fictício (não corresponde ao código real do INEP) — não é possível casar diretamente com bases externas a nível de escola.

**Nota de reprodutibilidade:** os caminhos deste script são relativos à raiz do repositório (`data/silver/alunos` e `data/externo/`), assumindo execução a partir da raiz (`python src/preprocessing/contrucao_master_table.py`). A pasta `data/externo/` não é versionada — cada pessoa que clonar o repositório deve baixar os arquivos do Censo Escolar e do Cadastro Único e salvá-los nesse caminho localmente (ver README).

**Saída:** `data/master/alunos_enriquecido.parquet` — arquivo local, fora do versionamento do repositório por depender de fontes externas (Censo Escolar e Cadastro Único) não incluídas no projeto.

---

## Modelagem

### `engenharia_features.py`

**Propósito:** Aplicar engenharia de features na master table, consolidando grupos de colunas correlacionadas em índices compostos e reduzindo redundância entre variáveis socioeconômicas, removendo em seguida as colunas originais substituídas.

**Fluxo de execução:**
```
1. indice_acessibilidade = média das 8 colunas pct_in_acessibilidade_*
2. indice_material_pedagogico = média das 5 colunas pct_in_material_ped_*
3. indice_infra_basica = média entre pct_in_agua_potavel, (1 - pct_in_energia_inexistente)
   e (1 - pct_in_esgoto_inexistente) -- as duas últimas são invertidas por serem
   indicadores de AUSÊNCIA do recurso, não de presença
4. alunos_por_sala = qt_mat_bas_media / qt_salas_media (proxy de superlotação)
5. log_n_escolas = log1p(n_escolas), reduzindo a assimetria da distribuição
6. Para cada variável socioeconômica (qtd_pes_pob, qtd_pes_baixa_renda,
   qtd_pes_acima_meio_sm), calcula o coeficiente de variação
   (_cv = desvio_padrao / media, nulos preenchidos com 0)
7. Remove todas as colunas originais consolidadas/substituídas nos passos acima
```

**Função principal:**

```python
engenharia_features(df) -> df
```

| Grupo de origem | Colunas removidas | Feature(s) criada(s) |
|---|---|---|
| Acessibilidade (8 colunas `pct_in_acessibilidade_*`) | 8 | `indice_acessibilidade` |
| Material pedagógico (5 colunas `pct_in_material_ped_*`) | 5 | `indice_material_pedagogico` |
| Infraestrutura básica (água, energia, esgoto) | 3 | `indice_infra_basica` |
| Porte da escola (`qt_mat_bas_media`, `qt_salas_media`) | 2 | `alunos_por_sala` |
| `n_escolas` | 1 | `log_n_escolas` |
| Socioeconômicas (min/max/desvio_padrão × 3 variáveis) | 9 | `qtd_pes_pob_cv`, `qtd_pes_baixa_renda_cv`, `qtd_pes_acima_meio_sm_cv` |

**Resultado observado** (`notebooks/02_modelagem.ipynb`): a master table entra com **44 colunas** e sai com **24 colunas** após a engenharia de features.

**Decisão de dados:** a opção por consolidar em índices/coeficiente de variação, em vez de simplesmente descartar as colunas originais, veio da análise exploratória (`01_explo_base.ipynb`), que identificou forte multicolinearidade dentro de cada grupo — preservando o sinal agregado sem manter a redundância.

---

### `preprocessamento.py`

**Propósito:** Criar o `ColumnTransformer` usado como primeira etapa das Pipelines de treino e otimização, padronizando o tratamento de variáveis numéricas e categóricas.

**Função principal:**

```python
criar_preprocessador(colunas_numericas, colunas_categoricas, escalar_numericas=False)
```
| Parâmetro | Tipo | Descrição |
|---|---|---|
| `colunas_numericas` | list[str] | Colunas numéricas a imputar (e, opcionalmente, padronizar) |
| `colunas_categoricas` | list[str] | Colunas categóricas a imputar e codificar |
| `escalar_numericas` | bool | Se `True`, aplica `StandardScaler` após a imputação. Default `False`, já que o LightGBM não é sensível à escala das variáveis |

**Tratamento numérico:** `SimpleImputer(strategy="median")` + (opcional) `StandardScaler`

**Tratamento categórico:** `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

**Saída:** um `ColumnTransformer` com dois ramos (`"num"` e `"cat"`), reutilizado tanto em `treinamento_modelo.py` quanto em `otimizacao_hiperparametros.py`.

---

### `treinamento_modelo.py`

**Propósito:** Treinar o modelo baseline — LightGBM com hiperparâmetros default — dentro de uma Pipeline scikit-learn completa (pré-processamento + modelo).

**Função principal:**

```python
treinar_modelo(df, colunas_numericas, colunas_categoricas, coluna_target,
                test_size=0.2, random_state=42, escalar_numericas=False)
```

**Fluxo de execução:**
```
1. Separa X (colunas_numericas + colunas_categoricas) e y (coluna_target)
2. Split treino/teste estratificado (train_test_split, stratify=y)
3. Monta o preprocessador via criar_preprocessador
4. Monta a Pipeline (preprocessamento + LGBMClassifier(random_state))
5. Treina (fit) na base de treino
6. Retorna um dicionário com o pipeline treinado e os 4 conjuntos
   (X_train, X_test, y_train, y_test), evitando a necessidade de
   retreinar para avaliação e interpretabilidade posteriores
```

**Colunas usadas no projeto** (definidas em `02_modelagem.ipynb`, a partir das 24 colunas resultantes de `engenharia_features`):
- **Categóricas (1):** `rede`
- **Numéricas (18):** todas as demais colunas de `df_final`, exceto identificadores (`id_aluno`, `id_escola`, `id_municipio`, `ano`) e o target (`alfabetizado`)

**Resultado observado:** split 80/20 estratificado gerou **2.684.657 registros de treino** e **671.165 de teste**, preservando a proporção de 59,14% alfabetizados / 40,86% não alfabetizados em ambos os conjuntos. ROC-AUC do baseline no teste: **0,6540**.

---

### `otimizacao_hiperparametros.py`

**Propósito:** Buscar os melhores hiperparâmetros do LightGBM dentro da mesma Pipeline (pré-processamento + modelo), usando `HalvingRandomSearchCV`.

**Funções principais:**

```python
amostrar_estratificado(X, y, frac, random_state=42)
```
Amostragem estratificada opcional, aplicada antes da busca — reduz ainda mais o custo total quando `frac_amostra` é informado.

```python
grid_padrao()
```
Espaço de busca padrão (ampla), usado quando `espaco_parametros=None`:

| Hiperparâmetro | Valores testados |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `num_leaves` | 15, 31, 63, 127 |
| `max_depth` | -1 (peso maior), 10, 20 |
| `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| `min_child_samples` | 50, 100, 200, 500, 1000 |
| `subsample` | 0.7, 0.8, 0.9, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 0.9, 1.0 |
| `reg_alpha` | 0, 0.1, 0.5, 1, 5 |
| `reg_lambda` | 0, 0.1, 0.5, 1, 5 |
| `is_unbalance` | True, False |

```python
otimizar_hiperparametros(X_train, y_train, colunas_numericas, colunas_categoricas,
                          espaco_parametros=None, frac_amostra=None, n_candidates=40, cv=3,
                          min_resources="exhaust", max_resources="auto", factor=3,
                          n_jobs=4, random_state=42)
```
| Parâmetro | Descrição |
|---|---|
| `espaco_parametros` | Espaço de busca customizado; se `None`, usa `grid_padrao()` |
| `frac_amostra` | Amostragem estratificada opcional antes da busca |
| `n_candidates` | Nº de combinações candidatas na primeira rodada do Halving |
| `cv` | Nº de folds por rodada (`StratifiedKFold`) |
| `min_resources` / `max_resources` | Piso/teto de amostras usadas pelo Halving por rodada |
| `factor` | Fração de candidatos eliminada a cada rodada |
| `n_jobs` | Paralelismo no nível da busca (o `LGBMClassifier` interno usa `n_jobs=1`) |

**Decisões de implementação:**
- **`HalvingRandomSearchCV`** em vez de `RandomizedSearchCV` tradicional: dado o volume da base (~3,36M de registros), o Halving aloca progressivamente mais dados às combinações mais promissoras, evitando treinar exaustivamente combinações ruins na base inteira.
- **`n_jobs=1` no `LGBMClassifier`**, com paralelismo controlado só no nível do `HalvingRandomSearchCV`: evita oversubscription de CPU (dois níveis de paralelismo disputando os mesmos núcleos).
- **`max_resources` exposto como parâmetro**: permite limitar o teto de memória usado nas rodadas finais — mitigação direta de um `MemoryError`/`BrokenProcessPool` observado empiricamente ao rodar com `n_jobs=-1` e `max_resources="auto"` na base completa.

**Resultados observados (rodadas registradas em `02_modelagem.ipynb`):**

| Rodada | Espaço de busca | Configuração | ROC-AUC (CV) | ROC-AUC (teste) |
|---|---|---|---|---|
| Baseline | — (`treinamento_modelo.py`) | — | — | 0,6540 |
| Busca ampla | `grid_padrao()`, 40 candidatos | `n_jobs=4`, `max_resources=1.000.000` | 0,6689 | — |
| Refinamento | espaço estreito ao redor dos vencedores | `n_jobs=4`, `max_resources=1.500.000` | 0,6702 | 0,6708 |

**Melhores hiperparâmetros encontrados (rodada de refinamento):**
```python
{
    "n_estimators": 300, "num_leaves": 63, "max_depth": -1,
    "learning_rate": 0.05, "min_child_samples": 20, "subsample": 1.0,
    "colsample_bytree": 0.6, "reg_alpha": 10, "reg_lambda": 0.5,
    "is_unbalance": True,
}
```

---

## Avaliação e Interpretabilidade

### `avaliacao_modelo.py`

**Propósito:** Calcular as métricas de avaliação do pipeline treinado sobre o conjunto de teste, sem retreinar o modelo.

**Função principal:**

```python
avaliar_modelo(resultado_treino)
```
Recebe o dicionário retornado por `treinar_modelo` (ou montado a partir de `busca.best_estimator_`) e calcula:
- Acurácia e ROC-AUC gerais
- Matriz de confusão
- Precisão, recall e F1 **separados por classe** (0 = não alfabetizado, 1 = alfabetizado)

**Decisão de dados:** métricas reportadas por classe, e não só agregadas, porque a classe "não alfabetizado" é o principal alvo de interesse para políticas públicas de risco educacional — uma métrica agregada como acurácia pode mascarar baixo desempenho justamente nessa classe (observado no baseline: recall de apenas 0,32 para "não alfabetizado", antes do tuning com `is_unbalance=True` elevá-lo para 0,62).

---

### `interpretabilidade.py`

**Propósito:** Fornecer as ferramentas de interpretabilidade do modelo final — curva ROC e análise de SHAP values — complementando as métricas agregadas de `avaliacao_modelo.py`.

**Funções principais:**

```python
plotar_curva_roc(resultado, titulo="Curva ROC")
```
Plota a curva ROC do pipeline no conjunto de teste e retorna o AUC.

```python
preparar_dados_shap(pipeline, X, sample_size=50_000, random_state=42)
```
Aplica o preprocessador do pipeline em uma amostra de `X` e devolve os dados já transformados (formato numérico que o modelo recebe de fato), junto com os nomes das colunas pós-transformação.

```python
calcular_shap_values(pipeline, X, sample_size=50_000, random_state=42)
```
Calcula os SHAP values do modelo LightGBM via `shap.TreeExplainer`, sobre uma amostra de `X`. Retorna um objeto `shap.Explanation`.

```python
plotar_shap_summary(explicacao, max_features=15)
```
Gráfico "beeswarm": mostra, além de quais features mais influenciam o modelo, a **direção** do efeito de cada uma.

```python
plotar_shap_bar(explicacao, max_features=15)
```
Gráfico de barras da importância média absoluta (`mean(|SHAP value|)`) por feature.

```python
plotar_shap_dependencia(explicacao, feature, feature_interacao=None)
```
Gráfico de dependência de uma feature específica, com possibilidade de colorir por uma segunda feature de interação.

**Decisão de dados:** o cálculo de SHAP é feito sobre uma amostra de 50 mil linhas do conjunto de teste (`sample_size`), e não sobre a base inteira (671 mil linhas), porque o custo computacional do `TreeExplainer` é bem maior que o de uma predição simples — 50 mil linhas já é suficiente para uma leitura estável dos padrões de interpretabilidade.

**Resultados observados:** o SHAP confirmou o padrão de importância difusa entre as features (a soma das features fora do top 15 supera, sozinha, o impacto médio da feature mais importante isolada) e reordenou o ranking em relação ao `gain`/`split` calculados via `feature_importances_` do LightGBM — ver `README.md`, seção "Interpretação dos resultados", para a análise completa e os gráficos gerados.

---
