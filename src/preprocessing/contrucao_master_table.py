"""
Construção da Master Table - Tech Challenge Fase 03
Enriquecimento da base Silver de alunos com Censo Escolar e Cadastro Único.
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# 1. Carregar base Silver de alunos (2023 e 2024 - Fase 02)
# =============================================================================

base_path = Path(r"data\silver\alunos")

dfs = []
for year_dir in sorted(base_path.glob("year=*")):
    year = year_dir.name.split("=")[1]
    df_year = pd.read_parquet(year_dir / "dados.parquet")
    df_year["ANO"] = year
    dfs.append(df_year)

df = pd.concat(dfs, ignore_index=True)

print(f"Total de registros base Alunos: {len(df):,}")
print(f"Total de colunas base Alunos: {df.shape[1]}")

# =============================================================================
# 2. Remover colunas de leakage e colunas constantes/sem utilidade
# =============================================================================

colunas_para_remover = [
    "proficiencia",
    "preenchimento_caderno",
    "caderno",
    "presenca",
    "serie",
    "peso_aluno",
]

df = df.drop(columns=colunas_para_remover)

print(f"Colunas restantes base Alunos após drop: {df.shape[1]}")
print(df.columns.tolist())

# =============================================================================
# 3. Carregar e filtrar Censo Escolar (2023 e 2024)
# =============================================================================

colunas_censo = [
    "CO_ENTIDADE", "CO_MUNICIPIO", "TP_LOCALIZACAO", "TP_DEPENDENCIA",
    "IN_BIBLIOTECA", "IN_SALA_LEITURA", "IN_INTERNET", "IN_LABORATORIO_INFORMATICA",
    "IN_AGUA_POTAVEL", "IN_ENERGIA_INEXISTENTE", "IN_ESGOTO_INEXISTENTE", "IN_ALIMENTACAO",
    "IN_ACESSIBILIDADE_CORRIMAO", "IN_ACESSIBILIDADE_ELEVADOR", "IN_ACESSIBILIDADE_PISOS_TATEIS",
    "IN_ACESSIBILIDADE_VAO_LIVRE", "IN_ACESSIBILIDADE_RAMPAS", "IN_ACESSIBILIDADE_SINAL_SONORO",
    "IN_ACESSIBILIDADE_SINAL_TATIL", "IN_ACESSIBILIDADE_SINAL_VISUAL", "IN_BANHEIRO_PNE",
    "IN_MATERIAL_PED_INFANTIL", "IN_MATERIAL_PED_CIENTIFICO", "IN_MATERIAL_PED_JOGOS",
    "IN_MATERIAL_PED_ARTISTICAS", "IN_MATERIAL_PED_MULTIMIDIA",
    "QT_MAT_BAS", "QT_SALAS_UTILIZADAS", "IN_EDUCACAO_INDIGENA",
]

base_path_externo = "data/externo"


def carregar_censo(ano):
    path = fr"{base_path_externo}\microdados_ed_basica_{ano}.csv"
    df_censo = pd.read_csv(path, sep=";", encoding="latin-1", usecols=colunas_censo)
    df_censo = df_censo.rename(columns={"CO_ENTIDADE": "id_escola", "CO_MUNICIPIO": "id_municipio_censo"})
    df_censo["ano"] = str(ano)
    return df_censo


df_censo_2023 = carregar_censo(2023)
df_censo_2024 = carregar_censo(2024)
df_censo_full = pd.concat([df_censo_2023, df_censo_2024], ignore_index=True)

print(f"Total Censo Escolar: {df_censo_full.shape}")
df_censo_full.to_csv(fr"{base_path_externo}\censo_escolar_filtrado.csv", index=False)

# =============================================================================
# 4. Agregar Censo Escolar por município (id_escola no Silver é mascarado/fictício,
#    join só é possível a nível de município)
# =============================================================================

colunas_indicadores = [
    "IN_BIBLIOTECA", "IN_SALA_LEITURA", "IN_INTERNET", "IN_LABORATORIO_INFORMATICA",
    "IN_AGUA_POTAVEL", "IN_ENERGIA_INEXISTENTE", "IN_ESGOTO_INEXISTENTE", "IN_ALIMENTACAO",
    "IN_ACESSIBILIDADE_CORRIMAO", "IN_ACESSIBILIDADE_ELEVADOR", "IN_ACESSIBILIDADE_PISOS_TATEIS",
    "IN_ACESSIBILIDADE_VAO_LIVRE", "IN_ACESSIBILIDADE_RAMPAS", "IN_ACESSIBILIDADE_SINAL_SONORO",
    "IN_ACESSIBILIDADE_SINAL_TATIL", "IN_ACESSIBILIDADE_SINAL_VISUAL", "IN_BANHEIRO_PNE",
    "IN_MATERIAL_PED_INFANTIL", "IN_MATERIAL_PED_CIENTIFICO", "IN_MATERIAL_PED_JOGOS",
    "IN_MATERIAL_PED_ARTISTICAS", "IN_MATERIAL_PED_MULTIMIDIA", "IN_EDUCACAO_INDIGENA",
]

df_censo_municipio = df_censo_full.groupby(["id_municipio_censo", "ano"]).agg(
    **{f"pct_{col.lower()}": (col, "mean") for col in colunas_indicadores},
    qt_mat_bas_media=("QT_MAT_BAS", "mean"),
    qt_salas_media=("QT_SALAS_UTILIZADAS", "mean"),
    n_escolas=("id_escola", "nunique"),
).reset_index()

df_censo_municipio = df_censo_municipio.rename(columns={"id_municipio_censo": "id_municipio"})

print(f"Total de registros Censo Escolar por município: {df_censo_municipio.shape[0]}")

# =============================================================================
# 5. Merge Silver + Censo Escolar (por município + ano)
# =============================================================================

df["ano"] = df["ano"].astype(int)
df_censo_municipio["ano"] = df_censo_municipio["ano"].astype(int)

df_enriquecido = df.merge(df_censo_municipio, on=["id_municipio", "ano"], how="left")

taxa_match = df_enriquecido["n_escolas"].notna().mean()
print(f"Taxa de match: {taxa_match:.2%}")

# Checagem de nulos nas colunas novas
colunas_novas = [c for c in df_enriquecido.columns if c not in df.columns]
print(df_enriquecido[colunas_novas].isnull().sum())

# Checkpoint: salvar base enriquecida só com Censo Escolar
df_enriquecido.to_parquet(
    fr"{base_path_externo}\alunos_enriquecido.parquet",
    index=False
)
print(f"Total de registros base Alunos enriquecida: {df_enriquecido.shape[0]}")

# =============================================================================
# 6. Ajustar id_municipio para o padrão de 6 dígitos (sem dígito verificador),
#    usado pelo Cadastro Único
# =============================================================================

df_enriquecido["id_municipio"] = df_enriquecido["id_municipio"] // 10

# =============================================================================
# 7. Carregar Cadastro Único (2023 e 2024)
# =============================================================================

cad_unico_2023 = pd.read_csv(
    fr"{base_path_externo}\cadunico_2023.txt", sep=None, engine="python"
)
cad_unico_2024 = pd.read_csv(
    fr"{base_path_externo}\cadunico_2024.txt", sep=None, engine="python"
)

cad_unico = pd.concat([cad_unico_2023, cad_unico_2024], ignore_index=True)

# =============================================================================
# 8. Agregar Cadastro Único por município + ano
# =============================================================================

variaveis_cad = [
    "qtd_pes_pob",
    "qtd_pes_baixa_renda",
    "qtd_pes_acima_meio_sm",
]

cad_unico["ano"] = cad_unico["anomes_s"].astype(str).str[:4].astype(int)

cad_unico_ano = (
    cad_unico
    .groupby(["codigo_ibge", "ano"], as_index=False)
    .agg(
        **{f"{col}_media": (col, "mean") for col in variaveis_cad},
        **{f"{col}_min": (col, "min") for col in variaveis_cad},
        **{f"{col}_max": (col, "max") for col in variaveis_cad},
        **{f"{col}_desvio_padrao": (col, "std") for col in variaveis_cad},
        qtd_meses=("anomes_s", "nunique"),
    )
)

# =============================================================================
# 9. Merge com o Cadastro Único
# =============================================================================

df_enriquecido_v2 = df_enriquecido.merge(
    cad_unico_ano,
    left_on=["ano", "id_municipio"],
    right_on=["ano", "codigo_ibge"],
    how="left",
    validate="many_to_one",
).drop(columns=["codigo_ibge"])

# Checagem de taxa de match
coluna_cad = "qtd_meses"

total = len(df_enriquecido_v2)
linhas_com_match = df_enriquecido_v2[coluna_cad].notna().sum()
linhas_sem_match = total - linhas_com_match

print(f"Total de linhas: {total:,}")
print(f"Linhas com match: {linhas_com_match:,}")
print(f"Linhas sem match: {linhas_sem_match:,}")
print(f"Taxa de match: {linhas_com_match / total:.2%}")

df_enriquecido_v2 = df_enriquecido_v2.drop(columns=[coluna_cad])

print(f"Total de registros base Alunos enriquecida com match: {len(df_enriquecido_v2):,}")

# =============================================================================
# 10. Salvar master table final
# =============================================================================

df_enriquecido_v2.to_parquet(
    fr"{base_path_externo}\alunos_enriquecido_v2.parquet",
    index=False
)

print(f"Shape da master table final: {df_enriquecido_v2.shape}")