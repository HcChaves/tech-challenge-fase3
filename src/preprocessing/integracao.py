"""
Integracao - Junta a base de Alunos com indicadores agregados do Municipio.
Versao local (sem AWS/S3), adaptada do pipeline original da Fase 2.

Esta e a tabela a nivel aluno que vamos usar para treinar o modelo.

Como rodar (a partir da raiz do repositorio, depois do silver.py):
    python src/preprocessing/integracao.py
"""
import pandas as pd
from pathlib import Path

SILVER_DIR = Path("data/silver")
ANOS = [2023, 2024]

# de-para de codigo id_municipio correspondente aos 2 primeiros digitos para uf e regiao
# fonte: https://www.ibge.gov.br/explica/codigos-dos-municipios.php
MAPA_UF = {
    11: ("RO", "Norte"), 12: ("AC", "Norte"), 13: ("AM", "Norte"),
    14: ("RR", "Norte"), 15: ("PA", "Norte"), 16: ("AP", "Norte"), 17: ("TO", "Norte"),
    21: ("MA", "Nordeste"), 22: ("PI", "Nordeste"), 23: ("CE", "Nordeste"),
    24: ("RN", "Nordeste"), 25: ("PB", "Nordeste"), 26: ("PE", "Nordeste"),
    27: ("AL", "Nordeste"), 28: ("SE", "Nordeste"), 29: ("BA", "Nordeste"),
    31: ("MG", "Sudeste"), 32: ("ES", "Sudeste"), 33: ("RJ", "Sudeste"), 35: ("SP", "Sudeste"),
    41: ("PR", "Sul"), 42: ("SC", "Sul"), 43: ("RS", "Sul"),
    50: ("MS", "Centro-Oeste"), 51: ("MT", "Centro-Oeste"),
    52: ("GO", "Centro-Oeste"), 53: ("DF", "Centro-Oeste"),
}

def le_silver(nome_base: str) -> pd.DataFrame:
    partes = []
    for ano in ANOS:
        caminho = SILVER_DIR / nome_base / f"year={ano}" / "dados.parquet"
        partes.append(pd.read_parquet(caminho))
    return pd.concat(partes, ignore_index=True)


def salva_silver(df: pd.DataFrame, nome_base: str) -> None:
    for ano in ANOS:
        df_ano = df[df["ano"] == ano]
        destino = SILVER_DIR / nome_base / f"year={ano}"
        destino.mkdir(parents=True, exist_ok=True)
        caminho_saida = destino / "dados.parquet"
        df_ano.to_parquet(caminho_saida, index=False)
        print(f"  OK Salvo: {caminho_saida} ({df_ano.shape[0]} registros)")


def adiciona_sigla_uf(alunos: pd.DataFrame) -> pd.DataFrame:
    """Deriva sigla_uf e regiao a partir dos 2 primeiros digitos de id_municipio."""
    alunos = alunos.copy()
    codigo_uf = (alunos["id_municipio"] // 100000).astype(int)
    alunos["sigla_uf"] = codigo_uf.map(lambda c: MAPA_UF.get(c, (None, None))[0])
    alunos["regiao"] = codigo_uf.map(lambda c: MAPA_UF.get(c, (None, None))[1])
    return alunos


def junta_municipio(alunos: pd.DataFrame, municipio: pd.DataFrame) -> pd.DataFrame:
    """
    Traz da base alfabetizacao_municipio:
    - mun_taxa_alfabetizacao, mun_media_portugues
    - mun_prop_nivel_0 a mun_prop_nivel_8: distribuicao de desempenho do
      municipio, mais informativa que so a media.
    """
    colunas_nivel = [f"proporcao_aluno_nivel_{i}" for i in range(9)]
    rename_nivel = {c: f"mun_prop_nivel_{c.split('_')[-1]}" for c in colunas_nivel}
 
    municipio_sel = municipio[[
        "ano", "id_municipio", "serie", "rede",
        "taxa_alfabetizacao", "media_portugues", *colunas_nivel,
    ]].rename(columns={
        "taxa_alfabetizacao": "mun_taxa_alfabetizacao",
        "media_portugues": "mun_media_portugues",
        **rename_nivel,
    })
 
    return pd.merge(
        alunos, municipio_sel,
        on=["ano", "id_municipio", "serie", "rede"],
        how="left",
    )


def junta_meta_municipio(alunos: pd.DataFrame, meta_municipio: pd.DataFrame) -> pd.DataFrame:
    """
    Traz da base meta_alfabetizacao_municipio:
    - mun_nivel_alfabetizacao, mun_percentual_participacao
    - gap_mun_meta_2030 = mun_taxa_alfabetizacao - meta_alfabetizacao_2030
    """
    meta_sel = meta_municipio[[
        "ano", "id_municipio", "rede",
        "meta_alfabetizacao_2030", "nivel_alfabetizacao", "percentual_participacao",
    ]].rename(columns={
        "nivel_alfabetizacao": "mun_nivel_alfabetizacao",
        "percentual_participacao": "mun_percentual_participacao",
    })
 
    alunos = pd.merge(
        alunos, meta_sel,
        on=["ano", "id_municipio", "rede"],
        how="left",
    )
 
    alunos["gap_mun_meta_2030"] = round(
        alunos["mun_taxa_alfabetizacao"] - alunos["meta_alfabetizacao_2030"], 2
    )
    alunos = alunos.drop(columns=["meta_alfabetizacao_2030"])
    return alunos


def junta_uf(alunos: pd.DataFrame, uf: pd.DataFrame) -> pd.DataFrame:
    """
    Traz da base alfabetizacao_uf:
    - uf_taxa_alfabetizacao, uf_media_portugues
    - gap_mun_vs_uf = mun_taxa_alfabetizacao - uf_taxa_alfabetizacao
    """
    uf_sel = uf[[
        "ano", "sigla_uf", "serie", "rede", "taxa_alfabetizacao", "media_portugues",
    ]].rename(columns={
        "taxa_alfabetizacao": "uf_taxa_alfabetizacao",
        "media_portugues": "uf_media_portugues",
    })
 
    alunos = pd.merge(
        alunos, uf_sel,
        on=["ano", "sigla_uf", "serie", "rede"],
        how="left",
    )
 
    alunos["gap_mun_vs_uf"] = round(
        alunos["mun_taxa_alfabetizacao"] - alunos["uf_taxa_alfabetizacao"], 2
    )
    return alunos


def main():
    print("Lendo bases da Silver\n")
    alunos = le_silver("alunos")
    municipio = le_silver("alfabetizacao_municipio")
    meta_municipio = le_silver("meta_alfabetizacao_municipio")
    uf = le_silver("alfabetizacao_uf")
 
    n_inicial = alunos.shape[0]
 
    print("Derivando sigla_uf e regiao a partir de id_municipio\n")
    alunos = adiciona_sigla_uf(alunos)
    print(f"  Municipios sem UF identificada: {alunos['sigla_uf'].isnull().sum()}")
 
    print("\nIndicadores de Municipio\n")
    alunos = junta_municipio(alunos, municipio)
 
    print("Metas de Municipio\n")
    alunos = junta_meta_municipio(alunos, meta_municipio)
 
    print("Indicadores de UF\n")
    alunos = junta_uf(alunos, uf)
 
    # Protecao contra duplicacao de linhas em qualquer um dos merges left
    assert alunos.shape[0] == n_inicial, (
        f"ERRO: algum join multiplicou linhas! Esperado {n_inicial}, "
        f"obtido {alunos.shape[0]}"
    )
 
    print(f"\nRegistros finais: {alunos.shape[0]}")
    print(f"Colunas finais: {alunos.shape[1]}")
    print(f"\n{alunos.columns.tolist()}")
 
    print("\nSalvando tabela integrada na Silver\n")
    salva_silver(alunos, "integrado")
 
    print("\nIntegracao concluida!")


if __name__ == "__main__":
    main()
