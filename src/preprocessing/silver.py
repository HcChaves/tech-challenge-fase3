"""
Camada Silver - Limpeza e padronizacao das bases da camada Bronze.
Versao local (sem AWS/S3), adaptada do pipeline original da Fase 2.

Como rodar (a partir da raiz do repositorio, depois do bronze.py):
    python src/preprocessing/silver.py
"""
import pandas as pd
from pathlib import Path

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")

ANOS = [2023, 2024]

# De-para retirado do dicionario de dados que acompanha as bases originais
MAPA_REDE_ALUNOS = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
MAPA_REDE_AGREGADO = {2: "Estadual", 3: "Municipal", 5: "Publica"}


def le_bronze(nome_base: str) -> pd.DataFrame:
    partes = []
    for ano in ANOS:
        caminho = BRONZE_DIR / nome_base / f"year={ano}" / "dados.parquet"
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


def dropa_duplicadas(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    antes = df.shape[0]
    df = df.drop_duplicates()
    print(f"  [{nome}] Duplicatas removidas: {antes - df.shape[0]}")
    return df


def limpa_alunos(alunos: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Tratando base Alunos ---")
    print(f"  Registros iniciais: {alunos.shape[0]}")

    alunos = alunos.copy()
    alunos["rede"] = alunos["rede"].map(MAPA_REDE_ALUNOS)

    # Remove rede Privada: nao possui meta de alfabetizacao no projeto
    alunos = alunos[alunos["rede"] != "Privada"]
    print(f"  Apos remover rede Privada: {alunos.shape[0]} registros")

    # O indicador mede apenas alunos avaliados (presentes na aplicacao)
    alunos = alunos[alunos["presenca"] == 1]
    print(f"  Apos remover ausentes: {alunos.shape[0]} registros")

    alunos["alfabetizado"] = alunos["alfabetizado"].astype(int)

    alunos = dropa_duplicadas(alunos, "alunos")
    return alunos


def limpa_agregado(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    print(f"\n--- Tratando base {nome} ---")
    print(f"  Registros iniciais: {df.shape[0]}")

    df = df.copy()
    # rede = 0 e valor invalido conforme dicionario de dados
    df = df[df["rede"] != 0]
    print(f"  Apos remover rede=0: {df.shape[0]} registros")

    df["rede"] = df["rede"].map(MAPA_REDE_AGREGADO)
    df = dropa_duplicadas(df, nome)
    return df


def limpa_meta(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    print(f"\n--- Tratando base {nome} ---")
    df = df.copy()
    # Corrige inconsistencia de tipo na fonte original
    if "meta_alfabetizacao_2030" in df.columns:
        df["meta_alfabetizacao_2030"] = df["meta_alfabetizacao_2030"].astype(float)
    df = dropa_duplicadas(df, nome)
    return df


def main():
    print("Lendo bases da camada Bronze\n")
    alunos = le_bronze("alunos")
    alfabetizacao_uf = le_bronze("alfabetizacao_uf")
    alfabetizacao_municipio = le_bronze("alfabetizacao_municipio")
    meta_brasil = le_bronze("meta_alfabetizacao_brasil")
    meta_uf = le_bronze("meta_alfabetizacao_uf")
    meta_municipio = le_bronze("meta_alfabetizacao_municipio")

    alunos = limpa_alunos(alunos)
    alfabetizacao_uf = limpa_agregado(alfabetizacao_uf, "alfabetizacao_uf")
    alfabetizacao_municipio = limpa_agregado(alfabetizacao_municipio, "alfabetizacao_municipio")
    meta_brasil = limpa_meta(meta_brasil, "meta_alfabetizacao_brasil")
    meta_uf = limpa_meta(meta_uf, "meta_alfabetizacao_uf")
    meta_municipio = limpa_meta(meta_municipio, "meta_alfabetizacao_municipio")

    print("\nSalvando bases na Silver\n")
    bases = {
        "alunos": alunos,
        "alfabetizacao_uf": alfabetizacao_uf,
        "alfabetizacao_municipio": alfabetizacao_municipio,
        "meta_alfabetizacao_brasil": meta_brasil,
        "meta_alfabetizacao_uf": meta_uf,
        "meta_alfabetizacao_municipio": meta_municipio,
    }
    for nome, df in bases.items():
        salva_silver(df, nome)

    print("\nCamada Silver concluida!")


if __name__ == "__main__":
    main()
