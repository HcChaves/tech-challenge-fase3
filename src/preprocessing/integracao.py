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


def main():
    print("Lendo bases da Silver\n")
    alunos = le_silver("alunos")
    municipio = le_silver("alfabetizacao_municipio")

    print("Integrando alunos com indicadores do municipio\n")

    municipio_renamed = municipio[[
        "ano", "id_municipio", "rede",
        "taxa_alfabetizacao", "media_portugues",
    ]].rename(columns={
        "taxa_alfabetizacao": "mun_taxa_alfabetizacao",
        "media_portugues": "mun_media_portugues",
    })

    df_integrado = pd.merge(
        alunos,
        municipio_renamed,
        on=["ano", "id_municipio", "rede"],
        how="left",
    )

    # Garante que o join nao multiplicou linhas (protecao contra chave duplicada)
    assert df_integrado.shape[0] == alunos.shape[0], (
        f"ERRO: join multiplicou linhas! Esperado {alunos.shape[0]}, "
        f"obtido {df_integrado.shape[0]}"
    )

    print(f"  Registros apos join: {df_integrado.shape[0]}")
    print(f"  Colunas finais: {df_integrado.shape[1]}")
    print(f"\n  {df_integrado.columns.tolist()}")

    print("\nSalvando tabela integrada na Silver\n")
    salva_silver(df_integrado, "integrado")

    print("\nIntegracao concluida!")


if __name__ == "__main__":
    main()
