"""
Camada Bronze - Le as bases brutas (CSV) e converte para Parquet particionado por ano.
Versao local, adaptada do pipeline original da Fase 2.

Como rodar (a partir da raiz do repositorio):
    python src/preprocessing/bronze.py
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
BRONZE_DIR = Path("data/bronze")

ANOS = [2023, 2024]

# nome_base -> (arquivo csv em data/raw, coluna que identifica o ano)
BASES = {
    "alunos": ("alunos.csv", "ano"),
    "meta_alfabetizacao_brasil": ("meta_alfabetizacao_brasil.csv", "ano"),
    "meta_alfabetizacao_municipio": ("meta_alfabetizacao_municipio.csv", "ano"),
    "meta_alfabetizacao_uf": ("meta_alfabetizacao_uf.csv", "ano"),
    "alfabetizacao_municipio": ("alfabetizacao_municipio.csv", "ano"),
    "alfabetizacao_uf": ("alfabetizacao_uf.csv", "ano"),
}


def processa_base(nome_base: str, arquivo_csv: str, col_ano: str) -> None:
    caminho_csv = RAW_DIR / arquivo_csv
    print(f"Lendo {caminho_csv} ...")
    df = pd.read_csv(caminho_csv)

    for ano in ANOS:
        df_ano = df[df[col_ano] == ano]

        destino = BRONZE_DIR / nome_base / f"year={ano}"
        destino.mkdir(parents=True, exist_ok=True)

        caminho_saida = destino / "dados.parquet"
        df_ano.to_parquet(caminho_saida, index=False)

        print(f"  OK {nome_base} year={ano}: {df_ano.shape[0]} registros -> {caminho_saida}")


def main():
    for nome_base, (arquivo_csv, col_ano) in BASES.items():
        processa_base(nome_base, arquivo_csv, col_ano)
    print("\nCamada Bronze concluida!")


if __name__ == "__main__":
    main()
