import numpy as np


def engenharia_features(df):
    """
    Aplica engenharia de features na master table e remove as colunas
    originais que foram consolidadas em índices compostos ou substituídas
    por versões transformadas, evitando redundância/multicolinearidade.
    """

    df = df.copy()

    # ---------------------------------------------------------------
    # 1. Índice composto de acessibilidade
    # ---------------------------------------------------------------
    colunas_acessibilidade = [
        "pct_in_acessibilidade_corrimao",
        "pct_in_acessibilidade_elevador",
        "pct_in_acessibilidade_pisos_tateis",
        "pct_in_acessibilidade_vao_livre",
        "pct_in_acessibilidade_rampas",
        "pct_in_acessibilidade_sinal_sonoro",
        "pct_in_acessibilidade_sinal_tatil",
        "pct_in_acessibilidade_sinal_visual",
    ]
    df["indice_acessibilidade"] = df[colunas_acessibilidade].mean(axis=1)

    # ---------------------------------------------------------------
    # 2. Índice composto de material pedagógico
    # ---------------------------------------------------------------
    colunas_material_ped = [
        "pct_in_material_ped_infantil",
        "pct_in_material_ped_cientifico",
        "pct_in_material_ped_jogos",
        "pct_in_material_ped_artisticas",
        "pct_in_material_ped_multimidia",
    ]
    df["indice_material_pedagogico"] = df[colunas_material_ped].mean(axis=1)

    # ---------------------------------------------------------------
    # 3. Índice composto de infraestrutura básica
    # ---------------------------------------------------------------
    colunas_infra_basica = [
        "pct_in_agua_potavel",
        "pct_in_energia_inexistente",
        "pct_in_esgoto_inexistente",
    ]
    df["indice_infra_basica"] = (
        df["pct_in_agua_potavel"]
        + (1 - df["pct_in_energia_inexistente"])
        + (1 - df["pct_in_esgoto_inexistente"])
    ) / 3

    # ---------------------------------------------------------------
    # 4. Razão aluno/sala (proxy de superlotação)
    # ---------------------------------------------------------------
    colunas_porte = ["qt_mat_bas_media", "qt_salas_media"]
    df["alunos_por_sala"] = df["qt_mat_bas_media"] / df["qt_salas_media"]

    # ---------------------------------------------------------------
    # 5. Log-transform em n_escolas
    # ---------------------------------------------------------------
    df["log_n_escolas"] = np.log1p(df["n_escolas"])

    # ---------------------------------------------------------------
    # 6. Redução de redundância socioeconômica (coeficiente de variação)
    # ---------------------------------------------------------------
    variaveis_socioeconomicas = ["qtd_pes_pob", "qtd_pes_baixa_renda", "qtd_pes_acima_meio_sm"]

    for var in variaveis_socioeconomicas:
        df[f"{var}_cv"] = df[f"{var}_desvio_padrao"] / df[f"{var}_media"].replace(0, np.nan)
        df[f"{var}_cv"] = df[f"{var}_cv"].fillna(0)

    colunas_socioeconomicas_redundantes = []
    for var in variaveis_socioeconomicas:
        colunas_socioeconomicas_redundantes += [f"{var}_min", f"{var}_max", f"{var}_desvio_padrao"]

    # ---------------------------------------------------------------
    # Remoção final de todas as colunas originais substituídas
    # ---------------------------------------------------------------
    colunas_para_remover = (
        colunas_acessibilidade
        + colunas_material_ped
        + colunas_infra_basica
        + colunas_porte
        + ["n_escolas"]
        + colunas_socioeconomicas_redundantes
    )

    df = df.drop(columns=colunas_para_remover)

    return df