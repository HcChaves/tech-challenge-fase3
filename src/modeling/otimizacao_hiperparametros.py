# src/modeling/otimizacao_hiperparametros.py

from sklearn.experimental import enable_halving_search_cv  # noqa: F401 (necessário para habilitar o HalvingRandomSearchCV)
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src.preprocessing.preprocessamento import criar_preprocessador


def amostrar_estratificado(X, y, frac, random_state=42):
    """
    Retorna uma amostra estratificada de X e y, preservando a proporção
    das classes do target. Útil para acelerar a busca de hiperparâmetros
    em bases muito grandes, sem distorcer o balanceamento das classes.
    """
    if frac is None or frac >= 1.0:
        return X, y

    X_amostra, _, y_amostra, _ = train_test_split(
        X, y,
        train_size=frac,
        stratify=y,
        random_state=random_state,
    )
    return X_amostra, y_amostra


def grid_padrao():
    """
    Espaço de busca padrão para o LightGBM. Serve como ponto de partida
    (busca ampla)
    """
    return {
        "modelo__n_estimators": [100, 200, 300, 500],
        "modelo__num_leaves": [15, 31, 63, 127],
        "modelo__max_depth": [-1, -1, -1, 10, 20],
        "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "modelo__min_child_samples": [50, 100, 200, 500, 1000],
        "modelo__subsample": [0.7, 0.8, 0.9, 1.0],
        "modelo__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "modelo__reg_alpha": [0, 0.1, 0.5, 1, 5],
        "modelo__reg_lambda": [0, 0.1, 0.5, 1, 5],
        "modelo__is_unbalance": [True, False],
    }


def otimizar_hiperparametros(X_train, y_train, colunas_numericas, colunas_categoricas,
                              espaco_parametros=None, frac_amostra=None, n_candidates=40, cv=3,
                              min_resources="exhaust", max_resources="auto", factor=3,
                              n_jobs=4, random_state=42):
    """
    Executa busca de hiperparâmetros para o LightGBM dentro da Pipeline
    completa (pré-processamento + modelo), usando HalvingRandomSearchCV.

    Diferente do RandomizedSearchCV tradicional, o Halving testa todas as
    combinações candidatas com poucos dados/recursos primeiro, elimina as
    piores e só dá mais dados às combinações promissoras nas rodadas
    seguintes 
    """

    if espaco_parametros is None:
        espaco_parametros = grid_padrao()

    X_busca, y_busca = amostrar_estratificado(X_train, y_train, frac_amostra, random_state)

    preprocessador = criar_preprocessador(colunas_numericas, colunas_categoricas)

    pipeline = Pipeline(steps=[
        ("preprocessamento", preprocessador),
        ("modelo", LGBMClassifier(random_state=random_state, n_jobs=1)),
    ])

    cv_estratificado = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    busca = HalvingRandomSearchCV(
        estimator=pipeline,
        param_distributions=espaco_parametros,
        n_candidates=n_candidates,
        cv=cv_estratificado,
        factor=factor,
        min_resources=min_resources,
        max_resources=max_resources,
        resource="n_samples",
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=2,
    )

    busca.fit(X_busca, y_busca)

    print(f"Melhores parâmetros: {busca.best_params_}")
    print(f"Melhor ROC-AUC (validação cruzada): {busca.best_score_:.4f}")
    print(f"Total de candidatos testados na busca: {n_candidates}")
    print(f"Recursos (amostras) usados na última rodada: {busca.n_resources_[-1]}")

    return busca