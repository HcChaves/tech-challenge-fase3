# src/modeling/otimizacao_hiperparametros.py

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src.preprocessing.preprocessamento import criar_preprocessador


def otimizar_hiperparametros(X_train, y_train, colunas_numericas, colunas_categoricas,
                              n_iter=20, cv=3, random_state=42):
    """
    Executa busca aleatória de hiperparâmetros para o LightGBM dentro
    da Pipeline completa (pré-processamento + modelo), usando validação
    cruzada estratificada.

    Retorna o RandomizedSearchCV já ajustado, do qual se pode extrair
    o melhor pipeline (`.best_estimator_`) e os melhores parâmetros
    (`.best_params_`).
    """

    preprocessador = criar_preprocessador(colunas_numericas, colunas_categoricas)

    pipeline = Pipeline(steps=[
        ("preprocessamento", preprocessador),
        ("modelo", LGBMClassifier(random_state=random_state)),
    ])

    espaco_parametros = {
        "modelo__n_estimators": [100, 200, 300, 500],
        "modelo__num_leaves": [15, 31, 63, 127],
        "modelo__max_depth": [-1, 5, 10, 15],
        "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "modelo__min_child_samples": [10, 20, 50, 100],
        "modelo__subsample": [0.7, 0.8, 0.9, 1.0],
        "modelo__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }

    cv_estratificado = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    busca = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=espaco_parametros,
        n_iter=n_iter,
        cv=cv_estratificado,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=random_state,
        verbose=2,
    )

    busca.fit(X_train, y_train)

    print(f"Melhores parâmetros: {busca.best_params_}")
    print(f"Melhor ROC-AUC (validação cruzada): {busca.best_score_:.4f}")

    return busca