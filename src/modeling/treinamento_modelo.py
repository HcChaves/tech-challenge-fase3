# src/modeling/treinamento_modelo.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from src.preprocessing.preprocessamento import criar_preprocessador


def treinar_modelo(df, colunas_numericas, colunas_categoricas, coluna_target,
                    test_size=0.2, random_state=42, escalar_numericas=False):
    """
    Recebe o dataframe completo e as listas de colunas, executa o split
    treino/teste, monta a Pipeline (pré-processamento + LightGBM) e treina.

    Retorna um dicionário com o pipeline treinado e os conjuntos usados,
    para permitir avaliação e interpretabilidade posteriores sem retreinar.
    """

    X = df[colunas_numericas + colunas_categoricas]
    y = df[coluna_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessador = criar_preprocessador(colunas_numericas, colunas_categoricas, escalar_numericas)

    modelo = LGBMClassifier(random_state=random_state)

    pipeline = Pipeline(steps=[
        ("preprocessamento", preprocessador),
        ("modelo", modelo),
    ])

    pipeline.fit(X_train, y_train)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }