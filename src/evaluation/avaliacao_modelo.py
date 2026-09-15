# src/evaluation/avaliacao_modelo.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def avaliar_modelo(resultado_treino):
    """
    Recebe o dicionário retornado por treinar_modelo() e calcula as
    métricas de avaliação no conjunto de teste, sem retreinar o modelo.

    Reporta métricas gerais e também separadas por classe (0 = não
    alfabetizado, 1 = alfabetizado), já que a classe 0 é o principal
    alvo de interesse para políticas públicas de risco educacional.
    """

    pipeline = resultado_treino["pipeline"]
    X_test = resultado_treino["X_test"]
    y_test = resultado_treino["y_test"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precisao_por_classe = precision_score(y_test, y_pred, average=None, labels=[0, 1])
    recall_por_classe = recall_score(y_test, y_pred, average=None, labels=[0, 1])
    f1_por_classe = f1_score(y_test, y_pred, average=None, labels=[0, 1])

    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "matriz_confusao": confusion_matrix(y_test, y_pred),
        "classe_0_nao_alfabetizado": {
            "precisao": precisao_por_classe[0],
            "recall": recall_por_classe[0],
            "f1": f1_por_classe[0],
        },
        "classe_1_alfabetizado": {
            "precisao": precisao_por_classe[1],
            "recall": recall_por_classe[1],
            "f1": f1_por_classe[1],
        },
    }

    print(classification_report(y_test, y_pred, target_names=["não alfabetizado", "alfabetizado"]))
    print(f"ROC-AUC: {metricas['roc_auc']:.4f}")

    return metricas