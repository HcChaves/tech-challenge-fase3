# src/evaluation/interpretabilidade.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plotar_curva_roc(resultado, titulo="Curva ROC"):
    """
    Plota a curva ROC do pipeline treinado sobre o conjunto de teste.

    Espera o mesmo formato de dicionário retornado por `treinar_modelo`
    (ou montado a partir de `busca.best_estimator_`): precisa das chaves
    "pipeline", "X_test" e "y_test".
    """
    pipeline = resultado["pipeline"]
    X_test = resultado["X_test"]
    y_test = resultado["y_test"]

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Modelo (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Classificador aleatório (AUC = 0.5)")
    plt.xlabel("Taxa de falsos positivos (FPR)")
    plt.ylabel("Taxa de verdadeiros positivos (TPR)")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return auc


def preparar_dados_shap(pipeline, X, sample_size=50_000, random_state=42):
    """
    Aplica o pré-processamento do pipeline em uma amostra de X e devolve os
    dados já transformados (formato numérico que o modelo realmente recebe),
    junto com os nomes das colunas pós-transformação.

    Usar uma amostra (em vez da base inteira) é recomendado para bases
    grandes: o cálculo de SHAP values é bem mais custoso que uma predição
    simples, e uma amostra de 30-50 mil linhas já é suficiente para uma
    leitura estável dos padrões de interpretabilidade.
    """
    if sample_size is not None and sample_size < len(X):
        X_amostra = X.sample(n=sample_size, random_state=random_state)
    else:
        X_amostra = X

    preprocessador = pipeline.named_steps["preprocessamento"]
    X_transformado = preprocessador.transform(X_amostra)
    nomes_features = preprocessador.get_feature_names_out()

    if hasattr(X_transformado, "toarray"):
        X_transformado = X_transformado.toarray()

    return X_transformado, nomes_features


def calcular_shap_values(pipeline, X, sample_size=50_000, random_state=42):
    """
    Calcula os SHAP values do modelo LightGBM dentro do pipeline, sobre uma
    amostra de X. Retorna o objeto de explicação do shap (`shap.Explanation`),
    """
    import shap

    X_transformado, nomes_features = preparar_dados_shap(pipeline, X, sample_size, random_state)

    modelo = pipeline.named_steps["modelo"]
    explainer = shap.TreeExplainer(modelo)
    explicacao = explainer(X_transformado)
    explicacao.feature_names = list(nomes_features)

    return explicacao


def plotar_shap_summary(explicacao, max_features=15):
    """
    Gráfico "beeswarm" do SHAP: mostra, para cada feature, como os valores
    (cores) se relacionam com o impacto na predição (posição no eixo X) --
    além de quais features mais influenciam o modelo, revela a *direção*
    do efeito
    """
    import shap

    shap.summary_plot(explicacao, max_display=max_features, show=False)
    plt.tight_layout()
    plt.show()


def plotar_shap_bar(explicacao, max_features=15):
    """
    Gráfico de barras do SHAP: importância média absoluta de cada feature.
    """
    import shap

    shap.plots.bar(explicacao, max_display=max_features, show=False)
    plt.tight_layout()
    plt.show()


def plotar_shap_dependencia(explicacao, feature, feature_interacao=None):
    """
    Gráfico de dependência do SHAP para uma única feature
    """
    import shap

    shap.plots.scatter(explicacao[:, feature], color=explicacao[:, feature_interacao] if feature_interacao else None, show=False)
    plt.tight_layout()
    plt.show()