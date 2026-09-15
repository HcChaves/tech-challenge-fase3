from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def criar_preprocessador(colunas_numericas, colunas_categoricas, escalar_numericas=False):
    """
    Cria o ColumnTransformer responsável por:
    - Imputar valores faltantes em variáveis numéricas (mediana)
    - Opcionalmente padronizar variáveis numéricas (StandardScaler)
    - Aplicar One-Hot Encoding em variáveis categóricas

    escalar_numericas: defina True para modelos sensíveis a escala
    (ex: Regressão Logística, SVM, KNN). Modelos baseados em árvore
    (LightGBM, Random Forest) não precisam, mas não são prejudicados.
    """

    etapas_numericas = [("imputer", SimpleImputer(strategy="median"))]
    if escalar_numericas:
        etapas_numericas.append(("scaler", StandardScaler()))

    transformador_numerico = Pipeline(steps=etapas_numericas)

    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, colunas_numericas),
        ("cat", transformador_categorico, colunas_categoricas),
    ])

    return preprocessador