
import os
import json
import gzip
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

def pregunta_modelo():

    # ---------------------------
    # 1. Crear carpetas necesarias
    # ---------------------------
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

    # ---------------------------
    # 2. Cargar datos
    # ---------------------------
    train = pd.read_csv("files/input/train_data.csv.zip")
    test = pd.read_csv("files/input/test_data.csv.zip")

    # ---------------------------
    # 3. Limpieza de datos
    # ---------------------------

    # Renombrar columna objetivo
    train = train.rename(columns={"default payment next month": "default"})
    test = test.rename(columns={"default payment next month": "default"})

    # Remover ID
    if "ID" in train.columns:
        train = train.drop(columns=["ID"])
    if "ID" in test.columns:
        test = test.drop(columns=["ID"])

    # EDUCATION > 4 → "others" (categoría 4)
    train.loc[train["EDUCATION"] > 4, "EDUCATION"] = 4
    test.loc[test["EDUCATION"] > 4, "EDUCATION"] = 4

    # Remover registros nulos
    train = train.dropna()
    test = test.dropna()

    # ---------------------------
    # 4. Dividir X, y
    # ---------------------------
    x_train = train.drop(columns=["default"])
    y_train = train["default"]

    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    # ---------------------------
    # 5. Crear pipeline
    # ---------------------------

    # Variables categóricas (por tipo)
    cat_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    # pero también EDUCATION, SEX, MARRIAGE se consideran categóricas
    for c in ["SEX", "EDUCATION", "MARRIAGE"]:
        if c not in cat_cols:
            cat_cols.append(c)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rf", RandomForestClassifier()),
        ]
    )

    param_grid = {
        "rf__n_estimators": [50, 100],
        "rf__max_depth": [5, 10, None],
    }

    model = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        refit=True,
    )

    # Entrenar
    model.fit(x_train, y_train)

    # ---------------------------
    # 6. Guardar modelo comprimido
    # ---------------------------
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

    # ---------------------------
    # 7. Calcular métricas
    # ---------------------------
    metrics_output = []

    def compute_metrics(dataset_name, y_true, y_pred):
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    # train metrics
    y_train_pred = model.predict(x_train)
    metrics_output.append(compute_metrics("train", y_train, y_train_pred))

    # test metrics
    y_test_pred = model.predict(x_test)
    metrics_output.append(compute_metrics("test", y_test, y_test_pred))

    # ---------------------------
    # 8. Matrices de confusión
    # ---------------------------

    def compute_cm(dataset_name, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
            "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
        }

    metrics_output.append(compute_cm("train", y_train, y_train_pred))
    metrics_output.append(compute_cm("test", y_test, y_test_pred))

    # ---------------------------
    # 9. Guardar metrics.json (line-delimited)
    # ---------------------------
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for m in metrics_output:
            f.write(json.dumps(m) + "\n")

    print("Modelo entrenado y métricas generadas correctamente.")
pregunta_modelo()