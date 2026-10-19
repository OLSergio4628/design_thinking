import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from loadData import RUTA_SALIDA_DATASET, cargar_y_validar_dataset


RUTA_RAIZ = Path(__file__).resolve().parent.parent
RUTA_REGISTRO = RUTA_RAIZ / "datos" / "model_registry.json"
RUTA_MODELO = RUTA_RAIZ / "datos" / "best_fraud_model.joblib"
RUTA_BASE_MLFLOW = RUTA_RAIZ / "mlflow.db"
EXPERIMENTO_MLFLOW = "credit-card-fraud"
METRICAS = {
    "average_precision": average_precision_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "balanced_accuracy": balanced_accuracy_score,
    "accuracy": accuracy_score,
}
SCORING = {
    "average_precision": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
}
SEMILLA = 42


def _crear_busquedas(device="cpu"):
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=SEMILLA,
                        ),
                    ),
                ]
            ),
            {"classifier__C": [0.1, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=100,
                n_jobs=-1,
                random_state=SEMILLA,
            ),
            {"max_depth": [None, 12]},
        ),
        "xgboost": (
            XGBClassifier(
                device=device,
                tree_method="hist",
                eval_metric="logloss",
                n_estimators=100,
                n_jobs=-1,
                random_state=SEMILLA,
            ),
            {"max_depth": [3, 6]},
        ),
    }


def _metricas_prueba(modelo, caracteristicas, etiquetas):
    probabilidades = modelo.predict_proba(caracteristicas)[:, 1]
    predicciones = (probabilidades >= 0.5).astype("int8")
    return {
        nombre: float(
            funcion(
                etiquetas,
                probabilidades
                if nombre in {"average_precision", "roc_auc"}
                else predicciones,
            )
        )
        for nombre, funcion in METRICAS.items()
    }


def _estadisticas_cv(busqueda):
    return {
        metrica: {
            "media": float(busqueda.cv_results_[f"mean_test_{metrica}"][
                busqueda.best_index_
            ]),
            "desviacion_estandar": float(
                busqueda.cv_results_[f"std_test_{metrica}"][
                    busqueda.best_index_
                ]
            ),
        }
        for metrica in SCORING
    }


def entrenar_y_registrar(
    dataframe,
    ruta_registro=RUTA_REGISTRO,
    ruta_modelo=RUTA_MODELO,
    n_splits=5,
    test_size=0.2,
    tracking_uri=None,
    device="cpu",
):
    """Ajusta clasificadores, evalúa un holdout y registra sus resultados."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device debe ser 'cpu' o 'cuda'.")
    if n_splits < 2:
        raise ValueError("La validación cruzada requiere al menos 2 particiones.")
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    caracteristicas = dataframe.drop(columns=["Class"])
    etiquetas = dataframe["Class"]
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        caracteristicas,
        etiquetas,
        test_size=test_size,
        random_state=SEMILLA,
        stratify=etiquetas,
    )
    menor_clase = int(y_entrenamiento.value_counts().min())
    if menor_clase < n_splits:
        raise ValueError(
            "La clase minoritaria del conjunto de entrenamiento debe tener al "
            f"menos {n_splits} ejemplos para la validación cruzada."
        )

    validacion = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEMILLA,
    )
    uri_tracking = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{RUTA_BASE_MLFLOW.as_posix()}",
    )
    mlflow.set_tracking_uri(uri_tracking)
    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    resultados_modelos = {}
    mejores_estimadores = {}
    for nombre, (estimador, parametros) in _crear_busquedas(device).items():
        busqueda = GridSearchCV(
            estimator=estimador,
            param_grid=parametros,
            scoring=SCORING,
            refit="average_precision",
            cv=validacion,
            n_jobs=1,
            return_train_score=False,
        )
        busqueda.fit(x_entrenamiento, y_entrenamiento)
        resultados_modelos[nombre] = {
            "mejores_parametros": busqueda.best_params_,
            "validacion_cruzada": _estadisticas_cv(busqueda),
            "prueba": _metricas_prueba(
                busqueda.best_estimator_, x_prueba, y_prueba
            ),
        }
        mejores_estimadores[nombre] = busqueda.best_estimator_

    mejor_nombre = max(
        resultados_modelos,
        key=lambda nombre: resultados_modelos[nombre]["validacion_cruzada"][
            "average_precision"
        ]["media"],
    )
    for nombre, resultados in resultados_modelos.items():
        with mlflow.start_run(run_name=nombre) as run:
            mlflow.log_params(
                {
                    "model_name": nombre,
                    "cv_strategy": "StratifiedKFold",
                    "cv_folds": n_splits,
                    "random_seed": SEMILLA,
                    "selection_metric": "average_precision",
                    "training_rows": len(y_entrenamiento),
                    "test_rows": len(y_prueba),
                    "xgboost_device": device,
                    "best_params": json.dumps(
                        resultados["mejores_parametros"], sort_keys=True
                    ),
                }
            )
            metricas_cv = {
                f"cv_{metrica}_{estadistica}": valor
                for metrica, estadisticas in resultados[
                    "validacion_cruzada"
                ].items()
                for estadistica, valor in estadisticas.items()
            }
            metricas_prueba = {
                f"test_{metrica}": valor
                for metrica, valor in resultados["prueba"].items()
            }
            mlflow.log_metrics({**metricas_cv, **metricas_prueba})
            mlflow.set_tag("selected_model", str(nombre == mejor_nombre))
            mlflow.sklearn.log_model(
                sk_model=mejores_estimadores[nombre],
                name="model",
                serialization_format="cloudpickle",
            )
            mlflow.log_dict(
                {
                    "model_name": nombre,
                    "best_params": resultados["mejores_parametros"],
                    "cross_validation": resultados["validacion_cruzada"],
                    "test": resultados["prueba"],
                    "selected_model": nombre == mejor_nombre,
                },
                "evaluation.json",
            )
            resultados["mlflow_run_id"] = run.info.run_id

    registro = {
        "run_id": str(uuid.uuid4()),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": RUTA_SALIDA_DATASET.name,
        "mlflow_experiment": EXPERIMENTO_MLFLOW,
        "mlflow_tracking_uri": uri_tracking,
        "xgboost_device": device,
        "registros": int(len(dataframe)),
        "registros_entrenamiento": int(len(y_entrenamiento)),
        "registros_prueba": int(len(y_prueba)),
        "validacion_cruzada": {
            "estrategia": "StratifiedKFold",
            "particiones": n_splits,
            "semilla": SEMILLA,
            "metrica_seleccion": "average_precision",
        },
        "modelos": resultados_modelos,
        "mejor_modelo": mejor_nombre,
    }

    ruta_registro = Path(ruta_registro)
    ruta_modelo = Path(ruta_modelo)
    ruta_registro.parent.mkdir(parents=True, exist_ok=True)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    if ruta_registro.exists():
        contenido = json.loads(ruta_registro.read_text(encoding="utf-8"))
        if not isinstance(contenido, dict) or not isinstance(
            contenido.get("experimentos"), list
        ):
            raise ValueError(
                f"El registro de modelos tiene un formato inválido: {ruta_registro}"
            )
    else:
        contenido = {"experimentos": []}

    joblib.dump(mejores_estimadores[mejor_nombre], ruta_modelo)
    contenido["experimentos"].append(registro)
    ruta_registro.write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registro


def main():
    parser = argparse.ArgumentParser(
        description="Entrena y compara modelos para detectar fraude."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=RUTA_SALIDA_DATASET,
        help="Ruta al CSV limpio de transacciones.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Número de particiones para StratifiedKFold.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Dispositivo de entrenamiento de XGBoost (cuda requiere GPU NVIDIA compatible).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="URI de tracking de MLflow; también se puede definir MLFLOW_TRACKING_URI.",
    )
    argumentos = parser.parse_args()
    dataframe = cargar_y_validar_dataset(argumentos.data)
    registro = entrenar_y_registrar(
        dataframe,
        n_splits=argumentos.folds,
        tracking_uri=argumentos.tracking_uri,
        device=argumentos.device,
    )
    mejor = registro["mejor_modelo"]
    metricas = registro["modelos"][mejor]["prueba"]
    print(
        f"Mejor modelo por average precision CV: {mejor}; "
        f"average precision en prueba: {metricas['average_precision']:.4f}"
    )
    print(f"Registro: {RUTA_REGISTRO}")
    print(f"Modelo guardado: {RUTA_MODELO}")
    print(f"MLflow tracking: {registro['mlflow_tracking_uri']}")


if __name__ == "__main__":
    main()
