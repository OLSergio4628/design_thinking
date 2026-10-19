import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuentes"))
from loadData import COLUMNAS, cargar_y_validar_dataset, resumir_dataset
from trainModels import EXPERIMENTO_MLFLOW, _crear_busquedas, entrenar_y_registrar


def crear_dataframe():
    return pd.DataFrame(
        [
            [0, *([0.0] * 28), 10.0, 0],
            [8 * 60 * 60, *([0.1] * 28), 100.0, 1],
            [16 * 60 * 60, *([0.2] * 28), 50.0, 1],
        ],
        columns=COLUMNAS,
    )


class CargaDatasetTests(unittest.TestCase):
    def test_configura_xgboost_para_gpu_cuda(self):
        estimador, _ = _crear_busquedas(device="cuda")["xgboost"]

        self.assertEqual(estimador.get_params()["device"], "cuda")
        self.assertEqual(estimador.get_params()["tree_method"], "hist")

    def test_valida_esquema_y_conserva_etiqueta_binaria(self):
        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "creditcard.csv"
            crear_dataframe().to_csv(ruta, index=False)

            dataframe = cargar_y_validar_dataset(ruta)

        self.assertEqual(len(dataframe), 3)
        self.assertEqual(dataframe["Class"].tolist(), [0, 1, 1])
        self.assertEqual(str(dataframe["Class"].dtype), "int8")

    def test_rechaza_valores_nulos(self):
        dataframe = crear_dataframe()
        dataframe.loc[0, "V1"] = float("nan")
        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "creditcard.csv"
            dataframe.to_csv(ruta, index=False)

            with self.assertRaisesRegex(ValueError, "valores nulos"):
                cargar_y_validar_dataset(ruta)

    def test_rechaza_etiquetas_fuera_de_binario(self):
        dataframe = crear_dataframe()
        dataframe.loc[0, "Class"] = 2
        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "creditcard.csv"
            dataframe.to_csv(ruta, index=False)

            with self.assertRaisesRegex(ValueError, "etiquetas 0 y 1"):
                cargar_y_validar_dataset(ruta)

    def test_rechaza_dataset_que_no_contiene_ambas_clases(self):
        dataframe = crear_dataframe()
        dataframe["Class"] = 0
        with tempfile.TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "creditcard.csv"
            dataframe.to_csv(ruta, index=False)

            with self.assertRaisesRegex(ValueError, "ambas clases"):
                cargar_y_validar_dataset(ruta)

    def test_resume_desbalance_montos_y_bloques_temporales(self):
        resumen = resumir_dataset(crear_dataframe())

        self.assertEqual(resumen["registros"], 3)
        self.assertEqual(resumen["clases"], {"0": 1, "1": 2})
        self.assertAlmostEqual(resumen["porcentaje_fraude"], 200 / 3)
        self.assertEqual(resumen["mediana_amount_por_clase"], {"0": 10.0, "1": 75.0})
        self.assertEqual(
            resumen["fraudes_por_bloque_8h_relativo"],
            [
                {"intervalo": "0–8 h", "conteo": 0},
                {"intervalo": "8–16 h", "conteo": 1},
                {"intervalo": "16–24 h", "conteo": 1},
            ],
        )

    def test_entrena_compara_y_registra_dos_modelos(self):
        dataframe = pd.DataFrame(
            {
                "feature_a": [float(numero) for numero in range(60)],
                "feature_b": [float(numero % 7) for numero in range(60)],
                "Class": [0] * 50 + [1] * 10,
            }
        )
        cpu_busquedas = _crear_busquedas(device="cpu")
        with tempfile.TemporaryDirectory() as directorio:
            ruta_registro = Path(directorio) / "model_registry.json"
            ruta_modelo = Path(directorio) / "best_model.joblib"
            uri_tracking = "sqlite:///:memory:"
            mlflow.set_tracking_uri(uri_tracking)
            cliente_mlflow = MlflowClient(tracking_uri=uri_tracking)
            cliente_mlflow.create_experiment(
                EXPERIMENTO_MLFLOW,
                artifact_location=(Path(directorio) / "artifacts").as_uri(),
            )

            registro = entrenar_y_registrar(
                dataframe,
                ruta_registro=ruta_registro,
                ruta_modelo=ruta_modelo,
                n_splits=2,
                tracking_uri=uri_tracking,
            )
            contenido = json.loads(ruta_registro.read_text(encoding="utf-8"))
            modelo_guardado = joblib.load(ruta_modelo)
            modelo_guardado = joblib.load(ruta_modelo)

            self.assertEqual(
                set(registro["modelos"]),
                {"logistic_regression", "random_forest", "xgboost"},
            )
            self.assertIn(registro, contenido["experimentos"])
            self.assertEqual(registro["validacion_cruzada"]["particiones"], 2)
            self.assertEqual(registro["registros_prueba"], 12)
            self.assertIn(registro["mejor_modelo"], registro["modelos"])
            for resultados in registro["modelos"].values():
                ejecucion = cliente_mlflow.get_run(resultados["mlflow_run_id"])
                artefactos = {
                    artefacto.path
                    for artefacto in cliente_mlflow.list_artifacts(
                        ejecucion.info.run_id
                    )
                }
                modelos_registrados = ejecucion.outputs.model_outputs
                self.assertEqual(
                    ejecucion.data.params["model_name"],
                    ejecucion.info.run_name,
                )
                self.assertEqual(ejecucion.data.params["xgboost_device"], "cpu")
                self.assertIn("test_average_precision", ejecucion.data.metrics)
                self.assertIn("cv_average_precision_media", ejecucion.data.metrics)
                self.assertIn("evaluation.json", artefactos)
                self.assertEqual(len(modelos_registrados), 1)
                artefactos_modelo = cliente_mlflow.list_logged_model_artifacts(
                    modelos_registrados[0].model_id
                )
                self.assertIn(
                    "MLmodel",
                    {artefacto.path for artefacto in artefactos_modelo},
                )
                self.assertEqual(
                    set(resultados["prueba"]),
                    {
                        "average_precision",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "balanced_accuracy",
                        "accuracy",
                    },
                )
                self.assertEqual(
                    set(resultados["validacion_cruzada"]),
                    {
                        "average_precision",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                        "balanced_accuracy",
                    },
                )
            self.assertEqual(len(modelo_guardado.classes_), 2)
        self.assertEqual(
            cpu_busquedas["xgboost"][0].get_params()["device"],
            "cpu",
        )


if __name__ == "__main__":
    unittest.main()
