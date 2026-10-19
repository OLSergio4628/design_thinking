import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuentes"))
from loadData import COLUMNAS, cargar_y_validar_dataset, resumir_dataset


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


if __name__ == "__main__":
    unittest.main()
