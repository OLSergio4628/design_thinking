from datetime import datetime
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


def es_nulo(valor):
    """Indica si un valor escalar es nulo, incluyendo NaN y pd.NA."""
    return bool(pd.isna(valor))


def normalizar_texto(valor):
    """Quita acentos, pasa a minúsculas y elimina caracteres especiales."""
    if es_nulo(valor):
        return None

    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(
        caracter for caracter in texto if not unicodedata.combining(caracter)
    )
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return " ".join(texto.split())


def _convertir_unidad(valor, factor):
    if es_nulo(valor):
        return np.nan
    try:
        return float(valor) * factor
    except (TypeError, ValueError) as error:
        raise ValueError(f"No se puede convertir el valor {valor!r} a número.") from error


def centimetros_a_pulgadas(centimetros):
    return _convertir_unidad(centimetros, 1 / 2.54)


def pulgadas_a_centimetros(pulgadas):
    return _convertir_unidad(pulgadas, 2.54)


def kilogramos_a_libras(kilogramos):
    return _convertir_unidad(kilogramos, 1 / 0.45359237)


def libras_a_kilogramos(libras):
    return _convertir_unidad(libras, 0.45359237)


RUTA_ARCHIVO = (
    Path(__file__).resolve().parent.parent
    / "datos"
    / "ia_project_dirty_dataset_6000.csv"
)

df = pd.read_csv(RUTA_ARCHIVO)

print(f"Total de registros: {len(df)}")
print("\nVista preliminar:")
print(df.head())

metricas_calidad = {}
reporte_markdown = {
    "titulo": "Reporte de calidad de datos",
    "fecha_generacion": datetime.now(),
    "secciones": [],
}