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


def _normalizar_medida(valor, unidad_base, unidad_alternativa, convertir):
    if es_nulo(valor):
        return np.nan
    if isinstance(valor, (int, float, np.number)):
        return float(valor)

    coincidencia = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*", str(valor)
    )
    if coincidencia is None:
        raise ValueError(f"Formato de medida no reconocido: {valor!r}.")

    cantidad = float(coincidencia.group(1))
    unidad = coincidencia.group(2).lower()
    if unidad == unidad_base:
        return cantidad
    if unidad == unidad_alternativa:
        return convertir(cantidad)
    raise ValueError(f"Unidad no reconocida en la medida {valor!r}.")


def _parsear_fecha(valor, formatos, columna):
    if es_nulo(valor):
        return pd.NaT

    texto = str(valor).strip()
    for formato in formatos:
        try:
            return datetime.strptime(texto, formato)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha no reconocido en {columna}: {valor!r}.")


def _normalizar_monto(valor):
    if es_nulo(valor):
        return np.nan

    texto = re.sub(r"(?i)\b(?:MXN|USD)\b", "", str(valor))
    texto = texto.replace("$", "").replace(",", "").strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", texto):
        raise ValueError(f"Formato de monto no reconocido: {valor!r}.")
    return float(texto)


def _mapear_categoria(serie, equivalencias, columna):
    valores = serie.map(normalizar_texto)
    resultado = valores.map(equivalencias)
    no_reconocidos = valores[resultado.isna() & valores.notna()].unique()
    if len(no_reconocidos):
        raise ValueError(
            f"Valores no reconocidos en {columna}: {no_reconocidos.tolist()}"
        )
    return resultado


def limpiar_dataframe(dataframe):
    dataframe = dataframe.copy()

    for columna in ("full_name", "state_province", "city"):
        dataframe[columna] = dataframe[columna].map(normalizar_texto)

    dataframe["email"] = dataframe["email"].map(
        lambda valor: valor.strip().lower() if isinstance(valor, str) else valor
    )
    dataframe["phone"] = dataframe["phone"].map(
        lambda valor: valor.strip() if isinstance(valor, str) else valor
    )

    dataframe["country"] = _mapear_categoria(
        dataframe["country"],
        {
            "mexico": "México",
            "estados unidos": "Estados Unidos",
            "canada": "Canadá",
            "brasil": "Brasil",
            "espana": "España",
            "argentina": "Argentina",
        },
        "country",
    )
    dataframe["customer_status"] = _mapear_categoria(
        dataframe["customer_status"],
        {
            "activo": "Activo",
            "active": "Activo",
            "inactivo": "Inactivo",
            "inactive": "Inactivo",
            "prueba": "Prueba",
            "en prueba": "Prueba",
            "trial": "Prueba",
        },
        "customer_status",
    )
    dataframe["plan"] = _mapear_categoria(
        dataframe["plan"],
        {
            "basico": "Básico",
            "basic": "Básico",
            "pro": "Pro",
            "empresarial": "Empresarial",
            "enterprise": "Empresarial",
        },
        "plan",
    )
    dataframe["preferred_language"] = _mapear_categoria(
        dataframe["preferred_language"],
        {"es": "es", "espanol": "es", "en": "en", "ingles": "en"},
        "preferred_language",
    )
    dataframe["marketing_consent"] = _mapear_categoria(
        dataframe["marketing_consent"],
        {
            "si": True,
            "s": True,
            "yes": True,
            "y": True,
            "true": True,
            "1": True,
            "no": False,
            "n": False,
            "false": False,
            "0": False,
        },
        "marketing_consent",
    ).astype("boolean")
    dataframe["timezone"] = dataframe["timezone"].map(
        lambda valor: valor.strip().upper() if isinstance(valor, str) else valor
    )
    dataframe["postal_code"] = dataframe["postal_code"].astype("string").str.strip()

    dataframe["height"] = dataframe["height"].map(
        lambda valor: _normalizar_medida(
            valor, "cm", "in", pulgadas_a_centimetros
        )
    )
    dataframe["weight"] = dataframe["weight"].map(
        lambda valor: _normalizar_medida(
            valor, "kg", "lb", libras_a_kilogramos
        )
    )

    formatos_fecha = {
        "date_of_birth": (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%d-%b-%Y",
            "%d.%m.%Y",
        ),
        "signup_date": (
            "%m/%d/%y %I:%M %p",
            "%d.%m.%Y",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
            "%d-%m-%Y %H:%M:%S",
            "%m-%d-%Y %H:%M:%S",
        ),
        "last_login": (
            "%d.%m.%Y",
            "%Y-%m-%d",
            "%d-%m-%Y %H:%M:%S",
            "%m/%d/%y %I:%M %p",
            "%Y/%m/%d %H:%M",
        ),
    }
    for columna, formatos in formatos_fecha.items():
        dataframe[columna] = pd.to_datetime(
            dataframe[columna].map(
                lambda valor: _parsear_fecha(valor, formatos, columna)
            )
        )

    dataframe["amount_spent"] = dataframe["amount_spent"].map(_normalizar_monto)
    return dataframe


RUTA_ARCHIVO = (
    Path(__file__).resolve().parent.parent
    / "datos"
    / "ia_project_dirty_dataset_6000.csv"
)

df = pd.read_csv(RUTA_ARCHIVO)
df = limpiar_dataframe(df)

print(f"Total de registros: {len(df)}")
print("\nVista preliminar:")
print(df.head())

metricas_calidad = {}
reporte_markdown = {
    "titulo": "Reporte de calidad de datos",
    "fecha_generacion": datetime.now(),
    "secciones": [],
}