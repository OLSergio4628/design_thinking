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


def eliminar_duplicados(dataframe):
    dataframe = dataframe.copy()
    claves = ["full_name", "date_of_birth"]
    clave_valida = dataframe[claves].notna().all(axis=1)
    registros_identificados = int(
        dataframe.loc[clave_valida].duplicated(claves, keep=False).sum()
    )
    grupos_duplicados = int(
        (
            dataframe.loc[clave_valida]
            .groupby(claves, dropna=False)
            .size()
            .gt(1)
        ).sum()
    )
    filas_antes = len(dataframe)
    dataframe["_completitud"] = dataframe.notna().sum(axis=1)
    dataframe["_orden_original"] = np.arange(filas_antes)
    con_clave = dataframe.loc[clave_valida].sort_values(
        ["_completitud", "last_login", "signup_date", "_orden_original"],
        ascending=[False, False, False, True],
        kind="mergesort",
        na_position="last",
    )
    sin_clave = dataframe.loc[~clave_valida]
    dataframe = pd.concat(
        [con_clave.drop_duplicates(claves, keep="first"), sin_clave]
    ).sort_values("_orden_original", kind="mergesort")
    eliminados = filas_antes - len(dataframe)
    dataframe = dataframe.drop(columns=["_completitud", "_orden_original"])
    metricas = {
        "registros_en_grupos_duplicados": registros_identificados,
        "grupos_duplicados": int(grupos_duplicados),
        "duplicados_eliminados": int(eliminados),
        "registros_finales": int(len(dataframe)),
    }
    return dataframe, metricas


def _resumen_nulos(dataframe):
    nulos_por_columna = dataframe.isna().sum()
    total_celdas = dataframe.size
    total_nulos = int(nulos_por_columna.sum())
    return {
        "cantidad": total_nulos,
        "porcentaje": (
            total_nulos / total_celdas * 100 if total_celdas else 0.0
        ),
        "por_columna": {
            columna: {
                "cantidad": int(nulos_por_columna[columna]),
                "porcentaje": (
                    float(nulos_por_columna[columna] / len(dataframe) * 100)
                    if len(dataframe)
                    else 0.0
                ),
            }
            for columna in dataframe.columns
        },
    }


def _firma_formato(valor):
    texto = str(valor)
    texto = re.sub(r"\d", "9", texto)
    texto = re.sub(r"[A-Za-z]+", "A", texto)
    return texto


def _metricas_consistencia_formatos(original, procesado):
    consistencia = {}
    columnas_fecha = ("date_of_birth", "signup_date", "last_login")
    columnas_categoria = (
        "country",
        "customer_status",
        "plan",
        "preferred_language",
        "marketing_consent",
    )

    for columna in columnas_fecha:
        variantes = int(
            original[columna].dropna().map(_firma_formato).nunique()
        )
        serie_final = procesado[columna]
        consistencia[columna] = {
            "representaciones_distintas_antes": variantes,
            "formatos_finales": 1 if serie_final.notna().any() else 0,
            "tipo_final": str(serie_final.dtype),
        }

    for columna in columnas_categoria:
        consistencia[columna] = {
            "representaciones_distintas_antes": int(
                original[columna].nunique(dropna=True)
            ),
            "valores_estandarizados_finales": int(
                procesado[columna].nunique(dropna=True)
            ),
            "tipo_final": str(procesado[columna].dtype),
        }

    for columna, unidad_final in (("height", "cm"), ("weight", "kg")):
        unidades = (
            original[columna]
            .dropna()
            .astype(str)
            .str.extract(r"([a-zA-Z]+)\s*$", expand=False)
            .str.lower()
        )
        consistencia[columna] = {
            "unidades_distintas_antes": int(unidades.nunique()),
            "unidad_final": unidad_final,
            "tipo_final": str(procesado[columna].dtype),
        }

    consistencia["amount_spent"] = {
        "representaciones_distintas_antes": int(
            original["amount_spent"]
            .dropna()
            .map(_firma_formato)
            .nunique()
        ),
        "tipo_final": str(procesado["amount_spent"].dtype),
    }
    return consistencia


def calcular_metricas_calidad(original, procesado, metricas_duplicados):
    nulos_antes = _resumen_nulos(original)
    nulos_despues = _resumen_nulos(procesado)
    return {
        "registros_antes": int(len(original)),
        "registros_despues": int(len(procesado)),
        "valores_nulos": {
            "antes": nulos_antes,
            "despues": nulos_despues,
        },
        **metricas_duplicados,
        "consistencia_formatos": _metricas_consistencia_formatos(
            original, procesado
        ),
    }


def generar_reporte_markdown(metricas):
    nulos_antes = metricas["valores_nulos"]["antes"]
    nulos_despues = metricas["valores_nulos"]["despues"]
    mejora_nulos = nulos_antes["porcentaje"] - nulos_despues["porcentaje"]

    lineas = [
        "# Reporte de calidad de datos",
        "",
        f"Fecha de generación: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Estado inicial",
        "",
        f"- Archivo de entrada: `{RUTA_ARCHIVO.name}`",
        f"- Registros: {metricas['registros_antes']:,}",
        f"- Columnas: {len(df_original.columns)}",
        (
            f"- Valores nulos: {nulos_antes['cantidad']:,} "
            f"({nulos_antes['porcentaje']:.2f}% de las celdas)"
        ),
        "- Formatos mixtos en fechas: "
        + ", ".join(
            f"{nombre} ({datos['representaciones_distintas_antes']})"
            for nombre, datos in metricas["consistencia_formatos"].items()
            if nombre in ("date_of_birth", "signup_date", "last_login")
        ),
        "",
        "## Transformaciones aplicadas",
        "",
        "- Normalización de nombres, ciudades y estados: minúsculas, sin acentos ni caracteres especiales.",
        "- Limpieza de espacios y conversión de emails a minúsculas.",
        "- Homogeneización de países, estados de cliente, planes, idiomas y consentimiento de marketing.",
        "- Conversión de estaturas a centímetros y pesos a kilogramos.",
        "- Conversión de fechas a valores datetime y montos a números decimales, sin conversión cambiaria.",
        (
            "- Deduplicación por nombre normalizado y fecha de nacimiento; "
            "se conservó el registro más completo y, en empate, el de actividad más reciente."
        ),
        "",
        "## Resultados y mejora de calidad",
        "",
        "| Métrica | Inicial | Final | Cambio |",
        "|---|---:|---:|---:|",
        (
            f"| Registros | {metricas['registros_antes']:,} | "
            f"{metricas['registros_despues']:,} | "
            f"-{metricas['duplicados_eliminados']:,} duplicados |"
        ),
        (
            f"| Valores nulos | {nulos_antes['cantidad']:,} "
            f"({nulos_antes['porcentaje']:.2f}%) | "
            f"{nulos_despues['cantidad']:,} "
            f"({nulos_despues['porcentaje']:.2f}%) | "
            f"{mejora_nulos:.2f} puntos porcentuales |"
        ),
        (
            f"| Grupos duplicados | — | {metricas['grupos_duplicados']:,} | "
            f"{metricas['duplicados_eliminados']:,} filas eliminadas |"
        ),
        "",
        "### Consistencia de formatos",
        "",
        "| Columna | Estado inicial | Estado final |",
        "|---|---|---|",
    ]

    formatos = metricas["consistencia_formatos"]
    for columna in ("date_of_birth", "signup_date", "last_login"):
        datos = formatos[columna]
        lineas.append(
            f"| `{columna}` | {datos['representaciones_distintas_antes']} "
            f"representaciones | {datos['tipo_final']} |"
        )
    for columna in ("customer_status", "plan", "preferred_language"):
        datos = formatos[columna]
        lineas.append(
            f"| `{columna}` | {datos['representaciones_distintas_antes']} "
            f"variantes | {datos['valores_estandarizados_finales']} valores estandarizados |"
        )
    for columna in ("height", "weight"):
        datos = formatos[columna]
        lineas.append(
            f"| `{columna}` | {datos['unidades_distintas_antes']} unidades | "
            f"Una unidad: {datos['unidad_final']} |"
        )
    datos_monto = formatos["amount_spent"]
    lineas.append(
        f"| `amount_spent` | {datos_monto['representaciones_distintas_antes']} "
        f"representaciones | {datos_monto['tipo_final']} |"
    )

    lineas.extend(
        [
            "",
            "### Valores nulos por columna",
            "",
            "| Columna | Inicial (cantidad, %) | Final (cantidad, %) |",
            "|---|---:|---:|",
        ]
    )
    for columna in df_original.columns:
        antes = nulos_antes["por_columna"][columna]
        despues = nulos_despues["por_columna"][columna]
        lineas.append(
            f"| `{columna}` | {antes['cantidad']:,} "
            f"({antes['porcentaje']:.2f}%) | {despues['cantidad']:,} "
            f"({despues['porcentaje']:.2f}%) |"
        )

    lineas.extend(
        [
            "",
            "## Archivos generados",
            "",
            f"- Dataset limpio: `{RUTA_SALIDA_DATASET.relative_to(RUTA_RAIZ).as_posix()}`",
            f"- Reporte: `{RUTA_SALIDA_REPORTE.relative_to(RUTA_RAIZ).as_posix()}`",
            "",
        ]
    )
    return "\n".join(lineas)


RUTA_ARCHIVO = (
    Path(__file__).resolve().parent.parent
    / "datos"
    / "ia_project_dirty_dataset_6000.csv"
)
RUTA_RAIZ = Path(__file__).resolve().parent.parent
RUTA_SALIDA_DATASET = (
    RUTA_RAIZ / "datos" / "ia_project_golden_clean.csv"
)
RUTA_SALIDA_REPORTE = RUTA_RAIZ / "ia_project_quality_report.md"

df_original = pd.read_csv(RUTA_ARCHIVO)
df = limpiar_dataframe(df_original)
metricas_calidad = {}
df, metricas_duplicados = eliminar_duplicados(df)
metricas_calidad = calcular_metricas_calidad(
    df_original, df, metricas_duplicados
)

print(
    f"Registros: {metricas_calidad['registros_antes']} antes / "
    f"{metricas_calidad['registros_despues']} después"
)
print(
    f"Nulos: {metricas_calidad['valores_nulos']['antes']['porcentaje']:.2f}% "
    f"antes / {metricas_calidad['valores_nulos']['despues']['porcentaje']:.2f}% después"
)
print(
    f"Duplicados eliminados: "
    f"{metricas_calidad['duplicados_eliminados']}"
)
print("\nVista preliminar:")
print(df.head())

reporte_markdown = {
    "titulo": "Reporte de calidad de datos",
    "fecha_generacion": datetime.now(),
    "secciones": [
        {
            "titulo": "Métricas de calidad",
            "criterio_deduplicacion": (
                "Nombre normalizado y fecha de nacimiento"
            ),
            **metricas_calidad,
        }
    ],
}

df.to_csv(
    RUTA_SALIDA_DATASET,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
RUTA_SALIDA_REPORTE.write_text(
    generar_reporte_markdown(metricas_calidad),
    encoding="utf-8",
)
print(f"\nDataset limpio guardado en: {RUTA_SALIDA_DATASET}")
print(f"Reporte guardado en: {RUTA_SALIDA_REPORTE}")