import argparse
import base64
import json
from pathlib import Path
import shutil
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd


RUTA_RAIZ = Path(__file__).resolve().parent.parent
RUTA_CREDENCIAL = RUTA_RAIZ / "config" / "kaggle.json"
RUTA_ARCHIVO = RUTA_RAIZ / "datos" / "creditcard.csv"
RUTA_SALIDA_DATASET = RUTA_RAIZ / "datos" / "creditcard_clean.csv"
RUTA_SALIDA_RESUMEN = RUTA_RAIZ / "datos" / "creditcard_summary.json"
RUTA_SALIDA_REPORTE = RUTA_RAIZ / "ia_project_quality_report.md"
URL_DESCARGA = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "mlg-ulb/creditcardfraud"
)

COLUMNAS = ["Time", *(f"V{numero}" for numero in range(1, 29)), "Amount", "Class"]


def descargar_dataset(forzar=False):
    """Descarga creditcard.csv usando config/kaggle.json si hace falta."""
    if RUTA_ARCHIVO.is_file() and not forzar:
        return
    if not RUTA_CREDENCIAL.is_file():
        raise FileNotFoundError(
            "No se encontró config/kaggle.json. Descarga tu token desde "
            "Kaggle y guárdalo en esa ruta."
        )

    try:
        credencial = json.loads(RUTA_CREDENCIAL.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "No se pudo leer config/kaggle.json como JSON válido."
        ) from error

    if not isinstance(credencial, dict):
        raise ValueError("config/kaggle.json debe contener un objeto JSON.")
    usuario = credencial.get("username")
    clave = credencial.get("key")
    if (
        not isinstance(usuario, str)
        or not usuario.strip()
        or not isinstance(clave, str)
        or not clave.strip()
    ):
        raise ValueError(
            "config/kaggle.json debe contener los campos username y key."
        )

    token = base64.b64encode(f"{usuario}:{clave}".encode("utf-8")).decode("ascii")
    solicitud = Request(
        URL_DESCARGA,
        headers={
            "Authorization": f"Basic {token}",
            "User-Agent": "design-thinking-data-loader",
        },
    )
    RUTA_ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=RUTA_ARCHIVO.parent) as directorio:
            ruta_zip = Path(directorio) / "creditcardfraud.zip"
            ruta_temporal = Path(directorio) / "creditcard.csv"
            try:
                with urlopen(solicitud, timeout=120) as respuesta:
                    with ruta_zip.open("wb") as archivo:
                        shutil.copyfileobj(respuesta, archivo)
            except HTTPError as error:
                raise RuntimeError(
                    f"Kaggle rechazó la descarga (HTTP {error.code}). "
                    "Verifica la credencial y que tengas acceso al dataset."
                ) from error
            except URLError as error:
                raise RuntimeError(
                    f"No se pudo conectar con Kaggle: {error.reason}"
                ) from error

            try:
                with zipfile.ZipFile(ruta_zip) as archivo_zip:
                    miembros = [
                        nombre
                        for nombre in archivo_zip.namelist()
                        if Path(nombre).name == "creditcard.csv"
                    ]
                    if len(miembros) != 1:
                        raise ValueError(
                            "El archivo descargado no contiene un único "
                            "creditcard.csv."
                        )
                    with archivo_zip.open(miembros[0]) as origen:
                        with ruta_temporal.open("wb") as destino:
                            shutil.copyfileobj(origen, destino)
            except zipfile.BadZipFile as error:
                raise ValueError(
                    "Kaggle no devolvió un archivo ZIP válido."
                ) from error

            ruta_temporal.replace(RUTA_ARCHIVO)
    except OSError as error:
        raise RuntimeError(
            f"No se pudo guardar el dataset en {RUTA_ARCHIVO}."
        ) from error


def cargar_y_validar_dataset(ruta=RUTA_ARCHIVO):
    """Lee el dataset y valida el esquema, las etiquetas y los valores."""
    dataframe = pd.read_csv(ruta)
    columnas_faltantes = [
        columna for columna in COLUMNAS if columna not in dataframe
    ]
    if columnas_faltantes:
        raise ValueError(
            "El dataset no tiene el esquema esperado. Faltan columnas: "
            f"{columnas_faltantes}"
        )
    columnas_extra = [columna for columna in dataframe if columna not in COLUMNAS]
    if columnas_extra:
        raise ValueError(
            f"El dataset contiene columnas inesperadas: {columnas_extra}"
        )

    dataframe = dataframe.loc[:, COLUMNAS].copy()
    for columna in COLUMNAS:
        dataframe[columna] = pd.to_numeric(dataframe[columna], errors="raise")

    if dataframe.empty:
        raise ValueError("El dataset no contiene transacciones.")
    if dataframe.isna().any().any():
        faltantes = dataframe.isna().sum()
        faltantes = faltantes[faltantes > 0].to_dict()
        raise ValueError(f"El dataset contiene valores nulos: {faltantes}")
    if not np.isfinite(dataframe.to_numpy(dtype=float)).all():
        raise ValueError("El dataset contiene valores infinitos o no finitos.")
    if not dataframe["Class"].isin([0, 1]).all():
        raise ValueError("La columna Class solo debe contener etiquetas 0 y 1.")
    if set(dataframe["Class"].unique()) != {0, 1}:
        raise ValueError("El dataset debe incluir ambas clases: 0 y 1.")
    dataframe["Class"] = dataframe["Class"].astype("int8")
    return dataframe


def resumir_dataset(dataframe):
    """Calcula métricas y agregaciones para el reporte y el panel."""
    conteos = dataframe["Class"].value_counts().reindex([0, 1], fill_value=0)
    bloques = (dataframe["Time"] // (8 * 60 * 60)).astype(int)
    bloques_fraude = (
        dataframe.loc[dataframe["Class"] == 1]
        .groupby(bloques, sort=True)
        .size()
    )
    ultimo_bloque = int(bloques.max()) if len(bloques) else -1
    fraude_por_bloque = [
        {
            "intervalo": f"{inicio * 8}–{(inicio + 1) * 8} h",
            "conteo": int(bloques_fraude.get(inicio, 0)),
        }
        for inicio in range(ultimo_bloque + 1)
    ]
    medianas_monto = {
        str(etiqueta): float(
            dataframe.loc[dataframe["Class"] == etiqueta, "Amount"].median()
        )
        for etiqueta in (0, 1)
    }
    return {
        "registros": int(len(dataframe)),
        "columnas": int(len(dataframe.columns)),
        "nulos": int(dataframe.isna().sum().sum()),
        "filas_duplicadas_exactas": int(dataframe.duplicated().sum()),
        "clases": {
            str(etiqueta): int(conteos.loc[etiqueta])
            for etiqueta in (0, 1)
        },
        "porcentaje_fraude": float(conteos.loc[1] / len(dataframe) * 100),
        "mediana_amount_por_clase": medianas_monto,
        "fraudes_por_bloque_8h_relativo": fraude_por_bloque,
    }


def generar_reporte_markdown(resumen):
    conteos = resumen["clases"]
    total = resumen["registros"]
    legitimas = conteos["0"]
    fraudes = conteos["1"]
    mediana = resumen["mediana_amount_por_clase"]
    return "\n".join(
        [
            "# Reporte de calidad de datos",
            "",
            "## Fuente",
            "",
            "- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)",
            "- Descarga autenticada con `config/kaggle.json` mediante la API de Kaggle.",
            f"- Archivo procesado: `{RUTA_ARCHIVO.relative_to(RUTA_RAIZ).as_posix()}`",
            "",
            "## Validación y preparación",
            "",
            f"- Registros: {total:,}",
            f"- Columnas: {resumen['columnas']} (Time, V1–V28, Amount y Class).",
            f"- Valores nulos: {resumen['nulos']:,}.",
            "- Variables convertidas a valores numéricos; `Class` se conserva como etiqueta entera binaria.",
            "- No se imputaron ni eliminaron transacciones.",
            (
                f"- Filas exactamente duplicadas detectadas: "
                f"{resumen['filas_duplicadas_exactas']:,}; se conservaron para "
                "evitar descartar transacciones sin evidencia de que sean errores."
            ),
            "",
            "## Distribución de clases",
            "",
            "| Clase | Significado | Registros | Porcentaje |",
            "|---:|---|---:|---:|",
            f"| 0 | Transacción legítima | {legitimas:,} | {legitimas / total * 100:.4f}% |",
            f"| 1 | Fraude | {fraudes:,} | {fraudes / total * 100:.4f}% |",
            "",
            f"La proporción de fraude es {resumen['porcentaje_fraude']:.4f}%; "
            "el conjunto presenta un fuerte desbalance de clases.",
            "",
            "## Monto mediano por clase",
            "",
            "| Clase | Mediana de Amount |",
            "|---:|---:|",
            f"| 0 | {mediana['0']:,.2f} |",
            f"| 1 | {mediana['1']:,.2f} |",
            "",
            "`Amount` se informa en la unidad publicada por el dataset; no se "
            "asume una moneda.",
            "",
            "## Archivos generados",
            "",
            f"- Dataset limpio: `{RUTA_SALIDA_DATASET.relative_to(RUTA_RAIZ).as_posix()}`",
            f"- Resumen para el panel: `{RUTA_SALIDA_RESUMEN.relative_to(RUTA_RAIZ).as_posix()}`",
            "- Este reporte.",
            "",
        ]
    )


def procesar_dataset(forzar_descarga=False):
    descargar_dataset(forzar=forzar_descarga)
    original = cargar_y_validar_dataset()
    resumen = resumir_dataset(original)

    RUTA_SALIDA_DATASET.parent.mkdir(parents=True, exist_ok=True)
    original.to_csv(RUTA_SALIDA_DATASET, index=False, encoding="utf-8")
    RUTA_SALIDA_RESUMEN.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    RUTA_SALIDA_REPORTE.write_text(
        generar_reporte_markdown(resumen),
        encoding="utf-8",
    )
    return resumen


def main():
    parser = argparse.ArgumentParser(
        description="Descarga, valida y prepara Credit Card Fraud Detection."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Vuelve a descargar el dataset aunque ya exista una copia local.",
    )
    argumentos = parser.parse_args()
    resumen = procesar_dataset(forzar_descarga=argumentos.download)
    print(
        f"Dataset listo: {resumen['registros']:,} transacciones, "
        f"{resumen['clases']['1']:,} fraudes "
        f"({resumen['porcentaje_fraude']:.4f}%)."
    )
    print(f"Archivo limpio: {RUTA_SALIDA_DATASET}")
    print(f"Reporte: {RUTA_SALIDA_REPORTE}")


if __name__ == "__main__":
    main()
