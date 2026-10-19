# Reporte de calidad de datos

## Fuente

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Descarga autenticada con `config/kaggle.json` mediante la API de Kaggle.
- Archivo procesado: `datos/creditcard.csv`

## Validación y preparación

- Registros: 284,807
- Columnas: 31 (Time, V1–V28, Amount y Class).
- Valores nulos: 0.
- Variables convertidas a valores numéricos; `Class` se conserva como etiqueta entera binaria.
- No se imputaron ni eliminaron transacciones.
- Filas exactamente duplicadas detectadas: 1,081; se conservaron para evitar descartar transacciones sin evidencia de que sean errores.

## Distribución de clases

| Clase | Significado | Registros | Porcentaje |
|---:|---|---:|---:|
| 0 | Transacción legítima | 284,315 | 99.8273% |
| 1 | Fraude | 492 | 0.1727% |

La proporción de fraude es 0.1727%; el conjunto presenta un fuerte desbalance de clases.

## Monto mediano por clase

| Clase | Mediana de Amount |
|---:|---:|
| 0 | 22.00 |
| 1 | 9.25 |

`Amount` se informa en la unidad publicada por el dataset; no se asume una moneda.

## Archivos generados

- Dataset limpio: `datos/creditcard_clean.csv`
- Resumen para el panel: `datos/creditcard_summary.json`
- Este reporte.
