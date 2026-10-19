# Reporte de calidad de datos

Fecha de generación: 2026-10-18 23:04:13

## Estado inicial

- Archivo de entrada: `ia_project_dirty_dataset_6000.csv`
- Registros: 6,000
- Columnas: 19
- Valores nulos: 6,631 (5.82% de las celdas)
- Formatos mixtos en fechas: date_of_birth (5), signup_date (5), last_login (5)

## Transformaciones aplicadas

- Normalización de nombres, ciudades y estados: minúsculas, sin acentos ni caracteres especiales.
- Limpieza de espacios y conversión de emails a minúsculas.
- Homogeneización de países, estados de cliente, planes, idiomas y consentimiento de marketing.
- Conversión de estaturas a centímetros y pesos a kilogramos.
- Conversión de fechas a valores datetime y montos a números decimales, sin conversión cambiaria.
- Deduplicación por nombre normalizado y fecha de nacimiento; se conservó el registro más completo y, en empate, el de actividad más reciente.

## Resultados y mejora de calidad

| Métrica | Inicial | Final | Cambio |
|---|---:|---:|---:|
| Registros | 6,000 | 5,485 | -515 duplicados |
| Valores nulos | 6,631 (5.82%) | 5,905 (5.67%) | 0.15 puntos porcentuales |
| Grupos duplicados | — | 278 | 515 filas eliminadas |

### Consistencia de formatos

| Columna | Estado inicial | Estado final |
|---|---|---|
| `date_of_birth` | 5 representaciones | datetime64[ns] |
| `signup_date` | 5 representaciones | datetime64[ns] |
| `last_login` | 5 representaciones | datetime64[ns] |
| `customer_status` | 8 variantes | 3 valores estandarizados |
| `plan` | 6 variantes | 3 valores estandarizados |
| `preferred_language` | 6 variantes | 2 valores estandarizados |
| `height` | 2 unidades | Una unidad: cm |
| `weight` | 2 unidades | Una unidad: kg |
| `amount_spent` | 31 representaciones | float64 |

### Valores nulos por columna

| Columna | Inicial (cantidad, %) | Final (cantidad, %) |
|---|---:|---:|
| `record_id` | 0 (0.00%) | 0 (0.00%) |
| `full_name` | 0 (0.00%) | 0 (0.00%) |
| `email` | 192 (3.20%) | 165 (3.01%) |
| `phone` | 167 (2.78%) | 146 (2.66%) |
| `date_of_birth` | 0 (0.00%) | 0 (0.00%) |
| `signup_date` | 0 (0.00%) | 0 (0.00%) |
| `last_login` | 1,040 (17.33%) | 836 (15.24%) |
| `country` | 0 (0.00%) | 0 (0.00%) |
| `state_province` | 5,056 (84.27%) | 4,610 (84.05%) |
| `city` | 0 (0.00%) | 0 (0.00%) |
| `postal_code` | 176 (2.93%) | 148 (2.70%) |
| `customer_status` | 0 (0.00%) | 0 (0.00%) |
| `plan` | 0 (0.00%) | 0 (0.00%) |
| `preferred_language` | 0 (0.00%) | 0 (0.00%) |
| `timezone` | 0 (0.00%) | 0 (0.00%) |
| `height` | 0 (0.00%) | 0 (0.00%) |
| `weight` | 0 (0.00%) | 0 (0.00%) |
| `marketing_consent` | 0 (0.00%) | 0 (0.00%) |
| `amount_spent` | 0 (0.00%) | 0 (0.00%) |

## Archivos generados

- Dataset limpio: `datos/ia_project_golden_clean.csv`
- Reporte: `ia_project_quality_report.md`
