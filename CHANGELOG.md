# Changelo
## V2.1.0 2026-10-19
- Incorporación del reporte técnico de detección de fraude con dataset, preparación, modelos, métricas, MLflow y deuda técnica

## V1.1.0 2026-10-18
- Sustitución de la fuente de datos por `mlg-ulb/creditcardfraud` de Kaggle
- Descarga autenticada desde `config/kaggle.json`, validación y preparación del CSV
- Reporte y visualización adaptados a la detección de fraude y al desbalance de clases
- Exclusión de credenciales y datos descargados del control de versiones

## V1.0.0 2026-10-18 21:30
- Inicialización de scaffolding
- Carga inicial de datos
- Generación de directivas

## V1.0.1 2026-10-18 21:50
- Carga de librerías, lectura del dataset y métricas iniciales

## V1.0.2 2026-10-18 22:10
- Funciones de limpieza y normalización

## V1.0.3 2026-10-18 22:47
- Aplicación de limpieza y normalización al DataFrame
- Conversión de fechas, unidades, montos y categorías a formatos homogéneos

## V1.0.4 2026-10-18 22:51
- Identificación y eliminación de registros duplicados
- Registro de métricas de deduplicación para el reporte

## V1.0.5 2026-10-18 22:57
- Cálculo de registros, valores nulos y duplicados antes y después del proceso
- Medición de la estandarización de fechas, categorías, unidades y montos

## V1.0.6 2026-10-18 23:02
- Exportación del dataset limpio a CSV
- Generación del reporte Markdown con estado inicial, transformaciones y métricas

## V1.0.7 2026-10-18 23:36
- Presentación del reporte Markdown con estado inicial, transformaciones y métricas

## V2.0.0 2026-10-18 23:49
- Actualización a version 2.0.0 caso de uso 2

## V2.1.0 2026-10-18 23:49
- Integración de mlflow