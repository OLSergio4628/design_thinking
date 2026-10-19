# Detección de fraude con tarjetas de crédito

Proyecto de clasificación binaria con el dataset público [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) de Kaggle. El conjunto contiene transacciones legítimas y fraudulentas, con una fuerte desproporción entre ambas clases.

## Preparar los datos

1. Obtén una credencial de la API de Kaggle desde la configuración de tu cuenta.
2. Guarda el archivo `kaggle.json` en `config/kaggle.json`. No lo agregues al control de versiones; está excluido por `.gitignore`.
3. Desde la raíz del repositorio ejecuta:

   ```powershell
   python fuentes\loadData.py
   ```

El proceso descarga el dataset solo si todavía no existe una copia local, valida el esquema y los valores, y genera:

- `datos/creditcard.csv`: copia de origen descargada.
- `datos/creditcard_clean.csv`: datos numéricos validados, sin descartar transacciones.
- `datos/creditcard_summary.json`: agregaciones para el panel.
- `ia_project_quality_report.md`: métricas de calidad y distribución de clases.

Para reemplazar la copia local y descargar de nuevo:

```powershell
python fuentes\loadData.py --download
```

Los archivos descargados y generados están excluidos de Git por su tamaño. La preparación conserva las filas duplicadas detectadas y las clases originales; las transacciones no se deduplican ni se imputan, ya que eso podría eliminar casos de fraude válidos.

Los archivos `ia_project_*` que ya existían en `datos/` pertenecen al ejercicio anterior de limpieza y no son utilizados por este flujo.

## Visualizar resultados

Después de preparar los datos, desde la raíz del proyecto inicia un servidor local:

```powershell
python -m http.server 8000
```

Abre `http://localhost:8000/datos/` para consultar la distribución de clases, los fraudes por intervalos relativos de ocho horas y el monto mediano por clase.
