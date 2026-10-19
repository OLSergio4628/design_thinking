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

## Entrenar y comparar modelos

Instala las dependencias del proyecto y ejecuta el entrenamiento sobre el CSV limpio:

```powershell
python -m pip install -r requirements.txt
python fuentes\trainModels.py
```

El entrenamiento compara regresión logística (con estandarización dentro de un `Pipeline`), *random forest* y XGBoost. Para cada algoritmo ajusta hiperparámetros mediante *grid search* con validación cruzada estratificada de cinco particiones. XGBoost usa `tree_method="hist"` y permite seleccionar CPU o GPU NVIDIA. Selecciona el mejor por *average precision* —más informativa que la exactitud en un conjunto tan desbalanceado— y registra también precisión, *recall*, F1, ROC-AUC, exactitud balanceada y exactitud.

Para solicitar GPU en XGBoost, ejecuta:

```powershell
python fuentes\trainModels.py --device cuda
```

Se requiere una GPU NVIDIA visible para el proceso, controladores compatibles y una distribución de XGBoost con soporte CUDA. El dispositivo predeterminado es CPU para que el entrenamiento siga funcionando en equipos sin GPU; `--device cuda` selecciona la GPU explícitamente y no recurre silenciosamente a CPU si no está disponible. La regresión logística y el *random forest* de scikit-learn siguen ejecutándose en CPU.

El 20% de los datos se reserva como prueba independiente. Los resultados de validación y prueba se agregan a `datos/model_registry.json`; el estimador con mejor *average precision* de validación se guarda en `datos/best_fraud_model.joblib`. Ambos son artefactos locales excluidos de Git.

Cada modelo ajustado crea una ejecución en el experimento `credit-card-fraud` de MLflow. Registra los hiperparámetros y la configuración de validación cruzada, métricas de validación y prueba, el modelo serializado y `evaluation.json`. Por defecto, el tracking de MLflow usa la base SQLite local `mlflow.db` y guarda los artefactos en `mlartifacts/`. Los modelos se serializan con CloudPickle; carga únicamente modelos procedentes de fuentes confiables. Para consultarlos en la interfaz web:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre `http://localhost:5000`. Para enviar los registros a otro servidor, define `MLFLOW_TRACKING_URI` o usa `--tracking-uri`, por ejemplo `--tracking-uri http://localhost:5000`. Se puede elegir otro CSV validado o cantidad de particiones con `--data` y `--folds`.

## Visualizar resultados

Después de preparar los datos, desde la raíz del proyecto inicia un servidor local:

```powershell
python -m http.server 8000
```

Abre `http://localhost:8000/datos/` para consultar la distribución de clases, los fraudes por intervalos relativos de ocho horas y el monto mediano por clase.
