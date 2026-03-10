# 🧪 MLOps Open Source: Hands-on Lab

> **Seminario de Grado II - Especialización en Ciencia de Datos**
> Este portal es una guía interactiva para implementar un stack de MLOps reproducible, conectando la teoría de investigación con la puesta en producción.

---

## 📌 Tabla de Contenidos

1. [🚀 Introducción al Stack](#introduccion)
2. [📅 Semana 1: Reproducible Baseline (MLflow)](#semana-1)
3. [🛠️ Semana 2: Pipelines y Calidad](#semana-2)
4. [⚡ Semana 3: Serving con FastAPI](#semana-3)
5. [🐳 Semana 4: Docker y Despliegue](#semana-4)
6. [🏁 Entregable Final](#entregable-final)

---

<a name="introduccion"></a>
## 🚀 1. Introducción al Stack

Para este laboratorio utilizaremos herramientas que permiten la transición del modelo de Jupyter Notebook a un entorno productivo real. A lo largo del curso, construiremos un ejemplo completo de clasificación de flores Iris usando un modelo de Random Forest, aplicando cada herramienta de manera integrada.

| Herramienta | Función |
|-------------|---------|
| **Git** | Control de versiones de código y colaboración. |
| **MLflow** | Tracking de experimentos y registro de modelos (Model Registry). |
| **DVC** | Versionado de datos y modelos, integración con Git. |
| **pytest** | Framework para escribir y ejecutar pruebas automatizadas en Python. |
| **Great Expectations** | Validación de calidad de datos con expectativas declarativas. |
| **FastAPI** | Framework para servir predicciones mediante una interfaz API. |
| **Docker** | Empaquetamiento en contenedores para asegurar la portabilidad. |

---

<a name="semana-1"></a>
## 📅 2. Semana 1: Reproducible Baseline

El objetivo es registrar cada entrenamiento. Si un experimento no se puede rastrear, no es ciencia de datos profesional. Comenzamos con un modelo baseline de clasificación de flores Iris usando Random Forest.

### Código de ejemplo (train.py)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)
    
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "iris_model")
    
    print(f"Modelo registrado con accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

*Nota: Anota el Run ID mostrado en la consola, lo necesitarás en la Semana 3.*

---

<a name="semana-2"></a>
## 🛠️ 3. Semana 2: Pipelines y Calidad

En esta semana, automatizamos el proceso de la Semana 1 con pipelines y agregamos controles de calidad. Usaremos el mismo dataset de Iris para mantener la coherencia.

### Conceptos Clave

- **Pipelines Automatizados**: Secuencias de pasos que transforman datos crudos en modelos entrenados.
- **Versionado de Datos**: DVC rastrea cambios en datasets.
- **Testing y Validación**: Pruebas para asegurar calidad.

| Herramienta | Función |
|-------------|---------|
| **DVC** | Versionado de datos y modelos. |
| **pytest** | Pruebas automatizadas. |
| **Great Expectations** | Validación de datos. |

### Código de ejemplo: Pipeline con DVC y Testing

#### 1. Instalación de dependencias
Agrega a `requirements.txt`:
```
dvc
pytest
great-expectations
```

#### 2. Pipeline básico (dvc.yaml)
```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - src/prepare.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv
    outs:
      - models/model.pkl
  test:
    cmd: pytest tests/
    deps:
      - src/
      - tests/
      - models/model.pkl
```

#### 3. Script de preparación de datos (src/prepare.py)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Cargar datos de Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Dividir en train/test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Guardar
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)
```

#### 4. Script de entrenamiento (src/train.py)
```python
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)
    
    # Cargar datos procesados
    train = pd.read_csv('data/processed/train.csv')
    X_train = train.drop('species', axis=1)
    y_train = train['species']
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar con test
    test = pd.read_csv('data/processed/test.csv')
    X_test = test.drop('species', axis=1)
    y_test = test['species']
    accuracy = model.score(X_test, y_test)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "iris_model")
    
    # Guardar modelo localmente para pipeline
    joblib.dump(model, 'models/model.pkl')
    
    print(f"Modelo entrenado con accuracy: {accuracy}")
```

#### 5. Pruebas con pytest (tests/test_model.py)
```python
import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def test_data_loading():
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')
    assert not train.empty and not test.empty
    assert 'species' in train.columns

def test_model_exists():
    model = joblib.load('models/model.pkl')
    assert model is not None

def test_model_accuracy():
    model = joblib.load('models/model.pkl')
    test = pd.read_csv('data/processed/test.csv')
    X_test = test.drop('species', axis=1)
    y_test = test['species']
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8
```

#### 6. Validación de datos con Great Expectations
```python
import great_expectations as ge
import pandas as pd

# Cargar datos
df = pd.read_csv('data/processed/train.csv')
df_ge = ge.from_pandas(df)

# Definir expectativas
df_ge.expect_column_to_exist('sepal length (cm)')
df_ge.expect_column_values_to_be_between('sepal length (cm)', 4.0, 8.0)
df_ge.expect_column_values_to_not_be_null('species')
df_ge.expect_column_distinct_values_to_be_in_set('species', [0, 1, 2])

# Validar
results = df_ge.validate()
assert results.success, "Validación de datos fallida"
```

Este pipeline automatiza el flujo completo, asegurando calidad en cada paso.

---

<a name="semana-3"></a>
## ⚡ 4. Semana 3: Serving con FastAPI

Transformamos el modelo de la Semana 1/2 en una API web. Usa el Run ID de MLflow obtenido en la Semana 1.

### Implementación de la API (app.py)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI(title="Iris Classification API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Cargar modelo desde MLflow (reemplaza TU_RUN_ID_AQUI con el ID real)
model_uri = 'runs:/TU_RUN_ID_AQUI/iris_model'
model = mlflow.sklearn.load_model(model_uri)

@app.get("/health")
def health():
    return {"status": "online", "model": "iris_classifier_v1"}

@app.post("/predict")
def predict(data: IrisInput):
    prediction = model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    species = ["setosa", "versicolor", "virginica"][int(prediction[0])]
    return {"prediction": species, "class_id": int(prediction[0])}
```

*Nota: Para obtener el Run ID, revisa la UI de MLflow en http://localhost:5000 o la salida de train.py.*

---

<a name="semana-4"></a>
## 🐳 5. Semana 4: Docker y Despliegue

Empaquetamos la API completa en Docker.

### Archivo de configuración (Dockerfile)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Orquestación (docker-compose.yml)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
  mlflow:
    image: ghcr.io/mlflow/mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow ui --host 0.0.0.0
```

---

<a name="entregable-final"></a>
## 🏁 6. Entregable Final

El repositorio debe contener:

- [ ] `train.py`: Entrenamiento con MLflow.
- [ ] `src/prepare.py`: Preparación de datos.
- [ ] `src/train.py`: Entrenamiento en pipeline.
- [ ] `tests/test_model.py`: Pruebas con pytest.
- [ ] `dvc.yaml`: Configuración del pipeline.
- [ ] `app.py`: API con FastAPI.
- [ ] `Dockerfile` & `docker-compose.yml`: Despliegue.
- [ ] `requirements.txt`: Dependencias completas.
- [ ] `README.md`: Instrucciones.

### Requirements.txt completo
```
mlflow
scikit-learn
fastapi
uvicorn
dvc
pytest
great-expectations
pandas
joblib
```

Este flujo completo crea un sistema MLOps end-to-end para clasificación de Iris.