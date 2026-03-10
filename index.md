# 🧪 MLOps Open Source: Hands-on Lab
> **Seminario de Grado II - Especialización en Ciencia de Datos** > Este portal es una guía interactiva para implementar un stack de MLOps reproducible.

---

## 📌 Tabla de Contenidos
1. [🚀 Introducción al Stack](#introducción-al-stack)
2. [📅 Semana 1: Reproducible Baseline (MLflow)](#semana-1)
3. [🛠️ Semana 2: Pipelines y Calidad](#semana-2)
4. [⚡ Semana 3: Serving con FastAPI](#semana-3)
5. [🐳 Semana 4: Docker y Despliegue](#semana-4)
6. [🏁 Entregable Final](#entregable-final)

---

<a name="introducción-al-stack"></a>
## 🚀 1. Introducción al Stack
Para este laboratorio utilizaremos herramientas *Open Source* que permiten la transición del modelo de Jupyter Notebook a un entorno productivo:

| Herramienta | Función |
| :--- | :--- |
| **Git** | Control de versiones de código. |
| **MLflow** | Tracking de experimentos y registro de modelos. |
| **FastAPI** | Creación de interfaces de programación (API). |
| **Docker** | Empaquetamiento y portabilidad. |



---

<a name="semana-1"></a>
## 📅 2. Semana 1: Reproducible Baseline
El objetivo es registrar nuestro primer entrenamiento para que no se pierda en el historial.

### 💻 Ejemplo de Código (train.py)
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Iniciar experimento en MLflow
mlflow.set_experiment("Baseline_Model")

with mlflow.start_run():
    # Configuración de hiperparámetros
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)
    
    # Entrenamiento (Ejemplo simple)
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    # Registro de métrica y modelo
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "iris_model")
    
    print(f"Modelo registrado con accuracy: {accuracy}")
    
---
<a name="semana-3"></a>

⚡ 4. Semana 3: Serving con FastAPI
Ahora convertimos ese archivo .pkl guardado en MLflow en un servicio que responde preguntas.

🛠️ Implementación de la API
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI(title="MLOps API")

# Clase para validar datos de entrada
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Cargar el modelo (Asegúrate de tener el RUN_ID correcto)
logged_model = 'runs:/TU_RUN_ID_AQUI/iris_model'
model = mlflow.sklearn.load_model(logged_model)

@app.get("/health")
def health():
    return {"status": "online"}

@app.post("/predict")
def predict(data: IrisInput):
    prediction = model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {"prediction": int(prediction[0])}

---
<a name="entregable-final"></a>
🏁 6. Entregable Final
Para aprobar el componente técnico de la materia, el repositorio debe contener:

[ ] train.py: Script con logging de MLflow.

[ ] app.py: API funcional.

[ ] Dockerfile: Imagen del servicio.

[ ] README.md: Instrucciones claras.

# Para probar tu API localmente:
uvicorn app:app --reload
