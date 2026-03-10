🚀 MLOps Open Source: Guía Práctica de Implementación

Este portal documenta el stack técnico de MLOps para el Seminario de Grado II.

📦 El Stack Mínimo

Git: Control de versiones para código y modelos.

MLflow: Plataforma para el ciclo de vida de ML (Tracking, Registry).

FastAPI: Framework web moderno y rápido para servir predicciones.

Docker: Contenedores para asegurar que "funcione en mi máquina y en el servidor".

🗓️ Plan de 4 Semanas

Semana 1: Baseline Reproducible

Herramienta clave: MLflow

Concepto: Todo experimento debe ser rastreable. Si no se puede repetir, no es ciencia.

Paso a paso: 1. Configurar un entorno virtual: python -m venv venv.
2. Instalar dependencias: pip install mlflow scikit-learn.
3. Usar mlflow.log_params() y mlflow.log_metrics().

Semana 2: Pipeline y Calidad

Herramienta clave: Automation (Makefiles)

Concepto: Automatizar tareas repetitivas y validar la integridad de los datos de entrada.

Paso a paso:

Crear un archivo Makefile para estandarizar comandos.

Implementar validaciones de tipos (ej. verificar que la edad no sea negativa).

Semana 3: Serving con FastAPI

Herramienta clave: FastAPI + Pydantic

Concepto: Convertir un modelo (.pkl o .onnx) en un servicio web accesible vía HTTP.

Paso a paso:

Definir esquemas de datos con Pydantic.

Crear el endpoint /predict que cargue el modelo desde MLflow.

Semana 4: Dockerización

Herramienta clave: Docker & Docker Compose

Concepto: Empaquetar la aplicación con todas sus librerías para un despliegue sin fricciones.

Paso a paso:

Escribir el Dockerfile basado en python:3.9-slim.

Configurar docker-compose.yml para orquestar la API y la UI de MLflow.