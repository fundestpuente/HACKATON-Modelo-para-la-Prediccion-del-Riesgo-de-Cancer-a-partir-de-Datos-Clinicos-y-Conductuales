# 🧬 **Modelo de Machine Learning para la Predicción del Riesgo de Cáncer a partir de Datos Clínicos y Conductuales**

## 👥 Integrantes del Grupo
- Sergio Rodríguez 
- Arianna Feijoo
- Daniel Suárez 
- Edison Soto
- Anthony Baes 

## 📌 Descripción del proyecto
Este proyecto desarrolla un modelo de Machine Learning capaz de predecir el riesgo de cáncer utilizando variables clínicas y conductuales. Incluye análisis exploratorio, preprocesamiento, entrenamiento de modelos, comparación entre algoritmos y construcción de un pipeline para predicciones automatizadas.

## ❗ Problemática
El cáncer continúa siendo una de las principales causas de mortalidad a nivel mundial. Su detección temprana es crítica, pero en muchos casos los síntomas son silenciosos o inespecíficos, lo que dificulta una identificación oportuna.
Aunque se conocen factores de riesgo como el tabaquismo, altos niveles de IMC, sedentarismo o antecedentes familiares, su relación combinada es compleja y requiere herramientas avanzadas para ser analizada.

Los sistemas de salud carecen de soluciones automatizadas capaces de evaluar el riesgo de forma rápida y escalable. Este proyecto propone resolver esta necesidad mediante un modelo de aprendizaje automático entrenado específicamente para reconocer patrones asociados al desarrollo de cáncer.

## 🎯 Objetivos del Proyecto
### Objetivo general

Desarrollar un modelo de machine learning capaz de predecir el riesgo de cáncer a partir de variables clínicas y conductuales, con el fin de contribuir a la detección temprana y apoyar la toma de decisiones en el ámbito de la salud preventiva.

### Objetivos específicos

- Integrar, limpiar y estandarizar las variables del dataset.

- Realizar análisis exploratorio para comprender patrones y correlaciones entre factores de riesgo.

- Entrenar y comparar modelos supervisados de clasificación binaria para predecir la variable Diagnosis.

- Evaluar los modelos mediante métricas como accuracy, recall, precision, F1-score y AUC-ROC.

- Analizar la importancia e interpretabilidad de las variables más influyentes.

- Diseñar un pipeline automatizado que permita predecir riesgo en nuevos pacientes en tiempo real.

## 🎯 Público Objetivo
- Profesionales de la salud: médicos generales, oncólogos, nutricionistas.

- Instituciones médicas y de salud pública.

- Investigadores biomédicos y científicos de datos.

- Desarrolladores de aplicaciones médicas y plataformas de bienestar.

- Personas interesadas en conocer y monitorear su riesgo personal mediante modelos predictivos.

## ⚙️ Instrucciones de Instalación y Ejecución


1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/fundestpuente/SIC-Modelo-para-la-Prediccion-del-Riesgo-de-Cancer-a-partir-de-Datos-Clinicos-y-Conductuales.git
   cd "SIC-Modelo-para-la-Prediccion-del-Riesgo-de-Cancer-a-partir-de-Datos-Clinicos-y-Conductuales"
   ```

2. **Actualizar pip e instalar dependencias**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**
   ```bash
    streamlit run src/app.py
    ```

## 📂 Estructura del Código

- `data/`: Carpeta para almacenar los datasets.
- `notebooks/`: Jupyter notebooks para análisis exploratorio.
- `src/`: Código fuente del proyecto.
  - `preprocessing.ipynb`: Funciones de preprocesamiento de datos y entrenamiento del modelo.
  - `interpretation.ipynb`: Notebook para análisis de interpretabilidad del modelo.
  - `app.py`: Interfaz de usuario con Streamlit.


## ✅ Herramientas Implementadas

- **Python 3.9+**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Streamlit**
