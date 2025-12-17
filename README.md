# 🧬 **Modelo de Machine Learning para la Predicción del Riesgo de Cáncer a partir de Datos Clínicos y Conductuales**

## 👥 Integrantes del Grupo
- Sergio Rodríguez 
- Arianna Feijoo
- Daniel Suárez 
- Edison Soto
- Anthony Baes 

## 📌 Descripción del proyecto
Este proyecto desarrolla un sistema inteligente capaz de predecir el riesgo de cáncer utilizando variables clínicas y conductuales. A diferencia de enfoques genéricos, esta solución implementa modelos especializados por tipo de órgano y un asistente conversacional basado en inteligencia artificial generativa para la interpretación de resultados.

## ❗ Problemática
El cáncer es una de las principales causas de mortalidad mundial. Su detección temprana es crítica, pero la relación entre factores como el tabaquismo, IMC, sedentarismo y antecedentes familiares es compleja. Los sistemas de salud necesitan herramientas automatizadas, escalables y explicables que ayuden a identificar patrones de riesgo antes de que los síntomas sean críticos.

## 🎯 Objetivos del Proyecto
### Objetivo general

Desarrollar un modelo de machine learning capaz de predecir el riesgo de cáncer a partir de variables clínicas y conductuales, con el fin de contribuir a la detección temprana y apoyar la toma de decisiones en el ámbito de la salud preventiva.

### Objetivos específicos

- Entrenar modelos de clasificación específicos para distintos tipos de cáncer (Cervical, Gástrico, Mama, Próstata, Pulmón).
- Integrar un Chatbot de IA Generativa para mejorar la comunicación médico-paciente.
- Evaluar los modelos mediante métricas de precisión (Accuracy, Recall, AUC-ROC).
- Diseñar una interfaz interactiva en Streamlit para facilitar el uso del sistema.

## 🚀 Funcionalidades Destacadas
- Modelos por Órgano: Implementación de clasificadores independientes optimizados para las características únicas de cada patología.
- Chatbot Google AI (Gemini): Asistente virtual que explica los resultados obtenidos, resuelve dudas sobre factores de riesgo y ofrece recomendaciones preventivas personalizadas.

## 🎯 Público Objetivo
- Profesionales de la salud: médicos generales, oncólogos, nutricionistas.
- Instituciones médicas y de salud pública.
- Investigadores biomédicos y científicos de datos.
- Desarrolladores de aplicaciones médicas y plataformas de bienestar.
- Personas interesadas en conocer y monitorear su riesgo personal mediante modelos predictivos.

## ⚙️ Instrucciones de Instalación y Ejecución

### Configuración de Google AI
1. Obtén tu clave en [Google AI Studio](https://aistudio.google.com/).
2. Crea un archivo `.env` en la raíz del proyecto.
3. Agrega la siguiente línea a tu archivo `.env`:
```bash
API_KEY=tu_clave_aqui
```

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/fundestpuente/SIC-Modelo-para-la-Prediccion-del-Riesgo-de-Cancer-a-partir-de-Datos-Clinicos-y-Conductuales.git
   cd "https://github.com/fundestpuente/HACKATON-Modelo-para-la-Prediccion-del-Riesgo-de-Cancer-a-partir-de-Datos-Clinicos-y-Conductuales.git"
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

## 📂 Estructura del Proyecto

```text
.
├── 📂 data/                    # Datasets y archivos de datos crudos
├── 📂 notebooks/               # Jupyter Notebooks de experimentación
│   ├── 📂 pkl/                 # Modelos serializados por órgano
│   │   ├── 📄 columnas_*.pkl   # Variables por categoría
│   │   └── 📄 modelo_*_final.pkl
│   ├── 📓 01_EDA.ipynb         # Análisis Exploratorio de Datos
│   └── 📓 testNewData.ipynb    # Pruebas de nuevos datos y modelado
├── 📂 src/                     # Código fuente de la aplicación
│   ├── 📂 resources/           # Imágenes del sistema
│   ├── 🐍 app.py               # Aplicación principal (Streamlit)
│   ├── 📄 scaler_final.pkl
│   ├── 📄 modelo_cancer_final.pkl
│   ├── 🖼️ grafico_interpretacion_shap.png
│   └── 📓 preprocessing.ipynb  # Preparación de datos y modelado
├── ⚙️ .env                     # Variables de entorno
├── 🚫 .gitignore               # Archivos excluidos de Git
├── 📖 README.md                # Documentación del proyecto
└── 📋 requirements.txt         # Librerías y dependencias
```

## ✅ Herramientas Implementadas

- **Lenguaje**: Python 3.9+
- **IA Generativa**: Google AI SDK (Gemini Pro)
- **ML Frameworks**: Scikit-learn, Imbalanced-learn (SMOTE)
- **Análisis**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn, SHAP
- **Despliegue**: Streamlit
