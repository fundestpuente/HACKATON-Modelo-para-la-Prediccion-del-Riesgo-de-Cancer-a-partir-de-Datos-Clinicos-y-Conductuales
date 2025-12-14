import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Configuramos la página
st.set_page_config(
    page_title="Sistema de Diagnóstico IA",
    page_icon="resources/icon.png",
    layout="centered"
)

# Cargamos los pkl
@st.cache_resource
def cargar_archivos():
    try:
        modelo = joblib.load('modelo_cancer_final.pkl')
        scaler = joblib.load('scaler_final.pkl')
        columnas = joblib.load('columnas_finales.pkl')
        return modelo, scaler, columnas
    except FileNotFoundError:
        return None, None, None

modelo, scaler, columnas_finales = cargar_archivos()

st.title("Predicción de Cáncer con IA")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (Random Forest)** optimizado para detectar patrones de riesgo oncológico.
Por favor, ingrese los datos clínicos del paciente a continuación:
""")

if modelo is None:
    st.error(" Error: No se encuentran los archivos .pkl. Por favor, asegúrate de haber ejecutado el script de entrenamiento primero.")
    st.stop()

# Side bar
st.sidebar.header("Datos del Paciente")

# Variables base
gender = st.sidebar.selectbox("Género", options=[0, 1], format_func=lambda x: "Masculino" if x == 0 else "Femenino")
age = st.sidebar.slider("Edad", 20, 90, 45)
bmi = st.sidebar.slider("Índice de Masa Corporal (BMI)", 15.0, 40.0, 24.0)
smoking = st.sidebar.selectbox("¿Hábito de fumar?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
genetic = st.sidebar.selectbox("Riesgo Genético", options=[0, 1, 2], format_func=lambda x: ["Bajo", "Medio", "Alto"][x])
activity = st.sidebar.slider("Actividad Física (Horas/Semana)", 0.0, 20.0, 5.0)
alcohol = st.sidebar.slider("Consumo Alcohol (Unidades/Semana)", 0.0, 20.0, 2.0)
cancer_history = st.sidebar.selectbox("Historial de Cáncer Previo", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")


if st.button("Realizar Diagnóstico", type="primary"):
    
    # Dataframe con los datos
    input_dict = {
        'Gender': [gender],
        'Age': [age],
        'BMI': [bmi],
        'Smoking': [smoking],
        'GeneticRisk': [genetic],
        'PhysicalActivity': [activity],
        'AlcoholIntake': [alcohol],
        'CancerHistory': [cancer_history]
    }
    df_input = pd.DataFrame(input_dict)

    df_input["Age_Smoking"] = df_input["Age"] * df_input["Smoking"]
    df_input["BMI_Activity"] = df_input["BMI"] * df_input["PhysicalActivity"]
    df_input["Genetic_Smoking"] = df_input["GeneticRisk"] * df_input["Smoking"]
    df_input["Alcohol_Smoking"] = df_input["AlcoholIntake"] * df_input["Smoking"]
    
    df_input["Obese"] = (df_input["BMI"] >= 30).astype(int)
    df_input["LowActivity"] = (df_input["PhysicalActivity"] < 2).astype(int)
    df_input["HeavyDrinker"] = (df_input["AlcoholIntake"] > 3).astype(int)
    
    df_input["Log_Alcohol"] = np.log1p(df_input["AlcoholIntake"])
    df_input["Log_Activity"] = np.log1p(df_input["PhysicalActivity"])
    
    df_input["Age_squared"] = df_input["Age"] ** 2
    df_input["BMI_squared"] = df_input["BMI"] ** 2
    df_input["BMI_per_Age"] = df_input["BMI"] / df_input["Age"]


    try:
        cols_scaler = scaler.feature_names_in_
        df_para_escalar = df_input[cols_scaler]
        
        X_scaled_array = scaler.transform(df_para_escalar)
        
        df_scaled_completo = pd.DataFrame(X_scaled_array, columns=cols_scaler)
        
        X_final_model = df_scaled_completo[columnas_finales]
        
        prediccion = modelo.predict(X_final_model.values)[0]
        probabilidad = modelo.predict_proba(X_final_model.values)[0][1]

    except Exception as e:
        st.error(f"Error en el procesamiento de datos: {e}")
        st.stop()

    st.divider()
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if prediccion == 1:
            try:
                st.image("resources/danger.png", width=100)
            except:
                st.warning("⚠️") # En caso de que no cargue la imagen
        else:
            try:
                st.image("resources/check.png", width=100)
            except:
                st.success("✅") # En caso de que no cargue la imagen

    with col_res2:
        st.subheader("Resultado del Análisis:")
        if prediccion == 1:
            st.error(f" POSITIVO (Riesgo Alto) ")
            st.write(f"Probabilidad estimada: **{probabilidad:.2%}**")
            st.warning("Se sugiere revisión clínica detallada.")
        else:
            st.success(f" NEGATIVO (Riesgo Bajo) ")
            st.write(f"Probabilidad estimada: **{probabilidad:.2%}**")
            st.info("Mantener hábitos saludables.")

    # Detalle técnico para más info
    with st.expander("Ver variables internas usadas por el modelo"):
        st.dataframe(X_final_model)