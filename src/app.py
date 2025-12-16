import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sistema de Diagnóstico IA", layout="centered")

@st.cache_resource
def cargar_archivos_generales():
    try:
        modelo = joblib.load('modelo_cancer_final.pkl')
        scaler = joblib.load('scaler_final.pkl')
        columnas = joblib.load('columnas_finales.pkl')
        return modelo, scaler, columnas
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def cargar_especialistas():
    modelos = {}
    columnas = {}
    nombres = ["pulmon", "mama", "prostata", "gastrico", "cervical"]
    
    for n in nombres:
        try:
            modelos[n] = joblib.load(f'modelo_{n}_final.pkl')
            columnas[n] = joblib.load(f'columnas_{n}.pkl')
        except:
            pass
    return modelos, columnas

modelo_gen, scaler_gen, cols_gen = cargar_archivos_generales()
modelos_esp, cols_esp = cargar_especialistas()

if 'mostrar_detalle' not in st.session_state:
    st.session_state.mostrar_detalle = False

if 'diag_general_hecho' not in st.session_state:
    st.session_state.diag_general_hecho = False

st.title("Predicción de Cáncer con Inteligancia Artificial")

if modelo_gen is None:
    st.error("Error: Faltan archivos del modelo general.")
    st.stop()

st.sidebar.header("Datos del Paciente")

gender = st.sidebar.selectbox("Género", options=[0, 1], format_func=lambda x: "Masculino" if x == 0 else "Femenino")
age = st.sidebar.slider("Edad", 20, 90, 45)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 24.0)
smoking = st.sidebar.selectbox("Fuma", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
genetic = st.sidebar.selectbox("Riesgo Genético", options=[0, 1, 2], format_func=lambda x: ["Bajo", "Medio", "Alto"][x])
activity = st.sidebar.slider("Actividad Física", 0.0, 20.0, 5.0)
alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 2.0)
cancer_history = st.sidebar.selectbox("Historial Cáncer", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

if st.button("Diagnóstico General", type="primary"):
    st.session_state.diag_general_hecho = True
    st.session_state.mostrar_detalle = False

if st.session_state.diag_general_hecho:
    input_dict = {
        'Gender': [gender], 'Age': [age], 'BMI': [bmi], 'Smoking': [smoking],
        'GeneticRisk': [genetic], 'PhysicalActivity': [activity],
        'AlcoholIntake': [alcohol], 'CancerHistory': [cancer_history]
    }

    df = pd.DataFrame(input_dict)
    
    df["Age_Smoking"] = df["Age"] * df["Smoking"]
    df["BMI_Activity"] = df["BMI"] * df["PhysicalActivity"]
    df["Genetic_Smoking"] = df["GeneticRisk"] * df["Smoking"]
    df["Alcohol_Smoking"] = df["AlcoholIntake"] * df["Smoking"]
    df["Obese"] = (df["BMI"] >= 30).astype(int)
    df["LowActivity"] = (df["PhysicalActivity"] < 2).astype(int)
    df["HeavyDrinker"] = (df["AlcoholIntake"] > 3).astype(int)
    df["Log_Alcohol"] = np.log1p(df["AlcoholIntake"])
    df["Log_Activity"] = np.log1p(df["PhysicalActivity"])
    df["Age_squared"] = df["Age"] ** 2
    df["BMI_squared"] = df["BMI"] ** 2
    df["BMI_per_Age"] = df["BMI"] / df["Age"]

    try:
        X_scaled = scaler_gen.transform(df[scaler_gen.feature_names_in_])
        df_scaled = pd.DataFrame(X_scaled, columns=scaler_gen.feature_names_in_)
        pred = modelo_gen.predict(df_scaled[cols_gen])[0]
        prob = modelo_gen.predict_proba(df_scaled[cols_gen])[0][1]
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        if pred == 1:
            c1.error("POSITIVO")
            c2.error("Riesgo Alto Detectado")
            c2.write(f"Probabilidad: {prob:.2%}")
            st.markdown("---")
            if st.button("Análisis Diferencial"):
                st.session_state.mostrar_detalle = True
        else:
            c1.success("NEGATIVO")
            c2.success("Riesgo Bajo")
            c2.write(f"Probabilidad: {prob:.2%}")

    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.mostrar_detalle:
    st.markdown("---")
    st.header("Análisis Diferencial")
    
    tabs_names = ["Pulmón", "Estómago"]
    if gender == 1:
        tabs_names.extend(["Mama", "Cervical"])
    else:
        tabs_names.append("Próstata")
        
    tabs = st.tabs(tabs_names)
    
    # 1. PESTAÑA PULMÓN
    with tabs[0]:
        #TENGO QUE LLENAR CON LOS DATOS

    # 2. PESTAÑA ESTÓMAGO 
    with tabs[1]:
        #TENGO QUE LLENAR CON LOS DATOS

    # Si es mujer u hombre
    if gender == 1:
        # 3. PESTAÑA MAMA
        #TENGO QUE LLENAR CON LOS DATOS

        # 4. PESTAÑA CERVICAL
        with tabs[3]:
            st.write("TENGO QUE LLENAR CON LOS DATOS")

    else:
        # 3. PESTAÑA PRÓSTATA
        with tabs[2]:
            st.write("
            st.write("TENGO QUE LLENAR CON LOS DATOS")