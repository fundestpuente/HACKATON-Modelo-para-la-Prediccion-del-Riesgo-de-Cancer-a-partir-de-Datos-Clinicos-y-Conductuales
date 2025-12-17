import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64

st.set_page_config(page_title="Sistema de Predicción Oncológico", layout="wide", page_icon="resources/icon.png")

st.markdown("""
    <style>
    .result-box { 
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px; 
        text-align: center; 
        color: #000000 !important; /* TEXTO SIEMPRE NEGRO */
    }
    
    .high-risk { 
        background-color: #ffcccc; 
        color: #8a1f1f !important; /* Rojo oscuro para el texto */
        border: 1px solid #ebccd1; 
    }
    
    .low-risk { 
        background-color: #d6e9c6; 
        color: #2b542c !important; /* Verde oscuro para el texto */
        border: 1px solid #d6e9c6; 
    }
    
    .ranking-card { 
        background-color: #f0f8ff; 
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        border-left: 5px solid #2196F3;
        color: #000000 !important; /* TEXTO SIEMPRE NEGRO */
    }
    .ranking-card b, .ranking-card h2, .result-box h2 {
        color: inherit !important;
    }
    </style>
""", unsafe_allow_html=True)


# Cargar las imagenes
def obtener_imagen_base64(ruta_imagen):
    try:
        with open(ruta_imagen, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

# CARGA DE MODELOS
@st.cache_resource
def cargar_todo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_esp = os.path.join(base_dir, '..', 'notebooks', 'pkl')
    
    # General
    try:
        gen_model = joblib.load(os.path.join(base_dir, 'modelo_cancer_final.pkl'))
        scaler = joblib.load(os.path.join(base_dir, 'scaler_final.pkl'))
        cols_gen = joblib.load(os.path.join(base_dir, 'columnas_finales.pkl'))
    except:
        gen_model, scaler, cols_gen = None, None, None

    # Especialistas
    esp_models = {}
    esp_cols = {}
    nombres = ["pulmon", "mama", "prostata", "gastrico", "cervical"]
    
    for n in nombres:
        try:
            esp_models[n] = joblib.load(os.path.join(ruta_esp, f'modelo_{n}_final.pkl'))
            esp_cols[n] = joblib.load(os.path.join(ruta_esp, f'columnas_{n}.pkl'))
        except:
            pass
            
    return gen_model, scaler, cols_gen, esp_models, esp_cols

modelo_gen, scaler_gen, cols_gen, modelos_esp, cols_esp = cargar_todo()

if 'diagnostico_general_listo' not in st.session_state:
    st.session_state.diagnostico_general_listo = False


# SIDE BAR
st.sidebar.header("👤 Perfil Básico")
st.sidebar.info("Ingrese sus datos generales para iniciar la predicción.")

gender = st.sidebar.radio("Género", [0, 1], format_func=lambda x: "Masculino" if x == 0 else "Femenino")
age = st.sidebar.slider("Edad (Años)", 15, 95, 45)
bmi = st.sidebar.slider("Índice de Masa Corporal (BMI)", 15.0, 45.0, 24.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Estilo de Vida")

is_smoker = st.sidebar.checkbox("Fumador Activo")
genetic = st.sidebar.selectbox("Antecedentes Genéticos", [0, 1, 2], format_func=lambda x: ["Bajo", "Medio", "Alto"][x])
cancer_history = st.sidebar.checkbox("Historial personal de Cáncer")
activity_input = st.sidebar.slider("Actividad Física (Horas/Semana)", 0, 20, 5)
alcohol_input = st.sidebar.slider("Consumo Alcohol (Tragos/Semana)", 0, 20, 2)

smoking_val_gen = 1 if is_smoker else 0 
smoking_val_esp = 1 if is_smoker else 0 


# DIAGNÓSTICO GENERAL
st.title("🏥 Sistema de Predicción Oncológico")

st.markdown("1. Evaluación de Riesgo General")
st.write("Analizaremos su perfil base para detectar indicadores de riesgo.")

if st.button("Realizar Diagnóstico General", type="primary", use_container_width=True):
    if modelo_gen:
        input_gen = pd.DataFrame({
            'Gender': [gender], 'Age': [age], 'BMI': [bmi], 'Smoking': [smoking_val_gen],
            'GeneticRisk': [genetic], 'PhysicalActivity': [activity_input],
            'AlcoholIntake': [alcohol_input], 'CancerHistory': [1 if cancer_history else 0]
        })
        
        input_gen["Age_Smoking"] = input_gen["Age"] * input_gen["Smoking"]
        input_gen["BMI_Activity"] = input_gen["BMI"] * input_gen["PhysicalActivity"]
        input_gen["Genetic_Smoking"] = input_gen["GeneticRisk"] * input_gen["Smoking"]
        input_gen["Alcohol_Smoking"] = input_gen["AlcoholIntake"] * input_gen["Smoking"]
        input_gen["Obese"] = (input_gen["BMI"] >= 30).astype(int)
        input_gen["LowActivity"] = (input_gen["PhysicalActivity"] < 2).astype(int)
        input_gen["HeavyDrinker"] = (input_gen["AlcoholIntake"] > 3).astype(int)
        input_gen["Log_Alcohol"] = np.log1p(input_gen["AlcoholIntake"])
        input_gen["Log_Activity"] = np.log1p(input_gen["PhysicalActivity"])
        input_gen["Age_squared"] = input_gen["Age"] ** 2
        input_gen["BMI_squared"] = input_gen["BMI"] ** 2
        input_gen["BMI_per_Age"] = input_gen["BMI"] / input_gen["Age"]

        try:
            X_scaled = scaler_gen.transform(input_gen[scaler_gen.feature_names_in_])
            df_scaled = pd.DataFrame(X_scaled, columns=scaler_gen.feature_names_in_)
            prob_gen = modelo_gen.predict_proba(df_scaled[cols_gen])[0][1]
            
            st.session_state.diagnostico_general_listo = True
            st.session_state.prob_general = prob_gen
            
        except Exception as e:
            st.error(f"Error en el modelo general: {e}")
            st.session_state.diagnostico_general_listo = False

if st.session_state.diagnostico_general_listo:
    prob = st.session_state.prob_general

    img_danger = obtener_imagen_base64("resources/danger.png") 
    img_check = obtener_imagen_base64("resources/check.png")

    if prob > 0.4:
        st.markdown(f"""
        <div class="result-box high-risk">
            <img src="data:image/png;base64,{img_danger}" width="50" style="margin-bottom: 10px;">
            <h2>RIESGO ALTO DETECTADO ({prob:.1%})</h2>
            <p>Su perfil coincide con patrones de riesgo oncológico.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box low-risk">
            <img src="data:image/png;base64,{img_check}" width="50" style="margin-bottom: 10px;">
            <h2>RIESGO BAJO ({prob:.1%})</h2>
            <p>Su perfil general es saludable.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    path_pulmon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/lungs.png") 
    path_gastrico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/estomago.png")
    path_utero = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/utero.png")
    path_mama = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/mama.png")
    path_prostata = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/prostata.png")

    img_pulmon = obtener_imagen_base64(path_pulmon)
    img_gastrico = obtener_imagen_base64(path_gastrico)
    img_utero = obtener_imagen_base64(path_utero)
    img_mama = obtener_imagen_base64(path_mama)
    img_prostata = obtener_imagen_base64(path_prostata)

    # PARA ESPECIFICAR
    st.markdown("2. Análisis Específico (Opcional)")
    st.info("Complete los síntomas para identificar el tipo de riesgo más probable.")

    with st.expander("Desplegar Formulario de Síntomas", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <h3 style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_pulmon}" width="35" style="margin-right: 10px;">
                Respiratorio
            </h3>
            """, unsafe_allow_html=True)
            s_allergy = st.checkbox("Alergias Frecuentes")
            s_wheezing = st.checkbox("Sibilancias (Silbidos)")
            s_cough = st.checkbox("Tos Crónica")
            s_chest = st.checkbox("Dolor en el pecho")
            s_fatigue = st.checkbox("Fatiga Crónica")
            
        with col2:
            st.markdown(f"""
            <h3 style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_gastrico}" width="35" style="margin-right: 10px;">
                Gástrico / Otros
            </h3>
            """, unsafe_allow_html=True)
            g_pylori = st.checkbox("Diagnóstico H. Pylori")
            s_swallow = st.checkbox("Dificultad para tragar")
            s_chronic = st.checkbox("Enf. Crónica Previa")
            s_anxiety = st.checkbox("Ansiedad")

    
        if gender == 1: # Mujer
            st.markdown(f"""
            <h3 style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_utero}" width="35" style="margin-right: 10px;">
                Ginecológico
            </h3>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c_partners = c1.number_input("Parejas Sexuales", 0, 50, 2)
            c_first_sex = c1.number_input("Edad 1ra relación", 10, 30, 18)
            c_pregnancies = c2.number_input("Embarazos", 0, 15, 0)
            c_hormonal = c2.checkbox("Uso de Anticonceptivos")
            
            st.markdown(f"""
            <h3 style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_mama}" width="35" style="margin-right: 10px;">
                Autoexamen de mama
            </h3>
            """, unsafe_allow_html=True)
            
            m_lump = st.checkbox("¿Ha detectado algún bulto o masa?")
            
            m_pain_lump = False
            m_pain_general = False
            
            if m_lump:
                st.info("Un bulto indoloro (que no duele) suele ser más sospechoso.")
                m_pain_lump = st.checkbox("¿El bulto es doloroso al tacto?")
            else:
                m_pain_general = st.checkbox("¿Siente dolor o cambios en la piel?")
            
        else: # Hombre
            st.markdown(f"""
            <h3 style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_prostata}" width="35" style="margin-right: 10px;">
                Salud Prostática
            </h3>
            """, unsafe_allow_html=True)
            p_urine = st.checkbox("¿Dificultad al orinar?")
            p_night = st.checkbox("¿Se levanta mucho de noche a orinar?")

    if st.button("Analizar Tipos Específicos"):
        st.write("Ranking de Probabilidades según sus síntomas:")
        
        ranking = [] 

        # PULMÓN
        if "pulmon" in modelos_esp:
            try:          
                df_p = pd.DataFrame({
                    'GENDER': [1 if gender == 0 else 0],
                    'AGE': [age],
                    'SMOKING': [smoking_val_esp],
                    'YELLOW_FINGERS': [0], 
                    'ANXIETY': [0],
                    'PEER_PRESSURE': [0],
                    'CHRONIC DISEASE': [0],
                    'FATIGUE ': [1 if s_fatigue else 0],
                    'ALLERGY ': [1 if s_allergy else 0],
                    'WHEEZING': [1 if s_wheezing else 0],
                    'ALCOHOL CONSUMING': [1 if alcohol_input > 2 else 0],
                    'COUGHING': [1 if s_cough else 0],
                    'SHORTNESS OF BREATH': [1 if s_chest else 0],
                    'SWALLOWING DIFFICULTY': [1 if s_swallow else 0],
                    'CHEST PAIN': [1 if s_chest else 0]
                })

                df_p = df_p[cols_esp["pulmon"]]
                prob = modelos_esp["pulmon"].predict_proba(df_p)[0][1]
                ranking.append(("Pulmón", prob))

            except Exception as e:
                st.error(f"Error Pulmón: {e}")


        # GÁSTRICO
        if "gastrico" in modelos_esp:
            try:
                df_g = pd.DataFrame(0.0, index=[0], columns=cols_esp["gastrico"])
                
                df_g['age'] = age
                df_g['helicobacter_pylori_infection'] = 1 if g_pylori else 0
                df_g['alcohol_consumption'] = 1 if alcohol_input > 2 else 0
                df_g['smoking_habits'] = 1 if smoking_val_esp == 1 else 0
                
                if 'gender_Male' in df_g.columns:
                    df_g = df_g.drop(columns=['gender_Male'])

                prob = modelos_esp["gastrico"].predict_proba(df_g)[0][1]
                
                if g_pylori:
                    prob = max(prob, 0.65)
                if s_swallow:
                    prob = min(prob + 0.10, 0.95)

                ranking.append(("Gástrico", prob))
                
            except Exception as e:
                st.error(f"Error Gástrico: {e}")

        # GINECOLÓGICO
        if gender == 1:
            if "cervical" in modelos_esp:
                try:
                    df_c = pd.DataFrame(0.0, index=[0], columns=cols_esp["cervical"])
                    df_c['Age'] = age
                    df_c['Number of sexual partners'] = c_partners
                    df_c['First sexual intercourse'] = c_first_sex
                    df_c['Num of pregnancies'] = c_pregnancies
                    df_c['Smokes'] = 1 if is_smoker else 0
                    df_c['Hormonal Contraceptives'] = 1 if c_hormonal else 0
                    df_c['STDs'] = 1 if c_partners > 5 else 0 
                    prob = modelos_esp["cervical"].predict_proba(df_c)[0][1]
                    ranking.append(("Cervical", prob))
                except Exception as e: st.error(f"Error Cervical: {e}")
            
            # MAMA
            if "mama" in modelos_esp:
                try:
                    df_m = pd.DataFrame(0.0, index=[0], columns=cols_esp["mama"])
                    val_radius = 11.0
                    val_area = 400.0
                    val_perimeter = 75.0
                    val_concave = 0.01

                    if m_lump:
                        if m_pain_lump:
                            val_radius = 16.0 
                            val_area = 800.0
                            val_perimeter = 100.0
                            val_concave = 0.08
                        else:
                            val_radius = 20.0 
                            val_area = 1200.0 
                            val_perimeter = 130.0
                            val_concave = 0.18 
                    
                    elif m_pain_general:
                        val_radius = 13.0
                        val_area = 550.0
                        val_perimeter = 85.0
                        
                    df_m['radius_mean'] = val_radius
                    df_m['texture_mean'] = 20.0 
                    
                    if 'area_mean' in df_m.columns: df_m['area_mean'] = val_area
                    if 'perimeter_mean' in df_m.columns: df_m['perimeter_mean'] = val_perimeter
                    if 'concave points_mean' in df_m.columns: df_m['concave points_mean'] = val_concave

                    if 'area_worst' in df_m.columns: df_m['area_worst'] = val_area * 1.3
                    if 'perimeter_worst' in df_m.columns: df_m['perimeter_worst'] = val_perimeter * 1.2
                    if 'radius_worst' in df_m.columns: df_m['radius_worst'] = val_radius * 1.2

                    prob = modelos_esp["mama"].predict_proba(df_m)[0][1]
                    
                    if m_lump:
                        if not m_pain_lump:
                            prob = max(prob, 0.92)
                        else:
                            prob = max(prob, 0.80)
                    
                    ranking.append(("Mama", prob))
                except Exception as e: st.error(f"Error Mama: {e}")

        # PROSTATA
        else:
            if "prostata" in modelos_esp:
                try:
                    df_pr = pd.DataFrame(0.0, index=[0], columns=cols_esp["prostata"])
                    if p_urine or p_night:
                        val_r, val_a = 19.0, 900.0
                    else:
                        val_r, val_a = 10.0, 400.0
                        
                    df_pr['radius'] = val_r
                    df_pr['perimeter'] = val_r * 6.28
                    df_pr['area'] = val_a
                    
                    if hasattr(modelos_esp["prostata"], "predict_proba"):
                        prob = modelos_esp["prostata"].predict_proba(df_pr)[0][1]
                    else:
                        pred = modelos_esp["prostata"].predict(df_pr)[0]
                        prob = 0.95 if pred == 1 else 0.05
                    ranking.append(("Próstata", prob))
                except Exception as e: st.error(f"Error Próstata: {e}")


        # MOSTRAR RESULTADOS
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        for nombre, prob in ranking:
            st.markdown(f"""
            <div class="ranking-card">
                <b>{nombre}:</b> {prob:.1%}
                <div style="background-color: #e0e0e0; border-radius: 5px;">
                    <div style="width: {prob*100}%; background-color: #2196F3; height: 10px; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)