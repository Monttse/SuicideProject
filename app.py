¡Excelente! Ahora sí podemos hacer la corrección definitiva para eliminar el error de duplicidad y el error de tiempo de espera (EOF) causado por la doble carga.

El problema en el código que enviaste es que la función load_data está definida dos veces, y Streamlit se detiene al ver esta duplicación antes de que la aplicación siquiera empiece a dibujar.

He tomado tu código, eliminé las secciones duplicadas, aseguré que las funciones de carga solo usen gdown (ya que es la única manera de que los archivos grandes funcionen) y mantuve todos tus placeholders y lógica de visualización.

Reemplaza el contenido completo de tu app.py con este código corregido y limpio.

Código app.py Estabilizado (Versión Final)
Python

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
import json
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Perfiles de Riesgo de Suicidio MX", layout="wide")

# --- VARIABLES Y ARCHIVOS ---
K_OPTIMO = 4 
PERFILES_PATH = 'perfiles.csv'     
TSNE_PATH = '13.tsne.PNG'          

# IDs ÚNICAS DE GOOGLE DRIVE (Archivos grandes)
DF_FILE_ID = '1UM9B_EJ5K_D_H-XGYaGhX6IDP79Gki1M' 
GEOJSON_FILE_ID = '1mTqwYwgobCnZpdezVfLAxVyHYbV3DQDN'

# --- 1. CARGA DE DATOS PRINCIPALES (PARQUET) con gdown ---
@st.cache_data
def load_data(file_id):
    """Descarga y carga el DataFrame principal usando gdown."""
    output_path_df = "temp_df.parquet"
    
    # Usamos st.spinner para avisar al servidor que debe esperar
    with st.spinner('Cargando datos principales desde Google Drive... (Puede tardar hasta 1 min)'):
        try:
            # DESCARGA: gdown
            gdown.download(id=file_id, output=output_path_df, quiet=True, fuzzy=True)
            
            # CARGA LOCAL
            df = pd.read_parquet(output_path_df)
            
            # Limpieza y Verificación
            # CRÍTICO: Asegurarse de que el ID del estado sea string de dos dígitos ('01', '09')
            df['ent_resid'] = df['ent_resid'].astype(str).str.zfill(2) 
            
            if 'cluster' not in df.columns or 'ent_resid' not in df.columns:
                st.error("Error: El archivo PARQUET no tiene las columnas clave.")
                return None
            
            # Limpiar el archivo temporal
            os.remove(output_path_df)
            return df
            
        except Exception as e:
            st.error(f"Error fatal al cargar el DataFrame principal: {e}")
            return None

df_final = load_data(DF_FILE_ID)

# --- 2. CARGA DE GEOJSON DESDE GOOGLE DRIVE (con gdown) ---
@st.cache_data
def load_geojson(file_id):
    """Descarga y carga el GeoJSON usando gdown."""
    output_path = "mexico_map_data.json"
    
    try:
        # DESCARGA: gdown
        gdown.download(id=file_id, output=output_path, quiet=True, fuzzy=True)
        
        # CARGA: Leemos el archivo localmente (con encoding='utf-8')
        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)
        
        # Limpiar el archivo temporal
        os.remove(output_path)
        return data
        
    except Exception as e:
        # Quitamos st.error para evitar el fallo "Oh no" y solo registramos.
        print(f"ERROR GDOWN/JSON: {e}") 
        return None

mx_geojson = load_geojson(GEOJSON_FILE_ID)

# ====================================================================
# --- COMIENZA LA INTERFAZ (UI) ---
# ====================================================================

st.title("Sistema de Identificación de Perfiles de Riesgo de Suicidio (2020-2023)")
st.subheader("Modelado no Supervisado (K-Means) en Casos de Suicidio en México")
st.markdown("---")

# --- SECCIÓN 1: PERFILES DE RIESGO ---
st.header("1. Perfiles de Riesgo Identificados (K=4)")

try:
    df_perfiles = pd.read_csv(PERFILES_PATH)
    st.dataframe(
        df_perfiles.style.background_gradient(cmap='YlOrRd', subset=['Tamaño del Cluster']),
        hide_index=True,
        use_container_width=True
    )
    st.caption("Los 4 perfiles se definen por la moda de variables sociodemográficas (sexo, ocupación, horario) y el promedio de la edad.")

except FileNotFoundError:
    st.error(f"⚠️ Error: No se pudo cargar la tabla de perfiles en {PERFILES_PATH}.")
st.markdown("---")


# --- SECCIÓN 2: MAPA Y ANÁLISIS GEOGRÁFICO ---
st.header("2. Foco de Intervención Geográfica")

try:
    if df_final is not None and mx_geojson is not None:
        
        st.subheader("Mapa de Riesgo Dominante por Entidad")
        
        # 1. Calcular el CLUSTER DOMINANTE por estado
        df_mapa = df_final.groupby('ent_resid')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
        df_mapa.rename(columns={'ent_resid': 'CVE_ENT', 'cluster': 'Cluster Dominante'}, inplace=True)
        
        nombres_perfil = {
            0: "0. Joven Inactivo (Desempleo)",
            1: "1. Adulto Mayor Ocupado",
            2: "2. Adulto Joven Ocupado (Foco)",
            3: "3. Riesgo Desconocido (Madrugada)"
        }
        df_mapa['Perfil Dominante'] = df_mapa['Cluster Dominante'].map(nombres_perfil)
        
        # 2. Creación del Mapa (Choropleth)
        fig = px.choropleth(
            df_mapa, 
            geojson=mx_geojson, 
            locations='CVE_ENT', 
            color='Perfil Dominante',
            featureidkey='properties.CVE_ENT', 
            projection="mercator",
            color_discrete_map={
                '0. Joven Inactivo (Desempleo)': 'yellow',
                '1. Adulto Mayor Ocupado': 'green',
                '2. Adulto Joven Ocupado (Foco)': 'red',
                '3. Riesgo Desconocido (Madrugada)': 'purple'
            }
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada estado está coloreado por el Perfil de Riesgo que es estadísticamente dominante en esa entidad.")

        st.markdown("---")
        st.subheader("Análisis Detallado por Entidad")
        
        # 3. Selector de Entidad (Análisis Detallado)
        lista_entidades = sorted(df_final['ent_resid'].unique().tolist())
        entidad_seleccionada = st.selectbox(
            'Selecciona una Entidad de Residencia:',
            lista_entidades,
            format_func=lambda x: f"Entidad {x}"
        )

        # Análisis por la Entidad Seleccionada
        df_filtrado = df_final[df_final['ent_resid'] == entidad_seleccionada]
        distribucion_cluster = df_filtrado['cluster'].value_counts(normalize=True).mul(100).sort_index()
        
        st.bar_chart(distribucion_cluster)
        st.caption(f"Distribución porcentual de los 4 perfiles en la entidad {entidad_seleccionada}.")

    elif df_final is None:
         st.error("No se pudo cargar el DataFrame principal. Revisa la ID del archivo PARQUET en Google Drive.")
    elif mx_geojson is None:
         st.error("El mapa no se pudo cargar. Revisa la ID del GeoJSON en Google Drive.")
         
except Exception as e:
    st.error(f"Error crítico al generar la sección geográfica: {e}")

st.markdown("---")


# --- SECCIÓN 3: VALIDACIÓN DEL MODELO (t-SNE) ---
st.header("3. Validación y Caracterización del Modelo (t-SNE)")

st.markdown("La visualización t-SNE comprime las múltiples dimensiones en dos. La **superposición** de los grupos indica que el modelo es mejor para la segmentación de políticas públicas que para la predicción individual.")

try:
    st.image(TSNE_PATH, caption="Visualización de Clusters con t-SNE", use_container_width=True)
except FileNotFoundError:
    st.error(f"Error: No se encontró la imagen del t-SNE en {TSNE_PATH}.")
