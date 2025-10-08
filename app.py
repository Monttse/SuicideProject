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

# GeoJSON ahora es local, ya no necesita ID de Drive
GEOJSON_PATH = 'mexico.json' # <--- ¡ESTO CAMBIÓ!

# ID DE GOOGLE DRIVE para el DataFrame principal (Sigue siendo GDrive)
DF_FILE_ID = '1UM9B_EJ5K_D_H-XGYaGhX6IDP79Gki1M' 

# -----------------------------------------------------------
# --- FUNCIONES DE CARGA DE DATOS ---
# -----------------------------------------------------------

@st.cache_data
def load_geojson(path):
    """Carga el GeoJSON simplificado directamente desde el repositorio."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data
        
    except Exception as e:
        st.error(f"Error fatal al cargar GeoJSON desde GitHub: {e}")
        return None

@st.cache_data
def load_data(file_id):
    """Descarga y carga el DataFrame principal usando gdown (la única descarga lenta)."""
    output_path_df = "temp_df.parquet"
    
    with st.spinner('Cargando datos principales desde Google Drive...'):
        try:
            gdown.download(id=file_id, output=output_path_df, quiet=True, fuzzy=True)
            df = pd.read_parquet(output_path_df)
            df['ent_resid'] = df['ent_resid'].astype(str).str.zfill(2) 
            # ... (otras verificaciones)
            os.remove(output_path_df)
            return df
        except Exception as e:
            st.error(f"Error fatal al cargar el DataFrame principal: {e}")
            return None

# --- LLAMADA INICIAL DE DATOS ---
df_final = load_data(DF_FILE_ID) # Sigue siendo la única llamada lenta


# ====================================================================
# --- COMIENZA LA INTERFAZ (UI) ---
# ====================================================================

st.title("Sistema de Identificación de Perfiles de Riesgo de Suicidio (2020-2023)")
st.subheader("Modelado no Supervisado (K-Means) en Casos de Suicidio en México")
st.markdown("---")

# --- SECCIÓN 1: PERFILES DE RIESGO ---
# ... (Igual)
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
    mx_geojson = load_geojson(GEOJSON_PATH) # <-- Carga instantánea desde GitHub

    if df_final is not None and mx_geojson is not None:
        
        st.subheader("Mapa de Concentración del Riesgo Principal (Cluster 2)")
        
        # --- LÓGICA: CALCULAR PORCENTAJE DE CONCENTRACIÓN DEL CLUSTER 2 ---
        df_conteo_total = df_final.groupby('ent_resid').size().reset_index(name='Total Casos')
        df_conteo_cluster2 = df_final[df_final['cluster'] == 2].groupby('ent_resid').size().reset_index(name='Casos Cluster 2')
        df_mapa = pd.merge(df_conteo_total, df_conteo_cluster2, on='ent_resid', how='left').fillna(0)
        df_mapa['Porcentaje Cluster 2'] = (df_mapa['Casos Cluster 2'] / df_mapa['Total Casos']) * 100
        df_mapa.rename(columns={'ent_resid': 'CVE_ENT'}, inplace=True)

        # --- CREACIÓN DEL MAPA (CHOROPLETH) ---
        fig = px.choropleth(
            df_mapa, 
            geojson=mx_geojson, 
            locations='CVE_ENT', 
            color='Porcentaje Cluster 2', 
            color_continuous_scale="Reds", 
            featureidkey='properties.CVE_ENT', # <--- REVISA ESTA CLAVE en tu GeoJSON de 6MB
            projection="mercator",
            labels={'Porcentaje Cluster 2':'% Cluster 2'}
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada estado está coloreado por la concentración porcentual del Perfil de Riesgo Principal (Cluster 2).")

        st.markdown("---")
        st.subheader("Análisis Detallado por Entidad")
        
        lista_entidades = sorted(df_final['ent_resid'].unique().tolist())
        entidad_seleccionada = st.selectbox(
            'Selecciona una Entidad de Residencia:',
            lista_entidades,
            format_func=lambda x: f"Entidad {x}"
        )

        df_filtrado = df_final[df_final['ent_resid'] == entidad_seleccionada]
        distribucion_cluster = df_filtrado['cluster'].value_counts(normalize=True).mul(100).sort_index()
        
        st.bar_chart(distribucion_cluster)
        st.caption(f"Distribución porcentual de los 4 perfiles en la entidad {entidad_seleccionada}.")

    elif df_final is None:
         st.error("No se pudo cargar el DataFrame principal.")
    elif mx_geojson is None:
         st.error("Error al cargar el GeoJSON. Asegúrate que el archivo esté en GitHub y se llame 'mexico_map_data.json'.")
         
except Exception as e:
    st.error(f"Error crítico al generar la sección geográfica: {e}")

st.markdown("---")

# --- SECCIÓN 3: VALIDACIÓN DEL MODELO (t-SNE) ---
# ... (Igual)
st.header("3. Validación y Caracterización del Modelo (t-SNE)")

# ... (El resto de la Sección 3) ...
try:
    st.image(TSNE_PATH, caption="Visualización de Clusters con t-SNE", use_container_width=True) 
except FileNotFoundError:
    st.error(f"Error: No se encontró la imagen del t-SNE en {TSNE_PATH}.")


