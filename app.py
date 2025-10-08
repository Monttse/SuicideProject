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
PERFILES_PATH = 'perfiles.csv'     # Archivo de perfiles (pequeño, debe estar en GitHub)
TSNE_PATH = '13.tsne.PNG'          # Imagen t-SNE (debe estar en GitHub)

# IDs DE GOOGLE DRIVE (¡REEMPLAZA ESTAS ID!)
DF_FILE_ID = '1UM9B_EJ5K_D_H-XGYaGhX6IDP79Gki1M'    # ID de tu archivo PARQUET
GEOJSON_FILE_ID = '1mTqwYwgobCnZpdezVfLAxVyHYbV3DQDN' # ID de tu archivo GeoJSON SIMPLIFICADO de 6MB

# -----------------------------------------------------------
# --- FUNCIONES DE CARGA DE DATOS (CACHEADAS) ---
# -----------------------------------------------------------

@st.cache_data
def load_geojson(file_id):
    """Descarga y carga el GeoJSON simplificado usando gdown."""
    output_path = "mexico_map_data.json"
    
    try:
        gdown.download(id=file_id, output=output_path, quiet=True, fuzzy=True)
        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)
        os.remove(output_path)
        return data
        
    except Exception as e:
        print(f"ERROR GDOWN/JSON: {e}") 
        return None

@st.cache_data
def load_data(file_id):
    """Descarga y carga el DataFrame principal usando gdown."""
    output_path_df = "temp_df.parquet"
    
    with st.spinner('Cargando datos principales desde Google Drive... (puede tardar hasta 1 minuto)'):
        try:
            gdown.download(id=file_id, output=output_path_df, quiet=True, fuzzy=True)
            df = pd.read_parquet(output_path_df)
            
            # CRÍTICO: Aseguramos que la clave del estado sea texto de dos dígitos ('01', '09')
            df['ent_resid'] = df['ent_resid'].astype(str).str.zfill(2) 
            
            if 'cluster' not in df.columns or 'ent_resid' not in df.columns:
                st.error("Error: El archivo PARQUET no tiene las columnas clave.")
                return None
            
            os.remove(output_path_df)
            return df
            
        except Exception as e:
            st.error(f"Error fatal al cargar el DataFrame principal: {e}")
            return None

# --- LLAMADA INICIAL DE DATOS ---
# Solo cargamos el DataFrame aquí para no exceder el timeout inicial
df_final = load_data(DF_FILE_ID)


# ====================================================================
# --- COMIENZA LA INTERFAZ (UI) ---
# ====================================================================

st.title("Sistema de Identificación de Perfiles de Riesgo de Suicidio (2020-2023)")
st.subheader("Modelado no Supervisado (K-Means) en Casos de Suicidio en México")
st.markdown("---")

# --- SECCIÓN 1: PERFILES DE RIESGO ---
st.header("1. Perfiles de Riesgo Identificados (K=4)")

try:
    # Carga del archivo de perfiles desde GitHub
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
    mx_geojson = None
    if df_final is not None:
         # Carga el GeoJSON solo si el DF principal se cargó correctamente
         with st.spinner('Cargando datos geográficos para el mapa...'):
             mx_geojson = load_geojson(GEOJSON_FILE_ID)

    if df_final is not None and mx_geojson is not None:
        
        st.subheader("Mapa de Concentración del Riesgo Principal (Cluster 2)")
        
        # --- NUEVA LÓGICA: CALCULAR PORCENTAJE DE CONCENTRACIÓN DEL CLUSTER 2 ---
        
        # 1. Contar el total de casos por estado
        df_conteo_total = df_final.groupby('ent_resid').size().reset_index(name='Total Casos')

        # 2. Contar los casos del Cluster 2 por estado
        df_conteo_cluster2 = df_final[df_final['cluster'] == 2].groupby('ent_resid').size().reset_index(name='Casos Cluster 2')

        # 3. Fusionar y Calcular el Porcentaje
        df_mapa = pd.merge(df_conteo_total, df_conteo_cluster2, on='ent_resid', how='left').fillna(0)
        df_mapa['Porcentaje Cluster 2'] = (df_mapa['Casos Cluster 2'] / df_mapa['Total Casos']) * 100

        df_mapa.rename(columns={'ent_resid': 'CVE_ENT'}, inplace=True)

        # --- CREACIÓN DEL MAPA (CHOROPLETH) ---
        
        fig = px.choropleth(
            df_mapa, 
            geojson=mx_geojson, 
            locations='CVE_ENT', 
            color='Porcentaje Cluster 2', # Usamos el porcentaje como variable de color
            color_continuous_scale="Reds", # Escala de rojos para el riesgo
            range_color=(0, df_mapa['Porcentaje Cluster 2'].max()),
            featureidkey='properties.CVE_ENT', 
            projection="mercator",
            labels={'Porcentaje Cluster 2':'% Cluster 2'}
        )
        
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada estado está coloreado por la concentración porcentual del Perfil de Riesgo Principal (Cluster 2).")

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
         st.error("No se pudo cargar el DataFrame principal.")
    elif mx_geojson is None:
         st.error("El mapa no se pudo cargar. Hubo un error al descargar el GeoJSON. Revisa la ID y los permisos de Drive.")
         
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

