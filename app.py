import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
import json
import os
import pydeck as pdk

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
        st.subheader("Mapa de Concentración del Riesgo Principal (Cluster 2)")

        # --- LÓGICA: CALCULAR PORCENTAJE DE CONCENTRACIÓN DEL CLUSTER 2 ---
        df_conteo_total = df_final.groupby('ent_resid').size().reset_index(name='Total Casos')
        df_conteo_cluster2 = df_final[df_final['cluster'] == 2].groupby('ent_resid').size().reset_index(name='Casos Cluster 2')
        df_mapa = pd.merge(df_conteo_total, df_conteo_cluster2, on='ent_resid', how='left').fillna(0)
        df_mapa['Porcentaje Cluster 2'] = (df_mapa['Casos Cluster 2'] / df_mapa['Total Casos']) * 100
        df_mapa.rename(columns={'ent_resid': 'CVE_ENT'}, inplace=True)
        
        # Necesitamos la tabla completa con el GeoJSON
        geojson_data = mx_geojson # Tu GeoJSON que ya cargaste
        
        # --- CREACIÓN DEL MAPA CON PYDECK (LA SOLUCIÓN ESTABLE) ---
        
        # Encuentra el centro de México para la vista inicial
        lat_centro = 23.6345
        lon_centro = -102.5528
        
        # Generamos la información de color. Pydeck usa el formato R, G, B, A (0-255)
        # Queremos Rojo, basado en el porcentaje (50% = rojo medio, 100% = rojo fuerte)
        # La escala será: 0-100% de Porcentaje Cluster 2 -> 0-255 en el canal R
        
        df_mapa['red_intensity'] = (df_mapa['Porcentaje Cluster 2'] / df_mapa['Porcentaje Cluster 2'].max() * 255).astype(int)
        
        # Creamos una capa GeoJson con Pydeck
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            get_fill_color="[red_intensity, 0, 0, 160]", # [R, G, B, Alpha] - Usamos la intensidad roja calculada
            get_line_color=[0, 0, 0], # Líneas negras
            line_width_min_pixels=1,
            pickable=True, # Permite la interacción
        )

        # Definimos la vista inicial del mapa (Centrado en México)
        view_state = pdk.ViewState(
            latitude=lat_centro,
            longitude=lon_centro,
            zoom=4.5,
            min_zoom=4,
            max_zoom=10,
            pitch=0,
        )

        # Renderizamos el mapa en Streamlit
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9", # Estilo de mapa claro
            initial_view_state=view_state,
            layers=[geojson_layer],
        ))
        
        # Agregamos una leyenda simple
        st.caption("Intensidad de color rojo = Mayor concentración del Perfil de Riesgo Principal (Cluster 2).")

        st.markdown("---")
        # El resto de la sección de análisis detallado con st.selectbox sigue igual        
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





