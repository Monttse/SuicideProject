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

# RUTA LOCAL EN GITHUB (¡VERIFICA QUE EL ARCHIVO EXISTA Y ESTÉ BIEN NOMBRADO!)
GEOJSON_PATH = 'mexico.json' 

# ID DE GOOGLE DRIVE para el DataFrame principal
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
        # Aquí se mostrará el error si el JSON está mal
        st.error(f"Error fatal al cargar GeoJSON desde GitHub: {e}") 
        return None

@st.cache_data
def load_data(file_id):
    """Descarga y carga el DataFrame principal usando gdown."""
    output_path_df = "temp_df.parquet"
    
    with st.spinner('Cargando datos principales desde Google Drive...'):
        try:
            gdown.download(id=file_id, output=output_path_df, quiet=True, fuzzy=True)
            df = pd.read_parquet(output_path_df)
            df['ent_resid'] = df['ent_resid'].astype(str).str.zfill(2) 
            os.remove(output_path_df)
            return df
        except Exception as e:
            st.error(f"Error fatal al cargar el DataFrame principal: {e}")
            return None

# --- LLAMADA INICIAL DE DATOS ---
df_final = load_data(DF_FILE_ID)


# ====================================================================
# --- COMIENZA LA INTERFAZ (UI) ---
# ====================================================================

st.title("Sistema de Identificación de Perfiles de Riesgo de Suicidio (2020-2023)")
st.subheader("Modelado no Supervisado (K-Means) en Casos de Suicidio en México")
st.markdown("---")

# --- SECCIÓN 1: PERFILES DE RIESGO ---
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


# --- Mapeo de Nombres de Estado y Clusters (AGREGAR ESTO AL INICIO DE TU APP.PY, despuÉs de las variables) ---

ESTADO_NOMBRES = {
    '01': 'Aguascalientes', '02': 'Baja California', '03': 'Baja California Sur', 
    '04': 'Campeche', '05': 'Coahuila', '06': 'Colima', '07': 'Chiapas', 
    '08': 'Chihuahua', '09': 'Ciudad de México', '10': 'Durango', '11': 'Guanajuato', 
    '12': 'Guerrero', '13': 'Hidalgo', '14': 'Jalisco', '15': 'México', 
    '16': 'Michoacán', '17': 'Morelos', '18': 'Nayarit', '19': 'Nuevo León', 
    '20': 'Oaxaca', '21': 'Puebla', '22': 'Querétaro', '23': 'Quintana Roo', 
    '24': 'San Luis Potosí', '25': 'Sinaloa', '26': 'Sonora', '27': 'Tabasco', 
    '28': 'Tamaulipas', '29': 'Tlaxcala', '30': 'Veracruz', '31': 'Yucatán', 
    '32': 'Zacatecas'
}

CLUSTER_NOMBRES = {
    0: 'Joven Inactivo (Desempleo)',
    1: 'Adulto Mayor Ocupado (Riesgo Vespertino)',
    2: 'Adulto Joven Ocupado (Foco Principal)',
    3: 'Riesgo Desconocido (Madrugada)'
}

# --- SECCIÓN 2: MAPA Y ANÁLISIS GEOGRÁFICO (El bloque try/except) ---
st.header("2. Foco de Intervención Geográfica")

try:
    mx_geojson = load_geojson(GEOJSON_PATH) 

    if df_final is not None and mx_geojson is not None:
        
        # --- NUEVO SELECTOR DE CLUSTER PARA EL MAPA ---
        st.subheader("Selección de Perfil de Riesgo")
        
        cluster_seleccionado_id = st.selectbox(
            'Visualizar Concentración de:',
            options=list(CLUSTER_NOMBRES.keys()),
            format_func=lambda x: f"Cluster {x}: {CLUSTER_NOMBRES[x]}"
        )
        
        cluster_seleccionado_nombre = CLUSTER_NOMBRES[cluster_seleccionado_id]
        
        st.subheader(f"Mapa de Concentración del Perfil: {cluster_seleccionado_nombre}")
        
        # --- 1. PREPARACIÓN DE DATOS (DataFrame) ---
        # La lógica usa el cluster seleccionado
        df_conteo_total = df_final.groupby('ent_resid').size().reset_index(name='Total Casos')
        df_conteo_cluster = df_final[df_final['cluster'] == cluster_seleccionado_id].groupby('ent_resid').size().reset_index(name='Casos Cluster')
        
        df_mapa = pd.merge(df_conteo_total, df_conteo_cluster, on='ent_resid', how='left').fillna(0)
        df_mapa['Porcentaje Cluster'] = (df_mapa['Casos Cluster'] / df_mapa['Total Casos']) * 100
        df_mapa.rename(columns={'ent_resid': 'CVE_ENT'}, inplace=True)
        
        # Escala de color
        max_porcentaje = df_mapa['Porcentaje Cluster'].max()
        # Se usa un color diferente para cada cluster para mayor contraste (ej. Cluster 0 = Azul, Cluster 1 = Verde, etc.)
        df_mapa['color_intensity'] = (df_mapa['Porcentaje Cluster'] / max_porcentaje * 255).astype(int)

        # Mapeo de colores RGB basado en el Cluster ID
        if cluster_seleccionado_id == 0: # Joven Inactivo -> Azul
            color_formula = "[0, 0, properties.color_intensity, 160]"
        elif cluster_seleccionado_id == 1: # Adulto Mayor -> Verde
            color_formula = "[0, properties.color_intensity, 0, 160]"
        elif cluster_seleccionado_id == 2: # Adulto Joven (Foco) -> Rojo
            color_formula = "[properties.color_intensity, 0, 0, 160]"
        else: # Riesgo Desconocido -> Amarillo/Blanco
            color_formula = "[properties.color_intensity, properties.color_intensity, 0, 160]"


        # --- 2. FUSIÓN DE DATOS EN EL GEOJSON (Pydeck) ---
        color_lookup = df_mapa.set_index('CVE_ENT')['color_intensity'].to_dict()
        
        # Crear una copia para evitar modificar el objeto cacheado
        geojson_data = json.loads(json.dumps(mx_geojson)) 
        
        for feature in geojson_data['features']:
            # La clave del GeoJSON debe coincidir con el DF ('01', '02', etc.)
            entidad_id = feature['properties']['CVE_ENT'] 
            color_value = color_lookup.get(entidad_id, 0) 
            feature['properties']['color_intensity'] = color_value
            
        # --- 3. CREACIÓN DEL MAPA CON PYDECK ---
        
        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            get_fill_color=color_formula, # Usa la fórmula de color definida
            get_line_color=[0, 0, 0], 
            line_width_min_pixels=1,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=23.6345,
            longitude=-102.5528,
            zoom=4.5,
            min_zoom=4,
            max_zoom=10,
            pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9", 
            initial_view_state=view_state,
            layers=[geojson_layer],
        ))
        
        st.caption(f"El color en el mapa indica la concentración porcentual del Perfil {cluster_seleccionado_id} ({cluster_seleccionado_nombre}) por entidad.")
        st.markdown("---")
        
        # --- 4. ANÁLISIS DETALLADO (SELECTOR DE ESTADO CON NOMBRES) ---
        st.subheader("Análisis Detallado por Entidad")
        
        lista_codigos = sorted(df_final['ent_resid'].unique().tolist())
        
        # CORRECCIÓN: El selector ahora usa el nombre del estado
        entidad_seleccionada_codigo = st.selectbox(
            'Selecciona una Entidad de Residencia:',
            options=lista_codigos,
            # La función de formato usa el diccionario de nombres
            format_func=lambda codigo: f"{ESTADO_NOMBRES.get(codigo, f'Entidad {codigo}')}" 
        )

        df_filtrado = df_final[df_final['ent_resid'] == entidad_seleccionada_codigo]
        distribucion_cluster = df_filtrado['cluster'].value_counts(normalize=True).mul(100).sort_index()
        
        st.bar_chart(distribucion_cluster)
        # CORRECCIÓN: La leyenda muestra el nombre completo
        nombre_estado = ESTADO_NOMBRES.get(entidad_seleccionada_codigo, f'Entidad {entidad_seleccionada_codigo}')
        st.caption(f"Distribución porcentual de los 4 perfiles en {nombre_estado}.")

    elif df_final is None:
         st.error("No se pudo cargar el DataFrame principal.")
    elif mx_geojson is None:
         st.error("Error al cargar el GeoJSON. Asegúrate que el archivo esté en GitHub y se llame 'mexico.json'.")
         
except Exception as e:
    st.error(f"Error crítico al generar la sección geográfica: {e}")
    
st.markdown("---")

# --- SECCIÓN 3: VALIDACIÓN DEL MODELO (t-SNE) ---
st.header("3. Validación y Caracterización del Modelo (t-SNE)")

try:
    st.image(TSNE_PATH, caption="Visualización de Clusters con t-SNE", use_container_width=True) 
except FileNotFoundError:
    st.error(f"Error: No se encontró la imagen del t-SNE en {TSNE_PATH}.")






