import streamlit as st
import pandas as pd
import numpy as np # Necesario si usas np.nan o transformaciones
import requests
import json # Para cargar el GeoJSON
import plotly.express as px # Para crear el mapa

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Perfiles de Riesgo de Suicidio MX", layout="wide")

# --- VARIABLES Y ARCHIVOS ---
K_OPTIMO = 4 
DF_PATH = 'datos_agrupados.parquet'
PERFILES_PATH = 'perfiles.csv'
TSNE_PATH = '13.tsne.PNG' # Cambia este nombre si tu archivo es diferente (ej: 'image_eed6c1.jpg')

# --- FUNCIÓN DE CARGA CACHEADA (Para velocidad) ---
@st.cache_data
def load_data(path):
    # Intentamos cargar el DF completo (el que incluye 'ent_resid' y 'cluster')
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos principal en {path}.")
        return None

df_final = load_data(DF_PATH)

# --- 1. TÍTULO Y EXPLICACIÓN ---
st.title("Sistema de Identificación de Perfiles de Riesgo de Suicidio (2020-2023)")
st.subheader("Modelado no Supervisado (K-Means) en Casos de Suicidio en México")
st.markdown("---")


# --- 2. PERFILES DE RIESGO (La Conclusión) ---
st.header("1. Perfiles de Riesgo Identificados (K=4)")

try:
    # Cargar la tabla de perfiles (el resultado de tu análisis)
    df_perfiles = pd.read_csv(PERFILES_PATH)
    
    # Mostrar la tabla (con el nombre del perfil)
    st.dataframe(
        df_perfiles.style.background_gradient(cmap='YlOrRd', subset=['Tamaño del Cluster']),
        hide_index=True,
        use_container_width=True
    )
    st.caption("Los 4 perfiles se definen por la moda de variables sociodemográficas (sexo, ocupación, horario) y el promedio de la edad.")

except FileNotFoundError:
    st.error(f"⚠️ Error: No se pudo cargar la tabla de perfiles en {PERFILES_PATH}. Asegúrate de que el archivo existe.")
st.markdown("---")


st.header("2. Foco de Intervención Geográfica (Análisis por Entidad)")

if df_final is not None and 'cluster' in df_final.columns and 'ent_resid' in df_final.columns:
    
    # Obtener la lista única de Entidades de Residencia para el selector
    lista_entidades = ['General (Top 5)'] + sorted(df_final['ent_resid'].unique().tolist())
    
    # 1. Selector de Entidad (El elemento interactivo)
    entidad_seleccionada = st.selectbox(
        'Selecciona una Entidad de Residencia para analizar su riesgo:',
        lista_entidades
    )

    if entidad_seleccionada == 'General (Top 5)':
        # Muestra el cálculo anterior (Foco en Cluster 2 - más grande)
        st.markdown("**Foco Primario:** Entidades con mayor concentración del Cluster 2 (Adulto Joven Ocupado)")
        
        # Calcular el Top 5 general del Cluster 2
        df_cluster_2 = df_final[df_final['cluster'] == 2]
        top_entidades_c2 = df_cluster_2['ent_resid'].value_counts().head(5)
        
        st.bar_chart(top_entidades_c2)
        
    else:
        # 2. Análisis por la Entidad Seleccionada
        st.markdown(f"**Distribución de Perfiles de Riesgo en:** **{entidad_seleccionada}**")
        
        # Filtra el DF por la Entidad seleccionada
        df_filtrado = df_final[df_final['ent_resid'] == entidad_seleccionada]
        
        # Calcula el porcentaje de cada cluster en ESA entidad
        distribucion_cluster = df_filtrado['cluster'].value_counts(normalize=True).mul(100).sort_index()
        
        # Muestra el gráfico de distribución
        st.bar_chart(distribucion_cluster)
        st.caption(f"Distribución de los 4 perfiles en {entidad_seleccionada}. La barra más alta indica el perfil de riesgo dominante en ese estado.")

else:
    st.warning("Advertencia: No se pueden mostrar los datos geográficos. Verifica que el archivo PARQUET contenga las columnas 'cluster' y 'ent_resid'.")

# --- 4. VALIDACIÓN DEL MODELO (t-SNE) ---
st.header("3. Validación y Caracterización del Modelo (t-SNE)")

st.markdown("La visualización t-SNE comprime las múltiples dimensiones en dos. La **superposición** de los grupos indica que el modelo es mejor para la segmentación de políticas públicas que para la predicción individual.")

try:
    st.image(TSNE_PATH, caption="Visualización de Clusters con t-SNE", use_container_width=True)
except FileNotFoundError:
    st.error(f"Error: No se encontró la imagen del t-SNE en {TSNE_PATH}. Asegúrate de guardar la imagen con el nombre correcto.")

# ... (resto de tu código de app.py)

GEOJSON_URL = 'https://drive.google.com/uc?id=1mTqwYwgobCnZpdezVfLAxVyHYbV3DQDN&confirm=t'
# ... (carga de datos)

# --- FUNCIÓN DE CARGA CACHEADA para GeoJSON ---
@st.cache_data
def load_geojson(url):
    try:
        # Descarga el archivo de la URL
        response = requests.get(url)
        
        if response.status_code == 200:
            # Si la descarga es exitosa, usamos el contenido (decodificado como UTF-8 si fuera necesario)
            # Como es un JSON, podemos usar response.json()
            return response.json() 
        else:
            # Esto atraparía errores 404, 403, etc.
            st.error(f"Error al descargar GeoJSON. Código HTTP: {response.status_code}. Verifica permisos.")
            return None
    except Exception as e:
        # Esto atrapa errores de conexión
        st.error(f"Error de conexión: {e}")
        return None

# Usamos la URL en lugar de la ruta local
mx_geojson = load_geojson(GEOJSON_URL)

# --- 3. FOCO GEOGRÁFICO ACCIONABLE (MAPA INTERACTIVO) ---
st.header("2. Foco de Intervención Geográfica (Mapa de Riesgo Dominante)")

if df_final is not None and mx_geojson is not None:
    
    # 1. Calcular el CLUSTER DOMINANTE por estado
    # Calcula la moda (el cluster más frecuente) por cada entidad de residencia
    df_mapa = df_final.groupby('ent_resid')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
    df_mapa.rename(columns={'ent_resid': 'CVE_ENT', 'cluster': 'Cluster Dominante'}, inplace=True)
    
    # Mapear el número de cluster a su nombre interpretativo
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
        featureidkey='properties.CVE_ENT', # CRÍTICO: Debe ser el campo en tu GeoJSON
        projection="mercator",
        color_discrete_map={
            '0. Joven Inactivo (Desempleo)': 'yellow',
            '1. Adulto Mayor Ocupado': 'green',
            '2. Adulto Joven Ocupado (Foco)': 'red', # Resaltar el cluster más grande
            '3. Riesgo Desconocido (Madrugada)': 'purple'
        }
    )
    
    # Configuración de mapa (centrado en México)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    
    # 3. Mostrar el mapa interactivo
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada estado está coloreado por el Perfil de Riesgo (Cluster) que es estadísticamente dominante en esa entidad.")
    
    # 4. Sección Interactiva para la Distribución (Mantiene tu selector)
    st.subheader("Análisis Detallado por Entidad")
    # ... (Aquí va tu código anterior del st.selectbox para ver la distribución de clusters por estado)
    # Puedes simplificarlo si el mapa ya es suficiente, o mantenerlo para el detalle.
    
else:
    st.warning("Advertencia: No se pueden mostrar los datos geográficos. Verifica que los archivos y columnas estén presentes.")










