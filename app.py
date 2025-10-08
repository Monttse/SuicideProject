import streamlit as st
import pandas as pd
import numpy as np # Necesario si usas np.nan o transformaciones

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
    st.image(TSNE_PATH, caption="Visualización de Clusters con t-SNE", use_container_width=False)
except FileNotFoundError:
    st.error(f"Error: No se encontró la imagen del t-SNE en {TSNE_PATH}. Asegúrate de guardar la imagen con el nombre correcto.")





